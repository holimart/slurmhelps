#!/usr/bin/env python3
"""Per-project configuration for slurmhelps (stdlib only).

slurmhelps never hardcodes a cluster, a user, a mirror directory or a GPU host: every tool reads
them from the CALLING project. The project root is, in order:

  1. $SLURMHELPS_PROJECT (forwarders and the queue runner set it);
  2. the nearest directory at or above the cwd holding ``slurmhelps.json``;
  3. the nearest directory at or above the cwd holding ``.langywrap/lanes.json`` with a
     top-level ``"slurmhelps"`` block;
  4. ``git rev-parse --show-toplevel`` of the cwd, else the cwd.

The config is ``<root>/slurmhelps.json`` or, if absent, the ``"slurmhelps"`` block of
``<root>/.langywrap/lanes.json``. Keys missing there take the DEFAULTS below. See
``examples/slurmhelps.json`` and ``docs/config.md``.

Shell use (bin/*.sh):

  eval "$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" shell)"
  # -> SH_PROJECT, SH_HOME, SH_SSH, SH_USER, SH_MIRROR, ... (see shell_vars)

  python3 lib/slurmhelps/config.py get cluster.ssh
  python3 lib/slurmhelps/config.py show
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HOME = Path(__file__).resolve().parents[2]  # the slurmhelps checkout
CONFIG_NAME = "slurmhelps.json"
LANES_REL = Path(".langywrap") / "lanes.json"

DEFAULTS: dict[str, Any] = {
    # Slurm cluster reached over ssh (login node). Jobs run in `mirror`, a copy of the project.
    "cluster": {
        "ssh": "",  # user@login-host (required for cluster tools)
        "user": "",  # squeue/sacct -u; default: the part of ssh before '@'
        "account": "",
        "mirror": "",  # project mirror on the cluster (jobs cd here)
        "remote_tmp": "",  # staging dir for job batch dirs
        "slurmhelps_remote": "",  # where bin/hpc_run.sh rsyncs this checkout (lib/ goes on PYTHONPATH)
        "venv": ".venv",  # mirror-relative venv; slurmhelps.pth is installed into it
        "job_prefix": "sh-",  # our job names start with this
        "max_jobs": 1,  # at most this many of our jobs queued or running
        "gres": "gpu:1",
        "cpus": 4,
        "mem": "16G",
        "threads": 4,  # OMP/MKL threads exported in the job setup
        "env": {},  # extra env exported in the job setup (e.g. a CuPy pool limit)
        "allowed_partitions": [],
        "partition_rank": {},
        "exclude_node_prefixes": [],
        "exclude_nodes": [],
        "exclude_gpus": [],
        "gpu_rank": {},
        "poll_slack_s": 7200,
        "sacct_since": "",  # hpc_minutes default -S
    },
    # Where the project keeps cluster job dirs, chain files and logs (project-relative).
    "hpc_dir": "outputs/hpc",
    # sshfs mount of the mirror (bin/mirror_mount.sh); remote defaults to cluster.ssh:cluster.mirror
    "mount": {"remote": "", "mountpoint": ""},
    # LAN GPU workstation used by bin/gpu_run.sh and the queue runner.
    "lan_gpu": {
        "host": "",
        "base": "",  # host dir holding .venv/, repo/, runs/
        "ssh_key": "",  # explicit key (lane sandboxes have no ~/.ssh view); "" = ssh default
        "known_hosts": ".langywrap/gpu-known-hosts",  # project-relative unless absolute
        "threads": 2,
        "gpu": "0",
    },
    # The lane GPU queue (bin/gpu_queue.py) and its cluster bridge (bin/hpc_bridge_gpu_queue.py).
    "queue": {
        "dir": ".langywrap/gpu-queue",
        "bridge_note": "ran on the cluster via bridge",
        "bridge_default_minutes": 90,
        "bridge_max_sync_gb": 2,
        "lan_host_label": "LAN GPU host",
    },
}


def _git_toplevel(start: Path) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None


def _lanes_block(root: Path) -> dict | None:
    try:
        doc = json.loads((root / LANES_REL).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = doc.get("slurmhelps") if isinstance(doc, dict) else None
    return block if isinstance(block, dict) else None


def project_root(start: str | Path | None = None) -> Path:
    env = os.environ.get("SLURMHELPS_PROJECT")
    if env and start is None:
        return Path(env).resolve()
    here = Path(start or os.getcwd()).resolve()
    for d in (here, *here.parents):
        if (d / CONFIG_NAME).is_file():
            return d
    for d in (here, *here.parents):
        if _lanes_block(d) is not None:
            return d
    return _git_toplevel(here) or here


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(root: str | Path | None = None) -> dict[str, Any]:
    r = Path(root).resolve() if root is not None else project_root()
    path = r / CONFIG_NAME
    raw: dict = {}
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = _lanes_block(r) or {}
    cfg = _merge(DEFAULTS, {k: v for k, v in raw.items() if not k.startswith("_")})
    c = cfg["cluster"]
    if not c["user"] and "@" in c["ssh"]:
        c["user"] = c["ssh"].split("@", 1)[0]
    m = cfg["mount"]
    if not m["remote"] and c["ssh"] and c["mirror"]:
        m["remote"] = f"{c['ssh']}:{c['mirror']}"
    kh = cfg["lan_gpu"]["known_hosts"]
    if kh and not os.path.isabs(kh):
        cfg["lan_gpu"]["known_hosts"] = str(r / kh)
    cfg["_root"] = str(r)
    cfg["_home"] = str(HOME)
    return cfg


def get(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def require(cfg: dict, *dotted: str) -> None:
    missing = [d for d in dotted if get(cfg, d) in ("", None, [], {})]
    if missing:
        raise SystemExit(
            f"slurmhelps: {', '.join(missing)} not set in {cfg['_root']}/{CONFIG_NAME} "
            f"(or the 'slurmhelps' block of .langywrap/lanes.json)"
        )


def shell_vars(cfg: dict) -> dict[str, str]:
    c, g, q = cfg["cluster"], cfg["lan_gpu"], cfg["queue"]

    def words(xs: list) -> str:
        return " ".join(str(x) for x in xs)

    def ranks(d: dict) -> str:
        return " ".join(f"{k}={v}" for k, v in d.items())

    return {
        "SH_PROJECT": cfg["_root"],
        "SH_HOME": cfg["_home"],
        "SH_HPC_DIR": cfg["hpc_dir"],
        "SH_SSH": c["ssh"],
        "SH_USER": c["user"],
        "SH_ACCOUNT": c["account"],
        "SH_MIRROR": c["mirror"],
        "SH_REMOTE_TMP": c["remote_tmp"],
        "SH_REMOTE_HOME": c["slurmhelps_remote"],
        "SH_VENV": c["venv"],
        "SH_PREFIX": c["job_prefix"],
        "SH_MAX_JOBS": str(c["max_jobs"]),
        "SH_GRES": c["gres"],
        "SH_CPUS": str(c["cpus"]),
        "SH_MEM": c["mem"],
        "SH_THREADS": str(c["threads"]),
        "SH_ENV": " ".join(f"{k}={shlex.quote(str(v))}" for k, v in c["env"].items()),
        "SH_PARTITIONS": words(c["allowed_partitions"]),
        "SH_PARTITION_RANK": ranks(c["partition_rank"]),
        "SH_EXCLUDE_PREFIXES": words(c["exclude_node_prefixes"]),
        "SH_EXCLUDE_NODES": ",".join(c["exclude_nodes"]),
        "SH_EXCLUDE_GPUS": words(c["exclude_gpus"]),
        "SH_GPU_RANK": ranks(c["gpu_rank"]),
        "SH_POLL_SLACK": str(c["poll_slack_s"]),
        "SH_MOUNT_REMOTE": cfg["mount"]["remote"],
        "SH_MOUNT_POINT": os.path.expanduser(cfg["mount"]["mountpoint"]),
        "SH_GPU_HOST": g["host"],
        "SH_GPU_BASE": g["base"],
        "SH_GPU_SSH_KEY": g["ssh_key"],
        "SH_GPU_KNOWN_HOSTS": g["known_hosts"],
        "SH_GPU_THREADS": str(g["threads"]),
        "SH_GPU_DEVICE": str(g["gpu"]),
        "SH_QUEUE_DIR": q["dir"],
    }


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "show"
    cfg = load()
    if cmd == "shell":
        for k, v in shell_vars(cfg).items():
            print(f"{k}={shlex.quote(v)}")
        return 0
    if cmd == "get" and len(argv) == 2:
        v = get(cfg, argv[1])
        print(json.dumps(v) if isinstance(v, (dict, list)) else v)
        return 0
    if cmd == "show":
        print(json.dumps(cfg, indent=2))
        return 0
    print("usage: config.py [show|shell|get KEY.PATH]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
