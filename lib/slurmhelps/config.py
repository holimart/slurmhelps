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
  python3 lib/slurmhelps/config.py mounts   # name<TAB>remote<TAB>mountpoint<TAB>mode
"""

from __future__ import annotations

import copy
import json
import os
import re
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
        "nice": None,  # #SBATCH --nice for every job (>= 0), None = unset
        "sync_always": [],  # project-relative paths hpc_chain.sh ships with EVERY job (e.g. shared parent arms)
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
    # Named sshfs mounts (bin/mirror_mount.sh NAME): {"name": {"remote": "user@host:/dir",
    # "mountpoint": "~/mnt/x", "mode": "rw"|"ro"}}. The site profile's MOUNT_<name> add more.
    "mounts": {},
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
        "bridge_gpu_vram_gib": None,  # conservative capacity of ALL bridge-eligible GPUs; constrained jobs fail closed until configured
        "bridge_max_sync_gb": 2,
        "lan_host_label": "LAN GPU host",
    },
    # ---- cluster-side tools (fleet/, image/, bootstrap/, maint/; read through lib/common.sh) ----
    # Paths ON the cluster. Placeholders: {work_root}, {user}. See docs/config.md "site".
    "site": {
        "work_root": "",  # per-user root on the shared parallel FS (e.g. /work/ffa/{user}); never /home
        "agents_dir": "{work_root}/agents",  # fleet logs + state (alloc.env, store snapshots)
        "langywrap_checkout": "{work_root}/langywrap",  # shared by every project; never in an image
        "env_sh": "{work_root}/config/env.sh",  # toolchain + caches profile (env/env.sh via shim)
        "user_bin": "{work_root}/bin",
        "images_dir": "{work_root}/images",  # squashfs working-set + toolchain images
        "fleet_home": "{work_root}/home/compute",  # HOME of every fleet process
        "pi_agent_dir": "{work_root}/home/pi/.pi/agent",
        "codex_home": "{work_root}/home/codex",
        "xdg_cache_home": "{work_root}/cache/xdg",
        "node_tmp": "/tmp",  # node-local (job-private tmpfs on Troja: RAM, wiped at job end)
        "slurm_bin": "",
        "partition_long": "",  # the fleet allocation's partition (default for fleet.partition)
        "partition_short": "",  # helper jobs: builds, tests, image builds
        "short_time": 110,  # minutes
        "short_cpus": 4,
        "short_mem": "16G",
        "toolchain_profile": "",  # site toolchain shell profile sourced by env/env.sh
        "toolchain_dirs": [],  # user toolchain dirs relative to work_root (toolchain image)
        "node_version": "",
        "mpi_env": {},  # exported for every fleet process after env.sh (MPI landmines)
        "login_host_pinned": "",  # the login node holding FUSE views (workset.sh view)
        "cluster_host_glob": "",  # hostnames that mean "on the cluster" (workstation tools refuse)
    },
    # The renewable fleet allocation (fleet/supervisor.sbatch, fleet/ly, lane launchers).
    "fleet": {
        "project": "",  # default: the project root
        "jobname": "agent-supervisor",
        "partition": "",  # default: site.partition_long
        "time": "7-00:00:00",
        "renew_signal": 3600,  # seconds before the limit: USR1 -> self-requeue
        "cpus": 0,
        "mem": "",
        "extra_writable": "{work_root}/home",  # lane sandbox writable roots (colon list)
        "work_checkout_var": "",  # project-named alias of SH_WORK_CHECKOUT (node-local mode)
        "lane_model": "",
        "lane_prompt_file": "",  # project-relative; {lane} {model} replaced
        "lane_exclude": "steward",
    },
    # Defaults for the working-set images (image/workset.py); a project's
    # .langywrap/lanes.json fleet.workset block overrides them key by key.
    "workset": {},
    # Disaster recovery (bootstrap/*.sh): how to recreate the project, its lane
    # worktrees and the langywrap checkout from zero. See docs/config.md "bootstrap".
    "bootstrap": {
        "project_name": "",  # checkout dir $work_root/<name>; default: basename of the project root
        "origin": "",  # git URL of the project (clone + push)
        "upstream": "",  # optional second remote `upstream`
        "branch": "",  # branch to clone
        "lane_worktree_dir": ".lane-worktrees",  # lane worktrees inside the checkout
        "worktrees_file": "",  # `checkouts.sh snapshot` output; "" = lane/<id> on origin
        "langywrap_origin": "",
        "langywrap_branch": "master",
        "langywrap_submodules": ["pi", "rtk"],  # what the fleet needs (pi: node build; rtk: cargo)
        "pi_upstream": "",  # where the pi release tag (versions.lock PI_RELEASE_TAG) is fetched
        "lock_file": "{slurmhelps}/bootstrap/versions.lock",
        "env_local": [],  # lines of $work_root/config/env.local.sh (written only if absent)
        "bin_shims": {},  # {"name": "target"}: $work_root/bin/<name> exec shims; target
        #                   project-relative unless absolute; {slurmhelps} expands
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


def _subst(value: Any, work_root: str = "") -> Any:
    """Expand ``{user}``, ``$USER``, ``{home}``, ``$HOME`` and (when known)
    ``{work_root}``/``$WORK_ROOT`` in a config string.

    Values that legitimately contain shell syntax (``cluster.env``,
    ``site.mpi_env``, ``bootstrap.env_local``) are never passed through here.
    """
    if not isinstance(value, str):
        return value
    user = os.environ.get("USER", "")
    home = os.environ.get("HOME", str(Path.home()))
    out = (value.replace("${USER}", user).replace("${HOME}", home)
                .replace("{user}", user).replace("{home}", home))
    if work_root:
        out = out.replace("${WORK_ROOT}", work_root).replace("{work_root}", work_root)
    out = re.sub(r"\$USER\b", lambda _: user, out)
    out = re.sub(r"\$HOME\b", lambda _: home, out)
    if work_root:
        out = re.sub(r"\$WORK_ROOT\b", lambda _: work_root, out)
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
    _expand_placeholders(cfg)
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


def _expand_placeholders(cfg: dict) -> None:
    """Resolve placeholders in the site/fleet sections, then in the workstation-side keys.

    A public example config can therefore say ``${USER}@cluster`` and
    ``${WORK_ROOT}/mirror`` instead of embedding a real account and path.
    """
    site = cfg["site"]
    site["work_root"] = _subst(site.get("work_root") or "")
    wr = site["work_root"]

    for sec in ("site", "fleet"):
        for k, v in list(cfg[sec].items()):
            if isinstance(v, list):
                cfg[sec][k] = [_subst(x, wr) for x in v]
            else:
                cfg[sec][k] = _subst(v, wr)
    if not cfg["fleet"]["partition"]:
        cfg["fleet"]["partition"] = site["partition_long"]

    c = cfg["cluster"]
    for k in ("ssh", "user", "account", "mirror", "remote_tmp", "slurmhelps_remote"):
        c[k] = _subst(c[k], wr)
    m = cfg["mount"]
    for k in ("remote", "mountpoint"):
        m[k] = _subst(m[k], wr)
    for ent in (cfg.get("mounts") or {}).values():
        if isinstance(ent, dict):
            for k in ("remote", "mountpoint"):
                if k in ent:
                    ent[k] = _subst(ent[k], wr)
    g = cfg["lan_gpu"]
    for k in ("host", "base", "ssh_key", "known_hosts"):
        g[k] = _subst(g[k], wr)


def site_vars(cfg: dict) -> dict[str, str]:
    """The cluster-side view (lib/common.sh sh_load evals these)."""
    s, f = cfg["site"], cfg["fleet"]
    out = {
        "SH_WORK_ROOT": s["work_root"],
        "SH_AGENTS_DIR": s["agents_dir"],
        "SH_LANGYWRAP_CHECKOUT": s["langywrap_checkout"],
        "SH_ENV_SH": s["env_sh"],
        "SH_USER_BIN": s["user_bin"],
        "SH_IMAGES_DIR": s["images_dir"],
        "SH_FLEET_HOME": s["fleet_home"],
        "SH_PI_AGENT_DIR": s["pi_agent_dir"],
        "SH_CODEX_HOME": s["codex_home"],
        "SH_XDG_CACHE_HOME": s["xdg_cache_home"],
        "SH_NODE_TMP": s["node_tmp"],
        "SH_SLURM_BIN": s["slurm_bin"],
        "SH_PARTITION_LONG": s["partition_long"],
        "SH_PARTITION_SHORT": s["partition_short"],
        "SH_SHORT_TIME": str(s["short_time"]),
        "SH_SHORT_CPUS": str(s["short_cpus"]),
        "SH_SHORT_MEM": s["short_mem"],
        "SH_TOOLCHAIN_PROFILE": s["toolchain_profile"].replace("{slurmhelps}", str(HOME)),
        "SH_TOOLCHAIN_DIRS": " ".join(s["toolchain_dirs"]),
        "SH_NODE_VERSION": s["node_version"],
        "SH_MPI_ENV": " ".join(f"{k}={shlex.quote(str(v))}" for k, v in s["mpi_env"].items()),
        "SH_LOGIN_HOST_PINNED": s["login_host_pinned"],
        "SH_CLUSTER_HOST_GLOB": s["cluster_host_glob"],
        "SH_FLEET_PROJECT": f["project"] or cfg["_root"],
        "SH_FLEET_JOBNAME": f["jobname"],
        "SH_FLEET_PARTITION": f["partition"],
        "SH_FLEET_TIME": f["time"],
        "SH_FLEET_RENEW_SIGNAL": str(f["renew_signal"]),
        "SH_FLEET_CPUS": str(f["cpus"] or ""),
        "SH_FLEET_MEM": f["mem"],
        "SH_EXTRA_WRITABLE": f["extra_writable"],
        "SH_WORK_CHECKOUT_VAR": f["work_checkout_var"],
        "SH_LANE_MODEL": f["lane_model"],
        "SH_LANE_PROMPT_FILE": f["lane_prompt_file"],
        "SH_LANE_EXCLUDE": f["lane_exclude"],
    }
    return out


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
        "SH_SYNC_ALWAYS": words(c["sync_always"]),
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
        "SH_NICE": "" if c.get("nice") is None else str(int(c["nice"])),
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


def named_mounts(cfg: dict) -> list[tuple[str, str, str, str]]:
    """(name, remote, mountpoint, mode) for every entry of the ``mounts`` object."""
    out = []
    for name, m in sorted((cfg.get("mounts") or {}).items()):
        if str(name).startswith("_"):
            continue  # comment keys
        if not isinstance(m, dict):
            raise SystemExit(f"slurmhelps: mounts.{name} must be an object")
        out.append(
            (
                str(name),
                str(m.get("remote", "")),
                os.path.expanduser(str(m.get("mountpoint", ""))),
                str(m.get("mode", "rw")),
            )
        )
    return out


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "show"
    cfg = load()
    if cmd == "shell":
        for k, v in shell_vars(cfg).items():
            print(f"{k}={shlex.quote(v)}")
        return 0
    if cmd == "site-shell":  # cluster-side view (lib/common.sh)
        for k, v in {**shell_vars(cfg), **site_vars(cfg)}.items():
            print(f"{k}={shlex.quote(v)}")
        return 0
    if cmd == "get" and len(argv) == 2:
        v = get(cfg, argv[1])
        print(json.dumps(v) if isinstance(v, (dict, list)) else v)
        return 0
    if cmd == "mounts":
        for row in named_mounts(cfg):
            print("\t".join(row))
        return 0
    if cmd == "show":
        print(json.dumps(cfg, indent=2))
        return 0
    print("usage: config.py [show|shell|site-shell|mounts|get KEY.PATH]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
