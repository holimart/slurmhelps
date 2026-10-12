"""On-demand GPU Slurm jobs, configured per project from ``.langywrap/lanes.json``.

Moved from langywrap (``langywrap.jobs.slurm_gpu``, ``langywrap jobs gpu``) into slurmhelps;
CLI now: ``python3 -m slurmhelps.slurm_gpu_cli gpu plan|submit|status|wait|cancel``
(with ``../slurmhelps/lib`` on PYTHONPATH). The config location is unchanged.

A project may hold at most ``max_project_jobs`` Slurm jobs whose names match
``count_job_names`` (its long-lived fleet allocation plus at most one on-demand
GPU job). The GPU job is launched to compute one thing and then ends.

Config (``fleet.slurm.gpu`` in ``.langywrap/lanes.json``)::

    {"account": "ffa",
     "partitions": [{"name": "ffa-preempt", "notes": "CPU=0.10, REQUEUE"},
                    {"name": "ffa-checkpoint", "notes": "CPU=0.15, REQUEUE"},
                    {"name": "gpu-ffa", "notes": "full price, not preemptible"}],
     "exclude_nodes": "r50",
     "forbidden_node_patterns": ["fuuk-*"],
     "allowed_partition_patterns": ["*ffa*"],
     "forbidden_path_prefixes": ["/home"],
     "gres": "gpu:1", "cpus": 2, "mem": "8G", "time": "00:05:00",
     "job_name_prefix": "langywrap-gpu-",
     "max_project_jobs": 2,
     "count_job_names": ["langywrap-gpu-*", "<fleet-allocation-name>"],
     "env_setup": "module load cuda", "workdir": ".",
     "log_dir": "/work/ffa/holecem/agents/state/gpu-jobs/myproject",
     "ledger": ".langywrap/gpu-jobs.ndjson"}

Machine and project rules live in lanes.json, not in code:

- ``exclude_nodes``: Slurm hostlist always passed to ``--exclude`` (optional).
- ``forbidden_node_patterns``: fnmatch patterns. Matching nodes are never
  planned on, and the concrete names seen in sinfo are appended to ``--exclude``.
- ``allowed_partition_patterns``: fnmatch patterns. When set, every partition
  must match one of them or the config is refused.
- ``forbidden_path_prefixes``: path prefixes ``log_dir`` must not be under
  (e.g. a project may forbid its home directory). Default empty.

Partition order is the cost order (cheapest first). ``plan`` picks the first
partition with a free matching GPU on a non-excluded node; if none is free it
picks the first partition anyway and reports that the job will queue.
"""

from __future__ import annotations

import fnmatch
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_REL = Path(".langywrap/lanes.json")


def find_project_root(start: Path | str | None = None) -> Path:
    """Walk up from ``start`` (default cwd) to the dir holding ``.langywrap/lanes.json``.

    Falls back to the nearest dir holding ``.git``, then to ``start`` itself.
    """
    here = Path(start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / REGISTRY_REL).is_file():
            return cand
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand
    return here

ACTIVE_STATES = frozenset(
    {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}
)
TERMINAL_HINT = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
)
FREE_NODE_STATES = frozenset({"idle", "mix", "mixed"})

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class SlurmGpuError(RuntimeError):
    """Configuration or Slurm interaction failed; the message says what to fix."""


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    notes: str = ""


@dataclass(frozen=True)
class GpuPolicy:
    project_root: Path
    account: str
    partitions: tuple[PartitionSpec, ...]
    exclude_nodes: str
    forbidden_node_patterns: tuple[str, ...]
    allowed_partition_patterns: tuple[str, ...]
    forbidden_path_prefixes: tuple[str, ...]
    gres: str
    cpus: int
    mem: str
    time: str
    job_name_prefix: str
    max_project_jobs: int
    count_job_names: tuple[str, ...]
    env_setup: str
    workdir: Path
    log_dir: Path
    ledger: Path


@dataclass
class PartitionChoice:
    partition: str
    notes: str
    free_nodes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    partition: str
    notes: str
    queued: bool
    reason: str
    node_candidates: list[str]
    choices: list[PartitionChoice]
    forbidden_nodes: list[str]
    project_jobs: int
    max_project_jobs: int

    @property
    def cap_ok(self) -> bool:
        return self.project_jobs < self.max_project_jobs


# ---------------------------------------------------------------- config


def _req(section: dict[str, Any], key: str) -> Any:
    if key not in section or section[key] in (None, "", []):
        raise SlurmGpuError(f"fleet.slurm.gpu.{key} is required in lanes.json")
    return section[key]


def _str_list(gpu: dict[str, Any], key: str) -> tuple[str, ...]:
    value = gpu.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SlurmGpuError(f"fleet.slurm.gpu.{key} must be a list of strings")
    return tuple(value)


def _under(path: Path, prefix: str) -> bool:
    pre = Path(prefix.rstrip("/") or "/")
    return path == pre or pre in path.parents


def load_policy(project: str | Path | None = None) -> GpuPolicy:
    root = find_project_root(project)
    try:
        doc = json.loads((root / REGISTRY_REL).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SlurmGpuError(f"no {REGISTRY_REL} under {root}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SlurmGpuError(f"cannot read {root / REGISTRY_REL}: {exc}") from exc
    fleet = doc.get("fleet") if isinstance(doc, dict) else None
    slurm = (fleet or {}).get("slurm") if isinstance(fleet, dict) else None
    gpu = (slurm or {}).get("gpu") if isinstance(slurm, dict) else None
    if not isinstance(gpu, dict):
        raise SlurmGpuError(f"{root / REGISTRY_REL} has no fleet.slurm.gpu section")

    raw_parts = _req(gpu, "partitions")
    if not isinstance(raw_parts, list):
        raise SlurmGpuError("fleet.slurm.gpu.partitions must be a list (cheapest first)")
    parts: list[PartitionSpec] = []
    for p in raw_parts:
        spec = (
            PartitionSpec(p, "")
            if isinstance(p, str)
            else PartitionSpec(str(p.get("name", "")), str(p.get("notes", "")))
        )
        if not spec.name:
            raise SlurmGpuError("every fleet.slurm.gpu.partitions entry needs a name")
        parts.append(spec)

    allowed = _str_list(gpu, "allowed_partition_patterns")
    if allowed:
        for spec in parts:
            if not any(fnmatch.fnmatchcase(spec.name, pat) for pat in allowed):
                raise SlurmGpuError(
                    f"partition {spec.name!r} matches no allowed_partition_patterns {list(allowed)}"
                )
    forbidden_nodes = _str_list(gpu, "forbidden_node_patterns")
    forbidden_prefixes = _str_list(gpu, "forbidden_path_prefixes")
    exclude = str(gpu.get("exclude_nodes", "") or "")

    prefix = str(_req(gpu, "job_name_prefix"))
    # Default counts only this module's own on-demand jobs. A project that runs a
    # long-lived fleet allocation lists its name in ``count_job_names`` explicitly.
    count_names = gpu.get("count_job_names") or [f"{prefix}*"]
    log_dir = Path(str(_req(gpu, "log_dir")))
    if not log_dir.is_absolute():
        raise SlurmGpuError("fleet.slurm.gpu.log_dir must be an absolute path")
    for forbidden in forbidden_prefixes:
        if _under(log_dir.resolve(), forbidden):
            raise SlurmGpuError(
                f"fleet.slurm.gpu.log_dir must not be under {forbidden} (forbidden_path_prefixes)"
            )
    workdir = Path(str(gpu.get("workdir", "."))).expanduser()
    if not workdir.is_absolute():
        workdir = root / workdir
    ledger = Path(str(gpu.get("ledger", ".langywrap/gpu-jobs.ndjson")))
    if not ledger.is_absolute():
        ledger = root / ledger

    return GpuPolicy(
        project_root=root,
        account=str(_req(gpu, "account")),
        partitions=tuple(parts),
        exclude_nodes=exclude,
        forbidden_node_patterns=forbidden_nodes,
        allowed_partition_patterns=allowed,
        forbidden_path_prefixes=forbidden_prefixes,
        gres=str(_req(gpu, "gres")),
        cpus=int(_req(gpu, "cpus")),
        mem=str(_req(gpu, "mem")),
        time=str(_req(gpu, "time")),
        job_name_prefix=prefix,
        max_project_jobs=int(gpu.get("max_project_jobs", 2)),
        count_job_names=tuple(str(n) for n in count_names),
        env_setup=str(gpu.get("env_setup", "")),
        workdir=workdir,
        log_dir=log_dir,
        ledger=ledger,
    )


# ---------------------------------------------------------------- parsing helpers


def expand_hostlist(expr: str) -> set[str]:
    """Expand a Slurm hostlist such as ``fuuk-[13-15],r50`` into node names.

    Supports one bracket group per entry (the form Slurm prints for these nodes).
    """
    names: set[str] = set()
    depth, cur, items = 0, "", []
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur)
    for item in (i.strip() for i in items):
        if not item:
            continue
        m = re.fullmatch(r"([^\[\]]*)\[([^\]]+)\](.*)", item)
        if not m:
            names.add(item)
            continue
        pre, inner, post = m.groups()
        for sub in inner.split(","):
            if "-" in sub:
                lo, hi = sub.split("-", 1)
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    names.add(f"{pre}{i:0{width}d}{post}")
            else:
                names.add(f"{pre}{sub}{post}")
    return names


def parse_gpu_counts(text: str) -> dict[str, int]:
    """``gpu:L40:4(S:0),vram:48G`` -> ``{"L40": 4}``; untyped ``gpu:2`` -> ``{"": 2}``."""
    counts: dict[str, int] = {}
    for m in re.finditer(r"(?:^|,)gpu(?::([A-Za-z0-9_.\-]+))?:(\d+)", text):
        typ = m.group(1) or ""
        counts[typ] = counts.get(typ, 0) + int(m.group(2))
    return counts


def parse_request(gres: str) -> tuple[str, int]:
    """``gpu:1`` -> ("", 1); ``gpu:L40:2`` -> ("L40", 2)."""
    m = re.fullmatch(r"gpu(?::([A-Za-z0-9_.\-]+))?:(\d+)", gres.strip())
    if not m:
        raise SlurmGpuError(f"gres {gres!r} must look like 'gpu:1' or 'gpu:L40:1'")
    return (m.group(1) or ""), int(m.group(2))


# ---------------------------------------------------------------- slurm access


def default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    if shutil.which(argv[0]) is None:
        raise SlurmGpuError(f"{argv[0]} not found on PATH (run from a Slurm login node)")
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def list_free_gpu_nodes(
    partition: str,
    want_type: str,
    need: int,
    excluded: set[str],
    forbidden_patterns: tuple[str, ...],
    forbidden_seen: set[str],
    runner: Runner,
) -> list[str]:
    """Nodes in ``partition`` with ``need`` idle GPUs of ``want_type``, minus exclusions.

    Nodes matching ``forbidden_patterns`` are never returned; their names are added
    to ``forbidden_seen`` so submit can pass them to ``--exclude``.
    """
    cp = runner(
        [
            "sinfo",
            "-h",
            "-N",
            "-p",
            partition,
            "-O",
            "NodeList:40,Gres:80,GresUsed:80,StateCompact:20",
        ]
    )
    if cp.returncode != 0:
        raise SlurmGpuError(f"sinfo -p {partition} failed: {cp.stderr.strip()}")
    free: list[str] = []
    for line in cp.stdout.splitlines():
        tok = line.split()
        if len(tok) != 4:
            continue
        node, gres, used, state = tok
        if any(fnmatch.fnmatchcase(node, pat) for pat in forbidden_patterns):
            forbidden_seen.add(node)
            continue
        if node in excluded:
            continue
        if state not in FREE_NODE_STATES:
            continue
        total = parse_gpu_counts(gres)
        busy = parse_gpu_counts(used)
        if want_type:
            avail = total.get(want_type, 0) - busy.get(want_type, 0)
        else:
            avail = sum(total.values()) - sum(busy.values())
        if avail >= need:
            free.append(node)
    return sorted(free)


def count_project_jobs(policy: GpuPolicy, runner: Runner) -> tuple[int, list[str]]:
    cp = runner(["squeue", "-h", "-u", _user(), "-o", "%i|%j|%T"])
    if cp.returncode != 0:
        raise SlurmGpuError(f"squeue failed: {cp.stderr.strip()}")
    matched: list[str] = []
    for line in cp.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        jid, name, state = parts
        if state not in ACTIVE_STATES:
            continue
        if any(fnmatch.fnmatchcase(name, pat) for pat in policy.count_job_names):
            matched.append(f"{jid}:{name}:{state}")
    return len(matched), matched


def plan(policy: GpuPolicy, runner: Runner = default_runner) -> Plan:
    want_type, need = parse_request(policy.gres)
    excluded = expand_hostlist(policy.exclude_nodes)
    forbidden_seen: set[str] = set()
    choices: list[PartitionChoice] = []
    for spec in policy.partitions:
        free = list_free_gpu_nodes(
            spec.name,
            want_type,
            need,
            excluded,
            policy.forbidden_node_patterns,
            forbidden_seen,
            runner,
        )
        choices.append(PartitionChoice(spec.name, spec.notes, free))
    count, _ = count_project_jobs(policy, runner)

    chosen = next((c for c in choices if c.free_nodes), None)
    if chosen is not None:
        queued = False
        reason = (
            f"first partition in cost order with a free matching GPU; "
            f"{len(chosen.free_nodes)} candidate node(s) after exclusions"
        )
    else:
        chosen = choices[0]
        queued = True
        reason = (
            "no partition has a free matching GPU on a non-excluded node; "
            f"using the cheapest ({chosen.partition}), the job will QUEUE"
        )
    return Plan(
        partition=chosen.partition,
        notes=chosen.notes,
        queued=queued,
        reason=reason,
        node_candidates=chosen.free_nodes,
        choices=choices,
        forbidden_nodes=sorted(forbidden_seen),
        project_jobs=count,
        max_project_jobs=policy.max_project_jobs,
    )


def render_plan(policy: GpuPolicy, p: Plan) -> str:
    lines = [
        f"project: {policy.project_root}",
        f"request: {policy.gres}, {policy.cpus} cpu, {policy.mem}, {policy.time}, "
        f"account {policy.account}",
        f"exclude: {exclude_arg(policy, p)}",
        f"project jobs counted: {p.project_jobs}/{p.max_project_jobs} "
        f"({'ok' if p.cap_ok else 'AT CAP - submit would refuse'})",
    ]
    for c in p.choices:
        nodes = ", ".join(c.free_nodes) if c.free_nodes else "none free"
        lines.append(f"  {c.partition} [{c.notes or 'no notes'}]: {nodes}")
    lines.append(f"choice: partition {p.partition}" + (" (QUEUED)" if p.queued else ""))
    lines.append(f"reason: {p.reason}")
    if p.node_candidates:
        lines.append(
            f"node candidates: {', '.join(p.node_candidates)} "
            "(slurm picks the node; --exclude enforces the exclusions)"
        )
    return "\n".join(lines)


def _parse_job_id(text: str) -> str:
    m = re.match(r"\s*(\d+)", text)
    if not m:
        raise SlurmGpuError(f"could not parse job id from sbatch output: {text!r}")
    return m.group(1)


def _append_ledger(policy: GpuPolicy, record: dict[str, Any]) -> None:
    policy.ledger.parent.mkdir(parents=True, exist_ok=True)
    with policy.ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def exclude_arg(policy: GpuPolicy, p: Plan) -> str:
    """``exclude_nodes`` plus every concrete node matched by a forbidden pattern."""
    extra = sorted(set(p.forbidden_nodes) - expand_hostlist(policy.exclude_nodes))
    return ",".join(x for x in [policy.exclude_nodes, ",".join(extra)] if x)


def build_sbatch_argv(
    policy: GpuPolicy, job_name: str, partition: str, script: Path, exclude: str
) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        f"--account={policy.account}",
        f"--partition={partition}",
        f"--gres={policy.gres}",
        *([f"--exclude={exclude}"] if exclude else []),
        f"--cpus-per-task={policy.cpus}",
        f"--mem={policy.mem}",
        f"--time={policy.time}",
        f"--job-name={job_name}",
        f"--output={policy.log_dir / (job_name + '-%j.out')}",
        str(script),
    ]


def submit(
    policy: GpuPolicy, command: list[str], runner: Runner = default_runner, dry_run: bool = False
) -> dict[str, Any]:
    if not command:
        raise SlurmGpuError("submit needs a command after --")
    count, matched = count_project_jobs(policy, runner)
    if count >= policy.max_project_jobs:
        raise SlurmGpuError(
            f"refusing: {count} matching job(s) already active "
            f"(cap {policy.max_project_jobs}): {', '.join(matched)}"
        )
    p = plan(policy, runner)
    job_name = f"{policy.job_name_prefix}{uuid.uuid4().hex[:8]}"

    policy.log_dir.mkdir(parents=True, exist_ok=True)
    script = policy.log_dir / f"{job_name}.sh"
    body = ["#!/bin/bash", "set -o pipefail"]
    if policy.env_setup.strip():
        body.append(policy.env_setup.strip())
    body.append(f"cd {shlex.quote(str(policy.workdir))} || exit 1")
    body.append(shlex.join(command))
    script.write_text("\n".join(body) + "\n", encoding="utf-8")

    argv = build_sbatch_argv(policy, job_name, p.partition, script, exclude_arg(policy, p))
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": "submit",
        "job_name": job_name,
        "partition": p.partition,
        "queued": p.queued,
        "node_candidates": p.node_candidates,
        "command": command,
        "script": str(script),
        "account": policy.account,
        "gres": policy.gres,
        "cpus": policy.cpus,
        "mem": policy.mem,
        "time": policy.time,
        "user": _user(),
        "project": str(policy.project_root),
    }
    if dry_run:
        record.update(dry_run=True, sbatch=argv)
        return record
    cp = runner(argv)
    if cp.returncode != 0:
        raise SlurmGpuError(f"sbatch failed: {(cp.stderr or cp.stdout).strip()}")
    record["job_id"] = _parse_job_id(cp.stdout)
    record["log"] = str(policy.log_dir / f"{job_name}-{record['job_id']}.out")
    _append_ledger(policy, record)
    return record


def status(job_id: str, runner: Runner = default_runner) -> dict[str, Any]:
    cp = runner(["squeue", "-h", "-j", job_id, "-o", "%T|%P|%R|%N"])
    if cp.returncode == 0 and cp.stdout.strip():
        state, part, reason, nodes = (cp.stdout.strip().split("|") + ["", "", "", ""])[:4]
        return {
            "job_id": job_id,
            "state": state,
            "partition": part,
            "reason": reason,
            "node": nodes,
            "source": "squeue",
        }
    cp = runner(["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "State,ExitCode,NodeList"])
    if cp.returncode != 0:
        raise SlurmGpuError(f"sacct failed: {cp.stderr.strip()}")
    line = next((ln for ln in cp.stdout.splitlines() if ln.strip()), "")
    if not line:
        return {"job_id": job_id, "state": "UNKNOWN", "source": "none"}
    state, exit_code, nodes = (line.split("|") + ["", "", ""])[:3]
    return {
        "job_id": job_id,
        "state": state.split()[0] if state else "UNKNOWN",
        "exit_code": exit_code,
        "node": nodes,
        "source": "sacct",
    }


def wait(
    job_id: str,
    timeout_s: float = 3600.0,
    runner: Runner = default_runner,
    sleep: Callable[[float], None] = time.sleep,
    max_interval: float = 60.0,
) -> dict[str, Any]:
    """Poll until the job leaves the queue and sacct reports a terminal state."""
    deadline = time.monotonic() + timeout_s
    interval = 2.0
    while True:
        st = status(job_id, runner)
        state = st["state"]
        if st["source"] == "sacct" and state.split(" ")[0].rstrip("+") in TERMINAL_HINT:
            return st
        if time.monotonic() >= deadline:
            st["timed_out"] = True
            return st
        sleep(interval)
        interval = min(interval * 1.5, max_interval)


def cancel(job_id: str, runner: Runner = default_runner) -> dict[str, Any]:
    cp = runner(["scancel", job_id])
    if cp.returncode != 0:
        raise SlurmGpuError(f"scancel {job_id} failed: {cp.stderr.strip()}")
    return {"job_id": job_id, "cancelled": True}


def record_event(policy: GpuPolicy, event: str, **fields: Any) -> None:
    _append_ledger(
        policy, {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": event, **fields}
    )
