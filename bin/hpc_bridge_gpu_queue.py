#!/usr/bin/env python3
"""Run the oldest pending LANE GPU job (bin/gpu_queue.py) on the Slurm cluster when idle.

Configuration (cluster ssh/user/mirror/prefix/venv/threads, hpc_dir, queue.*) comes from the calling
project's slurmhelps.json (lib/slurmhelps/config.py). Paths below use whiteboxchallenge's values
(hpc_dir outputs/hpc, prefix wb-, mirror /work/ffa/holecem/whitebox, LAN host = the 1080 Ti).

Called by bin/hpc_chain.sh when chain.tsv has no pending line and outputs/hpc/bridge.enabled
exists (the steward switches the bridge on/off with that file). One call = at most one job:

  1. lock .langywrap/gpu-queue/bridge.lock (one bridge at a time) and recover stale bridge claims
     left by a bridge that died (not yet submitted -> back to pending; submitted -> done, exit 125);
  2. if any wb-* job of ours is queued/running on the cluster, do nothing (exit 10);
  3. atomically CLAIM the oldest pending job (by `submitted`) with gpu_queue.claim_job (rename
     pending -> running, claimed_by=hpc-bridge), so the local 1080 Ti runner cannot also take it;
  4. refuse (release back to pending, with a `released` note) a job whose --sync paths are missing
     locally or whose rsync transfer to the mirror would exceed 2 GB; a job refused in the last 6 h
     is skipped when picking;
  5. rsync scripts/ + the job's --sync paths (only those) to the mirror /work/ffa/holecem/whitebox
     (same relative paths, as gpu_run.sh does on the 1080 Ti host), then run it through
     bin/hpc_run.sh as wb-gq-<id> with the job's minutes (field minutes/limit_minutes, else 90);
     the command runs in the mirror root with the mirror .venv first on PATH; its exit code goes to
     {output} (-> outputs/hpc/<job>/eval.json) and its log to outputs/hpc/<job>/remote.log;
  6. rsync each --pull dir back to the same local path (where gpu_run.sh would have put it) and
     finish the job in done/ with the remote exit code and note 'ran on troja L40S via bridge'.
Failures before submission release the job. Owned submissions without a terminal receipt remain
reserved for explicit reconciliation; they are never automatically resubmitted.

  python3 ../slurmhelps/bin/hpc_bridge_gpu_queue.py            # run one job (exit 0 ran, 10 nothing to do)
  python3 ../slurmhelps/bin/hpc_bridge_gpu_queue.py --dry-run  # show the plan for the oldest pending job

For a reviewed spare-capacity dispatch, run this helper independently of the chain with
--active-job-limit 4 (for example, leaving two slots of a six-job fleet reserved).
It runs at most one job under the same bridge lock and atomic LAN claim.
Add --concurrent to opt into independently running workers sharing a short admission lock,
durable inherited owner leases and not-yet-visible slot reservations. Default remains serial.
See docs/concurrent-bridge.md for the reviewed rollout and reconciliation boundary. The configured
cluster.max_jobs remains the upper bound; hpc_run also performs its normal admission check.
The default chain invocation remains idle-only. Running this as a separate worker keeps
the chain free to notice new steward rows while this helper waits for its single job.

Long-job gate: a job whose expected runtime (declared minutes, else the median bridge run time of
its name family in outputs/hpc/bridge.log, else unknown = short) exceeds LONG_JOB_MINUTES is only
picked while the steward chain is idle (see chain_recently_active); otherwise the oldest short job
is taken, and if there is none the bridge exits 10.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import re
import shlex
import signal
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_queue  # noqa: E402
import bridge_dispatch as dispatch  # noqa: E402

WORKER_LEASE_FD = None

CONFIG = gpu_queue.CONFIG
ROOT = gpu_queue.ROOT
HPC_RUN = gpu_queue.SLURMHELPS_HOME / "bin" / "hpc_run.sh"
_C = CONFIG["cluster"]
SSH = _C["ssh"]
USER = _C["user"]
MIRROR = _C["mirror"]
PREFIX = _C["job_prefix"]
VENV = _C["venv"]
THREADS = int(_C["threads"])
HPC_DIR = CONFIG["hpc_dir"]
GPU_HOST_REPO = f"{CONFIG['lan_gpu']['base']}/repo"
GPU_HOST_VENV = f"{CONFIG['lan_gpu']['base']}/.venv"
CLAIMED_BY = "hpc-bridge"
NOTE = CONFIG["queue"]["bridge_note"]
DEFAULT_MINUTES = int(CONFIG["queue"]["bridge_default_minutes"])
MAX_SYNC_GB = float(CONFIG["queue"]["bridge_max_sync_gb"])
MAX_SYNC_BYTES = int(MAX_SYNC_GB * 1024**3)
REFUSAL_COOLDOWN_H = 6
FAIRNESS_WINDOW_H = 1
EXIT_NOTHING = 10
LONG_JOB_MINUTES = 30  # expected runtime above this = a long job
CHAIN_TSV_ACTIVE_MIN = 45  # chain.tsv modified more recently than this => steward chain active
CHAIN_FINISHED_ACTIVE_MIN = 20  # chain.log's last 'finished' line more recent than this => active
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


class Refuse(Exception):
    """The job cannot run on the cluster now: release it back to pending."""


class Interrupted(Exception):
    pass


# ---- pure parts ------------------------------------------------------------------------------


def terminal_exit_code(result: dict) -> int:
    """Exit code from a cluster eval.json, or KeyError when it carries none.

    A normal finish writes ``exit``. A job stopped by Slurm (time limit, scancel) runs the
    hpc_run.sh TERM trap, whose ``hpc_partial.py finalize-failure`` writes ``exit_code`` and
    ``terminated: true`` instead. That is a terminal receipt too: reading only ``exit`` left
    every timed-out job as an orphan reservation (whiteboxchallenge 2026-10-18/19, 3 jobs).
    """
    if "exit" in result:
        return int(result["exit"])
    if "exit_code" in result and (result.get("terminated") or result.get("status") == "failed"):
        return int(result["exit_code"])
    raise KeyError("exit")


def job_minutes(job: dict) -> int:
    for key in ("minutes", "limit_minutes", "timeout_minutes"):
        v = job.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return DEFAULT_MINUTES


def cluster_job_name(job: dict) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", job["id"])
    return ("gq-" + name)[:80]


def to_mirror_relative(token: str, local_root: Path = ROOT) -> str:
    """Rewrite absolute local-repo / 1080 Ti host paths inside one argv token to repo-relative."""
    token = token.replace(f"{GPU_HOST_VENV}/bin/", "")
    for prefix in (str(local_root).rstrip("/") + "/", GPU_HOST_REPO + "/"):
        token = token.replace(prefix, "")
    return token


def translate_command(command: list[str], local_root: Path = ROOT) -> list[str]:
    """The job's command as it must run in the mirror root (python resolves to the mirror venv)."""
    cmd = list(command)
    if cmd[:2] == ["uv", "run"]:
        cmd = cmd[2:]
    cmd = gpu_queue.normalize_command(cmd)
    cmd = [to_mirror_relative(t, local_root) for t in cmd]
    # The mirror has one venv (.venv); local lanes also call the starter-kit venv, absent there.
    if cmd and cmd[0] in (
        "python3",
        ".venv/bin/python",
        ".venv/bin/python3",
        "whest-starterkit/.venv/bin/python",
        "whest-starterkit/.venv/bin/python3",
    ):
        cmd[0] = "python"
    return cmd


def remote_run_cmd(job: dict, local_root: Path = ROOT) -> str:
    """The hpc_run.sh RUNCMD (runs in MIRROR after the setup cmd; writes the exit to {output})."""
    jn = cluster_job_name(job)
    logdir = f"{HPC_DIR}/{jn}"
    mk = " ".join(shlex.quote(p) for p in [logdir, *job.get("pull", [])])
    command = translate_command(job["command"], local_root)
    if job.get("gpu_memory"):
        command = [
            "python",
            "-m",
            "slurmhelps.gpu_memory",
            "--requirements",
            json.dumps(job["gpu_memory"], separators=(",", ":")),
            "--",
            *command,
        ]
    inner = shlex.join(command)
    return (
        f"mkdir -p {mk}; rc=0; "
        f"env PATH={MIRROR}/{VENV}/bin:$PATH OMP_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS} "
        f"bash -c {shlex.quote(inner)} > {logdir}/remote.log 2>&1 || rc=$?; "
        'printf \'{"exit": %d, "node": "%s"}\\n\' "$rc" "$(hostname)" > {output}'
    )


def declared_minutes(job: dict) -> int | None:
    for key in ("minutes", "limit_minutes", "timeout_minutes"):
        v = job.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def job_family(name: str) -> str:
    """Name family: strip a trailing -sNN shard and a trailing -<hex hash>.

    'corpus-grower-organiser128-dc759413-s01' -> 'corpus-grower-organiser128'.
    """
    fam = re.sub(r"-s\d+$", "", name)
    return re.sub(r"-[0-9a-f]{6,}$", "", fam)


def _job_name(job: dict) -> str:
    if job.get("name"):
        return job["name"]
    parts = str(job.get("id", "")).split("-", 2)
    return parts[2] if len(parts) == 3 else str(job.get("id", ""))


def bridge_runtime_history(root: Path | None = None) -> dict[str, float]:
    """Median wall minutes per job family, from bridge.log submit..done pairs."""
    log = (root or ROOT) / HPC_DIR / "bridge.log"
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return {}
    submit_re = re.compile(r"^(\S+) bridge submit (\S+) as \S+ \(")
    done_re = re.compile(r"^(\S+) bridge done (\S+) exit ")
    started: dict[str, dt.datetime] = {}
    durations: dict[str, list[float]] = {}
    for line in lines:
        m = submit_re.match(line)
        if m:
            t0 = _parse_ts(m.group(1))
            if t0 is not None:
                started[m.group(2)] = t0
            continue
        m = done_re.match(line)
        if m and m.group(2) in started:
            t0 = started.pop(m.group(2))
            t1 = _parse_ts(m.group(1))
            if t1 is None or t1 < t0:
                continue
            name = _job_name({"id": m.group(2)})
            durations.setdefault(job_family(name), []).append((t1 - t0).total_seconds() / 60)
    return {fam: statistics.median(v) for fam, v in durations.items()}


def _parse_ts(text: str) -> dt.datetime | None:
    try:
        t = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def expected_minutes(job: dict, history: dict[str, float]) -> float | None:
    """Declared minutes, else the family's past median, else None (unknown)."""
    dm = declared_minutes(job)
    if dm is not None:
        return float(dm)
    return history.get(job_family(_job_name(job)))


def is_long(job: dict, history: dict[str, float]) -> bool:
    exp = expected_minutes(job, history)
    return exp is not None and exp > LONG_JOB_MINUTES


def chain_recently_active(now: dt.datetime | None = None, root: Path | None = None) -> bool:
    """True if outputs/hpc/chain.tsv changed within CHAIN_TSV_ACTIVE_MIN, or chain.log's last
    'finished' line (local HH:MM) is within CHAIN_FINISHED_ACTIVE_MIN."""
    now = now or dt.datetime.now(dt.UTC)
    hpc = (root or ROOT) / HPC_DIR
    try:
        age = now.timestamp() - (hpc / "chain.tsv").stat().st_mtime
        if age < CHAIN_TSV_ACTIVE_MIN * 60:
            return True
    except OSError:
        pass
    try:
        lines = (hpc / "chain.log").read_text(errors="replace").splitlines()
    except OSError:
        return False
    fin_re = re.compile(r"^(\d{2}):(\d{2}) finished\b")
    for line in reversed(lines):
        m = fin_re.match(line)
        if not m:
            continue
        local = now.astimezone()
        when = local.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if when > local:  # the log has no date: a later clock time means yesterday
            when -= dt.timedelta(days=1)
        return local - when < dt.timedelta(minutes=CHAIN_FINISHED_ACTIVE_MIN)
    return False


def pick_oldest(
    pending: list[dict],
    now: dt.datetime | None = None,
    recent_lanes: set[str] | None = None,
    *,
    skip_long: bool = False,
    history: dict[str, float] | None = None,
) -> list[dict]:
    """Pending jobs minus those the bridge refused recently, oldest `submitted` first.

    Fairness: jobs whose lane had a bridged job in `recent_lanes` (the last FAIRNESS_WINDOW_H) go
    after jobs from lanes that have not, so one busy lane cannot starve the others.
    skip_long: drop jobs with is_long(job, history) (the steward chain is active).
    """
    now = now or dt.datetime.now(dt.UTC)
    recent_lanes = recent_lanes or set()
    history = history or {}
    out = []
    for j in pending:
        rel = j.get("released") or {}
        if rel.get("by") == CLAIMED_BY and rel.get("at"):
            try:
                at = dt.datetime.fromisoformat(rel["at"])
                if now - at < dt.timedelta(hours=REFUSAL_COOLDOWN_H):
                    continue
            except ValueError:
                pass
        if skip_long and is_long(j, history):
            continue
        out.append(j)
    return sorted(
        out,
        key=lambda j: (j.get("lane", "") in recent_lanes, j.get("submitted", ""), j["id"]),
    )


def recent_bridged_lanes(d: dict[str, Path], now: dt.datetime | None = None) -> set[str]:
    """Lanes with a job the bridge finished (claimed_by hpc-bridge) within FAIRNESS_WINDOW_H."""
    now = now or dt.datetime.now(dt.UTC)
    lanes = set()
    for p in d["done"].glob("*.json"):
        try:
            j = json.loads(p.read_text())
            finished = dt.datetime.fromisoformat(j["finished"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            continue
        if j.get("claimed_by") != CLAIMED_BY:
            continue
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=dt.UTC)
        if now - finished < dt.timedelta(hours=FAIRNESS_WINDOW_H):
            lanes.add(j.get("lane", ""))
    return lanes


def pick_for_bridge(d: dict[str, Path]) -> list[dict]:
    """Candidates in pick order, with the long-job gate applied from the current chain state."""
    ordered = pick_oldest(
        load_pending(d),
        recent_lanes=recent_bridged_lanes(d),
        skip_long=chain_recently_active(),
        history=bridge_runtime_history(),
    )
    device = {"total_gib": CONFIG["queue"].get("bridge_gpu_vram_gib")}
    return gpu_queue.gpu_memory.eligible_in_order(ordered, device, phase="route")


def load_pending(d: dict[str, Path]) -> list[dict]:
    rows = []
    for p in d["pending"].glob("*.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def set_stage(d: dict[str, Path], job: dict, stage: str, **extra) -> None:
    job["bridge_stage"] = stage
    job.update(extra)
    dispatch.atomic_json(d["running"] / f"{job['id']}.json", job)


def release(d: dict[str, Path], job: dict, reason: str) -> None:
    job.pop("bridge_owner", None)
    gpu_queue.release_job(d, job, reason)
    # mark who released it so pick_oldest can cool it down
    p = d["pending"] / f"{job['id']}.json"
    j = json.loads(p.read_text())
    j["released"]["by"] = CLAIMED_BY
    p.write_text(json.dumps(j, indent=1))


def fail(d: dict[str, Path], job: dict, code: int, reason: str, log: list[str]) -> None:
    job["bridge_error"] = reason
    job["note"] = f"{NOTE} (failed: {reason})"
    log.append(f"# hpc_bridge: FAILED exit {code}: {reason}")
    gpu_queue.finish_job(d, job, code, "\n".join(log) + "\n")


def ambiguous_failure(d, job, stage, code, reason, log):
    """A missing terminal receipt cannot prove an accepted allocation stopped."""
    if job.get("bridge_owner") is not None:
        set_stage(d, job, stage, bridge_error=reason, bridge_reconcile_required=True)
        return
    fail(d, job, code, reason, log)


def recover_stale(d: dict[str, Path], out=print, *, legacy_live=False) -> None:
    """Respect live/unknown owners; ambiguous submitted work is never requeued."""
    for p in sorted(d["running"].glob("*.json")):
        try:
            job = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if job.get("claimed_by") != CLAIMED_BY:
            continue
        owner = job.get("bridge_owner")
        if owner is not None:
            status = dispatch.owner_status(d, owner)
            if status == "unknown":
                raise RuntimeError("bridge owner liveness unknown: " + job["id"])
            if status == "alive":
                continue
            if job.get("bridge_stage") in ("submitted", "pulling"):
                out("orphan submitted reservation retained for explicit reconciliation: " + job["id"])
                continue
        elif legacy_live:
            continue  # An older serial worker holds bridge.lock for its whole run.
        stage = job.get("bridge_stage", "claimed")
        if stage not in ("submitted", "pulling"):
            release(d, job, f"stale bridge claim (stage {stage}) released")
            out(f"recovered {job['id']}: released (stage {stage})")
        else:
            fail(
                d,
                job,
                125,
                f"bridge died at stage {stage}; cluster job {PREFIX}{job.get('bridge_job', '?')} may "
                f"have run, see {HPC_DIR}/{job.get('bridge_job', '?')}",
                [f"# hpc_bridge stale-claim recovery {gpu_queue.now()}"],
            )
            out(f"recovered {job['id']}: marked failed (stage {stage})")


# ---- side-effecting parts --------------------------------------------------------------------


def ssh(cmd: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, SSH, cmd], capture_output=True, text=True, timeout=timeout
    )


def our_cluster_job_names() -> list[str]:
    r = ssh(f'squeue -h -u {USER} -o "%j"')
    if r.returncode != 0:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()}")
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


#: Slurm states that mean the allocation has ended for good (sacct State, first word).
SLURM_TERMINAL = {
    "TIMEOUT": 124,
    "CANCELLED": 130,
    "FAILED": 1,
    "OUT_OF_MEMORY": 137,
    "NODE_FAIL": 125,
    "PREEMPTED": 125,
    "BOOT_FAIL": 125,
    "DEADLINE": 124,
}


def slurm_terminal_code(cluster_name: str) -> int | None:
    """Exit code for a cluster job that Slurm reports as ended badly, else None.

    Used when no exit file came back: an allocation Slurm itself reports as TIMEOUT /
    CANCELLED / FAILED is proof it stopped, so the job is finished instead of being kept as an
    orphan reservation. Any doubt (ssh error, still running, unknown state) returns None.
    """
    try:
        r = ssh(f"sacct -n -X --name {cluster_name} -o State --parsable2", timeout=60)
    except Exception:  # noqa: BLE001 - any doubt keeps the old, conservative reservation
        return None
    if r.returncode != 0:
        return None
    states = [line.split()[0] for line in r.stdout.splitlines() if line.strip()]
    if not states:
        return None
    return SLURM_TERMINAL.get(states[-1].rstrip("+"))


def our_cluster_jobs() -> int:
    r = ssh(f'squeue -h -u {USER} -o "%j"')
    if r.returncode != 0:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()}")
    return sum(1 for line in r.stdout.splitlines() if line.startswith(PREFIX))


def rsync_up_argv(paths: list[str], dry: bool) -> list[str]:
    argv = ["rsync", "-azR", "--timeout=600", "-e", "ssh " + " ".join(SSH_OPTS)]
    if dry:
        argv += ["--dry-run", "--stats"]
    return [*argv, *paths, f"{SSH}:{MIRROR}/"]


def sync_paths(job: dict) -> list[str]:
    return ["scripts/", *[p.rstrip("/") for p in job.get("sync", [])]]


def transfer_bytes(job: dict) -> int:
    """Bytes rsync would send to the mirror (delta vs what is already there)."""
    r = subprocess.run(
        rsync_up_argv(sync_paths(job), dry=True),
        **dispatch.inherited_lease(WORKER_LEASE_FD),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if r.returncode != 0:
        raise Refuse(f"rsync size check failed (rc {r.returncode}): {r.stderr.strip()[-300:]}")
    m = re.search(r"Total transferred file size: ([\d,.]+)", r.stdout)
    if not m:
        raise Refuse("rsync size check: no 'Total transferred file size' in output")
    return int(m.group(1).replace(",", "").replace(".", ""))


def pull_back(job: dict, log: list[str]) -> None:
    jn = cluster_job_name(job)
    items = [*job.get("pull", []), f"{HPC_DIR}/{jn}"]
    for rel in items:
        rel = rel.rstrip("/")
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "rsync",
                "-az",
                "--timeout=600",
                "-e",
                "ssh " + " ".join(SSH_OPTS),
                f"{SSH}:{MIRROR}/{rel}/",
                f"{ROOT / rel}/",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
            **dispatch.inherited_lease(WORKER_LEASE_FD),
        )
        if r.returncode != 0:
            log.append(f"# pull of {rel} failed (rsync exit {r.returncode}): {r.stderr.strip()}")
        else:
            log.append(f"# pulled {rel}")


def describe(job: dict) -> list[str]:
    jn = cluster_job_name(job)
    return [
        f"job        : {job['id']}",
        f"lane       : {job.get('lane') or '-'}, submitted {job.get('submitted')}",
        f"cluster    : {PREFIX}{jn}, {job_minutes(job)} min, results {HPC_DIR}/{jn}/",
        f"command    : {shlex.join(job['command'])[:400]}",
        f"remote cmd : {remote_run_cmd(job)[:600]}",
        f"sync up    : {', '.join(sync_paths(job))[:600]}",
        f"pull back  : {', '.join(job.get('pull', [])) or '(none)'} + {HPC_DIR}/{jn}",
    ]


def run_job(d: dict[str, Path], job: dict, out=print) -> int:
    """Run one claim; ambiguous owned submissions remain reserved for reconciliation."""
    jn = cluster_job_name(job)
    log = [f"# hpc_bridge run {gpu_queue.now()} (cluster, {PREFIX}{jn})", *gpu_queue.input_drift(job)]
    stage = "claimed"
    try:
        set_stage(d, job, "claimed", bridge_job=jn)
        missing = gpu_queue.missing_sync_paths(job.get("sync", []))
        if missing:
            raise Refuse(f"--sync path(s) missing locally: {', '.join(missing)[:400]}")
        nbytes = transfer_bytes(job)
        if nbytes > MAX_SYNC_BYTES:
            raise Refuse(f"inputs to sync are {nbytes / 1e9:.2f} GB (> {MAX_SYNC_GB:g} GB)")
        stage = "syncing"
        set_stage(d, job, "syncing")
        r = subprocess.run(
            rsync_up_argv(sync_paths(job), dry=False),
            **dispatch.inherited_lease(WORKER_LEASE_FD),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3 * 3600,
        )
        if r.returncode != 0:
            raise Refuse(f"rsync up failed (rc {r.returncode}): {r.stderr.strip()[-300:]}")
        log.append(f"# synced {nbytes} bytes to {MIRROR}")
        res = ROOT / HPC_DIR / jn / "eval.json"
        res.parent.mkdir(parents=True, exist_ok=True)
        res.unlink(missing_ok=True)
        stage = "submitted"
        set_stage(d, job, "submitted")
        submit_log = ROOT / HPC_DIR / f"{jn}.submit.log"
        mins = job_minutes(job)
        out(f"{gpu_queue.now()} bridge submit {job['id']} as {PREFIX}{jn} ({mins} min)")
        with submit_log.open("w") as fh:
            rc = subprocess.run(
                ["bash", str(HPC_RUN), jn, str(mins), remote_run_cmd(job), "--yes"],
                cwd=ROOT,
                stdout=fh,
                stderr=subprocess.STDOUT,
                **dispatch.inherited_lease(WORKER_LEASE_FD),
            ).returncode
        stext = submit_log.read_text(errors="replace")
        if rc == 3 or "no suitable idle GPU node" in stext:
            stage = "claimed"  # nothing was submitted
            raise Refuse(f"cluster busy/no GPU (hpc_run rc {rc})")
        stage = "pulling"
        set_stage(d, job, "pulling")
        pull_back(job, log)
        rlog = ROOT / HPC_DIR / jn / "remote.log"
        if rlog.exists():
            lines = rlog.read_text(errors="replace").splitlines()
            log.append(f"# ---- remote log ({HPC_DIR}/{jn}/remote.log, last 200 lines) ----")
            log += lines[-200:]
            log.append("# ---- end remote log ----")
        try:
            result = json.loads(res.read_text())
            code = terminal_exit_code(result)
        except (OSError, ValueError, KeyError, TypeError):
            slurm_code = slurm_terminal_code(PREFIX + jn)
            if slurm_code is not None:
                reason = (
                    f"no exit file; Slurm reports the allocation ended (exit {slurm_code}), "
                    f"see {HPC_DIR}/{jn}.submit.log"
                )
                job.pop("bridge_stage", None)
                fail(d, job, slurm_code, reason, log)
                out(f"{gpu_queue.now()} bridge done {job['id']} exit {slurm_code} (slurm terminal)")
                return slurm_code
            ambiguous_failure(
                d,
                job,
                stage,
                rc or 125,
                f"no exit file from the cluster (hpc_run rc {rc}; slurm timeout/cancel?), "
                f"see {HPC_DIR}/{jn}.submit.log",
                log,
            )
            return rc or 125
        job["note"] = f"{NOTE} (node {result.get('node', '?')})"
        job["bridge_submit_log"] = f"{HPC_DIR}/{jn}.submit.log"
        for k in ("bridge_stage",):
            job.pop(k, None)
        log.append(f"gpu job {job['name']} exit {code} ({job['note']})")
        gpu_queue.finish_job(d, job, code, "\n".join(log) + "\n")
        out(f"{gpu_queue.now()} bridge done {job['id']} exit {code}")
        return code
    except Refuse as e:
        if stage in ("claimed", "syncing"):
            release(d, job, str(e))
            out(f"{gpu_queue.now()} bridge released {job['id']}: {e}")
            return -1
        ambiguous_failure(d, job, stage, 125, str(e), log)
        return 125
    except BaseException as e:  # noqa: BLE001 - preserve ambiguous submitted reservations
        reason = f"{type(e).__name__}: {e}"
        if stage in ("claimed", "syncing"):
            release(d, job, f"bridge error before submission: {reason}")
        else:
            ambiguous_failure(d, job, stage, 125,
                              f"bridge error at stage {stage}: {reason}", log)
        if isinstance(e, (Interrupted, KeyboardInterrupt)):
            raise
        out(f"{gpu_queue.now()} bridge error {job['id']}: {reason}")
        return 125


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan; claim nothing")
    ap.add_argument("--concurrent", action="store_true", help="opt-in short admission lock and durable worker reservations; default remains serial")
    ap.add_argument(
        "--active-job-limit", type=int, default=1,
        help="opt in to dispatch below this active wb-job count (default: idle only); cannot exceed cluster.max_jobs",
    )
    ap.add_argument(
        "--no-size-check", action="store_true", help="dry-run: skip the rsync size probe"
    )
    a = ap.parse_args(argv)
    if not 1 <= a.active_job_limit <= int(_C["max_jobs"]):
        ap.error("--active-job-limit must be between 1 and cluster.max_jobs")
    d = gpu_queue.dirs()

    if a.dry_run:
        cands = pick_for_bridge(d)
        if not cands:
            print("no pending gpu_queue job (or only long jobs while the chain is active)")
            return EXIT_NOTHING
        job = cands[0]
        print("[dry-run] oldest pending gpu_queue job (nothing claimed):")
        for line in describe(job):
            print("  " + line)
        exp = expected_minutes(job, bridge_runtime_history())
        print(f"  expected   : {'unknown (short)' if exp is None else f'{exp:.0f} min'}")
        missing = gpu_queue.missing_sync_paths(job.get("sync", []))
        print(f"  missing    : {', '.join(missing) or 'none'}")
        if not a.no_size_check and not missing:
            try:
                n = transfer_bytes(job)
                verdict = "OK" if n <= MAX_SYNC_BYTES else f"REFUSE (> {MAX_SYNC_GB:g} GB)"
                print(f"  to sync    : {n} bytes ({n / 1e6:.1f} MB) -> {verdict}")
            except Refuse as e:
                print(f"  to sync    : {e}")
        print(
            f"  would run  : {HPC_RUN} {cluster_job_name(job)} {job_minutes(job)} "
            "'<remote cmd>' --yes"
        )
        return 0

    return dispatch_one(d, a.active_job_limit, concurrent=a.concurrent)


def dispatch_one(d, limit, *, concurrent=False):
    global WORKER_LEASE_FD
    legacy = None
    if not concurrent:
        legacy = (gpu_queue.Q / "bridge.lock").open("a")
        try:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            legacy.close()
            print("another bridge is running")
            return EXIT_NOTHING

    def on_signal(signum, _frame):
        raise Interrupted(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    try:
        with dispatch.WorkerLease(d) as worker:
            WORKER_LEASE_FD = worker.fd
            for _attempt in range(3):
                try:
                    with dispatch.admission_lock(d):
                        if concurrent:
                            with dispatch.legacy_probe(d) as legacy_live:
                                recover_stale(d, legacy_live=legacy_live)
                                reserved = dispatch.claims(d)
                                occupied = dispatch.occupied_slots(
                                    our_cluster_job_names(), reserved, PREFIX, cluster_job_name)
                        else:
                            recover_stale(d)
                            reserved = dispatch.claims(d)
                            occupied = our_cluster_jobs() + len(reserved)
                        if occupied >= limit:
                            print(f"{occupied} cluster allocations/reservations; bridge limit {limit}, idle")
                            return EXIT_NOTHING
                        job = None
                        reserved_names = {cluster_job_name(row) for row in reserved}
                        for candidate in pick_for_bridge(d):
                            if cluster_job_name(candidate) in reserved_names:
                                continue  # Sanitization/truncation collision shares output paths.
                            job = gpu_queue.claim_job(d, candidate["id"], CLAIMED_BY)
                            if job is not None:
                                break
                        if job is None:
                            return EXIT_NOTHING
                        set_stage(d, job, "claimed", bridge_job=cluster_job_name(job),
                                  bridge_owner=worker.owner)
                except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    print(f"cannot reserve bridge capacity ({exc}); idle")
                    return EXIT_NOTHING
                if run_job(d, job) != -1:
                    return 0
            return EXIT_NOTHING
    finally:
        WORKER_LEASE_FD = None
        if legacy is not None:
            legacy.close()


if __name__ == "__main__":
    sys.exit(main())
