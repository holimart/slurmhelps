#!/usr/bin/env python3
"""GPU job queue for lanes that cannot reach the GPU host themselves.

Lane tool environments mask the user's SSH identity ("Identity file ... not accessible"), so
lanes cannot run bin/gpu_run.sh directly. Instead they SUBMIT a job description here and the
steward's runner (started once, outside any lane, with SSH access) executes jobs one at a time
through bin/gpu_run.sh and writes the results back. The queue belongs to the CALLING project
(project root and queue dir from slurmhelps.json, see lib/slurmhelps/config.py); run it from the
project root, where slurmhelps is the sibling checkout ../slurmhelps:

  # lane side
  uv run python ../slurmhelps/bin/gpu_queue.py submit --name screen-rfb4 [--priority high|normal|low] \
      [--sync DIR ...] [--pull DIR ...] \
      -- python scripts/eval_runset.py --fast-numpy --fast-backend cupy --jobs 1 ...
  uv run python ../slurmhelps/bin/gpu_queue.py status [--name screen-rfb4]   # pending / running / done + exit
  uv run python ../slurmhelps/bin/gpu_queue.py log --name screen-rfb4         # tail of the job's output

  # steward side (one process, nohup)
  uv run python ../slurmhelps/bin/gpu_queue.py run

Queue dirs: <queue.dir>/{pending,running,done}/<id>.json (+ <id>.log), default
.langywrap/gpu-queue. Jobs run in
submission order, serially (the GPU host is a shared workstation). A job's command runs inside
the host's repo mirror exactly as with gpu_run.sh; --sync/--pull are passed through. Lanes
may list their inputs with a project helper (whiteboxchallenge: scripts/gpu_screen_files.py --list)
and pass the listed dirs as --sync, or simply --sync the directories they need.

Submit-time checks (2026-10 hardening, after a 36 h audit of 41 failed jobs):
  * every --sync path must exist locally (relative to the repo root), else submit exits 2 with
    the missing paths; --allow-missing-sync skips this for a path a queued job will produce first;
  * a command whose first token is a .py file gets `python` prepended (the host mirror keeps git's
    0644 mode, so a bare `scripts/x.py` fails with exit 126); a note is printed to stderr;
  * files named by `--candidate X` / `--baseline X` (or `--candidate=X`) that exist locally are
    hashed into the optional record field `input_sha256` {path: sha256}; at run time the runner
    re-hashes them and writes a WARNING line to the job log for any file that changed or vanished.
Run-time: the runner executes a per-job snapshot copy of gpu_run.sh (running/<id>.sh), so editing
bin/gpu_run.sh while a job is in flight cannot break it. Records without the new fields run
exactly as before.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

SLURMHELPS_HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SLURMHELPS_HOME / "lib"))
from slurmhelps import config as _config
from slurmhelps import gpu_memory

CONFIG = _config.load()
ROOT = Path(CONFIG["_root"])
Q = ROOT / CONFIG["queue"]["dir"]
GPU_RUN = SLURMHELPS_HOME / "bin" / "gpu_run.sh"


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def dirs() -> dict[str, Path]:
    out = {k: Q / k for k in ("pending", "running", "done")}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


HASHED_FLAGS = ("--candidate", "--baseline")

# Submit-time dedupe (2026-10-11, after 15 input groups were resubmitted with identical inputs,
# 30 extra submits, 10 of them after a failure). A submission is refused when a job with the same
# input_sha256 AND command is pending/running, or finished OK within DEDUPE_OK_H, or FAILED within
# DEDUPE_FAIL_H. Each lane may also hold at most MAX_PENDING_PER_LANE pending jobs. --force skips
# both.
DEDUPE_OK_H = 24
DEDUPE_FAIL_H = 6
MAX_PENDING_PER_LANE = 4
EXIT_REFUSED = 4


def normalize_command(command: list[str]) -> list[str]:
    """Prepend `python` when the command starts with a bare .py file (no interpreter)."""
    if command and command[0].endswith(".py"):
        return ["python", *command]
    return list(command)


def missing_sync_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if not (ROOT / p).exists()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed_inputs(command: list[str]) -> list[str]:
    """Paths given to --candidate / --baseline in the command (both `--f X` and `--f=X`)."""
    out = []
    for i, tok in enumerate(command):
        for flag in HASHED_FLAGS:
            if tok == flag and i + 1 < len(command):
                out.append(command[i + 1])
            elif tok.startswith(flag + "="):
                out.append(tok[len(flag) + 1 :])
    return out


def input_hashes(command: list[str]) -> dict[str, str]:
    out = {}
    for rel in hashed_inputs(command):
        path = ROOT / rel
        if path.is_file():
            out[rel] = sha256_file(path)
    return out


def input_drift(job: dict) -> list[str]:
    """Warning lines for --candidate/--baseline files that changed since submit."""
    warnings = []
    for rel, want in (job.get("input_sha256") or {}).items():
        path = ROOT / rel
        if not path.is_file():
            warnings.append(f"# gpu_queue WARNING: input {rel} hashed at submit is now missing")
            continue
        got = sha256_file(path)
        if got != want:
            warnings.append(
                f"# gpu_queue WARNING: input {rel} changed since submit "
                f"(sha256 {want[:12]} at submit, {got[:12]} now)"
            )
    return warnings


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def dedupe_key(job: dict) -> tuple[str, tuple[str, ...]]:
    return (
        json.dumps(job.get("input_sha256") or {}, sort_keys=True),
        tuple(job.get("command") or ()),
    )


def submit_refusal(job: dict, rows: list[dict], lane: str, now: dt.datetime) -> str | None:
    """Why this submission must be refused (None = accept). `rows` come from _load_all()."""
    key = dedupe_key(job)
    for r in rows:
        if dedupe_key(r) != key:
            continue
        if r["state"] in ("pending", "running"):
            return f"the same inputs and command are already {r['state']} as {r['id']}"
        finished = _parse_ts(r.get("finished"))
        if finished is None:
            continue
        age_h = (now - finished).total_seconds() / 3600
        if r.get("exit") == 0 and age_h < DEDUPE_OK_H:
            return (
                f"the same inputs and command already finished OK as {r['id']} "
                f"{age_h:.1f} h ago (window {DEDUPE_OK_H} h)"
            )
        if r.get("exit") != 0 and age_h < DEDUPE_FAIL_H:
            return (
                f"the same inputs and command FAILED (exit {r.get('exit')}) as {r['id']} "
                f"{age_h:.1f} h ago (window {DEDUPE_FAIL_H} h)"
            )
    pending = sum(1 for r in rows if r["state"] == "pending" and r.get("lane", "") == lane)
    if pending >= MAX_PENDING_PER_LANE:
        return (
            f"lane {lane!r} already has {pending} pending jobs "
            f"(cap {MAX_PENDING_PER_LANE} per lane)"
        )
    return None


def cmd_submit(a: argparse.Namespace) -> None:
    if not a.command:
        raise SystemExit("give the command after `--`")
    memory = {
        k: getattr(a, k) for k in gpu_memory.FIELDS if getattr(a, k, None) is not None
    }
    try:
        gpu_memory.resources({"gpu_memory": memory})
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if "cupy_pool_limit_gib" in memory:
        for i, token in enumerate(a.command):
            value = None
            if token == "--cupy-pool-limit-gb" and i + 1 < len(a.command):
                value = a.command[i + 1]
            elif token.startswith("--cupy-pool-limit-gb="):
                value = token.split("=", 1)[1]
            if value is not None and float(value) != memory["cupy_pool_limit_gib"]:
                raise SystemExit("command pool contradicts GPU memory metadata")
    syncs = a.sync or []
    missing = missing_sync_paths(syncs)
    if missing and not getattr(a, "allow_missing_sync", False):
        print(
            "gpu_queue submit: --sync path(s) do not exist locally (relative to "
            f"{ROOT}): {', '.join(missing)}\n"
            "  create them first, or pass --allow-missing-sync if an earlier queued job "
            "produces them",
            file=sys.stderr,
        )
        raise SystemExit(2)
    command = normalize_command(a.command)
    if command != list(a.command):
        print(
            f"gpu_queue submit: {a.command[0]} has no interpreter; running it as "
            f"`python {a.command[0]}`",
            file=sys.stderr,
        )
    a.command = command
    d = dirs()
    lane = a.lane or os.environ.get("LANGYWRAP_LANE", "")
    hashes = input_hashes(a.command)
    if not getattr(a, "force", False):
        reason = submit_refusal(
            {"input_sha256": hashes, "command": a.command},
            _load_all(),
            lane,
            dt.datetime.now(dt.UTC),
        )
        if reason:
            print(
                f"gpu_queue submit refused: {reason}.\n"
                "  wait for the earlier job, change its inputs or command, or pass --force",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED)
    # Priority is the first character of the id: the runner picks the lexically smallest pending
    # file, so "0" (high: short screens) runs before "5" (normal) before "9" (low: bulk truth
    # shards).
    prefix = {"high": "0", "normal": "5", "low": "9"}[a.priority]
    job_id = f"{prefix}-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%S')}-{a.name}"
    job = {
        "id": job_id,
        "name": a.name,
        "lane": lane,
        "sync": syncs,
        "pull": a.pull or [],
        "command": a.command,
        "submitted": now(),
    }
    if memory:
        job["gpu_memory"] = memory
    if hashes:
        job["input_sha256"] = hashes
    (d["pending"] / f"{job_id}.json").write_text(json.dumps(job, indent=1))
    print(
        json.dumps(
            {"submitted": job_id, "position": len(list(d["pending"].glob("*.json")))}
        )
    )


def _load_all() -> list[dict]:
    d = dirs()
    rows = []
    for state in ("running", "pending", "done"):
        for p in sorted(d[state].glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            j["state"] = state
            rows.append(j)
    return rows


def cmd_status(a: argparse.Namespace) -> None:
    rows = [r for r in _load_all() if not a.name or r["name"] == a.name]
    if not a.all:
        rows = [
            r
            for r in rows
            if r["state"] != "done"
            or r.get("finished", "")
            >= (dt.datetime.now(dt.UTC) - dt.timedelta(hours=12)).isoformat()
        ]
    for r in rows:
        print(
            f"{r['state']:8s} {r['id']:48s} exit={r.get('exit', '-')!s:4s} "
            f"lane={r.get('lane', '')} started={r.get('started', '-')} "
            f"finished={r.get('finished', '-')}"
            + (f" via={r['claimed_by']}" if r.get("claimed_by") else "")
            + (f" gpu_memory={r['gpu_memory']}" if r.get("gpu_memory") else "")
        )
    if not rows:
        print("no jobs")


def cmd_log(a: argparse.Namespace) -> None:
    d = dirs()
    logs = sorted(
        list(d["done"].glob(f"*-{a.name}.log")) + list(d["running"].glob(f"*-{a.name}.log"))
    )
    if not logs:
        raise SystemExit(f"no log for {a.name}")
    text = logs[-1].read_text(errors="replace").splitlines()
    print("\n".join(text[-a.tail :]))


# Claim protocol (shared with bin/hpc_bridge_gpu_queue.py, 2026-10-10). A job is claimed by an
# atomic rename pending/<id>.json -> running/<id>.json: exactly one of the local runner and the
# cluster bridge wins; the loser gets None and moves on. A claim may carry `claimed_by` (the bridge
# sets "hpc-bridge"); the local runner's restart re-queue leaves such claims alone (the bridge
# recovers its own stale claims under its lock). release_job undoes a claim (back to pending);
# finish_job moves it to done/ with an exit code.
CLAIM_FIELDS = ("started", "claimed_by", "bridge_stage", "bridge_job")


def claim_job(d: dict[str, Path], job_id: str, claimed_by: str | None = None) -> dict | None:
    src = d["pending"] / f"{job_id}.json"
    dst = d["running"] / f"{job_id}.json"
    try:
        os.rename(src, dst)
    except FileNotFoundError:
        return None
    job = json.loads(dst.read_text())
    job["started"] = now()
    if claimed_by:
        job["claimed_by"] = claimed_by
    dst.write_text(json.dumps(job, indent=1))
    return job


def release_job(d: dict[str, Path], job: dict, reason: str = "") -> None:
    job_id = job["id"]
    for k in CLAIM_FIELDS:
        job.pop(k, None)
    if reason:
        job["released"] = {"at": now(), "reason": reason}
    run_json = d["running"] / f"{job_id}.json"
    run_json.write_text(json.dumps(job, indent=1))
    os.rename(run_json, d["pending"] / f"{job_id}.json")
    (d["running"] / f"{job_id}.log").unlink(missing_ok=True)


def finish_job(d: dict[str, Path], job: dict, code: int, log_text: str | None = None) -> None:
    job_id = job["id"]
    job["finished"] = now()
    job["exit"] = code
    (d["done"] / f"{job_id}.json").write_text(json.dumps(job, indent=1))
    run_log = d["running"] / f"{job_id}.log"
    if log_text is not None:
        (d["done"] / f"{job_id}.log").write_text(log_text)
        run_log.unlink(missing_ok=True)
    elif run_log.exists():
        run_log.rename(d["done"] / f"{job_id}.log")
    (d["running"] / f"{job_id}.json").unlink(missing_ok=True)


def lan_device():
    """Bounded read-only probe; no sync, claim or remote filesystem mutation."""
    try:
        r = subprocess.run(
            ["bash", str(GPU_RUN), "--memory-status"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=25,
            check=True,
        )
        return gpu_memory.parse_probe(r.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def pick_for_lan(jobs):
    if not any(j.get("gpu_memory") for j in jobs):
        return jobs
    device = lan_device()
    return gpu_memory.eligible_in_order(jobs, device, phase="execute")


def run_one(job: dict, d: dict[str, Path]) -> None:
    job_id = job["id"]
    run_log = d["running"] / f"{job_id}.log"
    if job.get("gpu_memory") and not pick_for_lan([job]):
        return
    claimed = claim_job(d, job_id)
    if claimed is None:  # the cluster bridge (or another runner) took it first
        print(f"{now()} skip {job_id}: claimed elsewhere", flush=True)
        return
    job = claimed
    # Run a per-job snapshot of gpu_run.sh: bash reads scripts incrementally, so an edit to the
    # live file mid-run used to break the job (exit 2). GPU_RUN_ROOT keeps the repo root.
    snap = d["running"] / f"{job_id}.sh"
    shutil.copy2(GPU_RUN, snap)
    env = {
        **os.environ,
        "GPU_RUN_ROOT": str(ROOT),
        "SLURMHELPS_HOME": str(SLURMHELPS_HOME),
    }
    argv = ["bash", str(snap), "--name", job["name"]]
    for s in job.get("sync", []):
        argv += ["--sync", s]
    for p in job.get("pull", []):
        argv += ["--pull", p]
    if job.get("gpu_memory"):
        argv += [
            "--gpu-memory-json",
            json.dumps(job["gpu_memory"], separators=(",", ":")),
        ]
    argv += ["--", *normalize_command(job["command"])]
    with run_log.open("w") as fh:
        fh.write(f"# gpu_queue run {job['started']} {shlex.join(argv)}\n")
        for line in input_drift(job):
            fh.write(line + "\n")
        fh.flush()
        try:
            proc = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=4 * 3600,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            fh.write("# gpu_queue: job exceeded 4 h and was stopped\n")
        finally:
            snap.unlink(missing_ok=True)
    finish_job(d, job, code)
    print(f"{job['finished']} done {job_id} exit {code}", flush=True)


def cmd_run(a: argparse.Namespace) -> None:
    d = dirs()
    pid_file = Q / "runner.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"{now()} gpu_queue runner pid {os.getpid()} watching {Q}", flush=True)
    # Jobs left in running/ by a previous runner are re-queued, not lost.
    # Claims held by the cluster bridge are not ours: the bridge recovers its own stale claims.
    for p in d["running"].glob("*.json"):
        try:
            if json.loads(p.read_text()).get("claimed_by"):
                continue
        except json.JSONDecodeError:
            pass
        p.rename(d["pending"] / p.name)
    while True:
        pending = sorted(d["pending"].glob("*.json"))
        if not pending:
            time.sleep(a.poll)
            continue
        jobs = []
        for path in pending:
            try:
                jobs.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, FileNotFoundError):
                if path.exists():
                    path.rename(d["done"] / (path.stem + ".broken.json"))
        eligible = pick_for_lan(jobs)
        if not eligible:
            time.sleep(a.poll)
            continue
        run_one(eligible[0], d)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--lane")
    for field in gpu_memory.FIELDS:
        s.add_argument("--" + field.replace("_", "-"), type=float, default=None)
    s.add_argument(
        "--priority",
        choices=("high", "normal", "low"),
        default="normal",
        help="high = short screens (minutes) run before normal; low = bulk truth shards run last",
    )
    s.add_argument("--sync", action="append")
    s.add_argument("--pull", action="append")
    s.add_argument(
        "--allow-missing-sync",
        action="store_true",
        help="do not reject --sync paths missing locally (an earlier job makes them)",
    )
    s.add_argument(
        "--force",
        action="store_true",
        help="skip the duplicate check and the per-lane pending cap (exit 4 otherwise)",
    )
    s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("status")
    s.add_argument("--name")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("log")
    s.add_argument("--name", required=True)
    s.add_argument("--tail", type=int, default=40)
    s.set_defaults(fn=cmd_log)
    s = sub.add_parser("run")
    s.add_argument("--poll", type=float, default=10.0)
    s.set_defaults(fn=cmd_run)
    a = p.parse_args()
    if a.cmd == "submit" and a.command and a.command[0] == "--":
        a.command = a.command[1:]
    a.fn(a)


if __name__ == "__main__":
    main()
