#!/usr/bin/env python3
"""Report stray processes of this user on the current host (read-only).

Steward check, run every tick via a project's `.langywrap/lanes.json`
fleet.commands (inside the Slurm allocation) and by the steward on the login
node. It never kills anything. Source of truth: slurmhelps fleet/stray-check.py.

Configuration (environment, normally from lib/common.sh sh_load):
  SH_WORK_ROOT    the per-user root on the shared filesystem (orphans whose
                  command line mentions it are flagged)
  AGENT_PROJECT   the fleet project; its daemon (and the node-local clone's
                  daemon, /tmp/<name>-<uid>/<name>) is never flagged

Classes flagged (thresholds in seconds of process age):
  pytest        any pytest on a login node; in the allocation, more than one at
                once or one older than --pytest-max-age
  test-daemon   `langywrap.sessions.supervisor --serve` for a pytest/tmp/worktree
                project (not the fleet daemon) older than --test-daemon-age
  poller        shell wait loops (`until`/`while` ... `sleep`) older than
                --poller-age; these polled Lustre for 26 h on 2026-10-11
  scan          ugrep / grep -r / find / du / git fsck / git log --all older
                than --scan-age
  opencode      `opencode serve` (not part of the fleet)
  stale-srun    `srun --jobid` steps (ly exec) older than --srun-age
  orphan        python/bash/node children of init (ppid 1) older than
                --orphan-age whose command line mentions $SH_WORK_ROOT
  d-state       processes in uninterruptible sleep (informational only)

Exit status: 0 clean, 1 strays found (so the steward tick reports it),
2 internal error.  --json prints a machine-readable report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")
WORK = os.environ.get("SH_WORK_ROOT") or os.environ.get("WORK_ROOT") or "/work"
PROJECT = os.environ.get("AGENT_PROJECT", "").rstrip("/")
PROJECT_NAME = os.path.basename(PROJECT) if PROJECT else ""

# Never flagged: interactive tools, the fleet itself, and lane solver runs.
ALLOW = [
    re.compile(p)
    for p in (
        r"\.vscode-server",
        r"(^|/)claude( |$)",
        r"^htop",
        r"systemd",
        r"sshd",
        r"tmux",
        r"slurmstepd",
        r"--mode rpc",
        r"stray-check\.py",
    )
]
if PROJECT:
    ALLOW.append(re.compile(r"langywrap\.sessions\.supervisor --serve " + re.escape(PROJECT) + r"( |$)"))
    # node-local working set (fleet.node_local_checkout): the fleet daemon
    # serves the node-local clone; must not be mistaken for a /tmp test daemon
    ALLOW.append(re.compile(r"langywrap\.sessions\.supervisor --serve /tmp/" + re.escape(PROJECT_NAME)
                            + r"-[0-9]+/" + re.escape(PROJECT_NAME) + r"( |$)"))

SCAN = re.compile(
    r"(^|/)(ugrep|rg)( |$)|(^|/)grep .*-[a-zA-Z]*r|(^|/)find /work|(^|/)du -s|git fsck|git log --all|git rev-list --all"
)
POLLER = re.compile(r"\b(until|while)\b.*\bsleep\b", re.S)
TEST_DAEMON = re.compile(r"langywrap\.sessions\.supervisor --serve (/tmp/|\S*pytest|\S*langywrap-worktrees)")


def is_login_node() -> bool:
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return "hpc" in socket.gethostname() or not os.path.exists("/var/spool/slurmd")


def boot_time() -> float:
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("btime "):
                return float(line.split()[1])
    return 0.0


def processes(uid: int):
    btime = boot_time()
    now = time.time()
    me = {os.getpid(), os.getppid()}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        p = int(pid)
        try:
            if os.stat(f"/proc/{p}").st_uid != uid or p in me:
                continue
            with open(f"/proc/{p}/stat") as fh:
                raw = fh.read()
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not cmd:
            continue
        rest = raw[raw.rfind(")") + 2 :].split()
        state, ppid, start = rest[0], int(rest[1]), int(rest[19])
        age = now - (btime + start / CLK_TCK)
        yield {"pid": p, "ppid": ppid, "state": state, "age_s": int(age), "cmd": cmd}


def classify(pr: dict, args, login: bool):
    cmd, age = pr["cmd"], pr["age_s"]
    if any(a.search(cmd) for a in ALLOW):
        return None
    if TEST_DAEMON.search(cmd):
        # Test supervisors detach (ppid 1) by design, so only age separates a
        # running test session's daemon from a leftover.
        if age > args.test_daemon_age:
            return "test-daemon"
        return None
    # Only the real interpreter counts; wrapper lines (bash -c ..., timeout ...,
    # ly exec / srun clients that ship a pytest command into the allocation)
    # merely mention pytest in their argv and must not be counted.
    if re.match(r"\S*python[0-9.]*\s+(-\S+\s+)*-m pytest\b|\S*/pytest( |$)", cmd):
        return "pytest" if (login or age > args.pytest_max_age) else "pytest-running"
    if POLLER.search(cmd) and re.search(r"(^|/)(ba|z)?sh ", cmd) and age > args.poller_age:
        return "poller"
    if SCAN.search(cmd) and age > args.scan_age:
        return "scan"
    if "opencode" in cmd and " serve" in cmd:
        return "opencode"
    if "srun --jobid" in cmd and age > args.srun_age:
        return "stale-srun"
    if pr["ppid"] == 1 and WORK in cmd and age > args.orphan_age and re.match(r"\S*(python|bash|node|sh)\b", cmd):
        return "orphan"
    if pr["state"] == "D":
        return "d-state"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pytest-max-age", type=int, default=7200)
    ap.add_argument("--poller-age", type=int, default=1800)
    ap.add_argument("--scan-age", type=int, default=600)
    ap.add_argument("--srun-age", type=int, default=10800)
    ap.add_argument("--orphan-age", type=int, default=1800)
    ap.add_argument("--test-daemon-age", type=int, default=1200)
    args = ap.parse_args()

    login = is_login_node()
    found: dict[str, list] = {}
    for pr in processes(os.getuid()):
        cls = classify(pr, args, login)
        if cls:
            found.setdefault(cls, []).append(pr)

    running = found.pop("pytest-running", [])
    if len(running) > 1:
        found["pytest"] = found.get("pytest", []) + running
    info_only = {"d-state"}
    strays = {k: v for k, v in found.items() if k not in info_only}

    report = {
        "host": socket.gethostname(),
        "login_node": login,
        "strays": {k: len(v) for k, v in strays.items()},
        "info": {k: len(v) for k, v in found.items() if k in info_only},
        "pytest_running": len(running),
        "details": {
            k: [{"pid": p["pid"], "age_s": p["age_s"], "cmd": p["cmd"][:160]} for p in v[:10]]
            for k, v in found.items()
        },
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        where = "LOGIN NODE" if login else "allocation"
        total = sum(report["strays"].values())
        print(f"stray-check {report['host']} ({where}): {total} stray process(es)"
              + (f", {len(running)} pytest running" if running else ""))
        for k, v in found.items():
            tag = "info" if k in info_only else "STRAY"
            print(f"  [{tag}] {k}: {len(v)}")
            for p in v[:5]:
                print(f"      pid={p['pid']} age={p['age_s'] // 60}m {p['cmd'][:120]}")
        if total:
            print("  -> inspect, then kill only what is ours (never the fleet daemon or lane pi/MPI runs)")
    return 1 if strays else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # report, never crash the tick
        print(f"stray-check: internal error: {exc}", file=sys.stderr)
        sys.exit(2)
