#!/usr/bin/env python3
"""Suggest a Slurm --time limit (minutes) per job type from sacct history.

suggested = max(15, ceil(p90(elapsed_minutes) * 1.5))

Data: `sacct -u <cluster.user> -S <cluster.sacct_since> -X` over read-only ssh to cluster.ssh (from the
calling project's slurmhelps.json) unless --sacct-file is given. Job types are recognised by the
project's cluster.job_prefix (whiteboxchallenge: wb-).
Included states: COMPLETED, FAILED, TIMEOUT (TIMEOUT elapsed == limit, so it is a
censored lower bound and is flagged). CANCELLED and RUNNING are skipped.

Usage:
  python ../slurmhelps/bin/hpc_minutes.py --type csave
  python ../slurmhelps/bin/hpc_minutes.py --job wb-gq-9-20261010T141012-relu-edge-shard2
  python ../slurmhelps/bin/hpc_minutes.py --all
  python ../slurmhelps/bin/hpc_minutes.py --all --sacct-file dump.txt   # offline, sacct -P -n -X output
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from slurmhelps import config as _config  # noqa: E402

_C = _config.load()["cluster"]
SSH = _C["ssh"]
P = re.escape(_C["job_prefix"])
SACCT_CMD = (
    f"sacct -u {_C['user']} -S {_C['sacct_since'] or 'now-7days'} -X -P -n "
    "-o JobID,JobName%60,Submit,Start,End,Elapsed,Timelimit,State,NodeList"
)
SACCT_FORMAT = [
    "JobID",
    "JobName",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "Timelimit",
    "State",
    "NodeList",
]
MIN_MINUTES = 15
FACTOR = 1.5
KEPT_STATES = {"COMPLETED", "FAILED", "TIMEOUT"}

# Ordered: first match wins.
TYPE_RULES = [
    ("gq-bridged", re.compile(rf"^{P}gq-")),
    ("csave", re.compile(rf"^{P}csave-")),
    ("gate", re.compile(rf"^{P}gate-|gate")),
    ("fresh48", re.compile(rf"^{P}fresh48-")),
    ("research", re.compile(rf"^{P}")),
    ("other", re.compile(r".*")),
]


def job_type(name: str) -> str:
    for t, rx in TYPE_RULES:
        if rx.search(name):
            return t
    return "other"


def parse_duration_minutes(s: str) -> float | None:
    """'[D-]HH:MM:SS' -> minutes. Returns None for unparseable values (UNLIMITED, None...)."""
    s = (s or "").strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", s)
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 1440 + h * 60 + mi + se / 60.0


def parse_ts(s: str) -> float | None:
    from datetime import datetime

    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


@dataclass
class Job:
    job_id: str
    name: str
    jtype: str
    elapsed: float
    limit: float
    state: str
    wait: float | None


def parse_rows(lines) -> list[Job]:
    jobs = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < len(SACCT_FORMAT):
            continue
        rec = dict(zip(SACCT_FORMAT, parts, strict=False))
        state = rec["State"].split(" ")[0]
        if state not in KEPT_STATES:
            continue
        elapsed = parse_duration_minutes(rec["Elapsed"])
        limit = parse_duration_minutes(rec["Timelimit"])
        if elapsed is None or limit is None or elapsed <= 0:
            continue
        sub, sta = parse_ts(rec["Submit"]), parse_ts(rec["Start"])
        wait = (sta - sub) / 60.0 if sub and sta and sta >= sub else None
        jobs.append(
            Job(rec["JobID"], rec["JobName"], job_type(rec["JobName"]), elapsed, limit, state, wait)
        )
    return jobs


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in (0, 1]."""
    xs = sorted(values)
    k = max(1, math.ceil(p * len(xs)))
    return xs[k - 1]


def suggest_from_elapsed(elapsed_minutes: list[float]) -> int:
    if not elapsed_minutes:
        return 90  # no history: keep the bridge's flat default
    return max(MIN_MINUTES, math.ceil(percentile(elapsed_minutes, 0.9) * FACTOR))


@dataclass
class Stats:
    jtype: str
    jobs: list[Job] = field(default_factory=list)

    def summary(self) -> dict:
        el = [j.elapsed for j in self.jobs]
        lim = [j.limit for j in self.jobs]
        waits = [j.wait for j in self.jobs if j.wait is not None]
        timeouts = sum(1 for j in self.jobs if j.state == "TIMEOUT")
        return {
            "type": self.jtype,
            "n": len(self.jobs),
            "timeouts": timeouts,
            "limit_med": statistics.median(lim) if lim else 0,
            "elapsed_med": statistics.median(el) if el else 0,
            "elapsed_p90": percentile(el, 0.9) if el else 0,
            "elapsed_max": max(el) if el else 0,
            "wait_med": statistics.median(waits) if waits else 0,
            "wait_p90": percentile(waits, 0.9) if waits else 0,
            "suggest": suggest_from_elapsed(el),
        }


def group(jobs: list[Job]) -> dict[str, Stats]:
    out: dict[str, Stats] = {}
    for j in jobs:
        out.setdefault(j.jtype, Stats(j.jtype)).jobs.append(j)
    return out


def fetch_sacct() -> list[str]:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH, SACCT_CMD],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        sys.exit(f"sacct over ssh failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout.splitlines()


def fmt_row(s: dict) -> str:
    return (
        f"{s['type']:<12} n={s['n']:<4} timeouts={s['timeouts']:<3} "
        f"limit_med={s['limit_med']:7.1f} elapsed_med={s['elapsed_med']:7.1f} "
        f"elapsed_p90={s['elapsed_p90']:7.1f} elapsed_max={s['elapsed_max']:7.1f} "
        f"wait_med={s['wait_med']:8.1f} wait_p90={s['wait_p90']:8.1f} "
        f"-> suggest={s['suggest']}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--type", dest="jtype", help="job type: gq-bridged, csave, gate, fresh48, research, other"
    )
    g.add_argument("--job", help="Slurm job name; its type is inferred")
    g.add_argument("--all", action="store_true", help="table for every type")
    ap.add_argument("--sacct-file", help="read sacct -P -n -X output from this file instead of ssh")
    args = ap.parse_args(argv)

    if args.sacct_file:
        with open(args.sacct_file, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = fetch_sacct()

    stats = group(parse_rows(lines))

    if args.all:
        for t in ["csave", "gate", "fresh48", "research", "gq-bridged", "other"]:
            if t in stats:
                print(fmt_row(stats[t].summary()))
        return 0

    jtype = job_type(args.job) if args.job else args.jtype
    if jtype not in stats:
        print(
            f"no completed/failed/timeout history for type '{jtype}'; default 90 min",
            file=sys.stderr,
        )
        print(90)
        return 0
    s = stats[jtype].summary()
    if s["timeouts"]:
        print(f"note: {s['timeouts']} TIMEOUT job(s) are censored at their limit", file=sys.stderr)
    if s["n"] < 5:
        print(f"note: only {s['n']} samples, treat suggestion as rough", file=sys.stderr)
    print(s["suggest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
