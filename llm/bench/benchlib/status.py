"""`benchctl status`: our jobs, result directories and the last experiment doc."""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import config as config_mod
from . import util

JOB_PREFIXES = ("strata-", "glm53-", "qwen38-", "qwen35-", "bench-")


def jobs(cfg):
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    out = util.run(["squeue", "-h", "-u", user,
                    "-o", "%i|%P|%j|%T|%M|%N|%R|%C|%m"], timeout=25)
    rows = []
    if out["rc"] != 0:
        return rows, out
    for line in out["out"].splitlines():
        bits = line.split("|")
        if len(bits) < 9:
            continue
        job = {
            "id": bits[0], "partition": bits[1], "name": bits[2], "state": bits[3],
            "elapsed": bits[4], "nodes": bits[5], "reason": bits[6],
            "cpus": bits[7], "mem": bits[8],
            "ours": any(bits[2].startswith(p) for p in JOB_PREFIXES),
        }
        rows.append(job)
    return rows, out


def _count_lines(path):
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _describe(directory):
    d = Path(directory)
    info = {
        "path": str(d),
        "name": d.name,
        "mtime": d.stat().st_mtime if d.exists() else 0,
        "cells": 0,
        "summary_rows": 0,
        "files": 0,
        "statuses": {},
    }
    try:
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            info["files"] += 1
            name = path.name
            if name == "cells.jsonl":
                info["cells"] += _count_lines(path)
            elif name == "summary.csv":
                info["summary_rows"] += max(0, _count_lines(path) - 1)
                for row in _tail_statuses(path):
                    info["statuses"][row] = info["statuses"].get(row, 0) + 1
            elif name == "knobs.csv":
                info["summary_rows"] += max(0, _count_lines(path) - 1)
    except OSError:
        pass
    return info


def _tail_statuses(path, column=8):
    """Read the status column from summary.csv rows (best effort)."""
    out = []
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\n").split(",")
            idx = header.index("status") if "status" in header else column
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) > idx and parts[idx]:
                    out.append(parts[idx])
    except OSError:
        pass
    return out


def result_dirs(cfg, limit=15):
    dirs = []
    roots = [("strata-runs", cfg.path("strata_runs")),
             ("results", cfg.path("results")),
             ("glm-runs", cfg.path("glm_runs"))]
    for label, root in roots:
        p = Path(root)
        if not p.is_dir():
            continue
        try:
            entries = [x for x in p.iterdir() if x.is_dir()]
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "runtime":
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            dirs.append((mtime, label, entry))
    dirs.sort(key=lambda x: x[0], reverse=True)
    out = []
    for mtime, label, entry in dirs[:limit]:
        info = _describe(entry)
        info["root"] = label
        info["age_hours"] = round((time.time() - mtime) / 3600.0, 1)
        info["modified"] = util.utc_iso(mtime)
        out.append(info)
    return out


def last_experiment_docs(cfg, limit=5):
    docs = []
    for pattern in (config_mod.REPO_ROOT / "llm").glob("*/experiments/*.md"):
        try:
            docs.append((pattern.stat().st_mtime, pattern))
        except OSError:
            continue
    docs.sort(key=lambda x: x[0], reverse=True)
    return [{"path": str(p), "modified": util.utc_iso(m),
             "relative": str(p.relative_to(config_mod.REPO_ROOT))}
            for m, p in docs[:limit]]


def collect(cfg, include_all_jobs=False, limit=15):
    rows, proc = jobs(cfg)
    if not include_all_jobs:
        rows = [r for r in rows if r["ours"]]
    return {
        "generated_at": util.utc_iso(),
        "user": os.environ.get("USER", ""),
        "jobs": rows,
        "squeue_ok": proc["rc"] == 0,
        "result_dirs": result_dirs(cfg, limit=limit),
        "last_experiments": last_experiment_docs(cfg),
    }


def render_text(data):
    lines = []
    lines.append("jobs for %s" % (data["user"] or "?"))
    if not data.get("squeue_ok"):
        lines.append("  (squeue failed; not on a submit host?)")
    elif not data["jobs"]:
        lines.append("  none")
    for job in data["jobs"]:
        lines.append("  %-10s %-12s %-22s %-10s %-9s %s"
                     % (job["id"], job["partition"], job["name"], job["state"],
                        job["elapsed"], job["nodes"] or job["reason"]))
    lines.append("")
    lines.append("result directories (newest first)")
    if not data["result_dirs"]:
        lines.append("  none")
    for d in data["result_dirs"]:
        statuses = ", ".join("%s=%d" % (k, v) for k, v in sorted(d["statuses"].items()))
        bits = ["%dh ago" % d["age_hours"], "%d cell(s)" % d["cells"],
                "%d summary row(s)" % d["summary_rows"]]
        if statuses:
            bits.append(statuses)
        lines.append("  [%s] %s" % (d["root"], d["path"]))
        lines.append("      %s" % ", ".join(bits))
    lines.append("")
    lines.append("last experiment docs")
    for d in data["last_experiments"]:
        lines.append("  %s  (%s)" % (d["relative"], d["modified"]))
    return "\n".join(lines)
