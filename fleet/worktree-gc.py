#!/usr/bin/env python3
"""Remove stale scratch git worktrees of a fleet project (dry run by default).

Lanes create per-turn worktrees (e.g. `.lane-worktrees/replay-turn-<id>`) and
seldom remove them. In working-set image mode every one of them (~230 MB,
~8k files) is packed into the image at each spill, so they slow every spill
and grow the image (ADGFEM 2026-10-13: 22 worktrees, 5.2 GiB, spill 17 s ->
42 s). Source of truth: slurmhelps fleet/worktree-gc.py.

A worktree is removed only when ALL hold:
  * it is not the main checkout, not a lane's own worktree (basename equals a
    lane id in .langywrap/lanes.json) and not locked;
  * `git status --porcelain` is empty (untracked files included);
  * its HEAD commit and the directory are older than --min-age-h;
  * its HEAD commit is reachable from some ref (a branch keeps it alive after
    the worktree goes), so no commit becomes unreachable;
  * no process of this user has its cwd inside it.
Removal is `git worktree remove` without --force (git refuses dirty trees).

Usage: worktree-gc.py [--project DIR] [--min-age-h H] [--apply] [--json]
Exit status: 0 nothing (left) to do, 1 candidates found in a dry run, 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def worktrees(project: Path) -> list[dict]:
    out = git("worktree", "list", "--porcelain", cwd=project).stdout
    rows, cur = [], {}
    for line in out.splitlines() + [""]:
        if not line:
            if cur:
                rows.append(cur)
            cur = {}
            continue
        key, _, val = line.partition(" ")
        cur[key] = val or True
    return rows


def lane_ids(project: Path) -> set[str]:
    try:
        doc = json.loads((project / ".langywrap" / "lanes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(lane.get("id")) for lane in doc.get("lanes", []) if isinstance(lane, dict)}


def busy_dirs() -> list[str]:
    uid, dirs = os.getuid(), []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            dirs.append(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            continue
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default=os.environ.get("AGENT_PROJECT") or ".")
    ap.add_argument("--min-age-h", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    project = Path(git("rev-parse", "--show-toplevel", cwd=Path(a.project)).stdout.strip() or a.project)
    if not (project / ".git").exists():
        print(f"worktree-gc: {project} is not a git checkout", file=sys.stderr)
        return 2
    lanes, busy, now = lane_ids(project), busy_dirs(), time.time()
    limit = a.min_age_h * 3600
    report = []
    for wt in worktrees(project):
        path = Path(wt["worktree"])
        row = {"path": str(path), "branch": wt.get("branch", "(detached)")}
        reason = None
        if path == project:
            reason = "main checkout"
        elif path.name in lanes:
            reason = "lane worktree"
        elif "locked" in wt:
            reason = "locked"
        elif "prunable" in wt or not path.is_dir():
            reason = "prunable (missing dir; `git worktree prune`)"
        elif any(d == str(path) or d.startswith(str(path) + "/") for d in busy):
            reason = "in use (a process cwd is inside)"
        else:
            st = git("status", "--porcelain", cwd=path)
            if st.returncode != 0 or st.stdout.strip():
                reason = "dirty" if st.returncode == 0 else "status failed"
            else:
                head = wt.get("HEAD", "")
                ct = git("log", "-1", "--format=%ct", head, cwd=project).stdout.strip()
                age = now - max(int(ct or 0), int(path.stat().st_mtime))
                row["age_h"] = round(age / 3600, 1)
                if age < limit:
                    reason = f"younger than {a.min_age_h} h"
                elif not git("for-each-ref", "--contains", head, "--count=1", cwd=project).stdout.strip():
                    reason = "HEAD reachable from no ref (would lose commits)"
        if reason:
            row["keep"] = reason
        else:
            row["remove"] = True
            if a.apply:
                rm = git("worktree", "remove", str(path), cwd=project)
                row["removed"] = rm.returncode == 0
                if rm.returncode != 0:
                    row["error"] = (rm.stderr or rm.stdout).strip()[-300:]
        report.append(row)
    todo = [r for r in report if r.get("remove") and not r.get("removed")]
    if a.json:
        print(json.dumps({"project": str(project), "apply": a.apply, "worktrees": report}, indent=1))
    else:
        for r in report:
            verdict = ("removed" if r.get("removed") else "REMOVE" if r.get("remove") else "keep")
            detail = r.get("error") or r.get("keep") or f"age {r.get('age_h')} h"
            print(f"{verdict:8} {r['path']}  [{r['branch']}]  {detail}")
        print(f"worktree-gc: {sum(bool(r.get('removed')) for r in report)} removed, "
              f"{len(todo)} candidate(s) left{' (dry run)' if not a.apply else ''}")
    return 1 if todo else 0


if __name__ == "__main__":
    sys.exit(main())
