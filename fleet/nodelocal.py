#!/usr/bin/env python3
"""nodelocal.py — the node-local working set of a langywrap fleet.

Source of truth: slurmhelps fleet/nodelocal.py. Driven by
supervisor.sbatch and `ly spill`; configured by `fleet.node_local_checkout`
in .langywrap/lanes.json. When that block is absent or `enabled` is false,
nothing here runs and the fleet works in the /work checkout as before.

Why: lane tool calls (edit/read/write/bash) on the Lustre checkout have a p95
of 30-43 s; a synced 4 KB write on /work takes 0.2-3.8 s versus 0 s on /tmp.
The lanes therefore work in a node-local clone, and this script guarantees that
nothing is lost when the allocation ends.

On this cluster `NamespaceType=job_container/tmpfs`: /tmp is private to the job
and is WIPED when the job ends (also on a requeue of the same JobID). The final
spill is therefore the only way node-local work survives.

Commands
  config   --lanes LANES_JSON
      Print shell assignments (NL_ENABLED, NL_ROOT, ...) for supervisor.sbatch.
      Never fails: an unreadable registry prints NL_ENABLED=0.
  stage-in --lanes LANES_JSON --work WORK --jobid ID --gen N --state STATE
      Build the node-local clone and its worktrees before the daemon starts:
        * from the newest spill under spill_root (if any), otherwise
        * from the /work checkout (first cut-over: copies the uncommitted state).
      Exit 0 = ready, 3 = refused but nothing to lose (no spill exists; the
      caller may fall back to /work), 1 = failed (caller must NOT start lanes).
  spill    --state STATE --mode periodic|final|manual [--deadline EPOCH]
           [--reason TEXT] [--store PY SYNC PROJECT LOCAL PERSIST]
      Save everything the node-local tree holds to /work, most valuable first:
        1. git-refs  push every local branch and worktree HEAD to the /work repo
                     under refs/nodelocal/<jobid>/ (forced; objects included);
                     if that fails, git-bundle writes the unpushed commits to
                     <spill>/commits.bundle instead
        2. tree:*    per worktree, every dirty/untracked/ignored file (not clean
                     tracked ones) -> <spill>/trees/<name>/, its git index, and
                     the path lists that make the restore exact
        3. gitdir-langywrap  .git/langywrap (lane-turns.sqlite3 via backup API)
        4. store     the existing store-sync.py snapshot (final mode)
        5. telemetry LANGYWRAP_PI_TELEMETRY_FILE directory
        6. runs      worktree runs/ dirs that are real dirs -> /work counterpart
        7. mirror    selected runtime paths -> the /work checkout (steward reads)
        8. sync      fast-forward the fleet branch between node-local and /work
        9. publish   fast-forward other branches into /work refs/heads
       10. push-remote (optional, off by default: the operator pushes)
       11. flush     fsync the files written above (parallel, bounded)
       12. MANIFEST.json, written atomically (tmp + fsync + rename)
      Status: periodic|manual|final when every critical step succeeded,
      otherwise "partial" with reasons. Exit 0, or 2 for partial.
  status   --state STATE
      Print the state file and the newest manifest summary.

Safe to re-run at any time: spills serialise on a lock; each one rewrites the
same per-allocation spill directory incrementally.
"""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "slurmhelps-nodelocal/v1"
# Manifests written before the move to slurmhelps (same format).
LEGACY_SCHEMAS = ("adgfem-nodelocal/v1",)
DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    # Must end in the basename of the /work checkout: the session store dir is
    # /tmp/langywrap-store-<uid>/<basename(PROJECT)>, so it stays the same.
    # Placeholders: {uid}, {project_name} (basename of the checkout), {agents}
    # ($AGENTS_DIR, the fleet state root from the site profile).
    "root": "/tmp/{project_name}-{uid}/{project_name}",
    "spill_interval_s": 300,
    "spill_root": "{agents}/state/runtime/{project_name}/spill",
    "branch": None,  # default: fleet.branch
    # /work worktrees imported on the FIRST cut-over (relative to the checkout).
    "import_worktree_globs": [".lane-worktrees/*"],
    # Never copied into a spill/import (relative path globs or dir prefixes).
    "exclude": [
        "runs",
        ".lane-worktrees",
        ".langywrap/worktrees",
        ".langywrap/sessions.sqlite3*",
        ".langywrap/langypi-mail.sqlite3*",
        ".git",
    ],
    # Untracked runtime paths copied to the /work checkout on every spill
    # (update-only, never deleting) so a login-node reader still sees them.
    "mirror_to_work": [
        ".langywrap/lane-reports",
        ".langywrap/turn-gate-reports",
        ".langywrap/turn-gates",
        ".langywrap/gpu-jobs.ndjson",
    ],
    "telemetry_dir": "/tmp/langywrap-telemetry-{uid}",
    "push_remote": None,  # e.g. "origin"; None = never push off-cluster
    "publish_branches": True,
    "min_free_gib": 20,
    "keep_spills": 3,
    "quiesce_kill_lanes": True,
    "final_budget_term_s": 25,  # KillWait (30 s) minus a margin
    "final_budget_usr1_s": 3000,  # --signal=B:USR1@3600 minus a margin
    "final_budget_end_s": 1800,  # normal end / crash budget / EXIT / INT (ly down)
    # The store snapshot (7.5 GB sessions.sqlite3) took 48-58 s on 2026-10-11.
    # A final spill with less budget left skips it (the periodic snapshot is at
    # most AGENTS_STORE_SNAPSHOT_SECS old) instead of leaving a partial tmp copy.
    "store_min_s": 90,
}
# NEVER copied in either direction (/work -> /tmp import or restore, /tmp ->
# /work spill or mirror), whatever lanes.json says: lane runtime spools and
# logs, ~345k small files on 2026-10-12 that made the first import crawl at
# ~5 files/s on Lustre. All are gitignored, so excluding them never drops a
# tracked file. Paths are relative to each tree root (main checkout and every
# worktree). Do NOT add a blanket "*.log": tracked fixtures end in .log.
NEVER_COPY = [
    ".log",
    ".langywrap/activity-spool",
    ".langywrap/artifacts",
    ".langywrap/.langypi-session.*",
    ".langywrap/session-supervisor.*",
    ".trash",
    # caches: regenerable; .cache/repo-search.db is a live 4.9 GB SQLite index
    # (2026-10-12). NOT .langywrap/pi-sessions: pi resumes a lane from its jsonl.
    ".cache",
]
# Directory names excluded at ANY depth.
NEVER_COPY_ANYWHERE = [".trash"]


def copy_excludes(cfg: Dict[str, Any]) -> List[str]:
    """cfg["exclude"] plus the non-overridable NEVER_COPY list."""
    out = list(cfg.get("exclude") or [])
    for p in NEVER_COPY:
        if p not in out:
            out.append(p)
    return out


CRITICAL_PREFIXES = ("git-refs", "git-bundle", "tree:", "gitdir-langywrap", "store", "telemetry", "runs", "push-remote")
RESERVE_S = 3.0  # always kept for the manifest


class NLError(RuntimeError):
    pass


class Fallback(NLError):
    """Refused, but no spill exists, so falling back to /work loses nothing."""


def now_iso(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def log(msg: str) -> None:
    print(f"{now_iso()} nodelocal: {msg}", flush=True)


def fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes, durable: bool = False) -> None:
    """tmp + rename. `durable` adds fsync of the file and its directory; on
    Lustre a synced small write costs 0.2-3.8 s, so bulk spill files are not
    synced inline: the spill's `flush` step fsyncs them 16 at a time before it
    writes the durable MANIFEST.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        if durable:
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, path)
    if durable:
        fsync_dir(path.parent)


def write_json(path: Path, obj: Any, durable: bool = False) -> None:
    write_atomic(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode(), durable)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise NLError(f"timeout after {timeout:.0f}s: {shlex.join(map(str, cmd))}") from exc
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
        raise NLError(f"rc={p.returncode}: {shlex.join(map(str, cmd))}: {' | '.join(tail)}")
    return p


def git(repo: Path, *args: str, timeout: Optional[float] = None, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(repo), *args], timeout=timeout, check=check)


def git_out(repo: Path, *args: str, timeout: Optional[float] = None) -> str:
    return git(repo, *args, timeout=timeout).stdout.strip()


def rev(repo: Path, ref: str) -> Optional[str]:
    p = git(repo, "rev-parse", "-q", "--verify", f"{ref}^{{commit}}", check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def is_ancestor(repo: Path, a: str, b: str) -> bool:
    return git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0


# ---------------------------------------------------------------- config


def load_config(lanes_json: Path) -> Dict[str, Any]:
    with open(lanes_json) as fh:
        reg = json.load(fh)
    fleet = reg.get("fleet") or {}
    block = fleet.get("node_local_checkout") or {}
    if not isinstance(block, dict):
        raise NLError("fleet.node_local_checkout must be an object")
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in block.items() if not k.startswith("why")})
    cfg["enabled"] = bool(cfg.get("enabled"))
    cfg["branch"] = cfg.get("branch") or fleet.get("branch")
    if cfg["enabled"] and not cfg["branch"]:
        raise NLError("fleet.branch (or fleet.node_local_checkout.branch) must name the fleet branch")
    uid = str(os.getuid())
    project_name = Path(lanes_json).resolve().parent.parent.name
    agents = os.environ.get("AGENTS_DIR", "")
    for key in ("root", "telemetry_dir", "spill_root"):
        v = str(cfg[key]).replace("{uid}", uid).replace("{project_name}", project_name)
        if "{agents}" in v:
            if not agents:
                raise NLError(f"{key} uses {{agents}} but AGENTS_DIR is unset (load the site profile)")
            v = v.replace("{agents}", agents)
        cfg[key] = v
    cfg["spill_interval_s"] = max(10, int(cfg["spill_interval_s"]))
    return cfg


def cmd_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(Path(args.lanes))
    except Exception as exc:  # never fail the batch script on a config read
        print(f"# nodelocal config unreadable ({exc}); node-local disabled", file=sys.stderr)
        print("NL_ENABLED=0")
        return 0
    if not cfg["enabled"]:
        print("NL_ENABLED=0")
        return 0
    out = {
        "NL_ENABLED": "1",
        "NL_ROOT": cfg["root"],
        "NL_BASE": str(Path(cfg["root"]).parent),
        "NL_SPILL_INTERVAL": str(cfg["spill_interval_s"]),
        "NL_SPILL_ROOT": cfg["spill_root"],
        "NL_TELEMETRY_DIR": cfg["telemetry_dir"],
        "NL_QUIESCE_KILL": "1" if cfg["quiesce_kill_lanes"] else "0",
        "NL_BUDGET_TERM": str(int(cfg["final_budget_term_s"])),
        "NL_BUDGET_USR1": str(int(cfg["final_budget_usr1_s"])),
        "NL_BUDGET_END": str(int(cfg["final_budget_end_s"])),
    }
    for k, v in out.items():
        print(f"{k}={shlex.quote(v)}")
    return 0


# ---------------------------------------------------------------- helpers


def excluded(rel: str, patterns: List[str]) -> bool:
    rel = rel.rstrip("/")
    if any(part in NEVER_COPY_ANYWHERE for part in rel.split("/")):
        return True
    for pat in patterns:
        p = pat.rstrip("/")
        if rel == p or rel.startswith(p + "/") or fnmatch.fnmatchcase(rel, p):
            return True
    return False


def tree_name(rel: str) -> str:
    return "main" if rel in ("", ".") else rel.replace("/", "__")


def ref_name(name: str) -> str:
    """A git-ref-safe form of a tree name (no leading dot, no '..', no '.lock')."""
    out = re.sub(r"[^A-Za-z0-9_\-]", "_", name).strip("_")
    return out or "main"


def worktrees(repo: Path) -> List[Dict[str, Any]]:
    """`git worktree list --porcelain` as dicts: path, head, branch (or None), main."""
    out = git_out(repo, "worktree", "list", "--porcelain")
    res: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for line in out.splitlines() + [""]:
        if not line:
            if cur:
                res.append(cur)
            cur = {}
            continue
        key, _, val = line.partition(" ")
        if key == "worktree":
            cur = {"path": val, "head": None, "branch": None, "bare": False, "prunable": False}
        elif key == "HEAD":
            cur["head"] = val
        elif key == "branch":
            cur["branch"] = val[len("refs/heads/"):] if val.startswith("refs/heads/") else val
        elif key == "bare":
            cur["bare"] = True
        elif key == "prunable":
            cur["prunable"] = True
    for i, w in enumerate(res):
        w["main"] = i == 0
    return res


def index_path(wt: Path) -> Path:
    p = git_out(wt, "rev-parse", "--path-format=absolute", "--git-path", "index")
    return Path(p)


def index_objects_ok(wt: Path) -> bool:
    """Every blob the index names must exist (a restored index may name blobs
    that only lived in the lost node-local object store)."""
    ls = git(wt, "ls-files", "-s", "-z", check=False)
    if ls.returncode != 0:
        return False
    shas = sorted({e.split()[1] for e in ls.stdout.split("\0") if e.strip() and len(e.split()) >= 2})
    if not shas:
        return True
    p = subprocess.run(
        ["git", "-C", str(wt), "cat-file", "--batch-check"],
        input="\n".join(shas) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode == 0 and " missing" not in p.stdout


def free_gib(path: Path) -> float:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize / 2**30


def sqlite_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for old in dst.parent.glob(f".{dst.name}.tmp*"):  # left by a killed copy
        old.unlink()
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, dst)


BIG_DB_BYTES = 64 * 2**20


def is_db(f: Path) -> bool:
    return f.suffix in (".sqlite3", ".sqlite", ".db")


def copy_gitdir_langywrap(src: Path, dst: Path, part: str = "all") -> List[Path]:
    """Copy .git/langywrap: SQLite DBs through the backup API (consistent even
    while open), every other file as is; -wal/-shm/-journal are skipped.
    part: "all", "small" (everything but DBs >= BIG_DB_BYTES) or "big" (only
    those; lane-turns.sqlite3 was 1.1 GB on 2026-10-11)."""
    out_files: List[Path] = []
    if not src.is_dir():
        return out_files
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir() or f.name.endswith(("-wal", "-shm", "-journal")) or ".tmp" in f.name:
            continue
        big = is_db(f) and f.stat().st_size >= BIG_DB_BYTES
        if (part == "small" and big) or (part == "big" and not big):
            continue
        out = dst / rel
        if is_db(f):
            sqlite_copy(f, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        out_files.append(out)
    return out_files


def read_list(p: Path) -> List[str]:
    try:
        return [x for x in p.read_bytes().decode().split("\0") if x]
    except OSError:
        return []


def write_list(p: Path, items: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes("\0".join(items).encode() + (b"\0" if items else b""))


def stamp_markers(base: Path, jobid: str, names: List[str]) -> None:
    """After an import/restore the tree equals its spill (plus base layer):
    the next spill only needs files changed from now on."""
    d = base / "markers" / str(jobid)
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.marker").write_text(now_iso())


def rsync(args: List[str], timeout: Optional[float]) -> subprocess.CompletedProcess:
    return run(["rsync", "-a", *args], timeout=timeout)


def runs_target(work: Path, rel: str) -> Path:
    """Where a node-local worktree's runs/ lives on /work: the same relative
    path in the /work checkout (the main checkout's runs/ is /work/.../runs)."""
    return (work / rel / "runs") if rel not in ("", ".") else (work / "runs")


def link_runs(wt: Path, work: Path, rel: str) -> str:
    """Make <wt>/runs a symlink to its /work counterpart so Slurm jobs on other
    nodes see the outputs. The tracked runs/** entries get skip-worktree so git
    neither reports them deleted nor writes through the link."""
    target = runs_target(work, rel)
    target.mkdir(parents=True, exist_ok=True)
    tracked = git(wt, "ls-files", "-z", "--", "runs", check=False).stdout.split("\0")
    tracked = [t for t in tracked if t]
    if tracked:
        git(wt, "update-index", "--skip-worktree", "--", *tracked, check=False)
    p = wt / "runs"
    if p.is_symlink():
        if os.readlink(p) == str(target):
            return "kept"
        p.unlink()
    elif p.exists():
        # A real dir: carry its files (tracked fixtures included) to /work first.
        rsync(["--ignore-existing", f"{p}/", f"{target}/"], timeout=None)
        shutil.rmtree(p)
    os.symlink(str(target), str(p))
    return f"-> {target}"


def git_status_paths(wt: Path, patterns: List[str], timeout: Optional[float]) -> Tuple[List[str], List[str], List[str]]:
    """(overlay, deleted, dirty) for one worktree. overlay = every existing
    file/symlink that is not a clean tracked file (modified, added, untracked,
    ignored); deleted = tracked files missing from the tree; dirty = the
    non-ignored subset (what a human calls 'uncommitted')."""
    p = git(wt, "status", "--porcelain=v1", "-z", "--ignored=traditional", "--untracked-files=all", "--no-renames", timeout=timeout)
    overlay: List[str] = []
    deleted: List[str] = []
    dirty: List[str] = []
    seen = set()
    for entry in p.stdout.split("\0"):
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        if excluded(path, patterns):
            continue
        full = wt / path
        if "D" in xy and not os.path.lexists(full):
            deleted.append(path)
            dirty.append(path)
            continue
        if path.endswith("/"):
            for root, dirs, files in os.walk(full):
                rroot = Path(root)
                dirs[:] = [d for d in dirs if not (rroot / d / ".git").exists()
                           and not excluded(str((rroot / d).relative_to(wt)), patterns)]
                for f in files:
                    r = str((rroot / f).relative_to(wt))
                    if not excluded(r, patterns) and r not in seen:
                        seen.add(r)
                        overlay.append(r)
            continue
        if os.path.lexists(full) and not full.is_dir() and path not in seen:
            seen.add(path)
            overlay.append(path)
            if xy != "!!":
                dirty.append(path)
    return sorted(overlay), sorted(deleted), sorted(dirty)


# ---------------------------------------------------------------- stage-in


def newest_manifest(spill_root: Path) -> Optional[Dict[str, Any]]:
    best = None
    for mf in glob.glob(str(spill_root / "*" / "MANIFEST.json")):
        m = read_json(Path(mf))
        if not m or m.get("schema") not in (SCHEMA, *LEGACY_SCHEMAS) or "written_at_epoch" not in m:
            continue
        m["_dir"] = str(Path(mf).parent)
        if best is None or m["written_at_epoch"] > best["written_at_epoch"]:
            best = m
    return best


def fresh_clone(work: Path, root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(work), str(root)])
    git(root, "remote", "rename", "origin", "work")
    url = git(work, "remote", "get-url", "origin", check=False).stdout.strip()
    if url:
        git(root, "remote", "add", "origin", url)
        git(root, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
    for key, val in (("gc.auto", "0"), ("core.untrackedcache", "true"), ("receive.denyCurrentBranch", "refuse")):
        git(root, "config", key, val)
    wtc = git(work, "config", "--local", "--get", "extensions.worktreeconfig", check=False).stdout.strip()
    if wtc:
        git(root, "config", "extensions.worktreeConfig", wtc)
    for name in ("user.name", "user.email"):
        v = git(work, "config", "--get", name, check=False).stdout.strip()
        if v:
            git(root, "config", name, v)
    # Local branches = the /work repo's branches (lanes and the steward use them by name).
    git(root, "fetch", "--quiet", "--update-head-ok", "work", "+refs/heads/*:refs/heads/*")
    excl = Path(git_out(work, "rev-parse", "--path-format=absolute", "--git-path", "info/exclude"))
    dst = root / ".git" / "info" / "exclude"
    dst.parent.mkdir(parents=True, exist_ok=True)
    body = excl.read_text() if excl.is_file() else ""
    dst.write_text(body + "\n# node-local: runs is a symlink to /work (nodelocal.py)\n/runs\n")


def set_head(wt: Path, branch: Optional[str], head: str) -> None:
    if branch:
        git(wt, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
    else:
        git(wt, "update-ref", "--no-deref", "HEAD", head)


def add_worktree(root: Path, path: Path, branch: Optional[str], head: str, checkout: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["worktree", "add", "--quiet", "--force"]
    if not checkout:
        args.append("--no-checkout")
    if branch and rev(root, f"refs/heads/{branch}"):
        git(root, *args, str(path), branch)
    else:
        git(root, *args, "--detach", str(path), head)


def install_index(wt: Path, src: Path, log_name: str, notes: List[str]) -> None:
    dst = index_path(wt)
    shutil.copy2(src, dst)
    if not index_objects_ok(wt):
        notes.append(f"{log_name}: index named missing objects; rebuilt from HEAD (staged state lost, tree kept)")
        git(wt, "reset", "-q")


def import_from_work(cfg: Dict[str, Any], work: Path, root: Path, notes: List[str], jobid_now: str) -> Dict[str, Any]:
    """First cut-over: copy the /work checkout and its lane worktrees exactly
    (tracked, untracked and ignored files, and the git index)."""
    patterns = copy_excludes(cfg)
    wts = worktrees(work)
    main = wts[0]
    chosen = [main]
    for w in wts[1:]:
        p = Path(w["path"])
        try:
            rel = str(p.relative_to(work))
        except ValueError:
            continue  # steward worktrees outside the checkout stay on /work
        if w.get("prunable") or not p.is_dir():
            continue
        if any(fnmatch.fnmatchcase(rel, g) for g in cfg["import_worktree_globs"]):
            chosen.append(w)
    fresh_clone(work, root)
    imported = []
    for w in chosen:
        src = Path(w["path"])
        rel = "" if w["main"] else str(src.relative_to(work))
        dst = root / rel if rel else root
        t0 = time.time()
        if w["main"]:
            set_head(root, w["branch"], w["head"])
        else:
            add_worktree(root, dst, w["branch"], w["head"], checkout=False)
        excl = ["--exclude=/.git"] + [f"--exclude=/{e.rstrip('/')}" for e in patterns if e != ".git"]
        excl += [f"--exclude={n}/" for n in NEVER_COPY_ANYWHERE]
        for o in chosen[1:]:
            orel = str(Path(o["path"]).relative_to(work))
            if w["main"]:
                excl.append(f"--exclude=/{orel}")
        rsync([*excl, f"{src}/", f"{dst}/"], timeout=None)
        install_index(dst, index_path(src), rel or "main", notes)
        link = link_runs(dst, work, rel)
        imported.append({"rel": rel, "branch": w["branch"], "head": w["head"], "runs": link,
                         "seconds": round(time.time() - t0, 1)})
        log(f"imported {'main' if not rel else rel} ({w['branch'] or 'detached ' + w['head'][:10]}) in {time.time() - t0:.1f}s; runs {link}")
    n = len(copy_gitdir_langywrap(work / ".git" / "langywrap", root / ".git" / "langywrap"))
    log(f"imported .git/langywrap ({n} files)")
    bases = {tree_name(t["rel"]): str(work / t["rel"]) if t["rel"] else str(work) for t in imported}
    stamp_markers(root.parent, jobid_now, list(bases))
    return {"source": "work", "trees": imported, "bases": bases}


def restore_from_spill(cfg: Dict[str, Any], work: Path, root: Path, m: Dict[str, Any], notes: List[str],
                       jobid_now: str) -> Dict[str, Any]:
    spill = Path(m["_dir"])
    jid = m["jobid"]
    fresh_clone(work, root)
    # Branch tips and worktree HEADs as the spill pushed them (refs/nodelocal/<jid>/).
    git(root, "fetch", "--quiet", "--update-head-ok", "work",
        f"+refs/nodelocal/{jid}/heads/*:refs/heads/*",
        f"+refs/nodelocal/{jid}/worktrees/*:refs/nodelocal-spill/worktrees/*")
    bundle = spill / "commits.bundle"
    if bundle.is_file():
        # The spill's push into /work failed; its commits are in the bundle.
        git(root, "fetch", "--quiet", str(bundle), "+refs/heads/*:refs/nodelocal-bundle/heads/*",
            "+refs/nodelocal-wt/*:refs/nodelocal-bundle/wt/*")
        moved = []
        for line in git_out(root, "for-each-ref", "--format=%(refname:strip=3) %(objectname)",
                            "refs/nodelocal-bundle/heads/").splitlines():
            b, _, sha = line.partition(" ")
            cur = rev(root, f"refs/heads/{b}")
            if cur != sha and (cur is None or is_ancestor(root, cur, sha)):
                git(root, "update-ref", f"refs/heads/{b}", sha)
                moved.append(b)
        notes.append(f"commits restored from {bundle} (push into /work had failed): {moved}")
    names = [t["name"] for t in m.get("trees", [])]
    if "main" in names:
        names.remove("main")
        names.insert(0, "main")
    restored = []
    bases: Dict[str, Optional[str]] = {}
    for name in names:
        meta = read_json(spill / "trees-meta" / f"{name}.json")
        if not meta:
            notes.append(f"{name}: no tree meta in spill; skipped")
            continue
        rel, branch, head = meta["rel"], meta.get("branch"), meta["head"]
        dst = root / rel if rel else root
        t0 = time.time()
        if not rev(root, head):
            notes.append(f"{name}: commit {head[:10]} missing from the /work repo; using the branch tip")
            head = rev(root, f"refs/heads/{branch}") if branch else None
            if not head:
                notes.append(f"{name}: no usable commit; skipped")
                continue
        if branch and not rev(root, f"refs/heads/{branch}"):
            notes.append(f"{name}: branch {branch} missing; restored detached at {head[:10]}")
            branch = None
        if not rel:
            set_head(root, branch, head)
            git(root, "reset", "-q", "--hard")
        else:
            add_worktree(root, dst, branch, head, checkout=True)
        tip = rev(dst, "HEAD")
        same_head = tip == head
        if not same_head:
            # A commit landed between the spill's ref push and its tree copy.
            # Never move the branch back; keep the tip, overlay the files, and
            # rebuild the index from the tip instead of installing the old one.
            notes.append(f"{name}: branch tip {str(tip)[:10]} != spilled HEAD {head[:10]}; kept the tip, index rebuilt")
        link_runs(dst, work, rel)
        ov = read_list(spill / "lists" / f"{name}.overlay")
        inm_p = spill / "lists" / f"{name}.inmirror"
        inm = set(read_list(inm_p)) if inm_p.is_file() else set(ov)
        from_m = [x for x in ov if x in inm]
        from_b = [x for x in ov if x not in inm]
        tmpd = root.parent / "restore-lists"
        if from_m:
            write_list(tmpd / f"{name}.mirror", from_m)
            p = run(["rsync", "-a", "--from0", f"--files-from={tmpd / (name + '.mirror')}",
                     f"{spill / 'trees' / name}/", f"{dst}/"], check=False)
            if p.returncode != 0:
                notes.append(f"{name}: overlay rsync rc={p.returncode}: {(p.stderr or '').strip().splitlines()[-1:]}")
        if from_b:
            bdir = meta.get("base")
            if bdir and Path(bdir).is_dir():
                write_list(tmpd / f"{name}.base", from_b)
                p = run(["rsync", "-a", "--ignore-missing-args", "--from0", f"--files-from={tmpd / (name + '.base')}",
                         f"{bdir}/", f"{dst}/"], check=False)
                if p.returncode != 0:
                    notes.append(f"{name}: base-layer rsync rc={p.returncode}: {(p.stderr or '').strip().splitlines()[-1:]}")
            else:
                notes.append(f"{name}: {len(from_b)} file(s) belong to a base layer {bdir!r} that is gone")
        bases[name] = meta.get("base")
        dele = spill / "lists" / f"{name}.deleted"
        if dele.is_file():
            for rp in dele.read_text().split("\0"):
                if rp and os.path.lexists(dst / rp) and not (dst / rp).is_dir():
                    (dst / rp).unlink()
        idx = spill / "index" / f"{name}.index"
        if idx.is_file() and same_head:
            install_index(dst, idx, name, notes)
            link_runs(dst, work, rel)  # re-assert skip-worktree on the restored index
        restored.append({"rel": rel, "branch": branch, "head": head, "seconds": round(time.time() - t0, 1)})
        log(f"restored {name} at {head[:10]} ({branch or 'detached'}) in {time.time() - t0:.1f}s")
    n = len(copy_gitdir_langywrap(spill / "gitdir-langywrap", root / ".git" / "langywrap"))
    log(f"restored .git/langywrap ({n} files)")
    stamp_markers(root.parent, jobid_now, list(bases))
    tel = spill / "telemetry"
    if tel.is_dir():
        Path(cfg["telemetry_dir"]).mkdir(parents=True, exist_ok=True)
        rsync([f"{tel}/", f"{cfg['telemetry_dir']}/"], timeout=None)
    # Per-file newest wins against the /work main checkout: a file edited on
    # /work (by the steward) after the spill was written replaces the spilled
    # one. Lane worktrees come from the spill only.
    newer = newer_work_files(work, float(m["written_at_epoch"]), copy_excludes(cfg), worktrees(work))
    for rp in newer:
        dst = root / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / rp, dst, follow_symlinks=False)
    if newer:
        notes.append(f"{len(newer)} /work file(s) newer than the spill replaced the spilled copy: {newer[:20]}")
    return {"source": "spill", "spill_dir": str(spill), "spill_id": m["spill_id"], "spill_status": m["status"],
            "trees": restored, "work_newer": newer, "bases": {k: v for k, v in bases.items() if v}}


def newer_work_files(work: Path, since: float, patterns: List[str], wts: List[Dict[str, Any]]) -> List[str]:
    skip = set()
    for w in wts[1:]:
        try:
            skip.add(str(Path(w["path"]).relative_to(work)))
        except ValueError:
            pass
    out: List[str] = []
    for root_, dirs, files in os.walk(work):
        r = Path(root_)
        keep = []
        for d in dirs:
            rel = str((r / d).relative_to(work))
            if rel in skip or excluded(rel, patterns) or (r / d).is_symlink():
                continue
            keep.append(d)
        dirs[:] = keep
        for f in files:
            rel = str((r / f).relative_to(work))
            if excluded(rel, patterns):
                continue
            try:
                if os.lstat(r / f).st_mtime > since + 1:
                    out.append(rel)
            except OSError:
                pass
    return sorted(out)


def cmd_stage_in(args: argparse.Namespace) -> int:
    t0 = time.time()
    work = Path(args.work).resolve()
    state_path = Path(args.state)
    notes: List[str] = []
    m = None
    try:
        cfg = load_config(Path(args.lanes))
        root = Path(cfg["root"])
        base = root.parent
        spill_root = Path(cfg["spill_root"])
        m = newest_manifest(spill_root)
        if root.name != work.name:
            raise NLError(f"root {root} must end in /{work.name} (the session store dir is keyed by that name)")
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(base, 0o700)
        gib = free_gib(base)
        if gib < float(cfg["min_free_gib"]):
            raise NLError(f"only {gib:.1f} GiB free under {base} (< min_free_gib {cfg['min_free_gib']})")
        marker = read_json(root / ".git" / "nodelocal.json") if root.exists() else None
        if root.exists() and marker and m and marker.get("last_spill_id") == m["spill_id"]:
            log(f"warm: {root} is the tree that wrote the newest spill {m['spill_id']}; reused as is")
            prev_state = read_json(state_path) or {}
            result = {"source": "warm", "spill_id": m["spill_id"], "trees": [], "bases": prev_state.get("bases", {})}
        else:
            if root.exists():
                aside = base / f"aside-{time.strftime('%Y%m%dT%H%M%S')}"
                for old in sorted(base.glob("aside-*"))[:-1]:
                    shutil.rmtree(old, ignore_errors=True)
                os.rename(root, aside)
                notes.append(f"stale node-local tree moved aside to {aside}")
            if m:
                log(f"restoring from spill {m['_dir']} (status={m['status']}, id={m['spill_id']}, written {m['written_at']})")
                # One spill lineage: the source dir becomes this job's spill dir
                # (a Lustre rename), so the first spill here stays incremental.
                target = spill_root / str(args.jobid)
                src = Path(m["_dir"])
                if src.resolve() != target.resolve():
                    if target.exists():
                        os.rename(target, spill_root / f"{args.jobid}.aside-{time.strftime('%Y%m%dT%H%M%S')}")
                    os.rename(src, target)
                    m["_dir"] = str(target)
                    notes.append(f"spill lineage: {src.name} renamed to {target.name}")
                    log(f"spill lineage: {src} -> {target}")
                result = restore_from_spill(cfg, work, root, m, notes, str(args.jobid))
            else:
                log(f"no spill under {spill_root}; importing the /work checkout {work} (first cut-over)")
                result = import_from_work(cfg, work, root, notes, str(args.jobid))
        frozen = base / "bin" / "nodelocal.py"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), frozen)
        Path(cfg["telemetry_dir"]).mkdir(parents=True, exist_ok=True, mode=0o700)
        state = {
            "schema": SCHEMA,
            "ready": True,
            "jobid": str(args.jobid),
            "gen": str(args.gen),
            "host": socket.getfqdn(),
            "work": str(work),
            "root": str(root),
            "base": str(base),
            "spill_root": str(spill_root),
            "spill_dir": str(spill_root / str(args.jobid)),
            "config": cfg,
            "frozen": str(frozen),
            "staged_at": now_iso(),
            "stage_in": {k: v for k, v in result.items() if k not in ("trees", "bases")},
            "bases": result.get("bases", {}),
            "stage_in_trees": result.get("trees", []),
            "stage_in_notes": notes,
            "stage_in_seconds": round(time.time() - t0, 1),
        }
        write_json(state_path, state)
        mk = read_json(root / ".git" / "nodelocal.json") or {}
        mk.update({"jobid": str(args.jobid), "gen": str(args.gen), "staged_at": state["staged_at"],
                   "source": result["source"]})
        write_json(root / ".git" / "nodelocal.json", mk)
        for n in notes:
            log(f"NOTE {n}")
        log(f"stage-in done: source={result['source']} root={root} in {time.time() - t0:.1f}s")
        return 0
    except Exception as exc:
        log(f"ERROR stage-in failed: {exc}")
        for n in notes:
            log(f"NOTE {n}")
        if m is None:
            log("no spill exists, so falling back to /work loses nothing (exit 3)")
            return 3
        return 1


# ---------------------------------------------------------------- spill


class Spill:
    def __init__(self, state: Dict[str, Any], mode: str, deadline: Optional[float], reason: str,
                 store: Optional[List[str]]) -> None:
        self.st = state
        self.cfg = state["config"]
        self.mode = mode
        self.reason = reason
        self.store = store
        self.t0 = time.time()
        self.deadline = deadline if deadline else self.t0 + (900 if mode != "final" else float(self.cfg["final_budget_end_s"]))
        self.root = Path(state["root"])
        self.work = Path(state["work"])
        self.dir = Path(state["spill_dir"])
        self.jid = state["jobid"]
        self.steps: List[Dict[str, Any]] = []
        self.reasons: List[str] = []
        self.trees: List[Dict[str, Any]] = []
        self.sync: Dict[str, Any] = {}
        self.markers = Path(state["base"]) / "markers" / self.jid
        self.moved_heads: List[str] = []
        self.written: List[Path] = []
        self.bases: Dict[str, str] = dict(state.get("bases") or {})

    def left(self) -> float:
        return self.deadline - time.time()

    def step(self, name: str, fn, *a) -> Any:
        critical = name.startswith(CRITICAL_PREFIXES)
        rec: Dict[str, Any] = {"name": name, "critical": critical}
        if self.left() < RESERVE_S:
            rec.update(ok=False, skipped=True, seconds=0.0, detail="deadline reached")
            self.steps.append(rec)
            if critical:
                self.reasons.append(f"{name}: skipped, deadline reached")
            log(f"spill step {name}: SKIPPED (deadline)")
            return None
        t = time.time()
        try:
            detail = fn(*a)
            rec.update(ok=True, detail=detail)
        except Exception as exc:
            rec.update(ok=False, detail=str(exc)[-600:])
            if critical:
                self.reasons.append(f"{name}: {str(exc)[-300:]}")
        rec["seconds"] = round(time.time() - t, 2)
        self.steps.append(rec)
        log(f"spill step {name}: {'ok' if rec['ok'] else 'FAILED'} in {rec['seconds']}s"
            + (f" ({rec['detail']})" if not rec["ok"] or isinstance(rec.get("detail"), str) else ""))
        return rec.get("detail")

    def tmo(self) -> float:
        return max(1.0, self.left() - RESERVE_S)

    # -- steps

    def local_worktrees(self) -> List[Dict[str, Any]]:
        out = []
        for w in worktrees(self.root):
            p = Path(w["path"])
            try:
                rel = str(p.relative_to(self.root))
            except ValueError:
                log(f"WARN worktree outside the node-local root is not spilled: {p}")
                continue
            if w.get("prunable") or not p.is_dir():
                continue
            w["rel"] = "" if rel == "." else rel
            w["name"] = tree_name(w["rel"])
            w["head"] = rev(p, "HEAD") or w["head"]
            out.append(w)
        return out

    def git_refs(self, wts: List[Dict[str, Any]]) -> str:
        specs = [f"+refs/heads/*:refs/nodelocal/{self.jid}/heads/*"]
        for w in wts:
            if w["head"]:
                specs.append(f"+{w['head']}:refs/nodelocal/{self.jid}/worktrees/{ref_name(w['name'])}")
        git(self.root, "push", "--quiet", "--prune", "work", *specs, timeout=self.tmo())
        stale = self.dir / "commits.bundle"
        if stale.exists():
            stale.unlink()  # every commit is in the /work repo now
        return f"{len(specs) - 1} worktree HEADs + all branches -> refs/nodelocal/{self.jid}/"

    def git_bundle(self, wts: List[Dict[str, Any]]) -> str:
        """The push into /work failed: keep the unpushed commits anyway, as a
        bundle of everything not reachable from the /work branches."""
        for w in wts:
            if w["head"]:
                git(self.root, "update-ref", f"refs/nodelocal-wt/{ref_name(w['name'])}", w["head"])
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".commits.bundle.tmp{os.getpid()}"
        p = git(self.root, "bundle", "create", "--quiet", str(tmp), "--branches", "--glob=refs/nodelocal-wt/*",
                "--not", "--remotes=work", timeout=self.tmo(), check=False)
        if p.returncode != 0:
            if tmp.exists():
                tmp.unlink()
            if "empty bundle" in (p.stderr or ""):
                return "no commit is missing from /work"
            raise NLError(f"git bundle rc={p.returncode}: {(p.stderr or '').strip()[-300:]}")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, self.dir / "commits.bundle")
        fsync_dir(self.dir)  # the bundle itself was fsynced above: it is the only copy
        return f"commits.bundle {(self.dir / 'commits.bundle').stat().st_size} B"

    def tree(self, w: Dict[str, Any]) -> str:
        wt = Path(w["path"])
        name = w["name"]
        head_now = rev(wt, "HEAD")
        if head_now and head_now != w["head"]:
            self.moved_heads.append(name)  # a commit landed after git-refs; pushed again below
            w["head"] = head_now
        self.markers.mkdir(parents=True, exist_ok=True)
        marker = self.markers / f"{name}.marker"
        new_marker = self.markers / f"{name}.marker.new"
        new_marker.write_text(now_iso())
        dest = self.dir / "trees" / name
        # ns precision on the node-local fs: the marker is written BEFORE the
        # scan, so a file touched during the scan is copied again next time.
        prev = marker.stat().st_mtime_ns if marker.exists() else None
        # Base layer: a tree imported from /work keeps its /work copy as the
        # layer under the spill; files unchanged since the import/restore are
        # read from there on restore and never copied (330k runtime files in
        # .langywrap on 2026-10-11). inmirror = paths whose spill copy is current.
        base = self.bases.get(name)
        inm_path = self.dir / "lists" / f"{name}.inmirror"
        inmirror = set(read_list(inm_path))
        overlay, deleted, dirty = git_status_paths(wt, copy_excludes(self.cfg), self.tmo())
        changed = []
        total = 0
        for rp in overlay:
            try:
                s = os.lstat(wt / rp)
            except OSError:
                continue
            total += s.st_size
            newer = prev is None or max(s.st_mtime_ns, s.st_ctime_ns) >= prev
            if newer or (rp not in inmirror and not base):
                changed.append(rp)
        dest.mkdir(parents=True, exist_ok=True)
        if changed:
            fl = self.markers / f"{name}.files"
            fl.write_bytes("\0".join(changed).encode() + b"\0")
            rsync(["--from0", f"--files-from={fl}", f"{wt}/", f"{dest}/"], timeout=self.tmo())
            self.written.extend(dest / rp for rp in changed)
        write_atomic(self.dir / "lists" / f"{name}.overlay", "\0".join(overlay).encode() + (b"\0" if overlay else b""))
        write_atomic(self.dir / "lists" / f"{name}.deleted", "\0".join(deleted).encode() + (b"\0" if deleted else b""))
        oset = set(overlay)
        new_inm = sorted((inmirror | set(changed)) & oset)
        write_atomic(inm_path, "\0".join(new_inm).encode() + (b"\0" if new_inm else b""))
        self.written += [self.dir / "lists" / f"{name}.overlay", self.dir / "lists" / f"{name}.deleted", inm_path]
        idx = index_path(wt)
        if idx.is_file():
            write_atomic(self.dir / "index" / f"{name}.index", idx.read_bytes())
            self.written.append(self.dir / "index" / f"{name}.index")
        meta = {"name": name, "rel": w["rel"], "branch": w["branch"], "head": w["head"], "written_at": now_iso(),
                "overlay_count": len(overlay), "deleted_count": len(deleted), "changed_count": len(changed),
                "in_base_count": len(oset) - len(new_inm), "base": base, "bytes": total}
        write_json(self.dir / "trees-meta" / f"{name}.json", meta)
        self.written.append(self.dir / "trees-meta" / f"{name}.json")
        os.replace(new_marker, marker)
        self.trees.append({"name": name, "rel": w["rel"], "branch": w["branch"], "head": w["head"],
                           "dirty_count": len(dirty), "dirty_paths": dirty[:500],
                           "overlay_count": len(overlay), "ignored_count": len(set(overlay) - set(dirty)),
                           "deleted_count": len(deleted), "changed_count": len(changed), "bytes": total})
        return (f"{len(changed)}/{len(overlay)} files copied ({'incremental' if prev else 'full'}), "
                f"{len(oset) - len(new_inm)} from the base layer, {len(deleted)} deleted, {total} B")

    def gitdir_langywrap(self) -> str:
        files = copy_gitdir_langywrap(self.root / ".git" / "langywrap", self.dir / "gitdir-langywrap", "small")
        self.written += [p for p in files if not is_db(p)]  # DB copies are fsynced already
        return f"{len(files)} files"

    def gitdir_langywrap_db(self) -> str:
        """Large SQLite DBs of .git/langywrap, skipped when unchanged since the
        last spill into this directory (signature: size + mtime of db and wal)."""
        src = self.root / ".git" / "langywrap"
        sig_path = self.markers / "gitdir-db.json"
        sig = {}
        for f in sorted(src.glob("*")) if src.is_dir() else []:
            if is_db(f) or f.name.endswith("-wal"):
                st = f.stat()
                sig[f.name] = [st.st_size, st.st_mtime_ns]
        old = read_json(sig_path) or {}
        if old.get("sig") == sig and old.get("dir") == str(self.dir):
            return "unchanged since the last spill"
        files = copy_gitdir_langywrap(src, self.dir / "gitdir-langywrap", "big")
        self.markers.mkdir(parents=True, exist_ok=True)
        write_json(sig_path, {"sig": sig, "dir": str(self.dir)})
        return f"{len(files)} DB(s): {[p.name for p in files]}"

    def big_db_bytes(self) -> int:
        src = self.root / ".git" / "langywrap"
        return sum(f.stat().st_size for f in src.glob("*")
                   if is_db(f) and f.stat().st_size >= BIG_DB_BYTES) if src.is_dir() else 0

    def store_snapshot(self) -> str:
        assert self.store
        py, sync, project, local, persist = self.store
        p = run([py, sync, "snapshot", project, local, persist], timeout=self.tmo(), check=False)
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            raise NLError(f"store snapshot rc={p.returncode}: {(p.stderr or p.stdout).strip()[-300:]}")
        return "store snapshot written"

    def telemetry(self) -> str:
        src = Path(self.cfg["telemetry_dir"])
        if not src.is_dir():
            return "no telemetry dir"
        rsync([f"{src}/", f"{self.dir / 'telemetry'}/"], timeout=self.tmo())
        return "copied"

    def runs(self, wts: List[Dict[str, Any]]) -> str:
        moved = []
        for w in wts:
            p = Path(w["path"]) / "runs"
            if p.is_dir() and not p.is_symlink():
                tgt = runs_target(self.work, w["rel"])
                tgt.mkdir(parents=True, exist_ok=True)
                rsync([f"{p}/", f"{tgt}/"], timeout=self.tmo())
                moved.append(w["name"])
        return f"real runs/ copied to /work for {moved}" if moved else "all runs/ are /work symlinks"

    def mirror(self) -> str:
        done = []
        for rp in self.cfg.get("mirror_to_work") or []:
            if excluded(rp, NEVER_COPY):
                continue  # never mirror runtime spools/logs, even if configured
            src = self.root / rp
            if not os.path.lexists(src):
                continue
            dst = self.work / rp
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                rsync(["-u", f"{src}/", f"{dst}/"], timeout=self.tmo())
            else:
                rsync(["-u", str(src), str(dst)], timeout=self.tmo())
            done.append(rp)
        return f"{done}"

    def checked_out(self, repo: Path) -> Dict[str, str]:
        return {w["branch"]: w["path"] for w in worktrees(repo) if w.get("branch")}

    def sync_branch(self) -> str:
        b = self.cfg["branch"]
        git(self.root, "fetch", "--quiet", "work", timeout=self.tmo())
        local = rev(self.root, f"refs/heads/{b}")
        remote = rev(self.root, f"refs/remotes/work/{b}")
        res: Dict[str, Any] = {"branch": b, "local": local, "work": remote}
        if not local or not remote:
            res["state"] = "missing"
        elif local == remote:
            res["state"] = "in-sync"
        elif is_ancestor(self.root, remote, local):
            res["state"] = self.ff_work(b, local, remote)
        elif is_ancestor(self.root, local, remote):
            if self.mode == "final":
                res["state"] = "local-behind (left for the next stage-in)"
            else:
                res["state"] = self.ff_local(b, remote)
        else:
            res["state"] = (f"diverged: merge refs/nodelocal/{self.jid}/heads/{b} into /work {b} "
                            f"(or merge work/{b} in the node-local checkout)")
        if self.mode != "final":
            res["new_from_work"] = self.adopt_work_branches()
        self.sync = res
        if res["state"].startswith(("diverged", "work-ff-blocked", "local-ff-blocked")):
            raise NLError(res["state"])
        return res["state"]

    def ff_work(self, b: str, local: str, remote: str) -> str:
        co = self.checked_out(self.work)
        if b in co:
            wt = Path(co[b])
            p = git(wt, "merge", "--ff-only", "--quiet", local, timeout=self.tmo(), check=False)
            if p.returncode != 0:
                return f"work-ff-blocked: {(p.stderr or p.stdout).strip()[-300:]}"
            return f"ff-work {remote[:10]}..{local[:10]} in {wt}"
        git(self.work, "update-ref", f"refs/heads/{b}", local, remote, timeout=self.tmo())
        return f"ff-work ref {remote[:10]}..{local[:10]}"

    def ff_local(self, b: str, remote: str) -> str:
        co = self.checked_out(self.root)
        if b in co:
            p = git(Path(co[b]), "merge", "--ff-only", "--quiet", remote, timeout=self.tmo(), check=False)
            if p.returncode != 0:
                return f"local-ff-blocked: {(p.stderr or p.stdout).strip()[-300:]}"
            return f"ff-local to {remote[:10]}"
        git(self.root, "update-ref", f"refs/heads/{b}", remote)
        return f"ff-local ref to {remote[:10]}"

    def adopt_work_branches(self) -> List[str]:
        """Branches the steward created/advanced on /work (prep/*) become
        visible to the lanes: create missing ones, fast-forward idle ones."""
        out = git_out(self.root, "for-each-ref", "--format=%(refname:strip=3) %(objectname)", "refs/remotes/work/")
        co = self.checked_out(self.root)
        changed = []
        for line in out.splitlines():
            name, _, sha = line.partition(" ")
            if not name or name == "HEAD" or name in co:
                continue
            cur = rev(self.root, f"refs/heads/{name}")
            if cur is None:
                git(self.root, "update-ref", f"refs/heads/{name}", sha, "")
                changed.append(f"+{name}")
            elif cur != sha and is_ancestor(self.root, cur, sha):
                git(self.root, "update-ref", f"refs/heads/{name}", sha, cur)
                changed.append(f"ff {name}")
        return changed

    def publish(self) -> str:
        b = self.cfg["branch"]
        heads = git_out(self.root, "for-each-ref", "--format=%(refname:strip=2)", "refs/heads/").split()
        specs = [f"refs/heads/{h}:refs/heads/{h}" for h in heads if h != b]
        if not specs:
            return "nothing to publish"
        p = git(self.root, "push", "--porcelain", "work", *specs, timeout=self.tmo(), check=False)
        rejected = [ln.split("\t")[1] for ln in p.stdout.splitlines() if ln.startswith("!") and "\t" in ln]
        return f"{len(specs) - len(rejected)} branch(es) fast-forwarded or current on /work, {len(rejected)} not (checked out there or diverged)"

    def push_remote(self) -> str:
        remote = self.cfg["push_remote"]
        b = self.cfg["branch"]
        git(self.root, "push", "--quiet", remote, f"refs/heads/{b}:refs/heads/{b}", timeout=self.tmo())
        return f"pushed {b} to {remote}"

    def flush(self) -> str:
        """fsync every file this spill wrote, 16 at a time, within the deadline.
        Not syncfs: on Lustre that flushes the whole client (every writer on
        the node) and failed with ESHUTDOWN on one OST in the 2026-10-11 bench.
        Durability only matters for a hard node crash right after the spill;
        on a normal job end the kernel writes the pages back anyway."""
        from concurrent.futures import ThreadPoolExecutor, wait

        def one(p: Path) -> None:
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        files = [p for p in self.written if p.is_file()]
        if not files:
            return "nothing written"
        ex = ThreadPoolExecutor(max_workers=16)
        futs = [ex.submit(one, p) for p in files]
        done, pending = wait(futs, timeout=max(1.0, self.left() - RESERVE_S))
        ex.shutdown(wait=False, cancel_futures=True)
        errors = sum(1 for f in done if f.exception() is not None)
        if pending or errors:
            raise NLError(f"fsync: {len(done) - errors}/{len(files)} done, {errors} errors, {len(pending)} unfinished at the deadline")
        return f"fsync of {len(files)} files"

    def prune(self) -> str:
        keep = max(1, int(self.cfg.get("keep_spills", 3)))
        dirs = []
        for d in Path(self.st["spill_root"]).iterdir():
            m = read_json(d / "MANIFEST.json")
            if m and d.resolve() != self.dir.resolve():
                dirs.append((m.get("written_at_epoch", 0), d, m))
        dirs.sort(reverse=True)
        gone = []
        for _, d, m in dirs[keep - 1:]:
            jid = str(m.get("jobid", d.name))
            refs = git_out(self.work, "for-each-ref", "--format=%(refname)", f"refs/nodelocal/{jid}/").split()
            if refs:
                p = subprocess.run(["git", "-C", str(self.work), "update-ref", "--stdin"],
                                   input="".join(f"delete {r}\n" for r in refs), text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if p.returncode != 0:
                    continue
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d.name)
        return f"pruned {gone}" if gone else "nothing to prune"

    # -- driver

    def run(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json(self.dir / "SPILL_IN_PROGRESS.json", {"mode": self.mode, "started_at": now_iso(self.t0),
                                                          "pid": os.getpid(), "host": socket.getfqdn()})
        wts = self.step("list-worktrees", self.local_worktrees) or []
        if not wts:
            self.reasons.append("no worktrees listed")
        self.step("git-refs", self.git_refs, wts)
        if not self.steps[-1]["ok"]:
            self.step("git-bundle", self.git_bundle, wts)
        # Most valuable first: main, then the lane trees touched most recently
        # (git index mtime), so a short TERM budget covers the active ones.
        def activity(w: Dict[str, Any]) -> float:
            try:
                return index_path(Path(w["path"])).stat().st_mtime
            except Exception:
                return 0.0
        for w in sorted(wts, key=lambda w: (w["name"] != "main", -activity(w))):
            self.step(f"tree:{w['name']}", self.tree, w)
        if self.moved_heads:
            self.step("git-refs:again", self.git_refs, wts)
            if not self.steps[-1]["ok"]:
                self.step("git-bundle:again", self.git_bundle, wts)
        self.step("gitdir-langywrap", self.gitdir_langywrap)
        self.step("telemetry", self.telemetry)
        self.step("runs", self.runs, wts)
        self.step("mirror-to-work", self.mirror)
        self.step("sync-branch", self.sync_branch)
        if self.cfg.get("publish_branches"):
            self.step("publish-branches", self.publish)
        if self.cfg.get("push_remote"):
            self.step("push-remote", self.push_remote)
        # tree meta of worktrees that no longer exist must not be restored
        live = {w["name"] for w in wts}
        for f in (self.dir / "trees-meta").glob("*.json") if (self.dir / "trees-meta").is_dir() else []:
            if f.stem not in live and wts:
                f.unlink()
        # fsync what this spill wrote before the durable manifest describes it.
        self.step("flush", self.flush)
        # Large .git/langywrap DBs (lane-turns.sqlite3, ~1.1 GB): after the
        # cheap steps; skipped when the budget cannot fit them (~100 MB/s).
        need_db = self.big_db_bytes() / 100e6 + 5
        if self.big_db_bytes() and self.left() - RESERVE_S < need_db:
            self.steps.append({"name": "gitdir-langywrap-db", "critical": True, "ok": False, "skipped": True,
                               "seconds": 0.0, "detail": f"{self.left():.0f}s left < ~{need_db:.0f}s needed"})
            self.reasons.append(f"gitdir-langywrap-db: skipped ({self.left():.0f}s left < ~{need_db:.0f}s); "
                                f"the previous copy in this spill dir stands")
            log("spill step gitdir-langywrap-db: SKIPPED (budget)")
        else:
            self.step("gitdir-langywrap-db", self.gitdir_langywrap_db)
        # Last: the store snapshot is the slowest step (~50 s) and has a
        # periodic copy; everything above is worth more per second.
        if self.store:
            need = float(self.cfg.get("store_min_s", 90))
            if self.left() - RESERVE_S < need:
                self.steps.append({"name": "store", "critical": True, "ok": False, "skipped": True, "seconds": 0.0,
                                   "detail": f"{self.left():.0f}s left < store_min_s {need:.0f}s"})
                self.reasons.append(f"store: skipped ({self.left():.0f}s left < store_min_s {need:.0f}s); "
                                    f"the periodic store snapshot stands")
                log("spill step store: SKIPPED (budget)")
            else:
                self.step("store", self.store_snapshot)
        if self.mode == "final" and not self.reasons:
            self.step("prune-old-spills", self.prune)
        return self.write_manifest(wts)

    def write_manifest(self, wts: List[Dict[str, Any]]) -> int:
        ok = not self.reasons
        status = {"periodic": "periodic", "final": "final", "manual": "manual"}[self.mode] if ok else "partial"
        t = time.time()
        main_head = next((w["head"] for w in wts if w["name"] == "main"), None)
        man = {
            "schema": SCHEMA,
            "status": status,
            "mode": self.mode,
            "reason": self.reason,
            "reasons": self.reasons,
            "spill_id": f"{self.jid}.{self.st['gen']}.{int(t * 1000)}",
            "jobid": self.jid,
            "gen": self.st["gen"],
            "host": socket.getfqdn(),
            "started_at": now_iso(self.t0),
            "written_at": now_iso(t),
            "written_at_epoch": t,
            "duration_s": round(t - self.t0, 2),
            "deadline": now_iso(self.deadline),
            "local_root": str(self.root),
            "work_project": str(self.work),
            "branch": self.cfg["branch"],
            "git_head": main_head,
            "refs_namespace": f"refs/nodelocal/{self.jid}/",
            "trees": self.trees,
            "sizes": {"dirty_bytes": sum(tr["bytes"] for tr in self.trees),
                      "files": sum(tr["overlay_count"] for tr in self.trees)},
            "sync": self.sync,
            "steps": self.steps,
            "restored_from": self.st.get("stage_in"),
        }
        write_json(self.dir / "MANIFEST.json", man, durable=True)
        try:
            (self.dir / "SPILL_IN_PROGRESS.json").unlink()
        except OSError:
            pass
        mk_path = self.root / ".git" / "nodelocal.json"
        mk = read_json(mk_path) or {}
        mk.update({"last_spill_id": man["spill_id"], "last_spill_status": status, "last_spill_at": man["written_at"]})
        try:
            write_json(mk_path, mk)
        except OSError:
            pass
        log(f"spill {self.mode} -> {status} in {man['duration_s']}s; manifest {self.dir / 'MANIFEST.json'}"
            + (f"; reasons: {self.reasons}" if self.reasons else ""))
        return 0 if ok else 2


def cmd_spill(args: argparse.Namespace) -> int:
    if args.mode == "final":
        # A second scancel/TERM must not cut the final spill short; children
        # inherit the ignored disposition.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    state = read_json(Path(args.state))
    if not state or not state.get("ready"):
        log(f"ERROR no ready node-local state at {args.state}; refusing to spill (stage-in never completed)")
        return 1
    lock_path = Path(state["base"]) / "spill.lock"
    fh = open(lock_path, "a")
    deadline = float(args.deadline) if args.deadline else None
    wait_until = (deadline - 2 * RESERVE_S) if deadline else time.time() + (600 if args.mode != "periodic" else 5)
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() >= wait_until:
                log(f"spill {args.mode}: another spill holds {lock_path}; giving up")
                return 3
            time.sleep(0.5)
    try:
        return Spill(state, args.mode, deadline, args.reason or args.mode, args.store).run()
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def cmd_status(args: argparse.Namespace) -> int:
    state = read_json(Path(args.state))
    if not state:
        print(f"no node-local state at {args.state}")
        return 1
    print(json.dumps({k: state[k] for k in ("jobid", "gen", "root", "work", "spill_dir", "staged_at", "stage_in")
                      if k in state}, indent=2))
    m = read_json(Path(state["spill_dir"]) / "MANIFEST.json")
    if m:
        print(json.dumps({k: m.get(k) for k in ("status", "mode", "written_at", "duration_s", "git_head", "reasons", "sync")},
                         indent=2))
        for t in m.get("trees", []):
            print(f"  {t['name']:<40} {(t['head'] or '')[:10]} dirty={t['dirty_count']} files={t['overlay_count']} bytes={t['bytes']}")
    else:
        print("no manifest yet")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--lanes", required=True)
    s = sub.add_parser("stage-in")
    s.add_argument("--lanes", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--jobid", required=True)
    s.add_argument("--gen", default="0")
    s.add_argument("--state", required=True)
    p = sub.add_parser("spill")
    p.add_argument("--state", required=True)
    p.add_argument("--mode", choices=("periodic", "final", "manual"), required=True)
    p.add_argument("--deadline", type=float)
    p.add_argument("--reason", default="")
    p.add_argument("--store", nargs=5, metavar=("PY", "SYNC", "PROJECT", "LOCAL", "PERSIST"))
    t = sub.add_parser("status")
    t.add_argument("--state", required=True)
    args = ap.parse_args(argv)
    return {"config": cmd_config, "stage-in": cmd_stage_in, "spill": cmd_spill, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
