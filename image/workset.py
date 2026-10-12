#!/usr/bin/env python3
"""workset.py — squashfs working-set and toolchain images for a Slurm fleet.

Source of truth: slurmhelps image/workset.py (entry point image/workset.sh).
Configuration: the "workset" section of the project's slurmhelps config
(defaults) overridden key by key by `fleet.workset` in the project's
.langywrap/lanes.json; paths come from the config's "site" section, exported
by lib/common.sh (SH_WORK_ROOT, SH_IMAGES_DIR, SH_NODE_TMP, ...).

Why: lane tool calls on the Lustre checkout have a p95 of 30-43 s. The working
set (the project checkout with .git, lane worktrees and .langywrap runtime,
plus agent homes) is packed into ONE squashfs file. A job copies it to
node-local tmp (one big sequential read), unpacks it there, and binds each tree
back over its ORIGINAL absolute path inside a private mount namespace
(`unshare -c -m`), so every absolute path (venv shebangs, HOME, git worktree
gitdir links) keeps working. Live Lustre dirs (results) are bound back on top.
A spill packs the node-local tree into a new image again.

Commands (common options: --project DIR, --lanes FILE, --kind data|toolchain)
  config                 shell assignments for supervisor.sbatch (never fails)
  build [--from-lustre] [--tolerance N] [--force]
                         image from the Lustre originals; publish
                         .new + fsync + rename, previous kept as .prev
  stage-in [--allow-prev]
                         copy images (toolchain, then data) to the stage dir,
                         unsquashfs, verify; copy secrets in. Exit 3 = no data
                         image published yet (caller may run on Lustre)
  hold                   start the namespace and stay in it (namespace holder);
                         publishes NS_PID in the stage state file
  enter -- CMD...        run CMD inside a fresh workset namespace
  exec -- CMD...         run CMD inside the holder's namespace (nsenter);
                         falls back to a fresh, equivalent namespace
  spill [--mode periodic|manual|final] [--final] [--deadline EPOCH]
        [--reason TEXT] [--pre SHELL]
                         pack the node-local tree into a new image. Exit 0 ok,
                         2 failed (published image untouched), 3 busy
  status                 image, manifest, stage and holder state
  view [--umount]        squashfuse-mount the newest image read-only at
                         <images>/view (login node)
  unpack DEST            unsquashfs the image into DEST (inspection)
  writeback [--apply]    rsync the image's trees back onto the Lustre
                         originals (cut-back from workset mode); dry run
                         unless --apply

Exit codes: 0 ok, 1 error, 2 spill/build failed with the published image
untouched, 3 refused/busy/nothing to do.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import fnmatch
import json
import os
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from slurmhelps import config as shconfig  # noqa: E402

SCHEMA = "slurmhelps-workset/v1"
MANIFEST_IN_IMAGE = ".workset/manifest.json"

DEFAULT_EXCLUDES = [
    # live logs, spools and runtime that are useless (or harmful) in an image
    ".log/",
    ".langywrap/activity-spool/",
    ".langywrap/artifacts/",
    ".langywrap/runtime-trash/",
    ".langywrap/.langypi-session.*",
    ".langywrap/*.lock",
    "runtime.sqlite3*",
    "sessions.sqlite3*",
    "*.sqlite3-wal",
    "*.sqlite3-shm",
    "*.sqlite3-journal",
    "*.sock",
    "index.lock",
    ".git/*.lock",
    "*.telemetry.jsonl",
    ".cache/",
    ".trash/",
    "uv/cache/",
]

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "image": "{images}/{project_name}.sqfs",
    "toolchain_image": "{images}/toolchain.sqfs",
    "stage_dir": "{node_tmp}/workset-{uid}",
    "trees": ["{project}"],
    "toolchain_dirs": None,  # default: config site.toolchain_dirs relative to {work_root}
    "live_binds": ["{project}/runs"],
    "secrets": [],
    "excludes": DEFAULT_EXCLUDES,
    "extra_excludes": [],
    "toolchain_excludes": ["uv/cache/", ".cache/", "*.sock"],
    "spill_interval_s": 900,
    "keep_prev": 1,
    "comp": "zstd",
    "comp_level": 3,
    "final_comp": "lz4",
    "spill_tolerance": 0.002,
    "final_budget_term_s": 25,
    "final_budget_usr1_s": 3000,
    "final_budget_end_s": 1800,
    "keep_stage_copy": False,
    "scan_threads": 16,
}


class WSError(RuntimeError):
    pass


class Refused(WSError):
    pass


def now_iso(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def log(msg: str) -> None:
    print(f"{now_iso()} workset: {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TiB"


def nproc() -> int:
    for k in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(k, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def fsync_path(p: Path) -> None:
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(d: Path) -> None:
    try:
        fsync_path(d)
    except OSError:
        pass


def write_atomic(path: Path, data: bytes, durable: bool = True) -> None:
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


def write_json(path: Path, obj: Any, durable: bool = True) -> None:
    write_atomic(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode(), durable)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def run(cmd: List[str], timeout: Optional[float] = None, check: bool = True, **kw: Any) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as exc:
        raise WSError(f"timeout after {timeout:.0f}s: {shlex.join(map(str, cmd))}") from exc
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-4:]
        raise WSError(f"rc={p.returncode}: {shlex.join(map(str, cmd))}: {' | '.join(tail)}")
    return p


# ---------------------------------------------------------------- config


def placeholders(project: Path, site: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ
    return {
        "work_root": env.get("SH_WORK_ROOT") or site.get("work_root", ""),
        "project": str(project),
        "project_name": project.name,
        "uid": str(os.getuid()),
        "images": env.get("SH_IMAGES_DIR") or site.get("images_dir", ""),
        "node_tmp": env.get("SH_NODE_TMP") or site.get("node_tmp", "") or "/tmp",
        "agents": env.get("AGENTS_DIR") or env.get("SH_AGENTS_DIR") or site.get("agents_dir", ""),
    }


def expand(v: str, ph: Dict[str, str]) -> str:
    def sub(m: "re.Match[str]") -> str:
        k = m.group(1)
        if k not in ph:
            raise WSError(f"unknown placeholder {{{k}}} in {v!r}")
        if not ph[k]:
            raise WSError(f"placeholder {{{k}}} in {v!r} is empty (config site section incomplete?)")
        return ph[k]

    return re.sub(r"\{([a-z_]+)\}", sub, v)


def load_config(project: Path, lanes: Optional[Path]) -> Dict[str, Any]:
    lanes = lanes or project / ".langywrap" / "lanes.json"
    with open(lanes) as fh:
        reg = json.load(fh)
    fleet = reg.get("fleet") or {}
    if "workset" not in fleet:
        raise Refused(f"{lanes} has no fleet.workset block")
    block = fleet["workset"]
    if not isinstance(block, dict):
        raise WSError("fleet.workset must be an object")
    try:
        shc = shconfig.load(project)
    except Exception as exc:  # a broken slurmhelps.json is reported, not fatal for defaults
        raise WSError(f"slurmhelps config of {project}: {exc}") from exc
    cfg = dict(DEFAULTS)
    for src in (shc.get("workset") or {}, block):
        cfg.update({k: v for k, v in src.items() if not k.startswith(("why", "_"))})
    cfg["enabled"] = bool(block.get("enabled", False))  # only the project registry switches it on
    ph = placeholders(project, shc.get("site") or {})
    if cfg.get("toolchain_dirs") is None:
        rel = os.environ.get("SH_TOOLCHAIN_DIRS", "").split() or list((shc.get("site") or {}).get("toolchain_dirs") or [])
        cfg["toolchain_dirs"] = [f"{{work_root}}/{d}" for d in rel]
    for k in ("image", "stage_dir"):
        cfg[k] = expand(str(cfg[k]), ph)
    if cfg.get("toolchain_image"):
        cfg["toolchain_image"] = expand(str(cfg["toolchain_image"]), ph)
    for k in ("trees", "toolchain_dirs", "live_binds", "secrets"):
        out = []
        for v in cfg.get(k) or []:
            e = os.path.normpath(expand(str(v), ph))
            if not e.startswith("/"):
                raise WSError(f"workset.{k}: {v!r} is not absolute")
            out.append(e)
        cfg[k] = out
    pats = list(cfg.get("excludes") or []) + list(cfg.get("extra_excludes") or [])
    cfg["excludes"] = [expand(p, ph) if "{" in p else p for p in pats]
    cfg["spill_interval_s"] = max(30, int(cfg["spill_interval_s"]))
    cfg["keep_prev"] = max(0, int(cfg["keep_prev"]))
    cfg["project"] = str(project)
    cfg["lanes"] = str(lanes)
    for a in cfg["trees"]:
        for b in cfg["trees"]:
            if a != b and b.startswith(a + "/"):
                raise WSError(f"workset.trees: {b} is inside {a}; list only the outer tree")
    return cfg


class Kind:
    """The two image kinds share one engine."""

    def __init__(self, cfg: Dict[str, Any], kind: str):
        self.cfg = cfg
        self.kind = kind
        self.stage = Path(cfg["stage_dir"])
        if kind == "data":
            self.image = Path(cfg["image"])
            self.trees = list(cfg["trees"])
            self.patterns = list(cfg["excludes"])
            self.live = list(cfg["live_binds"])
            self.secrets = list(cfg["secrets"])
            self.root = self.stage / "root"
        else:
            if not cfg.get("toolchain_image"):
                raise Refused("workset.toolchain_image is null: no toolchain image")
            self.image = Path(cfg["toolchain_image"])
            self.trees = list(cfg["toolchain_dirs"])
            self.patterns = list(cfg["toolchain_excludes"])
            self.live = []
            self.secrets = []
            self.root = self.stage / "toolchain-root"
        self.manifest = Path(str(self.image) + ".manifest.json")
        self.matcher = Matcher(self.patterns)


# ---------------------------------------------------------------- exclusion


class Matcher:
    """Exclude patterns, matched per entry:
      * trailing '/'        -> directories only
      * no '/' (basename)   -> fnmatch against the entry name at any depth
      * contains '/'        -> fnmatch against every component-aligned suffix
                               of the path relative to the tree root ('*'
                               crosses '/', so '.git/*.lock' matches
                               '.git/refs/heads/x.lock' and nested worktrees'
                               '.langywrap/...' match too)
      * leading '/'         -> fnmatch against the absolute path
    """

    def __init__(self, patterns: Iterable[str]):
        self.base: List[Tuple[str, bool]] = []
        self.rel: List[Tuple[str, bool]] = []
        self.abs: List[Tuple[str, bool]] = []
        for p in patterns:
            d = p.endswith("/")
            q = p.rstrip("/")
            if not q:
                continue
            if q.startswith("/"):
                self.abs.append((q, d))
            elif "/" in q:
                self.rel.append((q, d))
            else:
                self.base.append((q, d))

    def excluded(self, abspath: str, rel: str, name: str, is_dir: bool) -> bool:
        for q, d in self.base:
            if (is_dir or not d) and fnmatch.fnmatchcase(name, q):
                return True
        if self.rel:
            parts = rel.split("/")
            for q, d in self.rel:
                if d and not is_dir:
                    continue
                for i in range(len(parts)):
                    if fnmatch.fnmatchcase("/".join(parts[i:]), q):
                        return True
        for q, d in self.abs:
            if (is_dir or not d) and fnmatch.fnmatchcase(abspath, q):
                return True
        return False


class Counts:
    def __init__(self) -> None:
        self.files = 0
        self.dirs = 0
        self.symlinks = 0
        self.other = 0
        self.lock = threading.Lock()

    def as_dict(self) -> Dict[str, int]:
        return {"files": self.files, "dirs": self.dirs, "symlinks": self.symlinks, "other": self.other}


def git_tracked(tree: str, sub: str) -> List[str]:
    """Tracked files (relative to `tree`) under `sub` of the git work tree at `tree`."""
    if not (Path(tree) / ".git").exists():
        return []
    rel = os.path.relpath(sub, tree)
    p = run(["git", "-C", tree, "ls-files", "-z", "--", rel], check=False, timeout=600)
    if p.returncode != 0:
        return []
    return [x for x in p.stdout.split("\0") if x]


class Scan:
    """Walk the trees once, decide exclusions, count what the image will hold.

    `src_of(abs_tree)` maps a tree to the directory that holds its content
    (the Lustre original for build, the node-local root for spill). Excluded
    entries are recorded as absolute image paths, top-most only. The exclude
    file handed to mksquashfs lists them as absolute paths on the source side:
    mksquashfs (without -wildcards) excludes by device+inode of each listed
    path, which is exact for any file name and also holds across the build's
    bind mounts. Only directory entries' d_type is used (no stat per file):
    on Lustre a stat is the expensive metadata operation.
    """

    def __init__(self, k: Kind, src_of, threads: int, live_mode: str):
        self.k = k
        self.src_of = src_of
        self.threads = max(1, threads)
        self.live_mode = live_mode  # "tracked" (build) or "content" (spill)
        self.counts = Counts()
        self.excludes: List[str] = []
        self.sqlite: List[str] = []  # absolute (image) paths of *.sqlite3 kept
        self.notes: List[str] = []
        self.live = set(k.live)
        self.secrets = set(k.secrets)
        self._lock = threading.Lock()

    def _add_excl(self, abspath: str) -> None:
        with self._lock:
            self.excludes.append(abspath)

    def run(self) -> "Scan":
        jobs: List[Tuple[str, str, str]] = []
        for t in self.k.trees:
            src = self.src_of(t)
            if not os.path.isdir(src):
                raise WSError(f"tree {t}: source {src} is not a directory")
            jobs.append((t, src, t))
        with cf.ThreadPoolExecutor(max_workers=self.threads) as ex:
            pending = {ex.submit(self._dir, *j) for j in jobs}
            while pending:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    for sub in f.result():
                        pending.add(ex.submit(self._dir, *sub))
        self.excludes.sort()
        return self

    def _dir(self, abspath: str, src: str, tree: str) -> List[Tuple[str, str, str]]:
        subdirs: List[Tuple[str, str, str]] = []
        files = dirs = links = other = 0
        try:
            it = os.scandir(src)
        except OSError as exc:
            raise WSError(f"cannot list {src}: {exc}") from exc
        with it:
            for e in it:
                a = abspath + "/" + e.name
                rel = a[len(tree) + 1:]
                if a in self.secrets:
                    self._add_excl(a)
                    continue
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                    is_link = (not is_dir) and e.is_symlink()
                    is_file = (not is_dir) and (not is_link) and e.is_file(follow_symlinks=False)
                except OSError:
                    is_dir = is_link = is_file = False
                if self.k.matcher.excluded(a, rel, e.name, is_dir):
                    self._add_excl(a)
                    continue
                if is_dir:
                    dirs += 1
                    if a in self.live and self.live_mode == "tracked":
                        f2, d2 = self._live_tracked(a, e.path, tree)
                        files += f2
                        dirs += d2
                        continue
                    subdirs.append((a, e.path, tree))
                elif is_link:
                    links += 1
                elif is_file:
                    files += 1
                    if e.name.endswith(".sqlite3"):
                        with self._lock:
                            self.sqlite.append(a)
                else:
                    try:
                        m = os.lstat(e.path).st_mode
                    except OSError:
                        continue
                    if stat.S_ISSOCK(m) or stat.S_ISCHR(m) or stat.S_ISBLK(m):
                        self._add_excl(a)  # sockets/devices never belong in an image
                    else:
                        other += 1
        with self.counts.lock:
            c = self.counts
            c.files += files
            c.dirs += dirs
            c.symlinks += links
            c.other += other
        return subdirs

    def _live_tracked(self, abspath: str, src: str, tree: str) -> Tuple[int, int]:
        """A live (Lustre-only) dir in a build: keep only its git-tracked files
        (they belong to the checkout); exclude everything else, walking only
        along the tracked paths (results dirs can be huge)."""
        tracked = [os.path.relpath(os.path.join(tree, t), abspath) for t in git_tracked(tree, abspath)]
        keep_dirs = {"."}
        keep_files = set()
        for t in tracked:
            keep_files.add(t)
            d = os.path.dirname(t)
            while d:
                keep_dirs.add(d)
                d = os.path.dirname(d)
        files = dirs = 0
        for d in sorted(keep_dirs):
            full = os.path.normpath(os.path.join(src, d))
            try:
                names = os.listdir(full)
            except OSError:
                continue
            for n in names:
                r = os.path.normpath(os.path.join(d, n))
                if r in keep_dirs:
                    dirs += 1
                elif r in keep_files and not os.path.isdir(os.path.join(full, n)):
                    files += 1
                else:
                    self._add_excl(os.path.normpath(os.path.join(abspath, r)))
        if tracked:
            self.notes.append(f"live dir {abspath}: {len(tracked)} tracked file(s) kept in the image")
        return files, dirs


# ---------------------------------------------------------------- squashfs


_COMPS: Optional[List[str]] = None


def compressors() -> List[str]:
    global _COMPS
    if _COMPS is None:
        p = run(["mksquashfs", "-help"], check=False)
        text = (p.stdout or "") + (p.stderr or "")
        found = []
        i = text.find("Compressors available")
        if i >= 0:
            for line in text[i:].splitlines()[1:12]:
                w = line.strip().split()
                if w and w[0] in ("gzip", "lzma", "lzo", "lz4", "xz", "zstd"):
                    found.append(w[0])
        _COMPS = found or ["gzip"]
    return _COMPS


def comp_args(name: str, level: Optional[int]) -> List[str]:
    avail = compressors()
    for c in (name, "zstd", "lz4", "gzip"):
        if c in avail:
            args = ["-comp", c]
            if level is not None and c in ("zstd", "gzip"):
                args += ["-Xcompression-level", str(level)]
            return args
    return []


def image_listing_counts(img: Path, timeout: float = 1800) -> Dict[str, int]:
    """Entry counts from the image itself (unsquashfs -lls), root dir excluded."""
    p = subprocess.Popen(["unsquashfs", "-lls", "-d", "R", str(img)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, errors="replace")
    c = {"files": 0, "dirs": -1, "symlinks": 0, "other": 0, "bytes": 0}
    rx = re.compile(r"^([-dlcbps])[rwxsStT-]{9}\s+\S+\s+(\d+)")
    assert p.stdout is not None
    for line in p.stdout:
        m = rx.match(line)
        if not m:
            continue
        t = m.group(1)
        if t == "-":
            c["files"] += 1
            c["bytes"] += int(m.group(2))
        elif t == "d":
            c["dirs"] += 1
        elif t == "l":
            c["symlinks"] += 1
        else:
            c["other"] += 1
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise WSError(f"unsquashfs -lls {img} timed out")
    if rc != 0:
        raise WSError(f"unsquashfs -lls {img} rc={rc}: {(p.stderr.read() if p.stderr else '')[-300:]}")
    return c


def tree_counts(root: Path) -> Dict[str, int]:
    c = {"files": 0, "dirs": 0, "symlinks": 0, "other": 0}
    for dp, dns, fns in os.walk(root):
        for n in list(dns):
            if os.path.islink(os.path.join(dp, n)):
                c["symlinks"] += 1
                dns.remove(n)
            else:
                c["dirs"] += 1
        for n in fns:
            m = os.lstat(os.path.join(dp, n)).st_mode
            if stat.S_ISREG(m):
                c["files"] += 1
            elif stat.S_ISLNK(m):
                c["symlinks"] += 1
            else:
                c["other"] += 1
    return c


# ---- private mount namespaces without root ------------------------------------
# unshare(1) -c maps our uid to itself, so mount(8) refuses ("must be superuser":
# it is restricted for a non-root uid) and capabilities are dropped at its exec.
# We therefore create the user+mount namespace in THIS process (ctypes), keep
# the namespace's capabilities until the bind mounts are done, and only then
# exec the command: the command runs as our own uid, without capabilities, in
# the namespace. nsenter -U -m --preserve-credentials joins it later (also from
# another `srun --overlap` step: verified on the cluster 2026-10-12).
_CLONE_NEWNS, _CLONE_NEWUSER = 0x00020000, 0x10000000
_MS_BIND, _MS_REC, _MS_PRIVATE, _MNT_DETACH = 4096, 16384, 1 << 18, 2


def _libc():
    import ctypes

    return ctypes, ctypes.CDLL(None, use_errno=True)


def ns_create() -> None:
    ctypes, libc = _libc()
    uid, gid = os.getuid(), os.getgid()
    if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS) != 0:
        raise WSError(f"unshare(user+mount): {os.strerror(ctypes.get_errno())} "
                      f"(unprivileged user namespaces disabled on this host?)")
    with open("/proc/self/setgroups", "w") as fh:
        fh.write("deny")
    with open("/proc/self/uid_map", "w") as fh:
        fh.write(f"{uid} {uid} 1")
    with open("/proc/self/gid_map", "w") as fh:
        fh.write(f"{gid} {gid} 1")
    if libc.mount(None, b"/", None, _MS_REC | _MS_PRIVATE, None) != 0:
        raise WSError(f"make / private: {os.strerror(ctypes.get_errno())}")


def ns_apply(plan: List[List[str]]) -> None:
    """Run a mount plan inside the namespace just created: ["mkdir", p],
    ["bind", src, dst], ["umount", p] (lazy)."""
    ctypes, libc = _libc()
    for op in plan:
        if op[0] == "mkdir":
            os.makedirs(op[1], exist_ok=True)
        elif op[0] == "bind":
            if libc.mount(op[1].encode(), op[2].encode(), None, _MS_BIND | _MS_REC, None) != 0:
                raise WSError(f"bind {op[1]} -> {op[2]}: {os.strerror(ctypes.get_errno())}")
        elif op[0] == "umount":
            libc.umount2(op[1].encode(), _MNT_DETACH)
        else:
            raise WSError(f"bad mount op {op}")


def ns_run(plan: List[List[str]], cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run CMD in a fresh private namespace with PLAN applied (child process)."""
    planf = Path(os.environ.get("TMPDIR") or "/tmp") / f"ws-plan-{os.getpid()}-{time.monotonic_ns()}.json"
    planf.write_text(json.dumps(plan))
    try:
        return subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "_ns", str(planf), "--"] + cmd,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    finally:
        planf.unlink(missing_ok=True)


def q(s: Any) -> str:
    return shlex.quote(str(s))


def copy_file(src: Path, dst: Path, durable: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
        if durable:
            fo.flush()
            os.fsync(fo.fileno())


def publish(k: Kind, built: Path, manifest: Dict[str, Any], deadline: Optional[float]) -> None:
    """Copy `built` next to the image as .new, fsync, rotate .prev (hard link,
    so the image path never disappears), rename. A failure leaves the
    published image untouched."""
    img = k.image
    img.parent.mkdir(parents=True, exist_ok=True)
    new = img.with_name(img.name + ".new")
    try:
        if built.resolve() != new.resolve():
            t = time.monotonic()
            copy_file(built, new, durable=True)
            manifest["copy_s"] = round(time.monotonic() - t, 2)
        else:
            fsync_path(new)
        if new.stat().st_size != built.stat().st_size:
            raise WSError(f"copy of {built} to {new} is short")
        if deadline is not None and time.time() > deadline:
            raise WSError("deadline passed before the rename; not publishing")
        # the copy must be a readable squashfs before it can replace anything
        run(["unsquashfs", "-s", str(new)], timeout=120)
        if img.exists() and k.cfg["keep_prev"] > 0:
            for i in range(k.cfg["keep_prev"], 1, -1):  # .prev.N <- .prev.(N-1)
                a = img.with_name(img.name + (f".prev.{i - 1}" if i > 2 else ".prev"))
                b = img.with_name(img.name + f".prev.{i}")
                if a.exists():
                    os.replace(a, b)
                    am = Path(str(a) + ".manifest.json")
                    if am.exists():
                        os.replace(am, Path(str(b) + ".manifest.json"))
            prev = img.with_name(img.name + ".prev")
            tmp = img.with_name(img.name + f".prev.tmp{os.getpid()}")
            tmp.unlink(missing_ok=True)
            os.link(img, tmp)
            os.replace(tmp, prev)
            if k.manifest.exists():
                shutil.copyfile(k.manifest, Path(str(prev) + ".manifest.json"))
        manifest["image_bytes"] = new.stat().st_size
        manifest["published_at"] = now_iso()
        os.replace(new, img)
        fsync_dir(img.parent)
        write_json(k.manifest, manifest)
    except BaseException:
        try:
            new.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- build


def fleet_running() -> bool:
    name = os.environ.get("AGENTS_JOBNAME", "agent-supervisor")
    user = os.environ.get("USER", "")
    try:
        p = run(["squeue", "-h", "-u", user, "-n", name, "-t", "R,CG,PD", "-o", "%i"], check=False, timeout=60)
    except (WSError, FileNotFoundError):
        return False
    return bool(p.stdout.strip())


def cmd_build(k: Kind, a: argparse.Namespace) -> int:
    cur = read_json(k.manifest)
    if cur and cur.get("kind") == "spill" and not a.from_lustre:
        raise Refused(
            f"{k.image} was written by a spill ({cur.get('written_at')}): it is NEWER than the Lustre "
            f"originals. Rebuilding from Lustre would discard that work; pass --from-lustre to do it anyway "
            f"(after `workset.sh writeback --apply`, or deliberately)")
    if k.kind == "data" and fleet_running() and not a.force:
        raise Refused("the fleet allocation is queued or running; build from Lustre with the fleet down (or --force)")
    t0 = time.monotonic()
    k.stage.mkdir(parents=True, exist_ok=True)
    work = k.stage / f"build-{k.kind}"
    if work.exists():
        shutil.rmtree(work)
    sroot = work / "root"
    sroot.mkdir(parents=True)
    log(f"build {k.kind}: scanning {len(k.trees)} tree(s) on the source filesystem")
    sc = Scan(k, lambda t: t, threads=int(k.cfg["scan_threads"]), live_mode="tracked").run()
    log(f"scan: {sc.counts.files} files, {sc.counts.symlinks} symlinks, {sc.counts.dirs} dirs, "
        f"{len(sc.excludes)} excluded entries ({time.monotonic() - t0:.1f}s)")
    for n in sc.notes:
        log(n)
    excl = work / "excludes.txt"
    excl.write_text("".join(e + "\n" for e in sc.excludes))  # Lustre originals = the bind mounts' inodes
    manifest = {
        "schema": SCHEMA, "kind": "build", "image_kind": k.kind, "mode": "build", "source": "lustre",
        "trees": k.trees, "live_binds": k.live, "excludes": k.patterns, "scan": sc.counts.as_dict(),
        "written_at": now_iso(), "written_at_epoch": time.time(), "host": os.uname().nodename,
        "jobid": os.environ.get("SLURM_JOB_ID", ""), "spill_id": f"build.{int(time.time())}",
        "project": k.cfg["project"],
    }
    write_json(sroot / MANIFEST_IN_IMAGE, manifest, durable=False)
    out = work / "image.sqfs"
    plan: List[List[str]] = []
    for t in sorted(k.trees, key=lambda x: x.count("/")):
        plan.append(["mkdir", str(sroot) + t])
        plan.append(["bind", t, str(sroot) + t])
    comp = comp_args(k.cfg["comp"], k.cfg.get("comp_level"))
    mk = ["mksquashfs", str(sroot), str(out), "-noappend", "-no-xattrs", "-no-progress",
          "-processors", str(nproc()), "-ef", str(excl)] + comp
    log(f"mksquashfs ({' '.join(comp)}) -> {out}")
    t1 = time.monotonic()
    p = ns_run(plan, mk)
    if p.returncode != 0:
        raise WSError(f"mksquashfs failed rc={p.returncode}: {p.stdout[-1500:]}")
    manifest["mksquashfs_s"] = round(time.monotonic() - t1, 1)
    manifest["comp"] = comp
    got = image_listing_counts(out)
    manifest["image"] = got
    want_files = sc.counts.files + 1  # + the embedded manifest
    diff = abs(got["files"] - want_files) + abs(got["symlinks"] - sc.counts.symlinks)
    log(f"verify: image {got['files']} files / {got['symlinks']} symlinks / {human(got['bytes'])}; "
        f"source {want_files} / {sc.counts.symlinks}")
    if diff > a.tolerance:
        raise WSError(f"count mismatch image vs source ({diff} > tolerance {a.tolerance}); not publishing. "
                      f"Was something writing to the trees during the build?")
    publish(k, out, manifest, None)
    log(f"published {k.image} ({human(k.image.stat().st_size)}, {got['files']} files, "
        f"{human(got['bytes'])} uncompressed) in {time.monotonic() - t0:.1f}s")
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ---------------------------------------------------------------- stage-in


def free_bytes(p: Path) -> int:
    st = os.statvfs(str(p))
    return st.f_bavail * st.f_frsize


def stage_one(k: Kind, allow_prev: bool) -> Dict[str, Any]:
    candidates = [k.image] + ([k.image.with_name(k.image.name + ".prev")] if allow_prev else [])
    last: Optional[Exception] = None
    for img in candidates:
        if not img.exists():
            continue
        try:
            return _stage_from(k, img)
        except WSError as exc:
            last = exc
            log(f"ERROR stage-in of {img} failed: {exc}")
            if img == k.image and allow_prev:
                log(f"WARN --allow-prev: trying {img.name}.prev (work since that image was written is NOT restored)")
    if last:
        raise last
    raise Refused(f"no image at {k.image}")


def _stage_from(k: Kind, img: Path) -> Dict[str, Any]:
    t0 = time.monotonic()
    k.stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    man = read_json(Path(str(img) + ".manifest.json")) or {}
    size = img.stat().st_size
    need = size + int((man.get("image") or {}).get("bytes") or size * 3)
    if free_bytes(k.stage) < need:
        raise WSError(f"not enough node-local space in {k.stage}: need ~{human(need)}, "
                      f"free {human(free_bytes(k.stage))} (tmp counts against --mem)")
    stamp = k.stage / f"{k.kind}.staged.json"
    old = read_json(stamp)
    if old and k.root.is_dir() and old.get("image_mtime") == img.stat().st_mtime and old.get("image_size") == size:
        log(f"{k.kind}: {k.root} already holds {img} (same job); reusing it")
        return old
    copy = k.stage / f"{k.kind}.sqfs"
    copy_file(img, copy, durable=False)
    t_copy = time.monotonic() - t0
    tmp_root = k.stage / f"{k.kind}-root.unpacking"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    t1 = time.monotonic()
    p = subprocess.run(["unsquashfs", "-no-xattrs", "-no-progress", "-processors", str(nproc()),
                        "-d", str(tmp_root), str(copy)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise WSError(f"unsquashfs {copy} rc={p.returncode}: {p.stdout[-800:]}")
    t_unpack = time.monotonic() - t1
    want = image_listing_counts(copy)
    got = tree_counts(tmp_root)
    if (got["files"], got["symlinks"]) != (want["files"], want["symlinks"]):
        raise WSError(f"unpacked tree {got} does not match the image listing {want}")
    if k.root.exists():
        shutil.rmtree(k.root)
    os.replace(tmp_root, k.root)
    if not k.cfg.get("keep_stage_copy"):
        copy.unlink()
    inner = read_json(k.root / MANIFEST_IN_IMAGE) or {}
    rec = {"image": str(img), "image_mtime": img.stat().st_mtime, "image_size": size, "root": str(k.root),
           "files": got["files"], "spill_id": inner.get("spill_id"), "image_written_at": inner.get("written_at"),
           "staged_at": now_iso(), "copy_s": round(t_copy, 1), "unpack_s": round(t_unpack, 1)}
    write_json(stamp, rec, durable=False)
    log(f"{k.kind}: staged {img} ({human(size)}) -> {k.root}: {got['files']} files verified; "
        f"copy {t_copy:.1f}s, unsquashfs {t_unpack:.1f}s, total {time.monotonic() - t0:.1f}s")
    return rec


def copy_secret_in(path: str, root: Path) -> None:
    src = Path(path)
    dst = Path(str(root) + path)
    if not src.exists():
        log(f"secret {path} absent on the shared filesystem; skipped")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def cmd_stage_in(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    kinds = [a.kind] if a.kind_given else ["toolchain", "data"]
    for kind in kinds:
        try:
            k = Kind(cfg, kind)
        except Refused as exc:
            log(f"{kind}: {exc}; skipped")
            continue
        if kind == "toolchain" and not k.image.exists():
            log(f"WARN toolchain image {k.image} missing: the toolchain is used from the shared filesystem")
            continue
        if kind == "data" and not k.image.exists():
            log(f"no data image at {k.image}; nothing to stage (exit 3)")
            return 3
        stage_one(k, a.allow_prev)
        if kind == "data":
            for s in k.secrets:
                copy_secret_in(s, k.root)
            if k.secrets:
                log(f"copied {len(k.secrets)} secret path(s) into the node-local tree (never into an image)")
    return 0


# ---------------------------------------------------------------- namespace


def state_file(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["stage_dir"]) / "ns.env"


def ns_setup_plan(cfg: Dict[str, Any]) -> List[List[str]]:
    """Bind the staged trees over their absolute paths; live dirs back on top.
    Order: hold every live dir while it is still the shared-filesystem one,
    then cover the trees, then put the held live dirs back."""
    stage = Path(cfg["stage_dir"])
    binds: List[Tuple[str, str]] = []
    data_root = stage / "root"
    tc_root = stage / "toolchain-root"
    if not data_root.is_dir():
        raise WSError(f"{data_root} missing: run stage-in first")
    for t in cfg["trees"]:
        binds.append((str(data_root) + t, t))
    if tc_root.is_dir():
        for t in cfg["toolchain_dirs"]:
            if os.path.isdir(str(tc_root) + t):
                binds.append((str(tc_root) + t, t))
    binds.sort(key=lambda b: b[1].count("/"))
    plan: List[List[str]] = []
    hold = stage / "hold"
    live = list(cfg["live_binds"])
    for i, l in enumerate(live):
        plan += [["mkdir", l], ["mkdir", str(hold / str(i))], ["bind", l, str(hold / str(i))]]
    for src, dst in binds:
        plan += [["mkdir", dst], ["bind", src, dst]]
    for i, l in enumerate(live):
        plan += [["mkdir", l], ["bind", str(hold / str(i)), l], ["umount", str(hold / str(i))]]
    return plan


def ws_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["WORKSET_ACTIVE"] = "1"
    c = env.get("WORKSET_CACHE_DIR") or f"{env.get('SH_NODE_TMP') or '/tmp'}/workset-cache-{os.getuid()}"
    env.update({"WORKSET_CACHE_DIR": c, "UV_CACHE_DIR": f"{c}/uv", "PIP_CACHE_DIR": f"{c}/pip",
                "PYTHONPYCACHEPREFIX": f"{c}/pycache", "NPM_CONFIG_CACHE": f"{c}/npm",
                "npm_config_cache": f"{c}/npm", "YARN_CACHE_FOLDER": f"{c}/yarn", "XDG_CACHE_HOME": f"{c}/xdg"})
    for d in ("uv", "pip", "pycache", "npm", "xdg"):
        os.makedirs(f"{c}/{d}", exist_ok=True)
    return env


def enter_exec(cfg: Dict[str, Any], cmd: List[str], publish_pid: bool) -> None:
    """Create the workset namespace in this process and exec CMD in it (the
    PID stays the same: a holder's NS_PID is the process that lives in it)."""
    plan = ns_setup_plan(cfg)
    env = ws_env()
    cwd = os.getcwd()
    ns_create()
    ns_apply(plan)
    if publish_pid:
        import signal as _signal

        _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
        write_atomic(state_file(cfg), (f"NS_PID={os.getpid()}\nSTARTED={now_iso()}\n"
                                       f"HOST={os.uname().nodename}\n").encode(), durable=False)
    try:
        os.chdir(cwd)
    except OSError:
        os.chdir("/")
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)


def holder_pid(cfg: Dict[str, Any]) -> Optional[int]:
    st = state_file(cfg)
    try:
        for line in st.read_text().splitlines():
            if line.startswith("NS_PID="):
                pid = int(line.split("=", 1)[1])
                os.kill(pid, 0)
                return pid
    except (OSError, ValueError):
        return None
    return None


def cmd_exec(cfg: Dict[str, Any], cmd: List[str], fallback: bool) -> None:
    pid = holder_pid(cfg)
    if pid is not None:
        mine = os.readlink("/proc/self/ns/mnt")
        try:
            theirs = os.readlink(f"/proc/{pid}/ns/mnt")
        except OSError:
            theirs = ""
        env = ws_env()
        if theirs and theirs == mine:
            os.execvpe(cmd[0], cmd, env)
        nse = ["nsenter", "-t", str(pid), "-U", "-m", "--preserve-credentials"]
        probe = subprocess.run(nse + ["true"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if probe.returncode == 0:
            sys.stdout.flush()
            os.execvpe("nsenter", nse + ["--", "bash", "-c", 'cd -- "$1" 2>/dev/null || cd /; shift; exec "$@"',
                                         "workset-exec", os.getcwd()] + cmd, env)
        msg = f"nsenter into holder pid {pid} failed: {probe.stderr.strip()}"
    else:
        msg = f"no live namespace holder ({state_file(cfg)})"
    if not fallback:
        raise WSError(msg)
    print(f"{now_iso()} workset: WARN {msg}; using a fresh, equivalent namespace (same binds)", file=sys.stderr)
    enter_exec(cfg, cmd, publish_pid=False)


# ---------------------------------------------------------------- spill


def sqlite_backup(src: Path, dst: Path) -> None:
    from urllib.parse import quote as uq

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    s = sqlite3.connect(f"file:{uq(str(src))}?mode=ro", uri=True, timeout=20)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    shutil.copymode(src, dst)


def files_differ(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() != b.read_bytes()
    except OSError:
        return True


def write_back_secret(src: Path, dst: Path) -> bool:
    if not files_differ(src, dst):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.ws{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def cmd_spill(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    k = Kind(cfg, "data")
    mode = "final" if a.final else a.mode
    deadline = a.deadline
    if not k.root.is_dir():
        raise Refused(f"{k.root} missing: nothing staged in this job")
    lockf = open(k.stage / "spill.lock", "w")
    if mode == "periodic":
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("spill busy (another spill holds the lock); periodic spill skipped")
            return 3
    else:
        t = time.time()
        while True:
            try:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if deadline and time.time() > deadline - 5:
                    log("spill busy until the deadline; giving up")
                    return 3
                time.sleep(0.5)
        if time.time() - t > 1:
            log(f"waited {time.time() - t:.0f}s for a running spill")
    t0 = time.monotonic()
    w0 = time.time()
    reasons: List[str] = []
    jid = os.environ.get("SLURM_JOB_ID", "nojob")
    gen = os.environ.get("SLURM_RESTART_COUNT", "0")
    spill_id = f"{jid}.{gen}.{int(w0)}.{mode}"
    log(f"spill start (mode={mode}, reason={a.reason or '-'}, "
        f"{'deadline in %.0fs' % (deadline - w0) if deadline else 'no deadline'})")

    # 1. secrets back to the shared filesystem (small, most valuable)
    for s in k.secrets:
        src, dst = Path(str(k.root) + s), Path(s)
        try:
            if src.is_file():
                if write_back_secret(src, dst):
                    log(f"secret {s} changed on the node; written back")
            elif src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file():
                        write_back_secret(f, dst / f.relative_to(src))
        except OSError as exc:
            reasons.append(f"secret {s}: {exc}")
    # 2. pre hook (the session-store snapshot)
    if a.pre:
        left = (deadline - time.time() - 10) if deadline else 600
        if left > 3:
            try:
                p = subprocess.run(["bash", "-c", a.pre], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=max(3, min(600, left)))
                sys.stdout.write(p.stdout)
                if p.returncode != 0:
                    reasons.append(f"pre hook rc={p.returncode}")
            except subprocess.TimeoutExpired:
                reasons.append("pre hook timed out")
        else:
            reasons.append("pre hook skipped: deadline")
    # 3. scan the node-local tree
    sc = Scan(k, lambda t: str(k.root) + t, threads=4, live_mode="content").run()
    work = k.stage / "spill-work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    # 4. consistent copies of every SQLite DB in the tree (backup API), bound
    #    over the live file for mksquashfs only (private namespace)
    overlays: List[Tuple[str, str]] = []
    for i, dbabs in enumerate(sorted(sc.sqlite)):
        live = Path(str(k.root) + dbabs)
        snap = work / "sqlite" / f"{i}.sqlite3"
        try:
            sqlite_backup(live, snap)
            overlays.append((str(snap), str(live)))
        except (sqlite3.Error, OSError) as exc:
            reasons.append(f"sqlite {dbabs}: raw copy ({exc})")
    excl = work / "excludes.txt"
    excl.write_text("".join(str(k.root) + e + "\n" for e in sc.excludes))
    inner = {
        "schema": SCHEMA, "kind": "spill", "image_kind": "data", "mode": mode, "reason": a.reason,
        "trees": k.trees, "live_binds": k.live, "excludes": k.patterns, "scan": sc.counts.as_dict(),
        "written_at": now_iso(w0), "written_at_epoch": w0, "host": os.uname().nodename, "jobid": jid,
        "gen": gen, "spill_id": spill_id, "project": cfg["project"], "sqlite_snapshots": len(overlays),
    }
    write_json(k.root / MANIFEST_IN_IMAGE, inner, durable=False)
    img_size = k.image.stat().st_size if k.image.exists() else 0
    via_tmp = free_bytes(k.stage) > img_size * 2 + (1 << 30)
    out = work / "image.sqfs" if via_tmp else k.image.with_name(k.image.name + ".new")
    comp = comp_args(k.cfg["final_comp"] if mode == "final" else k.cfg["comp"],
                     None if mode == "final" else k.cfg.get("comp_level"))
    mk = ["mksquashfs", str(k.root), str(out), "-noappend", "-no-xattrs", "-no-progress",
          "-processors", str(nproc()), "-ef", str(excl)] + comp
    if mode == "final":
        # Measured on Troja (16 cores, 4.3 GB / 150k files in tmpfs): 8.4 s with the
        # duplicate scan, 4.8 s without. A final spill runs against KillWait.
        mk.append("-no-duplicates")
    plan = [["bind", s, d] for s, d in overlays]
    timeout = None
    if deadline:
        # keep time for the copy to the shared filesystem (~400 MB/s) + rename
        timeout = deadline - time.time() - (img_size / 4e8 if via_tmp else 0) - 3
        if timeout < 2:
            log("FAILED: no time left for mksquashfs before the deadline; the published image stands")
            return 2
    t1 = time.monotonic()
    try:
        p = ns_run(plan, mk, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"FAILED: mksquashfs did not finish within {timeout:.0f}s (deadline); the published image stands")
        Path(out).unlink(missing_ok=True)
        return 2
    if p.returncode != 0:
        log(f"FAILED: mksquashfs rc={p.returncode}: {p.stdout[-800:]}; the published image stands")
        Path(out).unlink(missing_ok=True)
        return 2
    t_mk = time.monotonic() - t1
    try:
        got = image_listing_counts(out)
        want = sc.counts.files + 1
        drift = abs(got["files"] - want) + abs(got["symlinks"] - sc.counts.symlinks)
        tol = max(50, int(want * float(k.cfg["spill_tolerance"])))
        if mode == "final":
            tol = max(5, tol // 10)
        if drift > tol:
            raise WSError(f"image holds {got['files']} files / {got['symlinks']} symlinks, scan {want} / "
                          f"{sc.counts.symlinks}: drift {drift} > {tol}")
        man = dict(inner)
        man.update({"image": got, "mksquashfs_s": round(t_mk, 1), "comp": comp, "reasons": reasons,
                    "drift": drift})
        publish(k, out, man, deadline)
    except (WSError, OSError) as exc:
        log(f"FAILED: {exc}; the published image stands")
        Path(out).unlink(missing_ok=True)
        return 2
    dur = time.monotonic() - t0
    size = k.image.stat().st_size
    write_json(k.stage / "last-spill.json", {"spill_id": spill_id, "mode": mode, "reason": a.reason,
                                             "duration_s": round(dur, 1), "image_bytes": size,
                                             "files": got["files"], "at": now_iso(), "reasons": reasons},
               durable=False)
    shutil.rmtree(work, ignore_errors=True)
    log(f"spill done (mode={mode}): {k.image} {human(size)}, {got['files']} files, "
        f"{human(got['bytes'])} uncompressed; mksquashfs {t_mk:.1f}s, total {dur:.1f}s"
        + (f"; notes: {reasons}" if reasons else ""))
    return 0


# ---------------------------------------------------------------- view / status / unpack / writeback


def mounted_at(path: Path) -> bool:
    try:
        with open("/proc/self/mounts") as fh:
            return any(len(line.split()) > 1 and line.split()[1] == str(path) for line in fh)
    except OSError:
        return False


def cmd_view(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    k = Kind(cfg, a.kind)
    view = Path(a.at) if a.at else k.image.parent / "view"
    if mounted_at(view):
        run(["fusermount", "-u", str(view)])
        log(f"unmounted {view}")
    if a.umount:
        return 0
    if not k.image.exists():
        raise Refused(f"no image at {k.image}")
    view.mkdir(parents=True, exist_ok=True)
    run(["squashfuse", str(k.image), str(view)])
    m = read_json(k.manifest) or {}
    log(f"mounted {k.image} read-only at {view} (written {m.get('written_at')}, {m.get('kind')}/{m.get('mode')}). "
        f"Only on host {os.uname().nodename}. Remove: workset.sh view --umount")
    return 0


def cmd_status(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    print(f"enabled={cfg['enabled']} project={cfg['project']} stage={cfg['stage_dir']}")
    for kind in ("data", "toolchain"):
        try:
            k = Kind(cfg, kind)
        except Refused as exc:
            print(f"[{kind}] {exc}")
            continue
        for img in (k.image, k.image.with_name(k.image.name + ".prev")):
            if img.exists():
                m = read_json(Path(str(img) + ".manifest.json")) or {}
                print(f"[{kind}] {img}: {human(img.stat().st_size)}, {m.get('kind')}/{m.get('mode')} "
                      f"written {m.get('written_at')} files={(m.get('image') or {}).get('files')} "
                      f"spill_id={m.get('spill_id')}")
            else:
                print(f"[{kind}] {img}: absent")
        st = read_json(k.stage / f"{kind}.staged.json")
        if st:
            print(f"[{kind}] staged {st.get('root')} from {st.get('image')} at {st.get('staged_at')}")
    pid = holder_pid(cfg)
    print(f"namespace holder: {pid if pid else 'none'} ({state_file(cfg)})")
    ls = read_json(Path(cfg["stage_dir"]) / "last-spill.json")
    if ls:
        print(f"last spill: {json.dumps(ls, sort_keys=True)}")
    return 0


def cmd_unpack(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    k = Kind(cfg, a.kind)
    dest = Path(a.dest)
    if dest.exists():
        raise Refused(f"{dest} exists")
    run(["unsquashfs", "-no-xattrs", "-no-progress", "-processors", str(nproc()), "-d", str(dest), str(k.image)])
    log(f"unpacked {k.image} into {dest}")
    return 0


def rsync_filters(k: Kind, tree: str) -> List[str]:
    """The image's exclusions as rsync filter rules relative to `tree`: excluded
    paths are protected on the receiving side (rsync never deletes them)."""
    out = []
    for p in k.patterns:
        d = "/" if p.endswith("/") else ""
        qq = p.rstrip("/")
        if qq.startswith("/"):
            if qq.startswith(tree + "/"):
                out.append("- /" + qq[len(tree) + 1:] + d)
        elif "/" in qq:
            out.append("- " + qq.replace("*", "**") + d)
        else:
            out.append("- " + qq + d)
    for p in list(k.live) + list(k.secrets):
        if p.startswith(tree + "/"):
            out.append("- /" + p[len(tree) + 1:])
    out.append("- /.workset/")
    return out


def cmd_writeback(cfg: Dict[str, Any], a: argparse.Namespace) -> int:
    k = Kind(cfg, "data")
    if holder_pid(cfg):
        raise Refused("a workset namespace is live on this node; stop the fleet first")
    if a.apply and fleet_running() and not a.force:
        raise Refused("the fleet allocation is queued or running; write back with the fleet down (or --force)")
    src_img = Path(a.image) if a.image else k.image
    work = k.stage / "writeback-root"
    if work.exists():
        shutil.rmtree(work)
    k.stage.mkdir(parents=True, exist_ok=True)
    run(["unsquashfs", "-no-xattrs", "-no-progress", "-processors", str(nproc()), "-d", str(work), str(src_img)])
    rc = 0
    for t in k.trees:
        src = str(work) + t
        if not os.path.isdir(src):
            log(f"{t}: not in the image; skipped")
            continue
        filt = k.stage / "writeback.filter"
        filt.write_text("\n".join(rsync_filters(k, t)) + "\n")
        cmd = ["rsync", "-aHS", "--delete", "--itemize-changes", f"--filter=merge {filt}", src + "/", t + "/"]
        if not a.apply:
            cmd.insert(1, "--dry-run")
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        changes = [line for line in p.stdout.splitlines() if line.strip()]
        dels = sum(1 for line in changes if line.startswith("*deleting"))
        log(f"{t}: {'applied' if a.apply else 'would apply'} {len(changes)} change(s) ({dels} deletion(s)); rsync rc={p.returncode}")
        for line in changes[: a.show]:
            print("   " + line)
        if p.returncode != 0:
            print(p.stderr[-800:], file=sys.stderr)
            rc = 1
    shutil.rmtree(work, ignore_errors=True)
    if not a.apply:
        log("dry run only; pass --apply to write")
    return rc


# ---------------------------------------------------------------- main


def cmd_config(project: Path, lanes: Optional[Path]) -> int:
    try:
        cfg = load_config(project, lanes)
    except Exception as exc:  # never fail the batch script on a config read
        print(f"# workset config unreadable or absent ({exc}); workset disabled", file=sys.stderr)
        print("WS_ENABLED=0")
        return 0
    if not cfg["enabled"]:
        print("WS_ENABLED=0")
        return 0
    out = {
        "WS_ENABLED": "1",
        "WS_IMAGE": cfg["image"],
        "WS_STAGE": cfg["stage_dir"],
        "WS_STATE": str(state_file(cfg)),
        "WS_SPILL_INTERVAL": str(cfg["spill_interval_s"]),
        "WS_BUDGET_TERM": str(int(cfg["final_budget_term_s"])),
        "WS_BUDGET_USR1": str(int(cfg["final_budget_usr1_s"])),
        "WS_BUDGET_END": str(int(cfg["final_budget_end_s"])),
    }
    for kk, v in out.items():
        print(f"{kk}={shlex.quote(v)}")
    return 0


def internal_ns(argv: List[str]) -> int:
    """`workset.py _ns PLAN.json -- CMD...`: fresh namespace, apply, exec."""
    plan = json.loads(Path(argv[0]).read_text())
    cmd = argv[2:] if argv[1:2] == ["--"] else argv[1:]
    try:
        ns_create()
        ns_apply(plan)
    except WSError as exc:
        print(f"workset _ns: {exc}", file=sys.stderr)
        return 1
    sys.stdout.flush()
    os.execvp(cmd[0], cmd)
    return 1


def main(argv: List[str]) -> int:
    if argv[:1] == ["_ns"]:
        return internal_ns(argv[1:])
    ap = argparse.ArgumentParser(prog="workset.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("SLURMHELPS_PROJECT") or os.environ.get("AGENT_PROJECT"))
    ap.add_argument("--lanes")
    ap.add_argument("--kind", choices=["data", "toolchain"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config")
    b = sub.add_parser("build")
    b.add_argument("--from-lustre", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--tolerance", type=int, default=0)
    si = sub.add_parser("stage-in")
    si.add_argument("--allow-prev", action="store_true")
    sub.add_parser("hold")
    for name in ("enter", "exec"):
        e = sub.add_parser(name)
        e.add_argument("--no-fallback", action="store_true")
        e.add_argument("argv", nargs=argparse.REMAINDER)
    sp = sub.add_parser("spill")
    sp.add_argument("--mode", choices=["periodic", "manual", "final"], default="manual")
    sp.add_argument("--final", action="store_true")
    sp.add_argument("--deadline", type=float)
    sp.add_argument("--reason", default="")
    sp.add_argument("--pre", default="")
    sub.add_parser("status")
    v = sub.add_parser("view")
    v.add_argument("--umount", action="store_true")
    v.add_argument("--at")
    u = sub.add_parser("unpack")
    u.add_argument("dest")
    wb = sub.add_parser("writeback")
    wb.add_argument("--apply", action="store_true")
    wb.add_argument("--force", action="store_true")
    wb.add_argument("--image")
    wb.add_argument("--show", type=int, default=20)
    a = ap.parse_args(argv)
    if not a.project:
        print("workset: --project (or SLURMHELPS_PROJECT / AGENT_PROJECT) is required", file=sys.stderr)
        return 1
    project = Path(a.project).absolute()
    lanes = Path(a.lanes) if a.lanes else None
    a.kind_given = a.kind is not None
    a.kind = a.kind or "data"
    if a.cmd == "config":
        return cmd_config(project, lanes)
    try:
        cfg = load_config(project, lanes)
        if a.cmd == "build":
            return cmd_build(Kind(cfg, a.kind), a)
        if a.cmd == "stage-in":
            return cmd_stage_in(cfg, a)
        if a.cmd == "hold":
            enter_exec(cfg, ["sleep", "infinity"], publish_pid=True)
        if a.cmd in ("enter", "exec"):
            cmd = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
            if not cmd:
                cmd = [os.environ.get("SHELL", "bash")]
            if a.cmd == "enter":
                enter_exec(cfg, cmd, publish_pid=False)
            cmd_exec(cfg, cmd, fallback=not a.no_fallback)
        if a.cmd == "spill":
            return cmd_spill(cfg, a)
        if a.cmd == "status":
            return cmd_status(cfg, a)
        if a.cmd == "view":
            return cmd_view(cfg, a)
        if a.cmd == "unpack":
            return cmd_unpack(cfg, a)
        if a.cmd == "writeback":
            return cmd_writeback(cfg, a)
    except Refused as exc:
        log(f"refused: {exc}")
        return 3
    except WSError as exc:
        log(f"FAILED {a.cmd}: {exc}")
        return 2 if a.cmd in ("build", "spill") else 1
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
