#!/usr/bin/env python3
"""Unstage index entries that are stale leftovers of private-GIT_INDEX_FILE commits.

Lanes commit through a private GIT_INDEX_FILE, which moves HEAD but leaves the
shared .git/index holding older blobs. A later bare commit then reverts that
work. This tool finds those entries and, with --apply, runs
``git restore --staged`` on them (index := HEAD for those paths; the worktree
is never touched).

Per staged path (``git diff --cached --name-status -z``):

* status M/A/T with index blob != worktree blob AND the index blob equals the
  blob of that path in one of the last WINDOW commits that touch it
  (``git log -n WINDOW -- path``, then ``<commit>:<path>``) -> STALE;
* status D with the file present in the worktree AND in HEAD -> STALE;
* anything else (genuine staging, a brand-new blob never committed, a real
  deletion) -> KEEP.

Output: one JSON object ``{"stale": [...], "kept": [...], "applied": bool}``.
``applied`` is true only when ``--apply`` was given and every restore succeeded.
Dry-run is the default. If ``.git/index.lock`` blocks the restore, the script
waits up to 60 s and never deletes the lock. Exit 0 unless git itself fails.
"""
import argparse
import json
import os
import subprocess
import sys
import time

WINDOW = 200
LOCK_WAIT_S = 60.0
LOCK_POLL_S = 1.0
CHUNK = 300


class GitError(RuntimeError):
    pass


def git(cwd, *args, input_bytes=None, check=True):
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git {' '.join(args)} failed (rc={proc.returncode}): {err}")
    return proc


def toplevel(start):
    return git(start, "rev-parse", "--show-toplevel").stdout.decode("utf-8", "surrogateescape").strip()


def staged_entries(top):
    """(status, path) for every staged path. --no-renames keeps one path per entry."""
    out = git(top, "diff", "--cached", "--name-status", "-z", "--no-renames").stdout
    tokens = out.split(b"\0")
    entries = []
    i = 0
    while i + 1 < len(tokens):
        status = tokens[i].decode("utf-8", "surrogateescape")[:1]
        path = tokens[i + 1].decode("utf-8", "surrogateescape")
        entries.append((status, path))
        i += 2
    return entries


def index_blobs(top):
    """path -> blob sha for every entry in the shared index (stage 0)."""
    out = git(top, "ls-files", "-s", "-z").stdout
    blobs = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        fields = meta.split()
        if len(fields) >= 2:
            blobs[path.decode("utf-8", "surrogateescape")] = fields[1].decode()
    return blobs


def worktree_blob(top, path):
    full = os.path.join(top, path)
    if not os.path.lexists(full) or not os.path.isfile(full):
        return None
    return git(top, "hash-object", "--", path).stdout.decode().strip()


def in_head(top, path):
    return git(top, "cat-file", "-e", f"HEAD:{path}", check=False).returncode == 0


def historical_blobs(top, path):
    """Blobs of ``path`` in the last WINDOW commits touching it (same as rev-parse c:path)."""
    commits = git(top, "log", "--format=%H", "-n", str(WINDOW), "--", path).stdout.decode().split()
    if not commits:
        return set()
    req = "".join(f"{c}:{path}\n" for c in commits).encode("utf-8", "surrogateescape")
    out = git(top, "cat-file", "--batch-check", input_bytes=req).stdout.decode("utf-8", "replace")
    blobs = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            blobs.add(parts[0])
    return blobs


def classify(top, entries):
    idx = index_blobs(top)
    stale, kept = [], []
    for status, path in entries:
        exists_wt = os.path.lexists(os.path.join(top, path))
        if status == "D":
            is_stale = exists_wt and in_head(top, path)
        else:
            ib = idx.get(path)
            wb = worktree_blob(top, path)
            is_stale = (
                ib is not None
                and ib != wb
                and ib in historical_blobs(top, path)
            )
        (stale if is_stale else kept).append(path)
    return sorted(stale), sorted(kept)


def unstage(top, paths):
    """Run ``git restore --staged`` in chunks, waiting out index.lock for up to 60 s."""
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            proc = git(top, "restore", "--staged", "--", *chunk, check=False)
            if proc.returncode == 0:
                break
            err = proc.stderr.decode("utf-8", "replace").strip()
            if "index.lock" not in err:
                raise GitError(f"git restore --staged failed (rc={proc.returncode}): {err}")
            if time.monotonic() >= deadline:
                print(f"index-hygiene: index.lock still held after {LOCK_WAIT_S:.0f}s; "
                      "not applied (lock left in place)", file=sys.stderr)
                return False
            time.sleep(LOCK_POLL_S)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="unstage stale entries (default: dry-run)")
    ap.add_argument("--repo", default=".", help="any path inside the repository (default: .)")
    args = ap.parse_args(argv)
    try:
        top = toplevel(os.path.abspath(args.repo))
        entries = staged_entries(top)
        stale, kept = classify(top, entries)
        applied = False
        if args.apply:
            applied = unstage(top, stale) if stale else True
    except GitError as exc:
        print(f"index-hygiene: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"stale": stale, "kept": kept, "applied": applied}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
