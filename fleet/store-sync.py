#!/usr/bin/env python3
"""store-sync.py — keep the langywrap session store on node-local tmpfs.

Source of truth: slurmhelps fleet/store-sync.py.

Why (2026-10-10, .langywrap/audits/store-stall-2026-10-10T1004Z.md): with the
store on Lustre, the daemon and its lanes never left SQLite a reader-free
moment, so the WAL was never reset (976 MB) and every -shm lookup paid Lustre
latency; the fleet stalled. The store now lives in a node-local directory
(LANGYWRAP_SESSIONS_DB_DIR) and this script keeps a durable copy under /work.

  stage-in  <project> <local-dir> <persist-dir>
      Run before the daemon starts. A regular-file store in <project>/.langywrap
      is the legacy Lustre store: checkpoint it (TRUNCATE), copy it into
      <local-dir> with the SQLite backup API, write the snapshot, then park the
      legacy file and its -wal/-shm under <persist-dir>/legacy-<ts>/ so nothing
      can open the stale copy. Otherwise an existing <local-dir> file wins (it
      is newer than any snapshot); failing that the newest snapshot is restored.
  snapshot  <project> <local-dir> <persist-dir>
      Consistent online copy (backup API) of <local-dir> into <persist-dir>,
      published by tmp + fsync + rename. Safe while the daemon runs.
  status    <project> <local-dir> <persist-dir>
      Print sizes and mtimes of the legacy path, local store and snapshot.

Ephemeral session store: lanes.json "fleet": {"session_store": {"persist": false}}
makes sessions.sqlite3 node-local only. stage-in never restores it (the daemon
creates a fresh one) and snapshot never copies it to /work; langypi-mail.sqlite3
is still persisted. Default persist=true keeps the behaviour above.

Symlinks are never created. langywrap refuses symlinked DB, -wal/-shm and
parent paths, so a symlink found in <project>/.langywrap makes stage-in fail.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote

DBS = ("sessions.sqlite3", "langypi-mail.sqlite3")
SIDECARS = ("", "-wal", "-shm")


class StoreError(RuntimeError):
    pass


def ephemeral_dbs(project: Path) -> set[str]:
    """DB names that are never restored or snapshotted (lanes.json fleet.session_store.persist=false)."""
    import json

    try:
        cfg = json.loads((project / ".langywrap" / "lanes.json").read_text())
        persist = (cfg.get("fleet") or {}).get("session_store", {}).get("persist", True)
    except (OSError, ValueError, AttributeError):
        return set()
    return set() if persist is not False else {"sessions.sqlite3"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"{now()} store-sync: {msg}", flush=True)


def size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))


def fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def backup(src: Path, dst: Path) -> None:
    """Consistent copy of src into dst: backup API, quick_check, tmp + fsync + rename."""
    dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    tmp.unlink(missing_ok=True)
    s = sqlite3.connect(f"file:{quote(str(src))}?mode=ro", uri=True, timeout=60)
    d = None
    try:
        d = sqlite3.connect(tmp)
        s.backup(d)  # one call = one read transaction, a consistent image
        ok = d.execute("PRAGMA quick_check").fetchone()[0]
        if ok != "ok":
            raise StoreError(f"quick_check of copy {tmp}: {ok}")
        d.execute("PRAGMA journal_mode=WAL")
    finally:
        if d is not None:
            d.close()  # explicit: a `with sqlite3.connect()` block does not close
        s.close()
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def checkpoint(db: Path) -> None:
    wal = db.with_name(db.name + "-wal")
    before = size(wal)
    c = sqlite3.connect(db, timeout=600)
    try:
        busy, _, _ = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        c.close()
    if busy:
        raise StoreError(f"checkpoint of {db} busy; a process still holds it open")
    log(f"checkpointed {db}: wal {before} -> {size(wal)} B, db {size(db)} B")


def refuse_symlinks(project: Path) -> None:
    lw = project / ".langywrap"
    if lw.is_symlink():
        raise StoreError(f"{lw} is a symlink -> {os.readlink(lw)}; replace it with a real directory")
    for name in DBS:
        for suffix in SIDECARS:
            p = lw / (name + suffix)
            if p.is_symlink():
                raise StoreError(
                    f"{p} is a symlink -> {os.readlink(p)}; langywrap refuses symlinked "
                    f"stores. Remove the symlink (its target is not the store) and retry."
                )


def remove_stale_tmp(d: Path) -> None:
    for name in DBS:
        for p in d.glob(name + ".tmp*"):
            if p.is_file() and not p.is_symlink():
                p.unlink()
                log(f"removed stale partial copy {p}")


def stage_in(project: Path, local: Path, persist: Path) -> None:
    refuse_symlinks(project)
    lw = project / ".langywrap"
    if local.is_symlink():
        raise StoreError(f"{local} is a symlink; the node-local store dir must be real")
    local.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(local, 0o700)
    if persist.is_symlink():
        raise StoreError(f"{persist} is a symlink; the persist dir must be real")
    persist.mkdir(parents=True, exist_ok=True, mode=0o700)
    remove_stale_tmp(local)
    remove_stale_tmp(persist)
    ephemeral = ephemeral_dbs(project)
    for name in DBS:
        legacy, target, snap = lw / name, local / name, persist / name
        if name in ephemeral and not legacy.is_file():
            if target.exists():
                log(f"keeping node-local {target} ({size(target)} B; same allocation, ephemeral store)")
            else:
                log(f"ephemeral {name} (fleet.session_store.persist=false): not restored; the daemon will create {target}")
            continue
        if legacy.is_file():
            if target.exists():
                raise StoreError(
                    f"both legacy {legacy} and node-local {target} exist; refusing to overwrite "
                    f"the local store. Move the legacy file aside by hand, then retry."
                )
            log(f"legacy store {legacy} ({size(legacy)} B, wal {size(lw / (name + '-wal'))} B)")
            checkpoint(legacy)
            backup(legacy, target)
            log(f"copied legacy {name} to {target} ({size(target)} B), quick_check ok")
            backup(target, snap)
            log(f"snapshot written {snap} ({size(snap)} B)")
            parked = persist / f"legacy-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            parked.mkdir(parents=True, exist_ok=True, mode=0o700)
            for suffix in SIDECARS:
                p = lw / (name + suffix)
                if p.exists():
                    shutil.move(str(p), parked / p.name)
            log(f"parked legacy store under {parked}")
        elif target.exists():
            log(f"keeping node-local {target} ({size(target)} B; newer than any snapshot)")
        elif snap.exists():
            backup(snap, target)
            log(f"restored {target} from {snap} ({size(snap)} B)")
        else:
            log(f"no store yet for {name}; the daemon will create {target}")
    log(f"stage-in done: local store {local}")


def snapshot(project: Path, local: Path, persist: Path) -> None:
    if not local.is_dir() or local.is_symlink():
        raise StoreError(f"node-local store dir {local} missing; run stage-in first")
    persist.mkdir(parents=True, exist_ok=True, mode=0o700)
    ephemeral = ephemeral_dbs(project)
    for name in DBS:
        target = local / name
        if not target.exists() or name in ephemeral:
            continue
        t0 = time.monotonic()
        backup(target, persist / name)
        log(
            f"snapshot {name}: db {size(target)} B, wal {size(local / (name + '-wal'))} B "
            f"-> {persist / name} in {time.monotonic() - t0:.2f}s"
        )


def status(project: Path, local: Path, persist: Path) -> None:
    lw = project / ".langywrap"
    print(f"now {now()}  project {project}")
    for name in DBS:
        link = lw / name
        if link.is_symlink():
            kind = f"SYMLINK -> {os.readlink(link)} (must be removed)"
        elif link.is_file():
            kind = f"legacy regular file, {size(link)} B"
        else:
            kind = "absent"
        print(f"legacy {link}: {kind}")
        for p in (local / name, local / (name + "-wal"), local / (name + "-shm"), persist / name):
            if p.exists():
                print(f"  {p}: {size(p)} B, mtime {mtime(p)}")
            else:
                print(f"  {p}: absent")
    parked = sorted(persist.glob("legacy-*")) if persist.is_dir() else []
    print(f"parked legacy stores under {persist}: {len(parked)}")


def main(argv: list[str]) -> int:
    if len(argv) != 5 or argv[1] not in {"stage-in", "snapshot", "status"}:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, project, local, persist = argv[1], Path(argv[2]), Path(argv[3]), Path(argv[4])
    try:
        {"stage-in": stage_in, "snapshot": snapshot, "status": status}[cmd](project, local, persist)
    except StoreError as exc:
        log(f"FAILED {cmd}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
