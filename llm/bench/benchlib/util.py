"""Shared helpers: subprocesses, sizes, hashes, timestamps. Stdlib only."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time


def which(name: str):
    return shutil.which(name)


def run(argv, env=None, timeout=30, cwd=None, input_text=None):
    """Run argv and capture text output. Never raises for a non-zero exit.

    Returns dict(rc, out, err, cmd). `cmd` is the shell-quoted line so callers
    can print exactly what was attempted (dry-run and error paths).
    """
    argv = [str(a) for a in argv]
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=merged,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "rc": proc.returncode,
            "out": proc.stdout,
            "err": proc.stderr,
            "cmd": " ".join(shlex.quote(a) for a in argv),
        }
    except FileNotFoundError:
        return {"rc": 127, "out": "", "err": "not found: %s" % argv[0],
                "cmd": " ".join(shlex.quote(a) for a in argv)}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "", "err": "timeout after %ss" % timeout,
                "cmd": " ".join(shlex.quote(a) for a in argv)}


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def utc_iso(ts=None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else time.time()))


def human_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TiB" % n


def human_minutes(minutes) -> str:
    if minutes is None:
        return "unknown"
    minutes = int(minutes)
    if minutes < 60:
        return "%d min" % minutes
    return "%dh%02dm" % (minutes // 60, minutes % 60)


def sha256_file(path, limit=None) -> str:
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20 if left is None else min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
                if left <= 0:
                    break
    return h.hexdigest()


def file_size(path) -> int:
    try:
        return os.path.getsize(str(path))
    except OSError:
        return -1


def read_json(path, default=None):
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def parse_kv_list(values):
    """['a=1','b=2'] -> {'a':'1','b':'2'}; rejects entries without '='."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("expected KEY=VALUE, got %r" % item)
        k, v = item.split("=", 1)
        out[k] = v
    return out


def eprint(*args):
    print(*args, file=sys.stderr)


def shell_join(argv) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


def tail(text, lines=20) -> str:
    rows = (text or "").strip().splitlines()
    return "\n".join(rows[-lines:])
