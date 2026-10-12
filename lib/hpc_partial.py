"""Partial-result checkpointing for steward HPC job cards (stdlib only).

Job authors:
    from hpc_partial import PartialResults
    with PartialResults(out_path, meta={"job": "..."}) as pr:
        for net in nets:
            pr.add(f"net{net}", record)      # atomic rewrite after every unit
        pr.finish(summary)                   # status "complete", summary merged at top level

An exception inside the `with` block is recorded as status "failed" and re-raised.

Wrapper use (slurmhelps bin/hpc_run.sh, after a non-zero exit or SIGTERM):
    python3 hpc_partial.py finalize-failure --eval E --log L --exit-code RC --terminated 0|1
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

LOG_TAIL_LINES = 60


def _default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def write_atomic(path: Path, doc: dict) -> None:
    """Write JSON to a temp file in the same directory, then os.replace over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, default=_default))
    os.replace(tmp, path)


def read_partial(path: Path) -> dict | None:
    """Return the parsed JSON dict at path, or None if missing or unreadable (resume-safe)."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


class PartialResults:
    def __init__(self, path: str | Path, meta: dict | None = None, resume: bool = False):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.units: dict[str, Any] = {}
        if resume:
            old = read_partial(self.path)
            if old and old.get("status") == "partial" and old.get("meta") == self.meta:
                self.units = dict(old.get("units") or {})

    def __enter__(self) -> PartialResults:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fail(exc)
        return False  # always re-raise

    def has(self, key: str) -> bool:
        return key in self.units

    def _doc(self, status: str, extra: dict | None = None) -> dict:
        doc = dict(extra or {})
        doc.update(
            {
                "status": status,
                "meta": self.meta,
                "units_done": len(self.units),
                "units": self.units,
                "updated_at": _now(),
            }
        )
        return doc

    def add(self, key: str, record: Any) -> None:
        self.units[str(key)] = record
        write_atomic(self.path, self._doc("partial"))

    def finish(self, summary: dict) -> None:
        write_atomic(self.path, self._doc("complete", summary))

    def fail(self, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_atomic(
            self.path,
            self._doc("failed", {"error": f"{type(exc).__name__}: {exc}", "traceback": tb}),
        )


def checkpointed(path: str | Path, meta: dict | None = None) -> Callable:
    """Decorator: calls fn(pr, *args, **kwargs) inside a PartialResults."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with PartialResults(path, meta) as pr:
                return fn(pr, *args, **kwargs)

        return wrapper

    return deco


def finalize_failure(eval_path: Path, log_path: Path, exit_code: int, terminated: bool) -> dict:
    """Add exit info and a log tail to eval.json, or create a minimal failed one."""
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
    except OSError:
        lines = []
    tail = "\n".join(lines)
    doc = read_partial(eval_path)
    if doc is None:
        doc = {"status": "failed"}
    doc.update({"exit_code": exit_code, "terminated": bool(terminated), "log_tail": tail})
    write_atomic(Path(eval_path), doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finalize-failure")
    f.add_argument("--eval", required=True)
    f.add_argument("--log", required=True)
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--terminated", type=int, choices=(0, 1), required=True)
    a = ap.parse_args(argv)
    doc = finalize_failure(Path(a.eval), Path(a.log), a.exit_code, bool(a.terminated))
    print(json.dumps({"eval": a.eval, "status": doc.get("status"), "exit_code": a.exit_code}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
