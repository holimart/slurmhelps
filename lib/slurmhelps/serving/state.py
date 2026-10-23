#!/usr/bin/env python3
"""Crash-safe serving state and load-duration history (stdlib only).

The worker writes ``state.json`` through :func:`atomic_write_json`; the
controller reads it plus ``history.jsonl`` to report an honest load estimate.
A missing or corrupt file is reported as ``None`` -- never as a guessed state.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import COLD_FALLBACK_S

SCHEMA = 1
DEFAULT_HEARTBEAT_TIMEOUT_S = 90.0

STATES = (
    "queued",
    "starting",
    "loading",
    "warming",
    "ready",
    "draining",
    "stopped",
    "failed",
    "preempted",
    "requeued",
)

# Exactly the transitions from docs/serving-gateway.md.  ``failed`` and
# ``preempted`` are reachable from any state (including a terminal one, which
# keeps a re-report idempotent).
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"starting", "failed", "preempted"}),
    "starting": frozenset({"loading", "failed", "preempted"}),
    "loading": frozenset({"warming", "failed", "preempted"}),
    "warming": frozenset({"ready", "failed", "preempted"}),
    "ready": frozenset({"draining", "failed", "preempted"}),
    "draining": frozenset({"stopped", "failed", "preempted"}),
    "stopped": frozenset({"failed", "preempted"}),
    "failed": frozenset({"failed", "preempted"}),
    "preempted": frozenset({"requeued", "failed", "preempted"}),
    "requeued": frozenset({"starting", "failed", "preempted"}),
}

STATE_FIELDS = (
    "job_id",
    "node",
    "state",
    "phase",
    "progress",
    "load_elapsed_s",
    "load_estimate_s",
    "load_estimate_source",
    "endpoint",
    "capacity",
    "active_requests",
    "queued_requests",
    "last_heartbeat",
    "error",
    "attempt",
    "updated_at",
)

_PHASE_ALIASES = {
    "weight": "weights",
    "weights": "weights",
    "weight_load": "weights",
    "loading_weights": "weights",
    "kv": "kv",
    "kv_init": "kv",
    "loading_kv": "kv",
    "warmup": "warmup",
    "warming": "warmup",
    "warmup_load": "warmup",
}


def utc_now(clock: Any = time.time) -> str:
    """ISO-8601 UTC with a ``Z`` suffix (the documented state format)."""
    stamp = clock() if callable(clock) else clock
    return datetime.fromtimestamp(float(stamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> float | None:
    """Parse ISO-8601 / epoch seconds into epoch seconds; ``None`` if invalid."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def heartbeat_age_s(last_heartbeat: Any, now: Any = None) -> float | None:
    """Seconds since ``last_heartbeat``; ``None`` when it is absent/invalid."""
    beat = parse_time(last_heartbeat)
    if beat is None:
        return None
    current = parse_time(now) if now is not None else time.time()
    if current is None:
        return None
    return max(0.0, current - beat)


def is_stale(
    last_heartbeat: Any = None,
    now: Any = None,
    timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
) -> bool:
    """True when a heartbeat is missing or older than ``timeout_s``."""
    age = heartbeat_age_s(last_heartbeat, now)
    return age is None or age > float(timeout_s)


def normalize_phase(phase: Any) -> str:
    """Map worker phase spellings onto the three measured load components."""
    text = str(phase or "").strip().lower()
    if text.startswith("loading:"):
        text = text.split(":", 1)[1]
    text = text.replace("-", "_").replace(" ", "_")
    if text in _PHASE_ALIASES:
        return _PHASE_ALIASES[text]
    if text.startswith("loading_"):
        text = text[len("loading_") :]
    return _PHASE_ALIASES.get(text, text)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (small samples should not imply precision)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def atomic_write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, fsync, replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(data), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # directory fsync is best-effort on filesystems that refuse it
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read a JSON object; missing, non-object or invalid JSON -> ``None``."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    schema = data.get("schema")
    if schema is not None and schema != SCHEMA:
        return None
    return data


def default_state() -> dict[str, Any]:
    """A complete state object with every documented field present."""
    state: dict[str, Any] = {field: None for field in STATE_FIELDS}
    state.update(
        {
            "schema": SCHEMA,
            "state": None,
            "phase": None,
            "progress": None,
            "active_requests": 0,
            "queued_requests": 0,
            "attempt": 0,
        }
    )
    return state


def normalize_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill missing documented fields without inventing a service state."""
    out = default_state()
    if state:
        out.update(dict(state))
    out["schema"] = SCHEMA
    if out.get("state") is not None:
        out["state"] = str(out["state"])
    if out.get("attempt") is None:
        out["attempt"] = 0
    return out


class StateStore:
    """``<dir>/state.json`` + ``<dir>/history.jsonl`` + a lock file."""

    def __init__(self, directory: str | os.PathLike[str], *, clock: Any = time.time) -> None:
        self.dir = Path(directory)
        self.path = self.dir / "state.json"
        self.history_path = self.dir / "history.jsonl"
        self.lock_path = self.dir / "lock"
        self.clock = clock
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- timestamps ----------------------------------------------------
    def now_iso(self) -> str:
        return utc_now(self.clock)

    # -- state ---------------------------------------------------------
    def read(self) -> dict[str, Any] | None:
        state = read_json(self.path)
        return normalize_state(state) if state is not None else None

    def write(self, state: Mapping[str, Any]) -> dict[str, Any]:
        out = normalize_state(state)
        out.setdefault("updated_at", None)
        if not out.get("updated_at"):
            out["updated_at"] = self.now_iso()
        atomic_write_json(self.path, out)
        return out

    def update(
        self, changes: Mapping[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Merge field changes.  A changed ``state`` goes through validation."""
        merged: dict[str, Any] = dict(changes or {})
        merged.update(fields)
        current = self.read() or default_state()
        new_state = merged.get("state")
        if new_state is not None and new_state != current.get("state"):
            return self.transition(
                str(new_state), **{k: v for k, v in merged.items() if k != "state"}
            )
        current.update(merged)
        current["updated_at"] = self.now_iso()
        return self.write(current)

    def transition(self, new_state: str, **changes: Any) -> dict[str, Any]:
        """Apply the documented state machine or raise ``ValueError``."""
        new_state = str(new_state)
        if new_state not in STATES:
            raise ValueError(f"unknown serving state: {new_state}")
        current = self.read()
        if current is None:
            current = default_state()
        old_state = current.get("state")
        if old_state is not None and new_state != old_state:
            allowed = ALLOWED_TRANSITIONS.get(str(old_state), frozenset())
            # "any -> failed/preempted" holds even for an unknown stale state.
            if new_state not in allowed and new_state not in {"failed", "preempted"}:
                raise ValueError(f"invalid serving transition: {old_state} -> {new_state}")
        current.update(changes)
        current["state"] = new_state
        current["updated_at"] = self.now_iso()
        return self.write(current)

    def heartbeat(
        self,
        *,
        phase: str | None = None,
        progress: float | None = None,
        **changes: Any,
    ) -> dict[str, Any]:
        """Record a worker heartbeat; never changes ``state`` by itself."""
        current = self.read() or default_state()
        if phase is not None:
            current["phase"] = phase
        if progress is not None:
            current["progress"] = progress
        current.update(changes)
        stamp = self.now_iso()
        current["last_heartbeat"] = stamp
        current["updated_at"] = stamp
        return self.write(current)

    def stale(self, now: Any = None, timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S) -> bool:
        current = self.read()
        if current is None:
            return True
        return is_stale(current.get("last_heartbeat"), self.clock() if now is None else now, timeout_s)

    # ``store.is_stale(now, timeout_s)`` convenience, matching the helper.
    def is_stale(self, now: Any = None, timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S) -> bool:
        return self.stale(now, timeout_s)

    # -- locking -------------------------------------------------------
    @contextlib.contextmanager
    def lock(self, *, blocking: bool = True) -> Iterator[None]:
        """Exclusive advisory lock so two controllers cannot own one job."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(fd, flags)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # -- load-duration history ----------------------------------------
    def history(self) -> list[dict[str, Any]]:
        """Read valid JSONL events; malformed lines are ignored, not trusted."""
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def load_history(self) -> list[dict[str, Any]]:
        """Alias for :meth:`history` (the on-disk ``history.jsonl`` events)."""
        return self.history()

    def append_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Append one load-history event and fsync it."""
        record = dict(event)
        record.setdefault("ts", self.now_iso())
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def record_load(
        self,
        phase: str,
        duration_s: float,
        ok: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        """Record ``{phase, duration_s, ok, ...}``; the load-estimate input."""
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise ValueError("duration_s must be a number")
        if not math.isfinite(float(duration_s)) or float(duration_s) < 0:
            raise ValueError("duration_s must be finite and >= 0")
        record = {
            "phase": normalize_phase(phase),
            "raw_phase": str(phase),
            "duration_s": float(duration_s),
            "ok": bool(ok),
        }
        if not ok:
            record["error"] = str(extra.get("error", "failed"))
        record.update({k: v for k, v in extra.items() if k not in record})
        return self.append_event(record)

    def estimate(self, phase: str, source: str = "auto") -> dict[str, Any]:
        """Median / p90 / sample count, or an explicit cold fallback.

        Only successful history samples are used.  The result always carries
        its ``source``; a fallback is a range, never an implied exact ETA.
        """
        canonical = normalize_phase(phase)
        events = []
        for event in self.history():
            if event.get("ok") is False:
                continue
            if normalize_phase(event.get("phase")) != canonical:
                continue
            duration = event.get("duration_s")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                continue
            if not math.isfinite(float(duration)) or float(duration) < 0:
                continue
            if source and source not in {"auto", "*"}:
                if str(event.get("source", "")) != source:
                    continue
            events.append(float(duration))

        if events:
            return {
                "phase": canonical,
                "p50": float(statistics.median(events)),
                "p90": _percentile(events, 0.90),
                "samples": len(events),
                "source": f"history:{source}" if source not in {"auto", "*"} else "history",
                "cold": False,
            }

        fallback = COLD_FALLBACK_S.get(canonical)
        if fallback is None:
            return {
                "phase": canonical,
                "p50": None,
                "p90": None,
                "samples": 0,
                "source": "no-history",
                "cold": True,
            }
        # The measured cold start is 9-10 min; p90 is deliberately a range
        # rather than a second exact-looking point estimate.
        return {
            "phase": canonical,
            "p50": float(fallback),
            "p90": round(float(fallback) * 1.15, 1),
            "samples": 0,
            "source": "cold-p50",
            "cold": True,
        }


__all__ = [
    "ALLOWED_TRANSITIONS",
    "DEFAULT_HEARTBEAT_TIMEOUT_S",
    "SCHEMA",
    "STATE_FIELDS",
    "STATES",
    "StateStore",
    "atomic_write_json",
    "default_state",
    "heartbeat_age_s",
    "is_stale",
    "normalize_phase",
    "normalize_state",
    "parse_time",
    "read_json",
    "utc_now",
]
