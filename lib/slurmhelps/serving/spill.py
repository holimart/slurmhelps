"""Idle checkpointing for the serving gateway (state-only by default).

This module deliberately does **not** pretend to serialize GPU KV tensors.
The SM80 vLLM fork has no KV export/import API; the safe implementation writes
an atomic, inspectable manifest of service/session metadata after an idle
period.  A future engine adapter can implement ``snapshot`` for raw blocks
without changing the idle/size/atomicity rules enforced here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .state import atomic_write_json, read_json


class SpillManager:
    def __init__(
        self,
        spill_dir: str | Path,
        *,
        idle_seconds: float = 600.0,
        max_gib: float = 96.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dir = Path(spill_dir)
        self.idle_seconds = float(idle_seconds)
        self.max_gib = float(max_gib)
        self.clock = clock
        self.last_activity = clock()
        self.last_spill: float | None = None
        if self.idle_seconds < 0:
            raise ValueError("idle_seconds must be >= 0")
        if self.max_gib < 0:
            raise ValueError("max_gib must be >= 0")

    def note_activity(self, when: float | None = None) -> None:
        self.last_activity = self.clock() if when is None else float(when)

    def idle_for(self, when: float | None = None) -> float:
        now = self.clock() if when is None else float(when)
        return max(0.0, now - self.last_activity)

    def due(self, *, active_requests: int = 0, when: float | None = None) -> bool:
        if active_requests > 0:
            return False
        return self.idle_for(when) >= self.idle_seconds

    def snapshot(self, payload: Mapping[str, Any], *, active_requests: int = 0,
                 when: float | None = None) -> Path | None:
        """Write an atomic state-only manifest; returns None when not due."""
        if not self.due(active_requests=active_requests, when=when):
            return None
        body = {
            "schema": 1,
            "mode": "state-only",
            "spilled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "idle_seconds": round(self.idle_for(when), 3),
            "payload": dict(payload),
        }
        encoded = json.dumps(body, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode("utf-8")) > self.max_gib * (1024 ** 3):
            raise ValueError("spill exceeds max_gib")
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "manifest.json"
        atomic_write_json(path, body)
        self.last_spill = self.clock() if when is None else float(when)
        self.note_activity(self.last_spill)
        return path

    def read(self) -> dict[str, Any] | None:
        return read_json(self.dir / "manifest.json")

    def status(self, *, active_requests: int = 0, when: float | None = None) -> dict[str, Any]:
        return {
            "mode": "state-only",
            "idle_seconds": self.idle_seconds,
            "idle_for_s": round(self.idle_for(when), 3),
            "due": self.due(active_requests=active_requests, when=when),
            "last_spill": self.last_spill,
            "dir": str(self.dir),
            "manifest_exists": (self.dir / "manifest.json").is_file(),
        }
