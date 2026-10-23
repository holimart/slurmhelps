"""State-only idle spill policy; never claims raw KV serialization."""
from __future__ import annotations

import json

import pytest

from slurmhelps.serving.spill import SpillManager


def test_spill_waits_for_idle_window(tmp_path):
    class Clock:
        now = 100.0

        def __call__(self):
            return self.now

    clock = Clock()
    spill = SpillManager(tmp_path, idle_seconds=600, clock=clock)
    spill.note_activity(100.0)
    clock.now = 699.0
    assert not spill.due()
    assert spill.snapshot({"state": "ready"}) is None
    clock.now = 700.0
    assert spill.due()
    path = spill.snapshot({"state": "ready", "capacity": {"gpu_kv_tokens": 1_765_762}})
    assert path and path.name == "manifest.json"
    data = json.loads(path.read_text())
    assert data["mode"] == "state-only"
    assert data["payload"]["capacity"]["gpu_kv_tokens"] == 1_765_762
    assert spill.read()["schema"] == 1


def test_active_requests_block_spill(tmp_path):
    spill = SpillManager(tmp_path, idle_seconds=0)
    spill.note_activity(0)
    assert not spill.due(active_requests=1)
    assert spill.snapshot({"state": "ready"}, active_requests=1, when=10) is None


def test_new_activity_cancels_due_spill(tmp_path):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    spill = SpillManager(tmp_path, idle_seconds=10, clock=clock)
    spill.note_activity(0)
    clock.now = 20
    spill.note_activity(15)
    clock.now = 20
    assert not spill.due()


def test_size_cap_rejects_oversize(tmp_path):
    spill = SpillManager(tmp_path, idle_seconds=0, max_gib=0.000001)
    with pytest.raises(ValueError):
        spill.snapshot({"state": "ready", "padding": "x" * 1000})
