"""Serving state: atomic writes, transitions, heartbeat and load estimates."""

from __future__ import annotations

import json

import pytest

from slurmhelps.serving.state import (
    DEFAULT_HEARTBEAT_TIMEOUT_S,
    SCHEMA,
    StateStore,
    atomic_write_json,
    heartbeat_age_s,
    is_stale,
    read_json,
)


class FakeClock:
    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_atomic_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"schema": SCHEMA, "state": "queued", "job_id": "1"})
    assert read_json(path)["job_id"] == "1"
    assert not list(path.parent.glob("*.tmp"))
    assert read_json(tmp_path / "missing.json") is None

    path.write_text("{not json", encoding="utf-8")
    assert read_json(path) is None
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert read_json(path) is None
    path.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    assert read_json(path) is None


def test_store_writes_all_documented_fields(tmp_path):
    store = StateStore(tmp_path)
    state = store.write({"state": "queued", "job_id": "42"})
    assert state["schema"] == SCHEMA
    for field in (
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
    ):
        assert field in state
    assert store.read()["state"] == "queued"


def test_transitions_accept_documented_path_and_reject_shortcuts(tmp_path):
    store = StateStore(tmp_path)
    store.write({"state": "queued", "job_id": "7"})
    for state in ("starting", "loading", "warming", "ready", "draining", "stopped"):
        assert store.transition(state)["state"] == state

    store.write({"state": "queued", "job_id": "8"})
    with pytest.raises(ValueError):
        store.transition("ready")
    with pytest.raises(ValueError):
        store.transition("nonsense")


def test_preemption_and_requeue_paths_from_any_state(tmp_path):
    store = StateStore(tmp_path)
    for initial in ("queued", "starting", "loading", "warming", "ready"):
        store.write({"state": initial})
        assert store.transition("preempted")["state"] == "preempted"
        assert store.transition("requeued")["state"] == "requeued"
        assert store.transition("starting")["state"] == "starting"
        assert store.transition("failed")["state"] == "failed"

    store.write({"state": "ready"})
    assert store.transition("failed")["state"] == "failed"


def test_heartbeat_and_staleness(tmp_path):
    clock = FakeClock()
    store = StateStore(tmp_path, clock=clock)
    state = store.write({"state": "loading"})
    assert state["last_heartbeat"] is None
    assert store.stale() is True

    store.heartbeat(phase="loading:weights", progress=0.4)
    assert store.stale() is False
    clock.advance(DEFAULT_HEARTBEAT_TIMEOUT_S + 1)
    assert store.stale() is True

    age = heartbeat_age_s(store.read()["last_heartbeat"], clock())
    assert age > DEFAULT_HEARTBEAT_TIMEOUT_S
    assert is_stale(store.read()["last_heartbeat"], clock(), DEFAULT_HEARTBEAT_TIMEOUT_S)
    assert not is_stale(store.read()["last_heartbeat"], clock(), timeout_s=1e9)
    assert heartbeat_age_s(None) is None


def test_load_history_median_p90_and_sources(tmp_path):
    store = StateStore(tmp_path)
    for duration in (100, 300, 200):
        store.record_load("loading:weights", duration, ok=True, source="cold")
    store.record_load("loading:weights", 999, ok=False, source="cold", error="oom")

    estimate = store.estimate("loading:weights")
    assert estimate["samples"] == 3
    assert estimate["p50"] == 200.0
    assert estimate["p90"] == 300.0
    assert estimate["source"] == "history"
    assert estimate["cold"] is False

    filtered = store.estimate("weights", source="cold")
    assert filtered["samples"] == 3
    assert filtered["source"] == "history:cold"
    assert store.estimate("weights", source="warm")["cold"] is True


@pytest.mark.parametrize(
    "phase,p50",
    [("weights", 540.0), ("loading:weights", 540.0), ("kv", 30.0), ("warmup", 45.0)],
)
def test_cold_fallback_constants_are_honest(tmp_path, phase, p50):
    store = StateStore(tmp_path)
    estimate = store.estimate(phase)
    assert estimate["p50"] == p50
    assert estimate["p90"] >= estimate["p50"]
    assert estimate["samples"] == 0
    assert estimate["source"] == "cold-p50"
    assert estimate["cold"] is True


def test_unknown_phase_has_no_invented_eta(tmp_path):
    estimate = StateStore(tmp_path).estimate("loading:mystery")
    assert estimate["p50"] is None
    assert estimate["source"] == "no-history"


def test_malformed_history_lines_are_ignored(tmp_path):
    store = StateStore(tmp_path)
    store.record_load("kv", 10)
    with store.history_path.open("a", encoding="utf-8") as handle:
        handle.write("{bad json\n")
    assert len(store.history()) == 1
    assert store.estimate("kv")["p50"] == 10.0


def test_exclusive_lock_is_usable(tmp_path):
    store = StateStore(tmp_path)
    with store.lock():
        store.write({"state": "starting"})
    assert store.read()["state"] == "starting"
