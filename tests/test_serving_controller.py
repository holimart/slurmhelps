"""Controller tests: fake Slurm runner, fake clock, no GPU/ssh/sbatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slurmhelps.serving.config import ServingConfig
from slurmhelps.serving.controller import Controller
from slurmhelps.serving.state import StateStore


class FakeClock:
    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


def ok(stdout: str = "", stderr: str = "", rc: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class ScriptedRunner:
    """Return a scripted response per command; record every argv."""

    def __init__(self, responses=None) -> None:
        self.responses = responses or {}
        self.calls: list[list[str]] = []
        self._used: dict[str, int] = {}

    def __call__(self, argv, timeout=None):
        argv = [str(x) for x in argv]
        self.calls.append(argv)
        entry = self.responses.get(argv[0], ok())
        if callable(entry):
            return entry(argv)
        if isinstance(entry, list):
            index = self._used.get(argv[0], 0)
            self._used[argv[0]] = index + 1
            return entry[min(index, len(entry) - 1)]
        return entry


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def store(tmp_path, clock):
    return StateStore(tmp_path, clock=clock)


def make_controller(store, runner, clock, **kwargs):
    return Controller(
        ServingConfig(),
        store,
        runner=runner,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_plan_is_config_derived_and_has_no_side_effects(store, clock):
    runner = ScriptedRunner()
    ctl = make_controller(store, runner, clock)
    plan = ctl.plan(["python3", "-m", "worker"])
    assert plan["partition"] == "ffa-preempt"
    assert plan["gres"] == "gpu:A100:2"
    assert (plan["cpus"], plan["mem"], plan["time"]) == (8, "220G", "04:00:00")
    assert "786432" in plan["backend_args"]
    assert "8192" in plan["backend_args"]
    assert "#SBATCH --requeue" in plan["sbatch"]
    assert "exec python3 -m worker" in plan["sbatch"]
    assert runner.calls == []
    assert store.read() is None


def test_submit_parses_job_id_writes_queued_and_refuses_live(store, clock):
    runner = ScriptedRunner({"sbatch": [ok("12345;cluster\n"), ok("12346\n")]})
    ctl = make_controller(store, runner, clock)

    state = ctl.submit("/tmp/serve.sbatch")
    assert state["job_id"] == "12345"
    assert state["state"] == "queued"
    assert state["attempt"] == 1
    assert runner.calls[0] == ["sbatch", "--parsable", "/tmp/serve.sbatch"]

    with pytest.raises(RuntimeError):
        ctl.submit("/tmp/serve.sbatch")

    forced = ctl.submit("/tmp/serve.sbatch", force=True)
    assert forced["job_id"] == "12346"
    assert forced["attempt"] == 2


def test_submit_wraps_a_command_in_a_generated_script(store, clock, tmp_path):
    runner = ScriptedRunner({"sbatch": ok("9001\n")})
    ctl = make_controller(store, runner, clock)
    state = ctl.submit(command=["python3", "-m", "worker"])
    script = store.dir / "serve.sbatch"
    assert state["job_id"] == "9001"
    assert runner.calls[0] == ["sbatch", "--parsable", str(script)]
    text = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "exec python3 -m worker" in text


def test_sbatch_failure_does_not_fake_a_job(store, clock):
    runner = ScriptedRunner({"sbatch": ok("", "invalid partition", 1)})
    ctl = make_controller(store, runner, clock)
    with pytest.raises(RuntimeError):
        ctl.submit("/tmp/x.sbatch")
    assert store.read() is None


def test_poll_maps_slurm_states_and_never_marks_ready(store, clock):
    store.write({"state": "queued", "job_id": "55", "node": "old"})
    runner = ScriptedRunner(
        {
            "squeue": [
                ok("PENDING|node01|0:00\n"),
                ok("CONFIGURING|node01|0:01\n"),
                ok("RUNNING|node01|0:02\n"),
                ok("PREEMPTED|node01|0:03\n"),
                ok("REQUEUED|node01|0:04\n"),
                ok("FAILED|node01|0:05\n"),
            ]
        }
    )
    ctl = make_controller(store, runner, clock)

    assert ctl.poll()["state"] == "queued"
    assert ctl.poll()["state"] == "starting"
    store.transition("loading")
    assert ctl.poll()["state"] == "loading"  # RUNNING alone is not readiness
    assert ctl.poll()["state"] == "preempted"
    assert ctl.poll()["state"] == "requeued"
    final = ctl.poll()
    assert final["state"] == "failed"
    assert final["error"] == "slurm:FAILED"
    assert "ready" not in [entry["state"] for entry in [final]]
    assert runner.calls[-1] == ["squeue", "-h", "-j", "55", "-o", "%T|%N|%M"]


def test_poll_uses_sacct_fallback_and_completed_maps_to_stopped(store, clock):
    store.write({"state": "ready", "job_id": "56", "endpoint": "http://127.0.0.1:8000/v1"})
    runner = ScriptedRunner(
        {"squeue": ok(""), "sacct": ok("COMPLETED\nCOMPLETED\n")}
    )
    ctl = make_controller(store, runner, clock)
    state = ctl.poll()
    assert state["state"] == "stopped"
    assert any(call[0] == "sacct" for call in runner.calls)


def test_status_keeps_loading_phase_eta_and_stale_heartbeat(store, clock):
    store.write(
        {
            "state": "loading",
            "phase": "loading:weights",
            "job_id": "57",
            "node": "ampere02",
            "endpoint": "http://old:8000/v1",
        }
    )
    runner = ScriptedRunner({"squeue": ok("RUNNING|ampere02|2:00\n")})
    ctl = make_controller(store, runner, clock)
    clock.advance(10_000)
    status = ctl.status()
    assert status["state"] == "loading"
    assert status["phase"] == "loading:weights"
    assert status["slurm_state"] == "RUNNING"
    assert status["stale"] is True
    assert status["endpoint"] is None  # stale endpoint is never advertised
    assert status["load_estimate_s"] == 540.0
    assert status["load_estimate_source"] == "cold-p50"

    store.heartbeat(phase="loading:weights", progress=0.5)
    fresh = ctl.status()
    assert fresh["stale"] is False
    assert fresh["heartbeat_age_s"] == 0.0


def test_status_without_state_is_unknown_not_ready(store, clock):
    runner = ScriptedRunner()
    ctl = make_controller(store, runner, clock)
    status = ctl.status()
    assert status["state"] is None
    assert status["stale"] is True
    assert runner.calls == []


def test_status_ready_returns_endpoint_and_load_estimate_from_history(store, clock):
    store.record_load("weights", 620)
    store.write(
        {
            "state": "ready",
            "phase": "ready",
            "job_id": "58",
            "node": "ampere02",
            "endpoint": "http://127.0.0.1:8000/v1",
        }
    )
    store.heartbeat()
    runner = ScriptedRunner({"squeue": ok("RUNNING|ampere02|3:00\n")})
    ctl = make_controller(store, runner, clock)
    status = ctl.status()
    assert status["state"] == "ready"
    assert status["endpoint"] == "http://127.0.0.1:8000/v1"
    assert status["stale"] is False
    assert ctl.load_estimate("loading:weights")["p50"] == 620.0


def test_wait_ready_returns_on_ready_and_times_out(store, clock):
    store.write({"state": "ready", "job_id": "59", "endpoint": "http://127.0.0.1:8000/v1"})
    runner = ScriptedRunner({"squeue": ok("RUNNING|n|1:00\n")})
    ctl = make_controller(store, runner, clock)
    status = ctl.wait_ready(timeout=30, poll_s=5)
    assert status["state"] == "ready"

    store.write({"state": "loading", "phase": "loading:weights", "job_id": "60"})
    with pytest.raises(TimeoutError):
        ctl.wait_ready(timeout=20, poll_s=5)


def test_wait_ready_observes_worker_transitions(store, clock):
    store.write({"state": "loading", "phase": "loading:weights", "job_id": "61"})
    runner = ScriptedRunner({"squeue": ok("RUNNING|n|1:00\n")})
    ctl = make_controller(store, runner, clock)

    original_sleep = clock.sleep

    def sleep_and_finish(seconds: float) -> None:
        original_sleep(seconds)
        if (store.read() or {}).get("state") == "loading":
            store.transition("warming")
            store.transition("ready", endpoint="http://127.0.0.1:8000/v1")

    ctl.sleep = sleep_and_finish
    status = ctl.wait_ready(timeout=60, poll_s=1)
    assert status["state"] == "ready"
    assert status["endpoint"] == "http://127.0.0.1:8000/v1"


def test_endpoint_only_when_ready_and_hint_otherwise(store, clock):
    runner = ScriptedRunner()
    ctl = make_controller(store, runner, clock)
    assert ctl.endpoint() is None

    store.write({"state": "loading", "job_id": "62", "node": "ampere02"})
    assert ctl.endpoint() is None
    hint = ctl.endpoint_hint()
    assert hint is not None and "-L 8000:ampere02:8000" in hint and "ampere02" in hint

    store.write({"state": "ready", "endpoint": "http://127.0.0.1:8000/v1"})
    assert ctl.endpoint() == "http://127.0.0.1:8000/v1"

    store.write({"state": "ready", "endpoint": {"host": "n", "port": 8123}})
    assert ctl.endpoint() == "http://127.0.0.1:8123/v1"


def test_down_cancels_and_transitions_stopped(store, clock):
    store.write(
        {"state": "ready", "job_id": "63", "endpoint": "http://127.0.0.1:8000/v1"}
    )
    runner = ScriptedRunner({"scancel": ok("")})
    ctl = make_controller(store, runner, clock)
    state = ctl.down()
    assert runner.calls == [["scancel", "63"]]
    assert state["state"] == "stopped"
    assert state["endpoint"] is None

    store.write({"state": "queued", "job_id": "64"})
    assert ctl.down()["state"] == "stopped"

    empty = StateStore(store.dir, clock=clock)
    empty.path.unlink()
    with pytest.raises(RuntimeError):
        make_controller(empty, runner, clock).down()


def test_ssh_runner_is_used_when_a_login_host_is_configured(store, clock):
    local = ScriptedRunner()
    remote = ScriptedRunner({"ssh": ok("PREEMPTED|node|1:00\n")})
    store.write({"state": "ready", "job_id": "65"})
    ctl = Controller(
        ServingConfig(),
        store,
        runner=local,
        ssh_runner=remote,
        ssh="me@login",
        clock=clock,
        sleep=clock.sleep,
    )
    state = ctl.poll()
    assert state["state"] == "preempted"
    assert local.calls == []
    assert remote.calls[0][0] == "ssh"
    assert remote.calls[0][-2] == "me@login"
    assert "squeue" in remote.calls[0][-1]
