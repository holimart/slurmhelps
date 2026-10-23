"""Worker lifecycle tests: no GPU, no real backend, no network."""
from __future__ import annotations

from pathlib import Path

from slurmhelps.serving.state import StateStore
from slurmhelps.serving.worker import Worker, WorkerSpec


class FakeProcess:
    def __init__(self, rc: int | None = None):
        self.returncode = rc
        self.terminated = False
        self.killed = False
        self.waited = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        self.waited += 1
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def make_worker(tmp_path: Path, *, health, proc=None, log_text=""):
    calls = {"n": 0}
    process = proc or FakeProcess()

    def spawn(*_args, **_kwargs):
        if log_text:
            (tmp_path / "server.log").write_text(log_text)
        return process

    def maybe_health(_url):
        calls["n"] += 1
        return health(calls["n"])

    spec = WorkerSpec(
        command=["fake-backend"],
        state_dir=tmp_path,
        health_url="http://127.0.0.1:1/health",
        job_id="123",
        node="ampere02",
        max_model_len=786432,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        startup_timeout_s=10,
    )
    return Worker(spec, spawn=spawn, health=maybe_health, sleep=lambda _s: None), process


def test_worker_reaches_ready_and_parses_capacity(tmp_path):
    log = """Loading safetensors checkpoint shards:  42% Completed | 26/62\n
GPU KV cache size: 1,949,696 tokens
Maximum concurrency for 327,680 tokens per request: 5.95x
"""
    worker, _ = make_worker(tmp_path, health=lambda n: n >= 2, log_text=log)
    state = worker.wait_ready(run_until_ready=True)
    assert state["state"] == "ready"
    assert state["phase"] == "ready"
    assert state["endpoint"] == "http://127.0.0.1:8000/v1"
    assert state["capacity"]["gpu_kv_tokens"] == 1_949_696
    assert state["capacity"]["max_model_len"] == 786432
    assert state["capacity"]["max_num_batched_tokens"] == 8192
    assert state["job_id"] == "123"
    assert state["node"] == "ampere02"
    assert StateStore(tmp_path).history()


def test_worker_marks_backend_exit_failed(tmp_path):
    proc = FakeProcess(rc=7)
    worker, _ = make_worker(tmp_path, health=lambda _n: False, proc=proc)
    state = worker.wait_ready(run_until_ready=True)
    assert state["state"] == "failed"
    assert "exited" in state["error"]


def test_worker_preemption_is_not_reported_as_stopped(tmp_path):
    worker, proc = make_worker(tmp_path, health=lambda _n: True)
    worker.wait_ready(run_until_ready=True)
    worker.request_stop("preempted")
    state = worker.wait_serving()
    assert state["state"] == "preempted"
    assert proc.terminated
    assert state["endpoint"] is None


def test_worker_stop_before_ready_is_failed(tmp_path):
    worker, proc = make_worker(tmp_path, health=lambda _n: False)
    worker.request_stop("stopped")
    state = worker.wait_ready(run_until_ready=True)
    assert state["state"] == "failed"
    assert state["phase"] == "stopped-before-ready"
    assert proc.terminated


def test_worker_startup_timeout(tmp_path):
    worker, _ = make_worker(tmp_path, health=lambda _n: False)
    worker.spec.startup_timeout_s = 0
    state = worker.wait_ready(run_until_ready=True)
    assert state["state"] == "failed"
    assert "timeout" in state["error"]
