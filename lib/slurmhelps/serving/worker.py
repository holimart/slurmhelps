"""Compute-side worker for the preemptible serving gateway.

The worker is deliberately backend-agnostic: it starts a command, watches
``/health``, publishes phases/load estimates, and forwards SIGTERM. It does not
contain model-specific arguments; the GLM helper supplies those.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .state import StateStore

PROGRESS_RE = re.compile(r"Loading safetensors checkpoint shards:\s*(\d+)%")
KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
CONCURRENCY_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
ACTIVE_RE = re.compile(r"vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.]+)")
QUEUED_RE = re.compile(r"vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.]+)")


@dataclass
class WorkerSpec:
    command: Sequence[str]
    state_dir: Path
    health_url: str = "http://127.0.0.1:8000/health"
    metrics_url: str = "http://127.0.0.1:8000/metrics"
    job_id: str = ""
    node: str = ""
    port: int = 8000
    served_model_name: str = "glm53-reap50"
    max_model_len: int = 786432
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    startup_timeout_s: float = 3600.0
    heartbeat_s: float = 5.0
    log_name: str = "server.log"
    metadata: dict = field(default_factory=dict)


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _read_tail(path: Path, limit: int = 262144) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _metric_counts(url: str, timeout: float = 2.0) -> tuple[int | None, int | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError, ValueError):
        return None, None
    active = ACTIVE_RE.search(text)
    queued = QUEUED_RE.search(text)
    return (
        int(float(active.group(1))) if active else None,
        int(float(queued.group(1))) if queued else None,
    )


def _capacity_from_log(text: str) -> dict:
    out: dict = {}
    match = KV_RE.search(text)
    if match:
        out["gpu_kv_tokens"] = int(match.group(1).replace(",", ""))
    match = CONCURRENCY_RE.search(text)
    if match:
        out["max_concurrency_tokens"] = int(match.group(1).replace(",", ""))
        try:
            out["max_concurrency"] = float(match.group(2))
        except ValueError:
            pass
    return out


class Worker:
    """Owns one backend process and its state file."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        store: StateStore | None = None,
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
        health: Callable[[str], bool] = _http_ok,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.spec = spec
        self.store = store or StateStore(spec.state_dir)
        self.spawn = spawn
        self.health = health
        self.clock = clock
        self.sleep = sleep
        self.proc: subprocess.Popen | None = None
        self.log_path = spec.state_dir / spec.log_name
        self._stop = threading.Event()
        self._signal: str = ""

    # -- state ---------------------------------------------------------
    def _set(self, **fields) -> dict:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fields.setdefault("last_heartbeat", stamp)
        fields.setdefault("updated_at", stamp)
        return self.store.update(**fields)

    # -- lifecycle -----------------------------------------------------
    def request_stop(self, reason: str = "stopped") -> None:
        self._signal = reason
        self._stop.set()

    def _handle_signal(self, signum, _frame) -> None:
        self.request_stop("preempted" if signum == signal.SIGTERM else "stopped")

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def start(self) -> subprocess.Popen:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        self.proc = self.spawn(
            list(self.spec.command),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # The child owns the descriptor after spawn; closing our copy avoids
        # keeping the log alive if the parent exits.
        log.close()
        return self.proc

    def wait_ready(self, run_until_ready: bool = False) -> dict:
        started = self.clock()
        self._set(
            state="starting",
            job_id=self.spec.job_id,
            node=self.spec.node,
            phase="starting",
            progress=0.0,
            load_elapsed_s=0.0,
            load_estimate_s=None,
            load_estimate_source="pending",
            endpoint=None,
            error="",
            metadata=self.spec.metadata,
        )
        if self.proc is None:
            self.start()
        self._set(state="loading", phase="loading:weights", progress=0.0,
                  load_elapsed_s=0.0)

        last_beat = started
        while not self._stop.is_set():
            elapsed = self.clock() - started
            if self.proc.poll() is not None:
                tail = _read_tail(self.log_path, 8000)
                self._set(state="failed", phase="failed", error=f"backend exited {self.proc.returncode}",
                          log_tail=tail)
                return self.store.read() or {}

            if self.health(self.spec.health_url):
                break

            if elapsed > self.spec.startup_timeout_s:
                self._set(state="failed", phase="failed",
                          error=f"startup timeout after {elapsed:.0f}s",
                          log_tail=_read_tail(self.log_path, 8000))
                return self.store.read() or {}

            if self.clock() - last_beat >= self.spec.heartbeat_s:
                last_beat = self.clock()
                tail = _read_tail(self.log_path)
                progress = 0.0
                # Both TP workers log their own 0% line; use the furthest
                # progress seen, never the last (older) worker's 0%.
                found = [int(x) for x in PROGRESS_RE.findall(tail)]
                if found:
                    progress = min(1.0, max(found) / 100.0)
                self._set(
                    state="loading",
                    phase="loading:weights",
                    progress=progress,
                    load_elapsed_s=round(elapsed, 1),
                    load_estimate_s=None,
                    load_estimate_source="elapsed",
                )
            self.sleep(min(1.0, self.spec.heartbeat_s))

        if self._stop.is_set():
            return self._drain()

        # health means the backend is serving. vLLM's health endpoint is
        # ready-only; a separate warmup phase is intentionally short and
        # explicit rather than pretending the engine is ready earlier.
        self._set(state="warming", phase="warming", progress=0.99,
                  load_elapsed_s=round(self.clock() - started, 1))
        tail = _read_tail(self.log_path)
        capacity = _capacity_from_log(tail)
        capacity.setdefault("gpu_kv_tokens", None)
        capacity["max_model_len"] = self.spec.max_model_len
        capacity["max_num_seqs"] = self.spec.max_num_seqs
        capacity["max_num_batched_tokens"] = self.spec.max_num_batched_tokens
        endpoint = f"http://127.0.0.1:{self.spec.port}/v1"
        ready = self._set(
            state="ready",
            phase="ready",
            progress=1.0,
            load_elapsed_s=round(self.clock() - started, 1),
            load_estimate_s=round(self.clock() - started, 1),
            load_estimate_source="measured",
            endpoint=endpoint,
            capacity=capacity,
            active_requests=0,
            queued_requests=0,
        )
        self.store.record_load("weights", round(self.clock() - started, 1), ok=True)
        if run_until_ready:
            return ready
        return self.wait_serving()

    def wait_serving(self) -> dict:
        while not self._stop.is_set():
            if self.proc is not None and self.proc.poll() is not None:
                self._set(state="failed", phase="failed",
                          error=f"backend exited {self.proc.returncode}",
                          log_tail=_read_tail(self.log_path, 8000))
                return self.store.read() or {}
            # Keep the heartbeat fresh and surface live vLLM queue depth.  The
            # counts are read from metrics, never invented.
            active, queued = _metric_counts(self.spec.metrics_url)
            changes = {}
            if active is not None:
                changes["active_requests"] = active
            if queued is not None:
                changes["queued_requests"] = queued
            self._set(**changes)
            self.sleep(self.spec.heartbeat_s)
        return self._drain()

    def _drain(self) -> dict:
        current = self.store.read() or {}
        preempted = self._signal == "preempted"
        if preempted:
            self._set(state="preempted", phase="preempted", endpoint=None)
        elif current.get("state") == "ready":
            self._set(state="draining", phase="draining")
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if preempted:
            return self.store.read() or {}
        if current.get("state") == "ready":
            return self._set(state="stopped", phase="stopped", endpoint=None)
        # Stopping before the engine is ready is not a successful transition;
        # record the failure explicitly rather than inventing a ready/stopped path.
        return self._set(state="failed", phase="stopped-before-ready",
                         endpoint=None, error="stopped before ready")


def run_worker(spec: WorkerSpec, *, run_until_ready: bool = False, **kwargs) -> dict:
    worker = Worker(spec, **kwargs)
    worker.install_signal_handlers()
    return worker.wait_ready(run_until_ready=run_until_ready)
