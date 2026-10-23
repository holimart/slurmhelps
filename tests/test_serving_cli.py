"""CLI surface tests: no Slurm, no GPU, no ssh."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from slurmhelps.serving.state import StateStore

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "lib"))


def run(args, state_dir):
    return subprocess.run(
        [PYTHON, "-m", "slurmhelps.serving.cli", "--state-dir", str(state_dir), *args],
        text=True, capture_output=True, timeout=60, env=ENV,
    )


def test_cli_help():
    proc = subprocess.run(
        [PYTHON, "-m", "slurmhelps.serving.cli", "--help"],
        text=True, capture_output=True, timeout=30, env=ENV,
    )
    assert proc.returncode == 0
    for name in ("up", "status", "wait", "endpoint", "tunnel", "relay", "worker", "mock", "smoke"):
        assert name in proc.stdout


def test_up_dry_run_contains_preemptible_defaults(tmp_path):
    proc = run(["up", "--dry-run"], tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["gres"] == "gpu:A100:2"
    assert plan["partition"] == "ffa-preempt"
    assert "--requeue" in plan["slurm_args"]
    assert "--nice" in plan["slurm_args"]
    assert "786432" in plan["backend_args"]
    assert "8192" in plan["backend_args"]


def test_status_and_endpoint_from_mock_state(tmp_path):
    store = StateStore(tmp_path / "state")
    store.write({
        "schema": 1,
        "state": "ready",
        "job_id": "",
        "node": "ampere02",
        "phase": "ready",
        "endpoint": "http://127.0.0.1:8000/v1",
        "last_heartbeat": store.now_iso(),
    })
    proc = run(["endpoint", "--json"], tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["endpoint"] == "http://127.0.0.1:8000/v1"

    proc = run(["status", "--json"], tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["state"] == "ready"
    assert data["node"] == "ampere02"


def test_endpoint_loading_returns_hint_and_nonzero(tmp_path):
    store = StateStore(tmp_path / "state")
    store.write({
        "schema": 1,
        "state": "loading",
        "job_id": "42",
        "node": "ampere02",
        "phase": "loading:weights",
        "last_heartbeat": store.now_iso(),
    })
    proc = run(["endpoint", "--json"], tmp_path / "state")
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert data["endpoint"] is None
    assert "ssh" in data["hint"]


def test_tunnel_print_uses_loopback_and_jump_option(tmp_path):
    store = StateStore(tmp_path / "state")
    store.write({
        "schema": 1,
        "state": "ready",
        "job_id": "42",
        "node": "ampere02",
        "endpoint": "http://127.0.0.1:8000/v1",
        "last_heartbeat": store.now_iso(),
    })
    proc = run(["tunnel", "--local-port", "8123", "--via", "holecem@login", "--print"],
               tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    assert "ssh" in proc.stdout
    assert "-J holecem@login" in proc.stdout
    assert "8123:ampere02:8000" in proc.stdout


def test_doctor_prints_defaults(tmp_path):
    proc = run(["doctor"], tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["config"]["max_model_len"] == 786432
    assert data["config"]["max_num_batched_tokens"] == 8192
    assert data["config"]["gpus"] == 2


def test_shim_help():
    proc = subprocess.run(
        [str(ROOT / "bin" / "slurmhelps-serve"), "--help"],
        text=True, capture_output=True, timeout=30, env=ENV,
    )
    assert proc.returncode == 0, proc.stderr
    assert "worker" in proc.stdout
