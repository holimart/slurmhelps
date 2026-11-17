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
    proc = run(["tunnel", "--local-port", "8123", "--via", "tester@login", "--print"],
               tmp_path / "state")
    assert proc.returncode == 0, proc.stderr
    assert "ssh" in proc.stdout
    assert "-J tester@login" in proc.stdout
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


def test_qwen38_strata_volta_config_and_dry_run(tmp_path):
    from slurmhelps.serving.config import ServingConfig

    config = ROOT / "llm" / "qwen38" / "config-strata-volta.json"
    cfg = ServingConfig.from_mapping(json.loads(config.read_text()))
    assert (cfg.model, cfg.engine, cfg.gpus, cfg.gpu_type) == ("qwen38-flash-next-strata-volta", "strata", 4, "V100")
    assert (cfg.partition, cfg.nice, cfg.time, cfg.requeue) == ("ffa-gpulab", 100000, "12:00:00", True)
    assert (cfg.port, cfg.relay_port, cfg.max_model_len) == (8030, 18030, 262144)

    env = dict(ENV, SLURMHELPS_SERVE_DIR=str(tmp_path / "state"))
    proc = subprocess.run(["bash", str(ROOT / "llm" / "qwen38" / "serve.sh"), "--target", "strata-volta",
                           "up", "--dry-run"], text=True, capture_output=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["partition"] == "ffa-gpulab" and plan["gres"] == "gpu:V100:4"
    script = plan["sbatch"]
    for line in ("#SBATCH --partition=ffa-gpulab", "#SBATCH --gres=gpu:V100:4", "#SBATCH --nice=100000",
                 "#SBATCH --time=12:00:00", "#SBATCH --requeue", "#SBATCH --mem=355G"):
        assert line in script.splitlines(), (line, script)
    assert "serve-strata.sh" in script and "PORT=8030" in script and "MAX_NUM_SEQS=4" in script
    assert "KV_DTYPE=int8" in script and "--health-url http://127.0.0.1:8030/health" in script
    for bad in ("ffa-preempt", "gpu-ffa", "gpu-long"):
        assert bad not in script
    # bw and hopper still render as before.
    for target, gres, port in (("bw", "gpu:BW6000:2", 8010), ("hopper", "gpu:H100:2", 8020)):
        proc = subprocess.run(["bash", str(ROOT / "llm" / "qwen38" / "serve.sh"), "--target", target,
                               "up", "--dry-run"], text=True, capture_output=True, timeout=60, env=env)
        plan = json.loads(proc.stdout)
        assert plan["partition"] == "ffa-preempt" and plan["gres"] == gres
        assert f"PORT={port}" in plan["sbatch"] and "#SBATCH --nice=100\n" in plan["sbatch"]
