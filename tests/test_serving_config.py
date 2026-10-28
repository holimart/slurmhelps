"""Serving configuration: defaults, 700K support, validation and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slurmhelps.serving.config import MAX_MODEL_LEN_LIMIT, ServingConfig


def test_documented_defaults():
    cfg = ServingConfig()
    assert cfg.model == "glm53-reap50-nvfp4"
    assert cfg.engine == "vllm-sm80"
    assert cfg.gpus == 2
    assert cfg.gpu_type == "A100"
    assert cfg.partition == "ffa-preempt"
    assert cfg.account == ""
    assert (cfg.cpus, cfg.mem, cfg.time) == (8, "220G", "04:00:00")
    assert cfg.requeue is True
    assert cfg.nice == 100
    assert cfg.exclusive is False
    assert cfg.max_model_len == 786_432
    assert cfg.max_num_seqs == 32
    assert cfg.max_num_batched_tokens == 8192
    assert cfg.gpu_memory_utilization == 0.94
    assert cfg.kv_cache_dtype == "bfloat16"
    assert cfg.block_size == 128
    assert cfg.host_kv_ram_gib == 96
    assert cfg.queue_limit == 64
    assert cfg.coalesce_ms == 0
    assert (cfg.bind, cfg.port) == ("127.0.0.1", 8000)
    assert cfg.key_env == "SLURMHELPS_API_KEY"
    assert cfg.relay_port == 18000
    assert cfg.spill_enabled is False
    assert cfg.idle_seconds == 600
    assert cfg.spill_dir == ""
    assert cfg.spill_max_gib == 128
    assert cfg.spill_mode == "state-only"


def test_backend_args_carry_700k_and_batch_defaults():
    args = ServingConfig().backend_args()
    assert "786432" in args
    assert "8192" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-prefix-caching" in args
    assert "--enable-chunked-prefill" in args
    assert args[args.index("--served-model-name") + 1] == "glm53-reap50-nvfp4"
    # No invented flags: the model path belongs to the worker.
    assert "--model" not in args


@pytest.mark.parametrize("length", [128, 327_680, 700_000, 786_432, MAX_MODEL_LEN_LIMIT])
def test_700k_and_limit_are_accepted(length):
    cfg = ServingConfig.from_mapping({"max_model_len": length})
    assert cfg.max_model_len == length


@pytest.mark.parametrize("length", [127, 0, -1, MAX_MODEL_LEN_LIMIT + 1])
def test_model_len_bounds(length):
    with pytest.raises(ValueError):
        ServingConfig.from_mapping({"max_model_len": length})


@pytest.mark.parametrize(
    "mapping",
    [
        {"gpus": 0},
        {"gpus": True},
        {"gpus": "2"},
        {"cpus": 0},
        {"max_num_seqs": 0},
        {"max_num_batched_tokens": 0},
        {"port": 0},
        {"port": 65536},
        {"relay_port": 70000},
        {"coalesce_ms": -1},
        {"coalesce_ms": 1001},
        {"block_size": 0},
        {"gpu_type": ""},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1.1},
        {"partition": ""},
        {"requeue": 1},
    ],
)
def test_validation_rejects_bad_values(mapping):
    with pytest.raises(ValueError):
        ServingConfig.from_mapping(mapping)


def test_loopback_default_and_explicit_public_bind():
    with pytest.raises(ValueError):
        ServingConfig.from_mapping({"api": {"bind": "0.0.0.0"}})
    with pytest.raises(ValueError):
        ServingConfig.from_mapping({"tunnel": {"bind": "0.0.0.0"}})
    public = ServingConfig.from_mapping({"api": {"bind": "0.0.0.0"}}, allow_public=True)
    assert public.bind == "0.0.0.0"
    explicit = ServingConfig.from_mapping(
        {"api": {"bind": "0.0.0.0", "allow_public": True}}
    )
    assert explicit.allow_public is True


def test_nested_merge_and_account_inheritance():
    cfg = ServingConfig.from_mapping(
        {
            "serving": {
                "model": "m",
                "account": "",
                "api": {"port": 9100, "key_env": "MY_KEY"},
                "tunnel": {"via": "login", "relay_port": 19000, "bind": "::1"},
                "spill": {"enabled": True, "idle_seconds": 42, "max_gib": 7, "mode": "state-only"},
            }
        },
        cluster={"account": "proj"},
    )
    assert cfg.model == "m"
    assert cfg.account == "proj"
    assert cfg.port == 9100
    assert cfg.key_env == "MY_KEY"
    assert cfg.via == "login"
    assert cfg.relay_port == 19000
    assert cfg.relay_bind == "::1"
    assert cfg.spill_enabled is True
    assert cfg.idle_seconds == 42
    assert cfg.spill_max_gib == 7
    assert cfg.spill["mode"] == "state-only"


def test_root_resolves_spill_dir_and_workdir(tmp_path):
    cfg = ServingConfig.from_mapping({"spill": {"dir": "cache/kv"}}, root=tmp_path)
    assert cfg.spill_dir == str(tmp_path / "cache" / "kv")
    assert cfg.workdir == str(tmp_path)


def test_slurm_args_are_preemptible_by_default():
    args = ServingConfig().slurm_args()
    assert args[args.index("--partition") + 1] == "ffa-preempt"
    assert "--account" not in args
    assert "--gres=gpu:A100:2" in args
    assert "--requeue" in args
    assert "--exclusive" not in args
    assert args[args.index("--nice") + 1] == "100"


def test_thinkingcap_a100_config_is_valid():
    path = (
        Path(__file__).resolve().parents[1]
        / "llm"
        / "thinkingcap-flash-next"
        / "config-a100.json"
    )
    cfg = ServingConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    assert cfg.model == "thinkingcap-flash-next"
    assert (cfg.gpus, cfg.gpu_type, cfg.partition) == (2, "A100", "ffa-preempt")
    assert cfg.nice == 100
    assert cfg.max_model_len == 262144
    assert cfg.max_num_seqs == 2
    assert cfg.port == 8011
    assert cfg.kv_cache_dtype == "bfloat16"


def test_sbatch_script_has_no_secret_and_renders_command():
    cfg = ServingConfig.from_mapping({"workdir": "/work/x", "account": "acct"})
    text = cfg.render_sbatch(["python3", "-m", "worker", "--flag"])
    assert text.startswith("#!/bin/bash")
    assert "#SBATCH --partition=ffa-preempt" in text
    assert "#SBATCH --account=acct" in text
    assert "#SBATCH --gres=gpu:A100:2" in text
    assert "#SBATCH --requeue" in text
    assert "set -euo pipefail" in text
    assert "cd /work/x" in text
    assert "exec python3 -m worker --flag" in text
    assert "SLURMHELPS_API_KEY" not in text
    assert "Bearer" not in text


def test_config_never_reads_or_stores_the_secret(monkeypatch):
    monkeypatch.setenv("MY_KEY", "super-secret-value")
    cfg = ServingConfig.from_mapping({"api": {"key_env": "MY_KEY"}})
    assert cfg.key_env == "MY_KEY"
    assert "super-secret-value" not in repr(cfg)
    assert "super-secret-value" not in cfg.render_sbatch("true")


def test_render_sbatch_keeps_combined_gres_directive_clean():
    cfg = ServingConfig.from_mapping({})
    text = cfg.render_sbatch(["true"])
    assert "#SBATCH --gres=gpu:A100:2" in text
    assert "=--cpus-per-task" not in text
    assert "\n#SBATCH 8\n" not in text
