"""Tests for llm/glm53-reap50: stdlib benchmark client and shell entry points.

No network, no GPU, no model. The bench client's --self-test starts a local mock
OpenAI server on 127.0.0.1 and exercises the full request/SSE/aggregation path.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "llm" / "glm53-reap50"
BENCH = HELPER / "bench.py"
SCRIPTS = ("install.sh", "serve.sh", "bench.sh", "run-job.sh", "submit.sh", "glm53-reap50.sbatch")


def load_bench():
    spec = importlib.util.spec_from_file_location("glm53_bench", BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["glm53_bench"] = module
    spec.loader.exec_module(module)
    return module


bench = load_bench()


def test_percentile_edges():
    assert bench.percentile([], 50) == 0.0
    assert bench.percentile([7], 99) == 7.0
    assert bench.percentile([1, 2, 3, 4], 50) == 2.5
    assert bench.percentile([1, 2, 3, 4], 0) == 1.0
    assert bench.percentile([1, 2, 3, 4], 100) == 4.0


def test_make_prompt_is_reproducible_and_shared_prefix():
    a = bench.make_prompt(32, bench.random.Random(1))
    b = bench.make_prompt(32, bench.random.Random(1))
    assert a == b
    assert len(a.split()) == 32
    prefix = bench.make_prompt(8, bench.random.Random(9)).split()
    c = bench.make_prompt(32, bench.random.Random(2), prefix)
    assert c.split()[:8] == prefix
    assert len(c.split()) == 32


def test_parse_concurrency_accepts_spaces_commas_plus():
    assert bench.parse_concurrency("1 2 4") == [1, 2, 4]
    assert bench.parse_concurrency("4,2,1,2") == [1, 2, 4]
    assert bench.parse_concurrency("1+2+4") == [1, 2, 4]


def test_self_test_runs():
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--self-test"],
        text=True, capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "self-test OK" in proc.stdout


def test_dry_run_does_not_need_a_server():
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--dry-run", "--concurrency", "1,2", "--workers", "2"],
        text=True, capture_output=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "C=1" in proc.stdout and "C=2" in proc.stdout


def test_shell_syntax():
    for name in SCRIPTS:
        path = HELPER / name
        assert path.exists(), path
        proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        assert proc.returncode == 0, f"{name}: {proc.stderr}"


def test_install_list_mentions_core_variants():
    proc = subprocess.run(
        ["bash", str(HELPER / "install.sh"), "--list"],
        text=True, capture_output=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    for token in ("nvfp4", "fp8", "q4_k_s", "q4_k_m", "sglang", "llamacpp"):
        assert token in proc.stdout
