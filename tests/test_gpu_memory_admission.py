"""Memory admission integration: real temporary claims, mocked SSH/Slurm, no GPU."""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import gpu_queue as q
import hpc_bridge_gpu_queue as bridge
import pytest
from slurmhelps import gpu_memory as m

ROOT = Path(__file__).resolve().parents[1]
BIG = {"min_vram_gib": 44, "min_free_vram_gib": 42, "cupy_pool_limit_gib": 40}


@pytest.mark.parametrize(
    "value", [None, True, "44", [], {}, float("nan"), float("inf")]
)
def test_invalid_device_fields_fail_closed(value):
    assert not m.admit(
        {"gpu_memory": BIG}, {"total_gib": value, "free_gib": 44}, phase="route"
    )[0]
    assert not m.admit(
        {"gpu_memory": BIG}, {"total_gib": 48, "free_gib": value}, phase="execute"
    )[0]
    if value is not None:
        assert not m.admit(
            {"gpu_memory": BIG}, {"total_gib": 48, "free_gib": value}, phase="route"
        )[0]


def test_minimum_is_not_exact_total():
    job = {
        "gpu_memory": {
            "min_vram_gib": 8,
            "min_free_vram_gib": 42,
            "cupy_pool_limit_gib": 40,
        }
    }
    assert m.admit(job, {"total_gib": 44.47}, phase="route")[0]
    assert not m.admit(job, {"total_gib": 11}, phase="route")[0]
    assert m.admit(job, {"total_gib": 44.47, "free_gib": 42}, phase="execute")[0]
    assert not m.admit(job, {"total_gib": 44.47, "free_gib": 41.99}, phase="execute")[0]
    assert m.admit({}, {}, phase="execute")[0]


@pytest.mark.parametrize(
    "r",
    [
        {"min_vram_gib": True},
        {"min_free_vram_gib": float("nan")},
        {"cupy_pool_limit_gib": 40},
        {"min_free_vram_gib": 6, "cupy_pool_limit_gib": 40},
    ],
)
def test_bad_schema(r):
    with pytest.raises(ValueError):
        m.resources({"gpu_memory": r})


def test_cli_metadata_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "Q", tmp_path / "queue")
    a = argparse.Namespace(
        name="j",
        lane="x",
        priority="high",
        command=["python", "x.py"],
        sync=[],
        pull=[],
        force=True,
        **BIG,
    )
    q.cmd_submit(a)
    record = json.loads(next((tmp_path / "queue/pending").glob("*.json")).read_text())
    assert record["gpu_memory"] == BIG
    a.command += ["--cupy-pool-limit-gb", "6"]
    with pytest.raises(SystemExit):
        q.cmd_submit(a)


def test_both_consumer_filters_preserve_order_and_skip_head(tmp_path, monkeypatch):
    big = {"id": "0-big", "gpu_memory": BIG}
    small = {"id": "1-small"}
    monkeypatch.setattr(q, "lan_device", lambda: {"total_gib": 11, "free_gib": 10})
    assert q.pick_for_lan([big, small]) == [small]
    monkeypatch.setattr(bridge, "load_pending", lambda d: [big, small])
    monkeypatch.setattr(bridge, "recent_bridged_lanes", lambda d: set())
    monkeypatch.setattr(bridge, "chain_recently_active", lambda: False)
    monkeypatch.setattr(bridge, "bridge_runtime_history", dict)
    for j in [big, small]:
        j.update(submitted="2026-10-15T00:00:00Z", lane="x")
    monkeypatch.setitem(bridge.CONFIG["queue"], "bridge_gpu_vram_gib", 44)
    assert bridge.pick_for_bridge({}) == [big, small]
    monkeypatch.setitem(bridge.CONFIG["queue"], "bridge_gpu_vram_gib", None)
    assert bridge.pick_for_bridge({}) == [small]


def _claim_worker(d, barrier, result, owner):
    barrier.wait()
    result.put(q.claim_job(d, "job", owner) is not None)


def test_actual_atomic_claim_competition(tmp_path):
    d = {s: tmp_path / s for s in ["pending", "running", "done"]}
    for p in d.values():
        p.mkdir()
    (d["pending"] / "job.json").write_text(json.dumps({"id": "job", "gpu_memory": BIG}))
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    result = ctx.Queue()
    ps = [
        ctx.Process(target=_claim_worker, args=(d, barrier, result, owner))
        for owner in ["hpc-bridge", None]
    ]
    for p in ps:
        p.start()
    for p in ps:
        p.join(10)
        assert p.exitcode == 0
    assert sorted([result.get(timeout=2), result.get(timeout=2)]) == [False, True]
    assert (
        not list(d["pending"].iterdir()) and len(list(d["running"].glob("*.json"))) == 1
    )


def fake_tools(tmp_path):
    bindir = tmp_path / "tools"
    bindir.mkdir()

    def tool(name, text):
        p = bindir / name
        p.write_text("#!/bin/bash\n" + text)
        p.chmod(0o755)

    tool(
        "ssh",
        'cmd="${@: -1}"\n'
        'if [[ "$cmd" == *"sysconfig"* ]]; then exit 0; fi\n'
        'exec bash -c "$cmd"\n',
    )
    tool("rsync", "exit 0\n")
    # Assert the selected LAN GPU probe is inside the same remote command flock.
    tool(
        "nvidia-smi",
        'if [[ "${CHECK_LOCK:-0}" == 1 ]]; then\n'
        ' if flock -n "$GPU_TEST_LOCK" true; then echo unlocked >&2; exit 8; fi\n'
        'fi\nprintf "%s\\n" "${GPU_TEST_MEMORY:-45539, 45000}"\n',
    )
    return bindir


@pytest.mark.parametrize(
    "memory,inline,expected",
    [
        ("45539, 45000", None, 0),
        ("45539, 40000", None, 78),
        ("45539, 45000", "40", 0),
        ("45539, 45000", "6", 78),
        ("45539, 45000", "80", 78),
        ("45539, 45000", "nan", 78),
    ],
)
def test_lan_guard_under_real_flock_and_pool(tmp_path, memory, inline, expected):
    tools = fake_tools(tmp_path)
    remote = tmp_path / "remote"
    (remote / ".venv/bin").mkdir(parents=True)
    (remote / ".venv/bin/python").symlink_to(sys.executable)
    (remote / "slurmhelps").mkdir()
    (remote / "slurmhelps/lib").symlink_to(ROOT / "lib", target_is_directory=True)
    (tmp_path / "slurmhelps.json").write_text(
        json.dumps({"lan_gpu": {"host": "fake", "base": str(remote), "gpu": 0}})
    )
    marker = tmp_path / "model-called"
    cmd = [
        sys.executable,
        "-c",
        "import os,pathlib;pathlib.Path("
        + repr(str(marker))
        + ').write_text(os.environ["EVAL_CUPY_POOL_LIMIT_GB"])',
    ]
    if inline is not None:
        cmd = ["env", "EVAL_CUPY_POOL_LIMIT_GB=" + inline, *cmd]
    env = {
        **os.environ,
        "PATH": str(tools) + ":" + os.environ["PATH"],
        "GPU_RUN_ROOT": str(tmp_path),
        "SLURMHELPS_HOME": str(ROOT),
        "CHECK_LOCK": "1",
        "GPU_TEST_LOCK": str(remote / ".job.lock"),
        "GPU_TEST_MEMORY": memory,
    }
    r = subprocess.run(
        [
            "bash",
            str(ROOT / "bin/gpu_run.sh"),
            "--name",
            "test",
            "--gpu-memory-json",
            json.dumps(BIG),
            "--",
            *cmd,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert r.returncode == expected, (r.stdout, r.stderr)
    assert marker.exists() == (expected == 0)
    if expected == 0:
        assert marker.read_text() == "40"
    assert "gpu_memory_admission" in r.stdout


def test_bridge_postallocation_free_change_blocks_model(tmp_path, monkeypatch):
    tools = fake_tools(tmp_path)
    marker = tmp_path / "model"
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    monkeypatch.setattr(bridge, "MIRROR", str(tmp_path))
    monkeypatch.setattr(bridge, "VENV", ".venv")
    job = {
        "id": "job",
        "command": [
            sys.executable,
            "-c",
            "from pathlib import Path;Path(" + repr(str(marker)) + ").touch()",
        ],
        "gpu_memory": BIG,
    }
    assert m.admit(job, {"total_gib": 44}, phase="route")[0]
    cmd = bridge.remote_run_cmd(job, tmp_path).replace(
        "{output}", str(tmp_path / "eval.json")
    )
    env = {
        **os.environ,
        "PATH": str(tools) + ":" + os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "lib"),
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_TEST_MEMORY": "45539, 40000",
    }
    r = subprocess.run(
        ["bash", "-c", cmd], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert (
        r.returncode == 0
        and json.loads((tmp_path / "eval.json").read_text())["exit"] == 78
    )
    assert not marker.exists()
    assert (
        "gpu_memory_admission"
        in next(tmp_path.glob("outputs/hpc/*/remote.log")).read_text()
    )


def test_mock_slurm_guard_failure_finishes_once_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "Q", tmp_path / "queue")
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    d = q.dirs()
    job = {
        "id": "5-guard",
        "name": "guard",
        "command": ["python", "never.py"],
        "submitted": "2026-10-15T00:00:00Z",
        "gpu_memory": BIG,
    }
    (d["pending"] / "5-guard.json").write_text(json.dumps(job))
    job = q.claim_job(d, "5-guard", "hpc-bridge")
    monkeypatch.setattr(bridge, "transfer_bytes", lambda j: 0)
    monkeypatch.setattr(bridge, "pull_back", lambda j, log: None)
    submissions = []

    def fake(argv, **kw):
        if argv[:2] == ["bash", str(bridge.HPC_RUN)]:
            submissions.append(argv)
            p = tmp_path / bridge.HPC_DIR / bridge.cluster_job_name(job) / "eval.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"exit": 78, "node": "mock-allocated-low-free"}))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake)
    assert bridge.run_job(d, job, out=lambda *a: None) == 78
    assert (
        len(submissions) == 1
        and json.loads((d["done"] / "5-guard.json").read_text())["exit"] == 78
    )
    assert not list(d["pending"].iterdir()) and not list(d["running"].iterdir())


@pytest.mark.parametrize("value", ["6", "80", "nan", "inf", "true", "", "nope"])
def test_explicit_pool_conflicts_submit_and_runtime(tmp_path, monkeypatch, value):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "Q", tmp_path / "queue")
    command = ["env", "EVAL_CUPY_POOL_LIMIT_GB=" + value, "python", "never.py"]
    a = argparse.Namespace(
        name="j",
        lane="x",
        priority="high",
        command=command,
        sync=[],
        pull=[],
        force=True,
        **BIG,
    )
    with pytest.raises(SystemExit):
        q.cmd_submit(a)
    # Persisted/manual jobs reject before GPU probe or exec, using the same check.
    monkeypatch.setattr(
        m, "probe", lambda: pytest.fail("conflict must reject before GPU probe")
    )
    monkeypatch.setattr(os, "execvpe", lambda *a: pytest.fail("must not exec"))
    assert m.guarded_exec(BIG, command) == 78
    assert not (tmp_path / "queue/pending").exists()


def test_supported_same_pool_and_legacy():
    for command in [
        ["env", "EVAL_CUPY_POOL_LIMIT_GB=40", "python", "x.py"],
        ["/usr/bin/env", "--", "EVAL_CUPY_POOL_LIMIT_GB=40.0", "python", "x.py"],
        ["env", "X=unrelated", "env", "EVAL_CUPY_POOL_LIMIT_GB=40", "python", "x.py"],
        ["python", "x.py", "--cupy-pool-limit-gb", "40"],
    ]:
        m.validate_command_pool({"gpu_memory": BIG}, command)
    m.validate_command_pool({}, ["env", "EVAL_CUPY_POOL_LIMIT_GB=6", "python", "x.py"])
    for command in [
        ["env", "-i", "python"],
        ["env", "-u", "EVAL_CUPY_POOL_LIMIT_GB", "python"],
        ["env", "--unset=EVAL_CUPY_POOL_LIMIT_GB", "python"],
        ["python", "--cupy-pool-limit-gb", True],
    ]:
        with pytest.raises(ValueError):
            m.validate_command_pool({"gpu_memory": BIG}, command)


@pytest.mark.parametrize("prefix", [["env","--"],["/usr/bin/env"],["env","X=x","env"]])
def test_env_launcher_forms_cannot_hide_override(prefix):
    with pytest.raises(ValueError):
        m.validate_command_pool({'gpu_memory':BIG},[*prefix,"EVAL_CUPY_POOL_LIMIT_GB=80","python","never.py"])
