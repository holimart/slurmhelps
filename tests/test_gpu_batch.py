"""Network-free tests for the generic Slurm/GPU helpers (lib/slurmhelps/gpu_batch.py).

The ssh/rsync layer is injected via the module-level ``run_ssh`` / ``run_local``
choke points, so every test serves canned ``sinfo``/``scontrol``/``squeue``
output and asserts on the pure logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpu_batch",
    Path(__file__).resolve().parents[1] / "lib" / "slurmhelps" / "gpu_batch.py",
)
assert _SPEC and _SPEC.loader
gpu_batch = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gpu_batch  # dataclasses resolves annotations via sys.modules
_SPEC.loader.exec_module(gpu_batch)

_SINFO = 'sinfo -h -N -o "%P|%N|%G|%t|%a"'
_SCONTROL = "scontrol show partition -o"


class _Recorder:
    """Canned ssh responses keyed by a substring of the command; records calls."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, ssh: str, command: str, *, timeout: float = 120) -> str:
        self.calls.append(command)
        for needle, reply in self.replies.items():
            if needle in command:
                return reply
        return ""


def _install(monkeypatch: pytest.MonkeyPatch, sinfo: str, scontrol: str = "") -> _Recorder:
    rec = _Recorder({_SINFO: sinfo})
    if scontrol:
        rec.replies[_SCONTROL] = scontrol
    monkeypatch.setattr(gpu_batch, "run_ssh", rec)
    return rec


def test_probe_excludes_prefix_nonidle_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    sinfo = "\n".join(
        [
            "gpu|goodA|gpu:V100:2|idle|up",
            "gpu|goodB|gpu:T600:1|idle|up",
            "gpu|goodC|gpu:V100:4|allocated|up",
            "cpu|cpuNode|cpu:64|idle|up",
            "gpu|badnode1|gpu:V100:2|idle|up",
            "gpu|goodD|gpu:A100:1|idle|up",
        ]
    )
    _install(monkeypatch, sinfo)
    nodes = gpu_batch.probe_gpus("user@host", exclude_node_prefix=["badnode"], exclude_gpu=["T600"])
    assert {n.node for n in nodes} == {"goodA", "goodD"}
    assert all(n.state == "idle" for n in nodes)


def test_probe_enforces_allow_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    sinfo = "\n".join(
        [
            "gpu|n1|gpu:V100:1|idle|up",
            "restricted|n2|gpu:V100:1|idle|up",
            "other|n3|gpu:V100:1|idle|up",
        ]
    )
    scontrol = "\n".join(
        [
            "PartitionName=gpu AllowAccounts=ALL",
            "PartitionName=restricted AllowAccounts=teamA",
            "PartitionName=other AllowAccounts=teamB",
        ]
    )
    rec = _install(monkeypatch, sinfo, scontrol)

    nodes = gpu_batch.probe_gpus("user@host", account="teamA")
    assert {n.node for n in nodes} == {"n1", "n2"}
    assert _SCONTROL in "".join(rec.calls)

    # An explicit allow-list takes precedence over the account lookup.
    nodes = gpu_batch.probe_gpus("user@host", account="teamA", allowed_partition=["gpu"])
    assert {n.node for n in nodes} == {"n1"}


def test_probe_orders_cheapest_first_and_dedups_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinfo = "\n".join(
        [
            "gpu|expNode|gpu:V100:2|idle|up",
            "fast|expNode|gpu:V100:2|idle|up",
            "gpu|cheapNode|gpu:T600:1|idle|up",
        ]
    )
    _install(monkeypatch, sinfo)
    nodes = gpu_batch.probe_gpus(
        "user@host",
        gpu_rank={"V100": 10, "T600": 1},
        partition_rank={"fast": 1, "gpu": 5},
    )
    assert [n.node for n in nodes] == ["cheapNode", "expNode"]
    # expNode de-duplicated to its best partition ("fast", rank 1).
    assert nodes[1].partition == "fast"
    assert nodes[1].partition_rank == 1


def test_run_dry_run_prints_plan_without_sbatch_or_rsync(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sinfo = "gpu|nodeA|gpu:V100:2|idle|up\n"
    rec = _install(monkeypatch, sinfo, "PartitionName=gpu AllowAccounts=ALL")

    local_calls: list[list[str]] = []
    monkeypatch.setattr(
        gpu_batch,
        "run_local",
        lambda argv, *, timeout=None: local_calls.append(list(argv)),
    )

    results = gpu_batch.submit_run(
        "user@host",
        account="acct",
        workdir="/scratch/env",
        remote_tmp="/scratch/tmp",
        local_batch="./batch",
        remote_name="job1",
        run_cmd="python run.py --out {output}",
        partition="gpu",
        yes=False,
    )
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "nodeA" in out
    assert results.endswith(".jsonl")
    # Read-only probe ran, but nothing was submitted or transferred.
    assert all("sbatch" not in c for c in rec.calls)
    assert local_calls == []


def test_dry_run_flag_forces_dry_run_even_with_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "gpu|nodeA|gpu:V100:2|idle|up\n")
    local_calls: list[list[str]] = []
    monkeypatch.setattr(
        gpu_batch, "run_local", lambda argv, *, timeout=None: local_calls.append(list(argv))
    )
    gpu_batch.submit_run(
        "user@host",
        account="acct",
        workdir="/scratch/env",
        remote_tmp="/scratch/tmp",
        local_batch="./batch",
        remote_name="job1",
        run_cmd="true",
        nodelist="nodeA",
        partition="gpu",
        yes=True,
        dry_run=True,
    )
    assert local_calls == []


def test_sbatch_script_contains_directives_and_substitutions() -> None:
    script = gpu_batch.build_sbatch_script(
        remote_name="job1",
        remote_dir="/scratch/tmp/job1",
        account="acct",
        minutes=17,
        partition="gpu",
        run_cmd="python run.py --batch {batch_dir} --out {output}",
        results_name="results.jsonl",
    )
    assert "#SBATCH --job-name=job1" in script
    assert "#SBATCH --account=acct" in script
    assert "#SBATCH --time=17" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --output=/scratch/tmp/job1/slurm-%j.out" in script
    assert "python run.py --batch /scratch/tmp/job1 --out /scratch/tmp/job1/results.jsonl" in script
    assert "{batch_dir}" not in script and "{output}" not in script


def test_sbatch_script_emits_exclude() -> None:
    script = gpu_batch.build_sbatch_script(
        remote_name="job1",
        remote_dir="/scratch/tmp/job1",
        account="acct",
        partition="gpu",
        exclude=["fuuk-13", "fuuk-14"],
        run_cmd="echo hi",
    )
    assert "#SBATCH --exclude=fuuk-13,fuuk-14" in script

    # No exclude -> no directive.
    plain = gpu_batch.build_sbatch_script(
        remote_name="j", remote_dir="/d", account="a", run_cmd="echo"
    )
    assert "--exclude" not in plain


def test_select_gpu_honors_nodelist_without_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _install(monkeypatch, "gpu|other|gpu:V100:1|idle|up\n")
    node = gpu_batch.select_gpu("user@host", nodelist="chosen", partition="gpu")
    assert node.node == "chosen"
    assert node.partition == "gpu"
    assert rec.calls == []
