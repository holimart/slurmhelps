"""--no-wait and the atomic capped sbatch (lib/slurmhelps/gpu_batch.py), network-free."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpu_batch_nw",
    Path(__file__).resolve().parents[1] / "lib" / "slurmhelps" / "gpu_batch.py",
)
assert _SPEC and _SPEC.loader
gpu_batch = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gpu_batch
_SPEC.loader.exec_module(gpu_batch)


class _Ssh:
    def __init__(self, sbatch_reply: str | Exception = "4242;cluster") -> None:
        self.calls: list[str] = []
        self.sbatch_reply = sbatch_reply

    def __call__(self, ssh: str, command: str, *, timeout: float = 120) -> str:
        self.calls.append(command)
        if command.startswith(("flock", "sbatch")):
            if isinstance(self.sbatch_reply, Exception):
                raise self.sbatch_reply
            return self.sbatch_reply
        return ""


def _submit(monkeypatch, ssh: _Ssh, **kw):
    monkeypatch.setattr(gpu_batch, "run_ssh", ssh)
    monkeypatch.setattr(gpu_batch, "run_local", lambda argv, *, timeout=None: subprocess.CompletedProcess(argv, 0, "", ""))
    return gpu_batch.submit_run(
        "u@h",
        account="acct",
        workdir="/w/repo",
        remote_tmp="/w/temp",
        local_batch="./batch",
        remote_name="cmp-gq-1",
        run_cmd="true",
        nodelist="n1",
        partition="p",
        yes=True,
        results_name="eval.json",
        **kw,
    )


def test_capped_command_counts_and_submits_under_one_lock() -> None:
    cmd = gpu_batch.capped_sbatch_command(
        "/w/temp/cmp-gq-1/job.sbatch", lock_path="/w/temp/.lock", user="u", job_prefix="cmp-", max_jobs=6
    )
    assert cmd.startswith("flock -w 120 /w/temp/.lock sh -c ")
    inner = cmd.split(" sh -c ", 1)[1]
    assert "squeue -h -u u -o %j" in inner and "^cmp-" in inner
    assert '-ge 6' in inner and "exit 3" in inner
    assert inner.index("squeue") < inner.index("sbatch --parsable")


def test_capped_command_runs_in_a_real_shell(tmp_path: Path) -> None:
    """Executed with fake squeue/sbatch/flock: 6 live jobs refuse with 3, 5 submit."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "flock").write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
    (bindir / "sbatch").write_text("#!/bin/sh\necho 777\n")
    for n in (5, 6):
        (bindir / "squeue").write_text("#!/bin/sh\n" + "".join(f"echo cmp-j{i}\n" for i in range(n)) + "echo other\n")
        for f in bindir.iterdir():
            f.chmod(0o755)
        cmd = gpu_batch.capped_sbatch_command(
            "x.sbatch", lock_path=str(tmp_path / "l"), user="u", job_prefix="cmp-", max_jobs=6
        )
        r = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True,
                           env={"PATH": f"{bindir}:/usr/bin:/bin"})
        if n == 5:
            assert r.returncode == 0 and r.stdout.strip() == "777"
        else:
            assert r.returncode == 3 and "cap: 6 of 6" in r.stderr


def test_no_wait_returns_after_sbatch_without_polling(monkeypatch, capsys) -> None:
    ssh = _Ssh()
    _submit(monkeypatch, ssh, no_wait=True, max_jobs=6, job_prefix="cmp-", user="u")
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["job_id"] == "4242"
    assert line["remote_output"] == "/w/temp/cmp-gq-1/eval.json"
    assert not any(c.startswith("squeue -h -j") for c in ssh.calls)  # no poll
    assert not any(c.startswith("rm -rf") for c in ssh.calls)  # no cleanup
    assert any(c.startswith("flock -w 120 /w/temp/.slurmhelps-submit.lock") for c in ssh.calls)


def test_cap_refusal_raises_cap_refused_and_cleans_batch_dir(monkeypatch) -> None:
    ssh = _Ssh(gpu_batch.HpcError("ssh 'u@h' command failed (rc=3): cap: 6 of 6 cmp-* jobs live"))
    with pytest.raises(gpu_batch.CapRefused):
        _submit(monkeypatch, ssh, no_wait=True, max_jobs=6, job_prefix="cmp-", user="u")
    assert ssh.calls[-1] == "rm -rf /w/temp/cmp-gq-1"


def test_max_jobs_needs_prefix_and_user(monkeypatch) -> None:
    with pytest.raises(gpu_batch.HpcError):
        _submit(monkeypatch, _Ssh(), no_wait=True, max_jobs=6)
