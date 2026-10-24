"""bin/hpc_chain.sh paired lines ('GPU &&& CPU') and the hpc_run.sh RUN they produce.

Dry-run: `hpc_chain.sh --dry-run` prints the RUN command per pending line (no ssh, no submit).
Functional: the RUN is taken from the dry-run, handed to hpc_run.sh with a fake `python3` that
captures gpu_batch.py's --run-cmd instead of submitting, then executed locally with gpu_batch's
placeholder substitution emulated and every cluster path redirected into pytest's tmp_path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent  # the slurmhelps checkout
TEST_USER = os.environ.get("USER") or "testuser"
CLUSTER_WB = f"/work/ffa/{TEST_USER}/whitebox"  # examples/slurmhelps.json cluster.mirror
CLUSTER_SH = f"/work/ffa/{TEST_USER}/slurmhelps"  # examples/slurmhelps.json cluster.slurmhelps_remote


@pytest.fixture
def scratch(tmp_path):
    sh = tmp_path / "slurmhelps"
    shutil.copytree(ROOT / "bin", sh / "bin")
    shutil.copytree(ROOT / "lib", sh / "lib", ignore=shutil.ignore_patterns("__pycache__"))
    repo = tmp_path / "repo"  # the calling project
    repo.mkdir()
    shutil.copy(ROOT / "examples" / "slurmhelps.json", repo / "slurmhelps.json")
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    # fake python3: gpu_batch.py submissions are captured, everything else runs the real interpreter
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.nul"
    fake = fake_bin / "python3"
    fake.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == *gpu_batch.py ]]; then shift; printf \'%s\\0\' "$@" > "$WB_TEST_CAPTURE"; exit 0; fi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    fake.chmod(0o755)
    return {"tmp": tmp_path, "repo": repo, "sh": sh, "work": work, "bin": fake_bin, "capture": capture}


def write_chain(s, lines):
    chain = s["tmp"] / "chain.tsv"
    chain.write_text("".join(f"{j}\t{m}\t{c}\n" for j, m, c in lines))
    return chain


def dry_run(s, chain):
    out = subprocess.run(
        ["bash", str(s["sh"] / "bin" / "hpc_chain.sh"), "--dry-run", str(chain)],
        cwd=s["repo"], capture_output=True, text=True, check=True,
        env={**{k: v for k, v in os.environ.items() if k != "SLURMHELPS_PROJECT"},
             "USER": TEST_USER},
    ).stdout
    rows = {}
    for line in out.splitlines():
        tag, job, mins, cmd = line.split("\t")
        assert tag == "DRY-RUN"
        rows[job] = (mins, cmd)
    return rows


def run_through_hpc_run(s, job, mins, run_cmd, env_extra=None):
    """hpc_run.sh (no --yes: gpu_batch dry path) -> captured --run-cmd -> local execution."""
    env = dict({k: v for k, v in os.environ.items() if k != "SLURMHELPS_PROJECT"}, PATH=f"{s['bin']}{os.pathsep}{os.environ['PATH']}",
               WB_TEST_CAPTURE=str(s["capture"]), SLURM_JOB_ID="t1", USER=TEST_USER)
    subprocess.run(["bash", str(s["sh"] / "bin" / "hpc_run.sh"), job, mins, run_cmd],
                   cwd=s["repo"], env=env, capture_output=True, text=True, check=True)
    argv = s["capture"].read_bytes().split(b"\0")
    run_cmd_sent = argv[argv.index(b"--run-cmd") + 1].decode()
    batch = s["tmp"] / "remote" / job / "batch"
    batch.mkdir(parents=True)
    # emulate gpu_batch._subst and the cluster paths
    cmd = (run_cmd_sent.replace("{batch_dir}", str(batch)).replace("{output}", str(batch / "eval.json"))
           .replace("{workdir}", str(s["work"])).replace(CLUSTER_SH, str(s["sh"]))
           .replace(CLUSTER_WB, str(s["work"]))
           .replace("/tmp/wb-", str(s["tmp"] / "jt" / "wb-")))
    proc = subprocess.run(["bash", "-c", cmd], cwd=s["tmp"], env=env, capture_output=True,
                          text=True, timeout=120, **(env_extra or {}))
    return proc, batch, s["tmp"] / "jt" / "wb-t1"


def test_dry_run_single_unchanged_pair_translated_malformed_refused(scratch):
    single = "PYTHONPATH=scripts python x.py --out {output}"
    pair = 'printf gpu > {output} &&& printf cpu > {output2}'
    chain = write_chain(scratch, [("j-single", "30", single), ("j-pair", "60", pair),
                                  ("j-bad", "60", "printf a > {output} &&& printf b > {output}")])
    rows = dry_run(scratch, chain)
    assert rows["j-single"] == ("30", single)  # byte-identical pass-through
    mins, cmd = rows["j-pair"]
    assert mins == "60"
    assert "export OMP_NUM_THREADS=4; printf gpu > {output} )" in cmd
    assert "export OMP_NUM_THREADS=10; printf cpu > {output2} )" in cmd
    assert "&&&" not in cmd
    assert rows["j-bad"][1] == "MALFORMED-PAIR (not submitted)"
    assert not (Path(str(chain) + ".done")).exists()  # dry-run never writes DONE


def test_paired_job_both_outputs_and_gpu_survives_cpu_exit(scratch):
    gpu = 'printf \'{"omp":%s}\' "$OMP_NUM_THREADS" > {output}; sleep 2; printf done > "$JOB_TMP/gpu-marker"'
    cpu = 'printf \'{"omp":%s}\' "$OMP_NUM_THREADS" > {output2}; exit 3'
    chain = write_chain(scratch, [("j-pair", "5", f"{gpu} &&& {cpu}")])
    mins, run_cmd = dry_run(scratch, chain)["j-pair"]
    proc, batch, jt = run_through_hpc_run(scratch, "j-pair", mins, run_cmd)
    # GPU part kept running after the CPU part exited with 3 and finished its own work
    assert (jt / "gpu-marker").read_text() == "done"
    assert json.loads((batch / "eval.json").read_text())["omp"] == 4
    assert json.loads((batch / "eval2.json").read_text())["omp"] == 10
    assert proc.returncode == 3  # GPU rc 0 -> job reports the CPU part's failure


def test_paired_job_success_and_gpu_failure_code(scratch):
    ok = 'printf \'{"omp":%s}\' "$OMP_NUM_THREADS" > {output}'
    chain = write_chain(scratch, [("j-ok", "5", f"{ok} &&& printf '{{\"omp\":%s}}' \"$OMP_NUM_THREADS\" > {{output2}}")])
    mins, run_cmd = dry_run(scratch, chain)["j-ok"]
    proc, batch, _ = run_through_hpc_run(scratch, "j-ok", mins, run_cmd)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((batch / "eval2.json").read_text())["omp"] == 10

    bad_gpu = "printf partial > {output}; exit 7"
    chain2 = write_chain(scratch, [("j-gfail", "5", f"{bad_gpu} &&& printf x > {{output2}}")])
    mins, run_cmd = dry_run(scratch, chain2)["j-gfail"]
    proc, _, _ = run_through_hpc_run(scratch, "j-gfail", mins, run_cmd)
    assert proc.returncode == 7


def test_single_line_functional_has_no_eval2(scratch):
    cmd = 'printf \'{"omp":%s}\' "$OMP_NUM_THREADS" > {output}'
    chain = write_chain(scratch, [("j-one", "5", cmd)])
    mins, run_cmd = dry_run(scratch, chain)["j-one"]
    assert run_cmd == cmd
    proc, batch, _ = run_through_hpc_run(scratch, "j-one", mins, run_cmd)
    assert proc.returncode == 0, proc.stderr
    assert (batch / "eval.json").exists()
    assert not (batch / "eval2.json").exists()
