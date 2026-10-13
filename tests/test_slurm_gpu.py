"""Tests for the on-demand GPU Slurm helper. Slurm is faked with scripts on PATH.

No test reaches a real scheduler: sbatch/squeue/sinfo/sacct/scancel are shell
stubs driven by files in ``FAKE_SLURM_DIR``.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner
from slurmhelps import slurm_gpu as sg
from slurmhelps.slurm_gpu_cli import jobs as main

FAKE_SH = {
    "sinfo": """#!/bin/bash
part=""; while [ $# -gt 0 ]; do [ "$1" = -p ] && part="$2"; shift; done
cat "$FAKE_SLURM_DIR/sinfo-$part.txt" 2>/dev/null
exit 0
""",
    "squeue": """#!/bin/bash
if [[ " $* " == *" -j "* ]]; then
  id="$(echo "$*" | sed -E 's/.*-j ([0-9]+).*/\\1/')"
  cat "$FAKE_SLURM_DIR/job-$id.state" 2>/dev/null
else
  cat "$FAKE_SLURM_DIR/squeue.txt" 2>/dev/null
fi
exit 0
""",
    "sbatch": """#!/bin/bash
echo "$*" >> "$FAKE_SLURM_DIR/sbatch.log"
n=$(cat "$FAKE_SLURM_DIR/next-id" 2>/dev/null || echo 500)
echo $((n+1)) > "$FAKE_SLURM_DIR/next-id"
echo "$n"
""",
    "sacct": """#!/bin/bash
id="$(echo "$*" | sed -E 's/.*-j ([0-9]+).*/\\1/')"
cat "$FAKE_SLURM_DIR/sacct-$id.txt" 2>/dev/null
exit 0
""",
    "scancel": """#!/bin/bash
echo "$*" >> "$FAKE_SLURM_DIR/scancel.log"
""",
}


def _config(root: Path, log_dir: Path, **over) -> None:
    gpu = {
        "account": "ffa",
        "partitions": [
            {"name": "ffa-preempt", "notes": "CPU=0.10, REQUEUE"},
            {"name": "ffa-checkpoint", "notes": "CPU=0.15, REQUEUE"},
            {"name": "gpu-ffa", "notes": "full price"},
        ],
        "exclude_nodes": "r50",
        "forbidden_node_patterns": ["fuuk-*"],
        "allowed_partition_patterns": ["*ffa*"],
        "forbidden_path_prefixes": ["/home"],
        "gres": "gpu:1",
        "cpus": 2,
        "mem": "8G",
        "time": "00:05:00",
        "job_name_prefix": "langywrap-gpu-",
        "max_project_jobs": 2,
        "count_job_names": ["agent-supervisor", "langywrap-gpu-*"],
        "env_setup": "",
        "workdir": str(root),
        "log_dir": str(log_dir),
    }
    gpu.update(over)
    gpu = {k: v for k, v in gpu.items() if v is not None}  # None in over drops a key
    (root / ".langywrap").mkdir(parents=True, exist_ok=True)
    (root / ".langywrap" / "lanes.json").write_text(
        json.dumps({"fleet": {"slurm": {"gpu": gpu}}}), encoding="utf-8"
    )


@pytest.fixture()
def slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    for name, body in FAKE_SH.items():
        p = bin_dir / name
        p.write_text(body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))
    monkeypatch.setenv("USER", "holecem")
    root = tmp_path / "proj"
    root.mkdir()
    log_dir = tmp_path / "logs"
    _config(root, log_dir)
    return {"root": root, "state": state, "log_dir": log_dir, "tmp": tmp_path}


def _set(state: Path, name: str, text: str) -> None:
    (state / name).write_text(text, encoding="utf-8")


NODES_HEADER = ""


def _sinfo(state: Path, part: str, lines: list[str]) -> None:
    _set(state, f"sinfo-{part}.txt", "\n".join(lines) + "\n")


def _row(node, gres, used, state_):
    return f"{node:<40}{gres:<80}{used:<80}{state_:<20}"


def test_hostlist_expansion():
    assert sg.expand_hostlist("fuuk-[13-15],r50") == {"fuuk-13", "fuuk-14", "fuuk-15", "r50"}
    assert sg.expand_hostlist("uk-g6-[01-02,04]") == {"uk-g6-01", "uk-g6-02", "uk-g6-04"}


def test_gpu_count_parsing():
    assert sg.parse_gpu_counts("gpu:L40:4(S:0),vram:no_consume:48G") == {"L40": 4}
    assert sg.parse_gpu_counts("gpu:L40:0(IDX:N/A),vram:0") == {"L40": 0}
    assert sg.parse_gpu_counts("gpu:2(S:0),mps:100") == {"": 2}
    assert sg.parse_request("gpu:1") == ("", 1)
    assert sg.parse_request("gpu:L40:2") == ("L40", 2)


def test_plan_skips_fuuk_and_r50_and_picks_first_free(slurm):
    st = slurm["state"]
    _sinfo(
        st,
        "ffa-preempt",
        [
            _row("fuuk-13", "gpu:L40:4(S:0)", "gpu:L40:0(IDX:N/A)", "idle"),
            _row("r50", "gpu:MI210:2(S:1)", "gpu:MI210:0", "idle"),
            _row("uk-g6-01", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:2(IDX:0-1)", "alloc"),
        ],
    )
    _sinfo(
        st,
        "ffa-checkpoint",
        [
            _row("fuuk-14", "gpu:L40:4(S:0)", "gpu:L40:0", "mix"),
            _row("uk-g6-04", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:0", "idle"),
        ],
    )
    _set(st, "squeue.txt", "")
    p = sg.plan(sg.load_policy(slurm["root"]))
    assert p.partition == "ffa-checkpoint"
    assert p.queued is False
    assert p.node_candidates == ["uk-g6-04"]
    assert p.cap_ok


def test_plan_partition_without_free_gpu_is_skipped_for_cheaper_free_one(slurm):
    st = slurm["state"]
    _sinfo(
        st,
        "ffa-preempt",
        [_row("uk-g6-01", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:2", "mix")],
    )
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(
        st, "gpu-ffa", [_row("uk-g6-02", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:1", "mix")]
    )
    p = sg.plan(sg.load_policy(slurm["root"]))
    assert p.partition == "gpu-ffa"
    assert p.queued is False


def test_plan_queues_on_first_partition_when_nothing_free(slurm):
    st = slurm["state"]
    for part in ("ffa-preempt", "ffa-checkpoint", "gpu-ffa"):
        _sinfo(st, part, [_row("fuuk-15", "gpu:L40:4(S:0)", "gpu:L40:4", "alloc")])
    p = sg.plan(sg.load_policy(slurm["root"]))
    assert p.partition == "ffa-preempt"
    assert p.queued is True
    assert "QUEUE" in p.reason


def test_submit_refuses_at_cap_and_never_calls_sbatch(slurm):
    st = slurm["state"]
    _set(
        st,
        "squeue.txt",
        "100|agent-supervisor|RUNNING\n101|langywrap-gpu-abc|PENDING\n102|wb-gate-c3lr|RUNNING\n",
    )
    with pytest.raises(sg.SlurmGpuError, match="refusing"):
        sg.submit(sg.load_policy(slurm["root"]), ["nvidia-smi", "-L"])
    assert not (st / "sbatch.log").exists()


def test_cap_counts_only_matching_names(slurm):
    st = slurm["state"]
    _sinfo(
        st,
        "ffa-preempt",
        [_row("uk-g6-01", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:0", "idle")],
    )
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(st, "gpu-ffa", [])
    # agent-supervisor counts (1); wb-gate belongs to another project and does not.
    _set(st, "squeue.txt", "100|agent-supervisor|RUNNING\n102|wb-gate-c3lr|RUNNING\n")
    rec = sg.submit(sg.load_policy(slurm["root"]), ["nvidia-smi", "-L"])
    assert rec["job_id"] == "500"
    log = (st / "sbatch.log").read_text()
    for flag in (
        "--account=ffa",
        "--partition=ffa-preempt",
        "--gres=gpu:1",
        "--exclude=r50",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:05:00",
        "--job-name=langywrap-gpu-",
        "--parsable",
    ):
        assert flag in log, flag
    script = Path(rec["script"])
    assert script.is_relative_to(slurm["log_dir"])
    assert "nvidia-smi -L" in script.read_text()
    ledger = slurm["root"] / ".langywrap" / "gpu-jobs.ndjson"
    line = json.loads(ledger.read_text().splitlines()[0])
    assert line["job_id"] == "500" and line["event"] == "submit"


def test_submit_dry_run_does_not_call_sbatch(slurm):
    st = slurm["state"]
    _set(st, "squeue.txt", "")
    for part in ("ffa-preempt", "ffa-checkpoint", "gpu-ffa"):
        _sinfo(st, part, [])
    rec = sg.submit(sg.load_policy(slurm["root"]), ["echo", "hi"], dry_run=True)
    assert rec["dry_run"] is True and rec["sbatch"][0] == "sbatch"
    assert not (st / "sbatch.log").exists()


def test_submit_workset_stages_image_and_runs_command_inside(slurm):
    """workset=true: the job script stages the image and enters the namespace;
    the user's command runs from a separate inner script."""
    root, st = slurm["root"], slurm["state"]
    _config(root, slurm["log_dir"], workset=True, env_setup="module load cuda")
    _set(st, "squeue.txt", "")
    for part in ("ffa-preempt", "ffa-checkpoint", "gpu-ffa"):
        _sinfo(st, part, [])
    policy = sg.load_policy(root)
    assert policy.workset is True
    rec = sg.submit(policy, ["python3", "probe.py", "--x", "a b"], dry_run=True)
    assert rec["workset"] is True
    outer = Path(rec["script"]).read_text(encoding="utf-8")
    inner = Path(rec["script"]).with_suffix(".inner.sh").read_text(encoding="utf-8")
    assert f"--project {root} stage-in" in outer
    assert f"enter -- bash {Path(rec['script']).with_suffix('.inner.sh')}" in outer
    assert "probe.py" not in outer
    assert inner.splitlines()[2:] == ["module load cuda", f"cd {root} || exit 1", "python3 probe.py --x 'a b'"]
    assert sg.WORKSET_SH.is_file()


def test_submit_without_workset_runs_command_directly(slurm):
    st = slurm["state"]
    _set(st, "squeue.txt", "")
    for part in ("ffa-preempt", "ffa-checkpoint", "gpu-ffa"):
        _sinfo(st, part, [])
    rec = sg.submit(sg.load_policy(slurm["root"]), ["echo", "hi"], dry_run=True)
    assert rec["workset"] is False
    assert Path(rec["script"]).read_text(encoding="utf-8").splitlines()[-1] == "echo hi"
    assert not Path(rec["script"]).with_suffix(".inner.sh").exists()


def test_config_rejects_path_under_forbidden_prefix(slurm):
    root = slurm["root"]
    _config(root, Path("/home/someone/gpu-logs"))
    with pytest.raises(sg.SlurmGpuError, match="forbidden_path_prefixes"):
        sg.load_policy(root)
    # prefixes are path-aware: /homework is not under /home
    _config(root, Path("/homework/gpu-logs"))
    assert sg.load_policy(root).log_dir == Path("/homework/gpu-logs")


def test_count_job_names_default_is_only_this_modules_jobs(slurm):
    """Default cap counts only ``job_name_prefix*``; a fleet allocation is opt-in."""
    root = slurm["root"]
    _config(root, slurm["log_dir"], count_job_names=None)
    assert sg.load_policy(root).count_job_names == ("langywrap-gpu-*",)
    st = slurm["state"]
    _sinfo(
        st,
        "ffa-preempt",
        [_row("uk-g6-01", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:0", "idle")],
    )
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(st, "gpu-ffa", [])
    # Two non-matching RUNNING jobs must not reach max_project_jobs=2 (old default counted them).
    _set(st, "squeue.txt", "100|agent-supervisor|RUNNING\n101|agent-supervisor|RUNNING\n")
    rec = sg.submit(sg.load_policy(root), ["true"])
    assert rec["job_id"] == "500"


def test_forbidden_path_prefixes_default_is_empty(slurm):
    root = slurm["root"]
    _config(root, Path("/home/someone/gpu-logs"), forbidden_path_prefixes=None)
    assert sg.load_policy(root).forbidden_path_prefixes == ()


def test_config_rejects_partition_outside_allowed_patterns(slurm):
    root = slurm["root"]
    _config(root, slurm["log_dir"], partitions=["cpu-long"])
    with pytest.raises(sg.SlurmGpuError, match="allowed_partition_patterns"):
        sg.load_policy(root)
    # no allowed patterns configured -> nothing is enforced
    _config(root, slurm["log_dir"], partitions=["cpu-long"], allowed_partition_patterns=None)
    assert sg.load_policy(root).partitions[0].name == "cpu-long"


def test_exclude_and_forbidden_patterns_are_optional_config(slurm):
    root = slurm["root"]
    _config(root, slurm["log_dir"], exclude_nodes=None, forbidden_node_patterns=None)
    pol = sg.load_policy(root)
    assert pol.exclude_nodes == "" and pol.forbidden_node_patterns == ()
    _config(root, slurm["log_dir"], forbidden_node_patterns="fuuk-*")
    with pytest.raises(sg.SlurmGpuError, match="must be a list"):
        sg.load_policy(root)


def test_no_forbidden_patterns_means_no_node_is_forbidden(slurm):
    st = slurm["state"]
    _sinfo(st, "ffa-preempt", [_row("fuuk-13", "gpu:L40:4(S:0)", "gpu:L40:0", "idle")])
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(st, "gpu-ffa", [])
    _set(st, "squeue.txt", "")
    _config(slurm["root"], slurm["log_dir"], forbidden_node_patterns=None)
    p = sg.plan(sg.load_policy(slurm["root"]))
    assert p.node_candidates == ["fuuk-13"]


def test_forbidden_nodes_seen_in_sinfo_are_appended_to_exclude(slurm):
    st = slurm["state"]
    _sinfo(
        st,
        "ffa-preempt",
        [
            _row("fuuk-13", "gpu:L40:4(S:0)", "gpu:L40:0", "idle"),
            _row("uk-g6-01", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:0", "idle"),
        ],
    )
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(st, "gpu-ffa", [])
    _set(st, "squeue.txt", "")
    rec = sg.submit(sg.load_policy(slurm["root"]), ["true"])
    assert rec["node_candidates"] == ["uk-g6-01"]
    assert "--exclude=r50,fuuk-13" in (st / "sbatch.log").read_text()


def test_wait_returns_terminal_sacct_state(slurm):
    st = slurm["state"]
    _set(st, "job-700.state", "RUNNING|ffa-preempt||uk-g6-01\n")
    _set(st, "sacct-700.txt", "COMPLETED|0:0|uk-g6-01\n")
    sleeps = []

    def finish(seconds):
        sleeps.append(seconds)
        (st / "job-700.state").unlink()  # the job leaves the queue

    res = sg.wait("700", sleep=finish)
    assert res["state"] == "COMPLETED" and res["exit_code"] == "0:0"
    assert sleeps  # it polled while RUNNING


def test_status_falls_back_to_sacct(slurm):
    _set(slurm["state"], "sacct-701.txt", "FAILED|1:0|uk-g6-02\n")
    st = sg.status("701")
    assert st["state"] == "FAILED" and st["source"] == "sacct"


def test_cli_plan_and_dry_run_submit(slurm):
    st = slurm["state"]
    _set(st, "squeue.txt", "")
    _sinfo(
        st,
        "ffa-preempt",
        [_row("uk-g6-05", "gpu:nvidia_l40s:2(S:0-1)", "gpu:nvidia_l40s:0", "idle")],
    )
    _sinfo(st, "ffa-checkpoint", [])
    _sinfo(st, "gpu-ffa", [])
    runner = CliRunner()
    res = runner.invoke(main, ["gpu", "plan", "--project", str(slurm["root"])])
    assert res.exit_code == 0, res.output
    assert "choice: partition ffa-preempt" in res.output
    res = runner.invoke(
        main,
        [
            "gpu",
            "submit",
            "--project",
            str(slurm["root"]),
            "--dry-run",
            "--",
            "nvidia-smi",
            "-L",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "dry-run: sbatch --parsable" in res.output
    assert not (st / "sbatch.log").exists()
