"""Hardening of bin/gpu_queue.py and bin/gpu_run.sh (no network: subprocess is faked)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import gpu_queue  # noqa: E402

GPU_RUN = ROOT / "bin" / "gpu_run.sh"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo root + queue dir, so no real .langywrap/gpu-queue record is touched."""
    monkeypatch.setattr(gpu_queue, "ROOT", tmp_path)
    monkeypatch.setattr(gpu_queue, "Q", tmp_path / "queue")
    (tmp_path / "data").mkdir()
    return tmp_path


def submit_args(command, sync=None, allow=False, name="job"):
    return argparse.Namespace(
        name=name,
        lane="test",
        priority="high",
        sync=sync,
        pull=None,
        allow_missing_sync=allow,
        command=list(command),
    )


def pending_records(repo: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in (repo / "queue" / "pending").glob("*.json")]


# 1. submit-time validation ---------------------------------------------------------------


def test_submit_rejects_missing_sync_path(repo, capsys):
    with pytest.raises(SystemExit) as exc:
        gpu_queue.cmd_submit(submit_args(["python", "x.py"], sync=["data", "nope/dir"]))
    assert exc.value.code == 2
    assert "nope/dir" in capsys.readouterr().err
    assert not (repo / "queue" / "pending").exists() or pending_records(repo) == []


def test_submit_allows_missing_sync_with_flag(repo):
    gpu_queue.cmd_submit(submit_args(["python", "x.py"], sync=["later"], allow=True))
    assert pending_records(repo)[0]["sync"] == ["later"]


def test_submit_accepts_existing_sync_file_and_dir(repo):
    (repo / "data" / "f.json").write_text("{}")
    gpu_queue.cmd_submit(submit_args(["python", "x.py"], sync=["data", "data/f.json"]))
    assert pending_records(repo)[0]["sync"] == ["data", "data/f.json"]


def test_submit_prepends_python_to_bare_py(repo, capsys):
    gpu_queue.cmd_submit(submit_args(["scripts/eval_runset.py", "--fast-numpy"]))
    rec = pending_records(repo)[0]
    assert rec["command"] == ["python", "scripts/eval_runset.py", "--fast-numpy"]
    assert "no interpreter" in capsys.readouterr().err


def test_submit_keeps_interpreted_command(repo):
    gpu_queue.cmd_submit(submit_args(["python", "scripts/a.py"]))
    rec = pending_records(repo)[0]
    assert rec["command"] == ["python", "scripts/a.py"]
    assert "input_sha256" not in rec  # old record shape when nothing is hashed


def test_submit_records_minutes_for_the_bridge(repo):
    # whiteboxchallenge 2026-10-18: a lane could not ask for more than the bridge default
    # (90 min), so a long oracle timed out on Slurm with no result.
    args = submit_args(["python", "scripts/a.py"])
    args.minutes = 240
    gpu_queue.cmd_submit(args)
    rec = pending_records(repo)[0]
    assert rec["minutes"] == 240
    sys.path.insert(0, str(ROOT / "bin"))
    import hpc_bridge_gpu_queue as bridge

    assert bridge.job_minutes(rec) == 240


def test_submit_without_minutes_keeps_old_shape(repo):
    gpu_queue.cmd_submit(submit_args(["python", "scripts/a.py"]))
    assert "minutes" not in pending_records(repo)[0]


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "1441"])
def test_minutes_flag_rejects_bad_values(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        gpu_queue._positive_minutes(bad)


# 2. candidate / baseline hashes ---------------------------------------------------------


def test_submit_records_input_sha256(repo):
    (repo / "data" / "cand.py").write_text("A = 1\n")
    (repo / "data" / "base.py").write_text("B = 2\n")
    cmd = ["python", "s.py", "--candidate", "data/cand.py", "--baseline=data/base.py"]
    cmd += ["--candidate", "data/not-local.py"]
    gpu_queue.cmd_submit(submit_args(cmd))
    rec = pending_records(repo)[0]
    assert rec["input_sha256"] == {
        "data/cand.py": hashlib.sha256(b"A = 1\n").hexdigest(),
        "data/base.py": hashlib.sha256(b"B = 2\n").hexdigest(),
    }


class FakeRun:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def __call__(self, argv, **kw):
        snap = Path(argv[1])
        self.calls.append({"argv": argv, "kw": kw, "snap_text": snap.read_text()})
        return subprocess.CompletedProcess(argv, self.rc)


def queue_job(repo, job):
    d = gpu_queue.dirs()
    (d["pending"] / f"{job['id']}.json").write_text(json.dumps(job))
    return d


def test_run_warns_when_input_changed(repo, monkeypatch):
    cand = repo / "data" / "cand.py"
    cand.write_text("A = 1\n")
    gpu_queue.cmd_submit(submit_args(["python", "s.py", "--candidate", "data/cand.py"]))
    job = pending_records(repo)[0]
    cand.write_text("A = 2\n")
    fake = FakeRun()
    monkeypatch.setattr(gpu_queue.subprocess, "run", fake)
    monkeypatch.setattr(gpu_queue, "GPU_RUN", GPU_RUN)
    gpu_queue.run_one(job, gpu_queue.dirs())
    log = (repo / "queue" / "done" / f"{job['id']}.log").read_text()
    assert "WARNING: input data/cand.py changed since submit" in log


def test_run_no_warning_when_input_unchanged(repo, monkeypatch):
    (repo / "data" / "cand.py").write_text("A = 1\n")
    gpu_queue.cmd_submit(submit_args(["python", "s.py", "--candidate", "data/cand.py"]))
    job = pending_records(repo)[0]
    monkeypatch.setattr(gpu_queue.subprocess, "run", FakeRun())
    monkeypatch.setattr(gpu_queue, "GPU_RUN", GPU_RUN)
    gpu_queue.run_one(job, gpu_queue.dirs())
    log = (repo / "queue" / "done" / f"{job['id']}.log").read_text()
    assert "WARNING" not in log


# 4. per-job snapshot of gpu_run.sh -------------------------------------------------------


def test_run_one_executes_snapshot(repo, monkeypatch):
    live = repo / "gpu_run.sh"
    live.write_text("#!/usr/bin/env bash\necho v1\n")
    monkeypatch.setattr(gpu_queue, "GPU_RUN", live)
    seen = {}

    def fake_run(argv, **kw):
        snap = Path(argv[1])
        live.write_text("#!/usr/bin/env bash\necho v2 (edited mid-run)\n")  # steward edits
        seen.update(argv=argv, env=kw["env"], snap=snap, text=snap.read_text())
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(gpu_queue.subprocess, "run", fake_run)
    # An old-format record (no input_sha256, bare .py command) must still run.
    job = {
        "id": "0-20261010T000000-old",
        "name": "old",
        "lane": "x",
        "sync": ["data"],
        "pull": ["out"],
        "command": ["scripts/eval_runset.py", "--x"],
        "submitted": "2026-10-10T00:00:00+00:00",
    }
    d = queue_job(repo, job)
    gpu_queue.run_one(job, d)
    assert seen["argv"][0] == "bash"
    assert seen["snap"] != live and seen["snap"].parent == d["running"]
    assert seen["text"] == "#!/usr/bin/env bash\necho v1\n"
    assert seen["env"]["GPU_RUN_ROOT"] == str(repo)
    assert seen["argv"][2:] == [
        "--name",
        "old",
        "--sync",
        "data",
        "--pull",
        "out",
        "--",
        "python",
        "scripts/eval_runset.py",
        "--x",
    ]
    assert not seen["snap"].exists()  # cleaned up after the run
    done = json.loads((d["done"] / f"{job['id']}.json").read_text())
    assert done["exit"] == 7
    assert list(d["running"].iterdir()) == []


# gpu_run.sh ------------------------------------------------------------------------------


def test_gpu_run_syntax():
    subprocess.run(["bash", "-n", str(GPU_RUN)], check=True)


def test_gpu_run_body_is_wrapped_in_main():
    lines = [ln for ln in GPU_RUN.read_text().splitlines() if ln.strip()]
    assert lines[-1] == 'main "$@"'


def test_gpu_run_missing_sync_exits_3_before_ssh(tmp_path):
    # ssh/rsync stubs that record any call: the missing-sync check must fire first.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    for tool in ("ssh", "rsync"):
        stub = bindir / tool
        stub.write_text(f'#!/bin/sh\necho {tool} >> "{calls}"\nexit 99\n')
        stub.chmod(0o755)
    (tmp_path / "slurmhelps.json").write_text(
        json.dumps({"lan_gpu": {"host": "u@gpu-host", "base": "/srv/gpu"}})
    )
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "GPU_RUN_ROOT": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(GPU_RUN), "--name", "t", "--sync", "absent/dir", "--", "true"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
    assert "absent/dir" in proc.stderr
    assert not calls.exists()


# 5. submit-time dedupe and per-lane pending cap (2026-10-11) --------------------------------


def _iso(hours_ago: float) -> str:
    import datetime as dt

    return (dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours_ago)).isoformat(timespec="seconds")


def _put(repo: Path, state: str, job_id: str, **fields) -> None:
    d = gpu_queue.dirs()
    rec = {"id": job_id, "name": job_id, "lane": "test", "command": ["python", "s.py"], **fields}
    (d[state] / f"{job_id}.json").write_text(json.dumps(rec))


_SEQ = iter(range(10**6))


def _submit(command=("python", "s.py"), lane="test", force=False, **kw):
    # unique names: job ids are per-second, so same-name submits in one second would collide
    kw.setdefault("name", f"j{next(_SEQ)}")
    a = submit_args(list(command), **kw)
    a.lane = lane
    a.force = force
    gpu_queue.cmd_submit(a)


def test_dedupe_refuses_same_pending_job(repo, capsys):
    _put(repo, "pending", "5-a", submitted=_iso(0.1))
    with pytest.raises(SystemExit) as exc:
        _submit()
    assert exc.value.code == 4
    assert "already pending as 5-a" in capsys.readouterr().err


def test_dedupe_refuses_same_running_job(repo):
    _put(repo, "running", "5-a", started=_iso(0.1))
    with pytest.raises(SystemExit) as exc:
        _submit()
    assert exc.value.code == 4


def test_dedupe_compares_input_hashes(repo):
    (repo / "data" / "c.py").write_text("A = 1\n")
    cmd = ["python", "s.py", "--candidate", "data/c.py"]
    _submit(command=cmd)
    (repo / "data" / "c.py").write_text("A = 2\n")
    _submit(command=cmd)  # changed inputs: different sha256, accepted
    assert len(pending_records(repo)) == 2


def test_dedupe_compares_command(repo):
    _put(repo, "pending", "5-a", submitted=_iso(0.1))
    _submit(command=("python", "s.py", "--other"))
    assert len(pending_records(repo)) == 2


def test_dedupe_refuses_ok_job_within_24h(repo):
    _put(repo, "done", "5-a", exit=0, finished=_iso(23))
    with pytest.raises(SystemExit) as exc:
        _submit()
    assert exc.value.code == 4


def test_dedupe_allows_ok_job_older_than_24h(repo):
    _put(repo, "done", "5-a", exit=0, finished=_iso(25))
    _submit()
    assert len(pending_records(repo)) == 1


def test_dedupe_refuses_failed_job_within_6h(repo):
    _put(repo, "done", "5-a", exit=1, finished=_iso(5))
    with pytest.raises(SystemExit) as exc:
        _submit()
    assert exc.value.code == 4


def test_dedupe_allows_failed_job_older_than_6h(repo):
    _put(repo, "done", "5-a", exit=1, finished=_iso(7))
    _submit()
    assert len(pending_records(repo)) == 1


def test_dedupe_force_overrides(repo):
    _put(repo, "pending", "5-a", submitted=_iso(0.1))
    _submit(force=True)
    assert len(pending_records(repo)) == 2


def test_lane_pending_cap_refuses_fifth(repo, capsys):
    for i in range(gpu_queue.MAX_PENDING_PER_LANE):
        _submit(command=("python", f"s{i}.py"))
    with pytest.raises(SystemExit) as exc:
        _submit(command=("python", "extra.py"))
    assert exc.value.code == 4
    assert "already has 4 pending" in capsys.readouterr().err
    assert len(pending_records(repo)) == 4


def test_lane_pending_cap_is_per_lane(repo):
    for i in range(gpu_queue.MAX_PENDING_PER_LANE):
        _submit(command=("python", f"s{i}.py"), lane="a")
    _submit(command=("python", "other.py"), lane="b")
    assert len(pending_records(repo)) == gpu_queue.MAX_PENDING_PER_LANE + 1


def test_lane_pending_cap_ignores_done_and_force_overrides(repo):
    for i in range(gpu_queue.MAX_PENDING_PER_LANE):
        _put(repo, "done", f"d{i}", exit=0, finished=_iso(48), command=["python", f"d{i}.py"])
    _submit(command=("python", "fresh0.py"))
    assert len(pending_records(repo)) == 1
    for i in range(gpu_queue.MAX_PENDING_PER_LANE - 1):
        _submit(command=("python", f"p{i}.py"))
    _submit(command=("python", "forced.py"), force=True)
    assert len(pending_records(repo)) == gpu_queue.MAX_PENDING_PER_LANE + 1


def test_submit_refusal_is_pure():
    import datetime as dt

    now = dt.datetime(2026, 10, 11, 12, tzinfo=dt.UTC)
    rows = [
        {
            "id": "x",
            "state": "done",
            "exit": 2,
            "command": ["a"],
            "input_sha256": {},
            "finished": "2026-10-11T10:00:00+00:00",
        }
    ]
    assert gpu_queue.submit_refusal({"command": ["a"], "input_sha256": {}}, rows, "l", now)
    assert gpu_queue.submit_refusal({"command": ["b"], "input_sha256": {}}, rows, "l", now) is None
