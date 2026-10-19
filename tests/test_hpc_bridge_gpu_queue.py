"""bin/hpc_bridge_gpu_queue.py: pure parts + claim/release state machine (no network)."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import gpu_queue  # noqa: E402
import hpc_bridge_gpu_queue as bridge  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_queue, "ROOT", tmp_path)
    monkeypatch.setattr(gpu_queue, "Q", tmp_path / "queue")
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.npz").write_bytes(b"x")
    return tmp_path


def make_job(d, jid, submitted, **kw):
    job = {
        "id": jid,
        "name": jid.split("-", 2)[-1],
        "lane": "test",
        "sync": ["data"],
        "pull": ["out/res"],
        "command": ["python", "scripts/s.py", "--out", "out/res"],
        "submitted": submitted,
        **kw,
    }
    (d["pending"] / f"{jid}.json").write_text(json.dumps(job))
    return job


# ---- pure parts ---------------------------------------------------------------------------


def test_translate_command_paths_and_interpreter(tmp_path):
    cmd = [
        "uv",
        "run",
        "python3",
        f"{tmp_path}/scripts/eval_runset.py",
        f"--out={tmp_path}/outputs/a.json",
        "--baseline",
        "/mnt/512ssd/whitebox-gpu/repo/outputs/b.py",
        "rel/path",
    ]
    assert bridge.translate_command(cmd, tmp_path) == [
        "python",
        "scripts/eval_runset.py",
        "--out=outputs/a.json",
        "--baseline",
        "outputs/b.py",
        "rel/path",
    ]
    assert bridge.translate_command(["scripts/x.py", "-a"], tmp_path) == [
        "python",
        "scripts/x.py",
        "-a",
    ]
    host_py = "/mnt/512ssd/whitebox-gpu/.venv/bin/python"
    assert bridge.translate_command([host_py, "s.py"], tmp_path)[0] == "python"


def test_remote_run_cmd_shape(tmp_path):
    job = {
        "id": "0-20261010T000000-my job",
        "pull": ["outputs/lanes/x/out"],
        "command": ["python", "scripts/s.py", "--msg", "it's"],
    }
    cmd = bridge.remote_run_cmd(job, tmp_path)
    jn = bridge.cluster_job_name(job)
    assert jn == "gq-0-20261010T000000-my-job"
    assert cmd.startswith(f"mkdir -p outputs/hpc/{jn} outputs/lanes/x/out; rc=0; ")
    assert f"PATH={bridge.MIRROR}/.venv/bin:$PATH" in cmd
    assert f"> outputs/hpc/{jn}/remote.log 2>&1 || rc=$?" in cmd
    assert cmd.endswith("> {output}")
    assert cmd.count("{output}") == 1
    # the inner command survives shell quoting
    import shlex

    toks = shlex.split(cmd.split("bash -c ", 1)[1].split(" > ", 1)[0])
    assert shlex.split(toks[0]) == ["python", "scripts/s.py", "--msg", "it's"]


def test_job_minutes_default_and_declared():
    assert bridge.job_minutes({}) == 90
    assert bridge.job_minutes({"minutes": 30}) == 30
    assert bridge.job_minutes({"limit_minutes": 200}) == 200
    assert bridge.job_minutes({"minutes": "x"}) == 90


def test_pick_oldest_by_submitted_and_cooldown():
    now = dt.datetime(2026, 10, 10, 12, tzinfo=dt.UTC)
    jobs = [
        {"id": "0-b", "submitted": "2026-10-10T10:00:00+00:00"},
        {"id": "9-a", "submitted": "2026-10-10T09:00:00+00:00"},
        {
            "id": "5-c",
            "submitted": "2026-10-10T08:00:00+00:00",
            "released": {"by": "hpc-bridge", "at": "2026-10-10T11:00:00+00:00"},
        },
        {
            "id": "5-d",
            "submitted": "2026-10-10T07:00:00+00:00",
            "released": {"by": "hpc-bridge", "at": "2026-10-10T01:00:00+00:00"},
        },
    ]
    assert [j["id"] for j in bridge.pick_oldest(jobs, now)] == ["5-d", "9-a", "0-b"]


# ---- claim / release / finish --------------------------------------------------------------


def test_claim_is_exclusive_and_release_restores(repo):
    d = gpu_queue.dirs()
    make_job(d, "5-j1", "2026-10-10T00:00:00+00:00")
    job = gpu_queue.claim_job(d, "5-j1", "hpc-bridge")
    assert job["claimed_by"] == "hpc-bridge" and "started" in job
    assert gpu_queue.claim_job(d, "5-j1") is None  # the local runner loses the race
    assert not (d["pending"] / "5-j1.json").exists()
    bridge.release(d, job, "because")
    back = json.loads((d["pending"] / "5-j1.json").read_text())
    assert "claimed_by" not in back and "started" not in back
    assert back["released"]["reason"] == "because" and back["released"]["by"] == "hpc-bridge"
    assert list(d["running"].iterdir()) == []


def test_local_runner_skips_job_claimed_by_bridge(repo, monkeypatch, capsys):
    d = gpu_queue.dirs()
    job = make_job(d, "5-j1", "2026-10-10T00:00:00+00:00")
    gpu_queue.claim_job(d, "5-j1", "hpc-bridge")
    called = []
    monkeypatch.setattr(gpu_queue.subprocess, "run", lambda *a, **k: called.append(a))
    gpu_queue.run_one(job, d)  # runner read the pending file before the bridge renamed it
    assert called == [] and "claimed elsewhere" in capsys.readouterr().out
    assert json.loads((d["running"] / "5-j1.json").read_text())["claimed_by"] == "hpc-bridge"


class FakeCluster:
    """Fakes rsync/ssh/hpc_run.sh for run_job."""

    def __init__(self, repo, nbytes=1000, exit_code=0, hpc_rc=0, write_result=True):
        self.repo, self.nbytes, self.exit_code = repo, nbytes, exit_code
        self.hpc_rc, self.write_result = hpc_rc, write_result
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] == "rsync" and "--dry-run" in argv:
            out = f"Total transferred file size: {self.nbytes:,} bytes\n"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["bash", str(bridge.HPC_RUN)]:
            jn = argv[2]
            base = self.repo / "outputs" / "hpc" / jn
            base.mkdir(parents=True, exist_ok=True)
            if self.write_result:
                (base / "eval.json").write_text(json.dumps({"exit": self.exit_code, "node": "n1"}))
                (base / "remote.log").write_text("hello from l40s\n")
            kw["stdout"].write("[hpc] job 1 finished\n")
            return subprocess.CompletedProcess(argv, self.hpc_rc)
        raise AssertionError(argv)


def claimed(repo, **kw):
    d = gpu_queue.dirs()
    make_job(d, "5-j1", "2026-10-10T00:00:00+00:00", **kw)
    return d, gpu_queue.claim_job(d, "5-j1", "hpc-bridge")


def test_run_job_success_marks_done(repo, monkeypatch):
    d, job = claimed(repo)
    fake = FakeCluster(repo, exit_code=7)
    monkeypatch.setattr(bridge.subprocess, "run", fake)
    assert bridge.run_job(d, job, out=lambda *_: None) == 7
    done = json.loads((d["done"] / "5-j1.json").read_text())
    assert done["exit"] == 7 and done["note"].startswith("ran on troja L40S via bridge")
    log = (d["done"] / "5-j1.log").read_text()
    assert "hello from l40s" in log and "# pulled out/res" in log
    hpc = [c for c in fake.calls if c[:2] == ["bash", str(bridge.HPC_RUN)]][0]
    assert hpc[3] == "90" and hpc[-1] == "--yes"
    up = [c for c in fake.calls if c[0] == "rsync" and "-azR" in c and "--dry-run" not in c][0]
    assert up[-3:] == ["scripts/", "data", f"{bridge.SSH}:{bridge.MIRROR}/"]
    assert list(d["running"].iterdir()) == [] and list(d["pending"].iterdir()) == []


def test_run_job_refuses_missing_sync(repo, monkeypatch):
    d, job = claimed(repo, sync=["data", "nope"])
    monkeypatch.setattr(bridge.subprocess, "run", FakeCluster(repo))
    assert bridge.run_job(d, job, out=lambda *_: None) == -1
    back = json.loads((d["pending"] / "5-j1.json").read_text())
    assert "missing locally: nope" in back["released"]["reason"]


def test_run_job_refuses_over_2gb(repo, monkeypatch):
    d, job = claimed(repo)
    fake = FakeCluster(repo, nbytes=3 * 1024**3)
    monkeypatch.setattr(bridge.subprocess, "run", fake)
    assert bridge.run_job(d, job, out=lambda *_: None) == -1
    assert "> 2 GB" in json.loads((d["pending"] / "5-j1.json").read_text())["released"]["reason"]
    assert not any(c[:2] == ["bash", str(bridge.HPC_RUN)] for c in fake.calls)


def test_run_job_cluster_busy_releases(repo, monkeypatch):
    d, job = claimed(repo)
    monkeypatch.setattr(bridge.subprocess, "run", FakeCluster(repo, hpc_rc=3, write_result=False))
    assert bridge.run_job(d, job, out=lambda *_: None) == -1
    assert (d["pending"] / "5-j1.json").exists()


def test_run_job_no_exit_file_marks_failed(repo, monkeypatch):
    d, job = claimed(repo)
    monkeypatch.setattr(bridge.subprocess, "run", FakeCluster(repo, hpc_rc=1, write_result=False))
    assert bridge.run_job(d, job, out=lambda *_: None) == 1
    done = json.loads((d["done"] / "5-j1.json").read_text())
    assert done["exit"] == 1 and "no exit file" in done["bridge_error"]


def test_run_job_interrupt_after_submit_marks_failed(repo, monkeypatch):
    d, job = claimed(repo)
    fake = FakeCluster(repo)

    def boom(argv, **kw):
        if argv[:2] == ["bash", str(bridge.HPC_RUN)]:
            raise bridge.Interrupted("signal 15")
        return fake(argv, **kw)

    monkeypatch.setattr(bridge.subprocess, "run", boom)
    with pytest.raises(bridge.Interrupted):
        bridge.run_job(d, job, out=lambda *_: None)
    done = json.loads((d["done"] / "5-j1.json").read_text())
    assert done["exit"] == 125 and "stage submitted" in done["bridge_error"]


def test_run_job_error_before_submit_releases(repo, monkeypatch):
    d, job = claimed(repo)

    def boom(argv, **kw):
        raise OSError("rsync not found")

    monkeypatch.setattr(bridge.subprocess, "run", boom)
    assert bridge.run_job(d, job, out=lambda *_: None) == 125
    back = json.loads((d["pending"] / "5-j1.json").read_text())
    assert "before submission" in back["released"]["reason"]


def test_recover_stale_claims(repo):
    d = gpu_queue.dirs()
    make_job(d, "5-a", "2026-10-10T00:00:00+00:00")
    make_job(d, "5-b", "2026-10-10T00:00:01+00:00")
    make_job(d, "5-c", "2026-10-10T00:00:02+00:00")
    a = gpu_queue.claim_job(d, "5-a", "hpc-bridge")
    bridge.set_stage(d, a, "syncing")
    b = gpu_queue.claim_job(d, "5-b", "hpc-bridge")
    bridge.set_stage(d, b, "submitted", bridge_job="gq-5-b")
    gpu_queue.claim_job(d, "5-c")  # the local runner's claim: not ours to touch
    bridge.recover_stale(d, out=lambda *_: None)
    assert (d["pending"] / "5-a.json").exists()
    assert json.loads((d["done"] / "5-b.json").read_text())["exit"] == 125
    assert (d["running"] / "5-c.json").exists()


def test_dry_run_claims_nothing(repo, monkeypatch, capsys):
    d = gpu_queue.dirs()
    make_job(d, "0-new", "2026-10-10T05:00:00+00:00")
    make_job(d, "9-old", "2026-10-10T01:00:00+00:00")
    monkeypatch.setattr(bridge.subprocess, "run", FakeCluster(repo))
    assert bridge.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "9-old" in out and "-> OK" in out
    assert sorted(p.name for p in d["pending"].iterdir()) == ["0-new.json", "9-old.json"]
    assert list(d["running"].iterdir()) == []


def test_pick_oldest_prefers_lanes_not_bridged_in_last_hour():
    now = dt.datetime(2026, 10, 10, 12, tzinfo=dt.UTC)
    jobs = [
        {"id": "5-busy-old", "lane": "busy", "submitted": "2026-10-10T06:00:00+00:00"},
        {"id": "5-quiet", "lane": "quiet", "submitted": "2026-10-10T09:00:00+00:00"},
        {"id": "5-busy-new", "lane": "busy", "submitted": "2026-10-10T10:00:00+00:00"},
    ]
    order = bridge.pick_oldest(jobs, now, recent_lanes={"busy"})
    assert [j["id"] for j in order] == ["5-quiet", "5-busy-old", "5-busy-new"]
    # with no recent lanes it is plain oldest-first
    assert bridge.pick_oldest(jobs, now)[0]["id"] == "5-busy-old"


def test_recent_bridged_lanes_counts_only_bridge_jobs_within_hour(repo):
    now = dt.datetime(2026, 10, 10, 12, tzinfo=dt.UTC)
    d = gpu_queue.dirs()

    def done(jid, lane, finished, claimed_by="hpc-bridge"):
        rec = {"id": jid, "lane": lane, "finished": finished, "exit": 0}
        if claimed_by:
            rec["claimed_by"] = claimed_by
        (d["done"] / f"{jid}.json").write_text(json.dumps(rec))

    done("a", "fresh", "2026-10-10T11:30:00+00:00")
    done("b", "old", "2026-10-10T10:00:00+00:00")
    done("c", "local", "2026-10-10T11:50:00+00:00", claimed_by=None)
    assert bridge.recent_bridged_lanes(d, now) == {"fresh"}


def test_main_bridge_run_takes_fair_lane_first(repo, monkeypatch):
    d = gpu_queue.dirs()
    make_job(d, "5-busy-old", "2026-10-10T06:00:00+00:00", lane="busy")
    make_job(d, "5-quiet", "2026-10-10T09:00:00+00:00", lane="quiet")
    (d["done"] / "x.json").write_text(
        json.dumps(
            {
                "id": "x",
                "lane": "busy",
                "claimed_by": "hpc-bridge",
                "finished": dt.datetime.now(dt.UTC).isoformat(),
                "exit": 0,
            }
        )
    )
    picked = []
    monkeypatch.setattr(bridge, "recover_stale", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "our_cluster_jobs", lambda: 0)
    monkeypatch.setattr(bridge, "run_job", lambda d_, job, out=print: picked.append(job["id"]) or 0)
    assert bridge.main([]) == 0
    assert picked == ["5-quiet"]


# ---- long-job gate -------------------------------------------------------------------------------


@pytest.mark.parametrize("argv,active,dispatch", [
    ([], 1, False),
    (["--active-job-limit", "4"], 3, True),
    (["--active-job-limit", "4"], 4, False),
    (["--active-job-limit", "6"], 6, False),
])
def test_capacity_opt_in_keeps_atomic_claim_and_default_idle_policy(repo, monkeypatch, argv, active, dispatch):
    monkeypatch.setitem(bridge._C, "max_jobs", 6)
    d = gpu_queue.dirs()
    make_job(d, "5-ready", "2026-10-10T09:00:00+00:00")
    monkeypatch.setattr(bridge, "recover_stale", lambda *_: None)
    monkeypatch.setattr(bridge, "our_cluster_jobs", lambda: active)
    picked = []
    def run_claimed(d_, job, out=print):
        assert not (d_["pending"] / "5-ready.json").exists()
        assert json.loads((d_["running"] / "5-ready.json").read_text())["claimed_by"] == "hpc-bridge"
        picked.append(job["id"])
        return 0
    monkeypatch.setattr(bridge, "run_job", run_claimed)
    assert bridge.main(argv) == (0 if dispatch else bridge.EXIT_NOTHING)
    assert picked == (["5-ready"] if dispatch else [])
    assert (d["pending"] / "5-ready.json").exists() is not dispatch


@pytest.mark.parametrize("limit", ["0", "7"])
def test_capacity_option_cannot_disable_or_exceed_configured_cap(repo, monkeypatch, limit):
    monkeypatch.setitem(bridge._C, "max_jobs", 6)
    with pytest.raises(SystemExit) as exc:
        bridge.main(["--active-job-limit", limit])
    assert exc.value.code == 2


def _hpc(repo):
    p = repo / "outputs" / "hpc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_age(path, minutes):
    t = dt.datetime.now().timestamp() - minutes * 60
    os.utime(path, (t, t))


def _chain_state(repo, tsv_age_min, finished_ago_min):
    hpc = _hpc(repo)
    tsv = hpc / "chain.tsv"
    tsv.write_text("job\t10\techo\n")
    _set_age(tsv, tsv_age_min)
    when = dt.datetime.now().astimezone() - dt.timedelta(minutes=finished_ago_min)
    (hpc / "chain.log").write_text(f"{when:%H:%M} finished old-job rc=0\n")


def test_chain_recently_active_by_tsv_or_finished_line(repo):
    now = dt.datetime.now(dt.UTC)
    assert bridge.chain_recently_active(now, root=repo) is False  # no files at all
    _chain_state(repo, tsv_age_min=5, finished_ago_min=600)
    assert bridge.chain_recently_active(now, root=repo) is True  # fresh chain.tsv
    _chain_state(repo, tsv_age_min=60, finished_ago_min=5)
    assert bridge.chain_recently_active(now, root=repo) is True  # recent finished line
    _chain_state(repo, tsv_age_min=60, finished_ago_min=30)
    assert bridge.chain_recently_active(now, root=repo) is False


def test_long_job_skipped_when_chain_recently_active(repo, monkeypatch):
    d = gpu_queue.dirs()
    make_job(d, "1-long", "2026-10-10T06:00:00+00:00", minutes=68)
    make_job(d, "2-short", "2026-10-10T09:00:00+00:00", minutes=10)
    now = dt.datetime(2026, 10, 10, 12, tzinfo=dt.UTC)
    jobs = bridge.load_pending(d)
    assert [j["id"] for j in bridge.pick_oldest(jobs, now, skip_long=True)] == ["2-short"]
    only_long = [j for j in jobs if j["id"] == "1-long"]
    assert bridge.pick_oldest(only_long, now, skip_long=True) == []

    picked = []
    monkeypatch.setattr(bridge, "recover_stale", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "our_cluster_jobs", lambda: 0)
    monkeypatch.setattr(bridge, "run_job", lambda d_, job, out=print: picked.append(job["id"]) or 0)
    _chain_state(repo, tsv_age_min=5, finished_ago_min=600)
    assert bridge.main([]) == 0
    assert picked == ["2-short"]


def test_long_job_taken_when_chain_idle(repo, monkeypatch):
    d = gpu_queue.dirs()
    make_job(d, "1-long", "2026-10-10T06:00:00+00:00", minutes=68)
    make_job(d, "2-short", "2026-10-10T09:00:00+00:00", minutes=10)
    picked = []
    monkeypatch.setattr(bridge, "recover_stale", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "our_cluster_jobs", lambda: 0)
    monkeypatch.setattr(bridge, "run_job", lambda d_, job, out=print: picked.append(job["id"]) or 0)
    _chain_state(repo, tsv_age_min=120, finished_ago_min=180)
    assert bridge.main([]) == 0
    assert picked == ["1-long"]  # idle chain: oldest pending job, long or not


def test_short_and_unknown_jobs_preferred_when_chain_active(repo, monkeypatch):
    d = gpu_queue.dirs()
    make_job(d, "1-long", "2026-10-10T06:00:00+00:00", minutes=68)
    make_job(d, "2-unknown", "2026-10-10T09:00:00+00:00")  # no minutes, no history -> short
    now = dt.datetime(2026, 10, 10, 12, tzinfo=dt.UTC)
    order = bridge.pick_oldest(bridge.load_pending(d), now, skip_long=True, history={})
    assert [j["id"] for j in order] == ["2-unknown"]
    # without the gate the oldest (long) job still comes first
    assert bridge.pick_oldest(bridge.load_pending(d), now)[0]["id"] == "1-long"


def test_expected_minutes_declared_then_bridge_history(repo):
    log = _hpc(repo) / "bridge.log"
    lines = [
        "2026-10-10T21:00:00+00:00 bridge submit 9-20261010T050201-corpus-grower-organiser128-"
        "dc759413-s01 as wb-x (90 min)",
        "2026-10-10T22:00:00+00:00 bridge done 9-20261010T050201-corpus-grower-organiser128-"
        "dc759413-s01 exit 0",
        "2026-10-10T23:00:00+00:00 bridge submit 9-20261010T050201-corpus-grower-organiser128-"
        "dc759413-s02 as wb-y (90 min)",
        "2026-10-11T00:10:00+00:00 bridge done 9-20261010T050201-corpus-grower-organiser128-"
        "dc759413-s02 exit 0",
        "2026-10-11T01:00:00+00:00 bridge submit 9-20261011T000000-rds-x-c1f60027 as wb-z (90 min)",
    ]
    log.write_text("\n".join(lines) + "\n")
    assert bridge.job_family("corpus-grower-organiser128-dc759413-s01") == (
        "corpus-grower-organiser128"
    )
    assert bridge.job_family("rds-corrbeta-shrink90-c1f60027") == "rds-corrbeta-shrink90"
    hist = bridge.bridge_runtime_history(repo)
    assert hist == {"corpus-grower-organiser128": 65.0}  # median of 60 and 70; unfinished skipped
    fam_job = {"name": "corpus-grower-organiser128-dc759413-s07"}
    assert bridge.expected_minutes(fam_job, hist) == 65.0
    assert bridge.is_long(fam_job, hist)
    assert bridge.expected_minutes({**fam_job, "minutes": 10}, hist) == 10.0  # declared wins
    assert not bridge.is_long({**fam_job, "minutes": 10}, hist)
    assert bridge.expected_minutes({"name": "never-seen"}, hist) is None  # unknown -> short
    assert not bridge.is_long({"name": "never-seen"}, hist)


# whiteboxchallenge 2026-10-18/19: Slurm TIMEOUT/scancel runs the TERM trap, which writes
# exit_code + terminated (hpc_partial finalize-failure), not "exit". The bridge must treat
# that as a terminal receipt instead of leaving an orphan reservation.


def test_terminal_exit_code_reads_normal_exit():
    assert bridge.terminal_exit_code({"exit": 0}) == 0
    assert bridge.terminal_exit_code({"exit": 2, "exit_code": 9}) == 2


def test_terminal_exit_code_reads_trap_receipt():
    rec = {"status": "failed", "exit_code": 143, "terminated": True, "log_tail": "x"}
    assert bridge.terminal_exit_code(rec) == 143


def test_terminal_exit_code_rejects_receiptless_doc():
    with pytest.raises(KeyError):
        bridge.terminal_exit_code({"status": "partial"})
    with pytest.raises(KeyError):
        bridge.terminal_exit_code({"exit_code": 1, "status": "partial", "terminated": False})
