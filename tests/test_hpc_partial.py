"""lib/hpc_partial.py: atomic rewrites, resume-safe reading, fail path, wrapper finalize."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

import hpc_partial as hp  # noqa: E402


def test_add_rewrites_atomically_and_leaves_no_tmp(tmp_path):
    out = tmp_path / "eval.json"
    pr = hp.PartialResults(out, meta={"job": "t"})
    pr.add("net0", {"mse": 1.5})
    doc = json.loads(out.read_text())
    assert doc["status"] == "partial"
    assert doc["units_done"] == 1 and doc["units"] == {"net0": {"mse": 1.5}}
    assert doc["meta"] == {"job": "t"} and "updated_at" in doc
    pr.add("net1", {"mse": 2.0})
    assert json.loads(out.read_text())["units_done"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["eval.json"]


def test_numpy_like_values_serialize(tmp_path):
    class Arr:
        def tolist(self):
            return [1.0, 2.0]

    out = tmp_path / "e.json"
    hp.PartialResults(out).add("k", {"v": Arr()})
    assert json.loads(out.read_text())["units"]["k"]["v"] == [1.0, 2.0]


def test_finish_merges_summary_and_keeps_units(tmp_path):
    out = tmp_path / "e.json"
    with hp.PartialResults(out, meta={"job": "t"}) as pr:
        pr.add("a", 1)
        pr.finish({"decision": "NOT-PROMISING", "status": "ignored"})
    doc = json.loads(out.read_text())
    assert doc["status"] == "complete"
    assert doc["decision"] == "NOT-PROMISING"
    assert doc["units"] == {"a": 1}


def test_resume_reads_partial_units_only_when_meta_matches(tmp_path):
    out = tmp_path / "e.json"
    first = hp.PartialResults(out, meta={"job": "t"})
    first.add("net0", {"x": 1})
    again = hp.PartialResults(out, meta={"job": "t"}, resume=True)
    assert again.has("net0") and not again.has("net1")
    other = hp.PartialResults(out, meta={"job": "other"}, resume=True)
    assert other.units == {}
    done = hp.PartialResults(out, meta={"job": "t"})
    done.finish({"x": 0})
    assert hp.PartialResults(out, meta={"job": "t"}, resume=True).units == {}


def test_read_partial_is_safe_on_missing_or_corrupt(tmp_path):
    assert hp.read_partial(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert hp.read_partial(bad) is None
    assert hp.PartialResults(bad, resume=True).units == {}


def test_exception_records_failed_and_reraises(tmp_path):
    out = tmp_path / "e.json"
    with pytest.raises(RuntimeError, match="boom"), hp.PartialResults(out, meta={"job": "t"}) as pr:
        pr.add("net0", {"ok": True})
        pr.add("net1", {"ok": True})
        raise RuntimeError("boom")
    doc = json.loads(out.read_text())
    assert doc["status"] == "failed"
    assert doc["error"] == "RuntimeError: boom"
    assert "Traceback" in doc["traceback"] and "boom" in doc["traceback"]
    assert doc["units_done"] == 2


def test_checkpointed_decorator_fail_path(tmp_path):
    out = tmp_path / "e.json"

    @hp.checkpointed(out, {"job": "d"})
    def job(pr, n):
        pr.add("u0", n)
        raise ValueError("late")

    with pytest.raises(ValueError):
        job(7)
    doc = json.loads(out.read_text())
    assert doc["status"] == "failed" and doc["units"] == {"u0": 7}


def test_finalize_failure_annotates_partial_and_tails_log(tmp_path):
    ev = tmp_path / "eval.json"
    log = tmp_path / "job.log"
    hp.PartialResults(ev).add("u0", 1)
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    doc = hp.finalize_failure(ev, log, exit_code=143, terminated=True)
    on_disk = json.loads(ev.read_text())
    assert on_disk == doc
    assert on_disk["status"] == "partial" and on_disk["units_done"] == 1
    assert on_disk["exit_code"] == 143 and on_disk["terminated"] is True
    tail = on_disk["log_tail"].splitlines()
    assert len(tail) == hp.LOG_TAIL_LINES and tail[-1] == "line 99"


def test_finalize_failure_minimal_when_no_json(tmp_path):
    ev = tmp_path / "eval.json"
    log = tmp_path / "job.log"  # missing log is fine
    doc = hp.finalize_failure(ev, log, exit_code=1, terminated=False)
    assert doc == json.loads(ev.read_text())
    assert doc["status"] == "failed" and doc["exit_code"] == 1
    assert doc["terminated"] is False and doc["log_tail"] == ""


def test_cli_finalize_failure(tmp_path, capsys):
    ev = tmp_path / "eval.json"
    log = tmp_path / "job.log"
    log.write_text("hello\n")
    assert (
        hp.main(
            [
                "finalize-failure",
                "--eval",
                str(ev),
                "--log",
                str(log),
                "--exit-code",
                "2",
                "--terminated",
                "0",
            ]
        )
        == 0
    )
    doc = json.loads(ev.read_text())
    assert doc["exit_code"] == 2 and doc["log_tail"] == "hello"
