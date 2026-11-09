"""workset.Scan under a live, changing tree: a directory that vanishes mid-scan is skipped, other errors stay fatal.

Regression for the 2026-11-09 final spill that failed with "cannot list .../.git/worktrees/wt1" because a lane
removed the worktree while the namespace was being quiesced.
"""
from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("workset_under_test_scan", HERE / "workset.py")
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)


class _NoExclude:
    def excluded(self, abspath, rel, name, is_dir):
        return False


def _tree(tmp_path: Path):
    root = tmp_path / "tree"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "one.txt").write_text("1")
    (root / "b" / "two.txt").write_text("2")
    k = types.SimpleNamespace(live=[], secrets=[], trees=[str(root)], matcher=_NoExclude())
    return root, k


def _scan(k):
    return ws.Scan(k, lambda t: t, threads=1, live_mode="content")


def test_directory_vanishing_between_listing_and_scan_is_skipped(tmp_path, monkeypatch):
    root, k = _tree(tmp_path)
    real = os.scandir

    def flaky(path):
        if str(path).endswith("/a"):
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real(path)

    monkeypatch.setattr(ws.os, "scandir", flaky)
    sc = _scan(k).run()
    assert any("vanished during scan" in n and n.endswith("/a") for n in sc.notes)
    assert sc.counts.files == 1, "the surviving sibling must still be counted"


def test_other_listing_errors_stay_fatal(tmp_path, monkeypatch):
    root, k = _tree(tmp_path)
    real = os.scandir

    def denied(path):
        if str(path).endswith("/a"):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(ws.os, "scandir", denied)
    with pytest.raises(ws.WSError, match="cannot list"):
        _scan(k).run()
