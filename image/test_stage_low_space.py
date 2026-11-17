"""workset.py stage-in under tight node-local space (real mksquashfs/unsquashfs, tiny tree).

Tight space: free < image + unpacked  ->  unpack straight from the shared image (no local copy),
and the shared image must never be deleted.  Too tight: free < unpacked + headroom -> refuse.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("workset_under_test", HERE / "workset.py")
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)

pytestmark = pytest.mark.skipif(not (shutil.which("mksquashfs") or Path("/usr/sbin/mksquashfs").exists()),
                                reason="squashfs tools missing")


def _build(tmp_path: Path):
    src = tmp_path / "src"
    (src / "a" / "b").mkdir(parents=True)
    (src / "a" / "one.txt").write_text("one\n")
    (src / "a" / "b" / "two.bin").write_bytes(b"\x00\x01" * 4096)
    (src / "link").symlink_to("a/one.txt")
    img = tmp_path / "images" / "data.sqfs"
    img.parent.mkdir()
    mk = shutil.which("mksquashfs") or "/usr/sbin/mksquashfs"
    subprocess.run([mk, str(src), str(img), "-no-progress", "-quiet"], check=True)
    unpacked = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    Path(str(img) + ".manifest.json").write_text(json.dumps({"image": {"bytes": unpacked}}))
    return img, unpacked


def _kind(tmp_path: Path, img: Path, headroom: int):
    return types.SimpleNamespace(kind="data", image=img, stage=tmp_path / "stage", root=tmp_path / "stage" / "data-root",
                                 cfg={"direct_unpack_headroom_bytes": headroom}, secrets=[])


def test_tight_space_unpacks_directly_and_keeps_the_shared_image(tmp_path, monkeypatch):
    img, unpacked = _build(tmp_path)
    size = img.stat().st_size
    k = _kind(tmp_path, img, headroom=1000)
    tight = unpacked + 2000  # >= unpacked + headroom (1000) but < unpacked + image size
    assert unpacked + 1000 <= tight < size + unpacked
    monkeypatch.setattr(ws, "free_bytes", lambda p: tight)
    ws._stage_from(k, img)
    assert img.exists() and img.stat().st_size == size, "the shared image must survive a direct unpack"
    assert (k.root / "a" / "one.txt").read_text() == "one\n"
    assert (k.root / "link").is_symlink()
    assert not (k.stage / "data.sqfs").exists(), "no node-local copy in direct mode"


def test_too_tight_refuses_and_leaves_the_image(tmp_path, monkeypatch):
    img, unpacked = _build(tmp_path)
    k = _kind(tmp_path, img, headroom=1000)
    monkeypatch.setattr(ws, "free_bytes", lambda p: unpacked + 10)  # < unpacked + headroom
    with pytest.raises(ws.WSError, match="not enough node-local space"):
        ws._stage_from(k, img)
    assert img.exists()
    assert not k.root.exists()


def test_roomy_node_still_copies_then_removes_its_local_copy(tmp_path, monkeypatch):
    img, unpacked = _build(tmp_path)
    k = _kind(tmp_path, img, headroom=1000)
    monkeypatch.setattr(ws, "free_bytes", lambda p: 10**12)
    ws._stage_from(k, img)
    assert img.exists()
    assert (k.root / "a" / "b" / "two.bin").stat().st_size == 8192
    assert not (k.stage / "data.sqfs").exists(), "the local copy is removed after unpacking, as before"
