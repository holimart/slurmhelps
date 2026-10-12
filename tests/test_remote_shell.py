"""Run the stub-based workstation-side shell tests (tests/shell/test-remote.sh) under pytest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "shell" / "test-remote.sh"


@pytest.mark.skipif(
    shutil.which("bash") is None or not Path("/usr/bin/rsync").exists(),
    reason="needs bash and /usr/bin/rsync",
)
def test_remote_shell(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "remote-test")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "0 failed" in proc.stdout
