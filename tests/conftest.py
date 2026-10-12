"""Tests run against the example project config (examples/slurmhelps.json, whiteboxchallenge's values).

bin/gpu_queue.py and bin/hpc_bridge_gpu_queue.py read their config at import time, so the project
is pinned here, before any test module imports them. Tests that need a different project pass
their own environment to subprocesses.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["SLURMHELPS_PROJECT"] = str(Path(__file__).resolve().parent.parent / "examples")
