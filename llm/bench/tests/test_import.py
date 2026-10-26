"""benchctl import: new-dir detection, dry-run, and the state file.

Uses a fake dataset importer; the real opportunityscan checkout is never run.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "llm" / "bench"
BENCHCTL = BENCH / "benchctl"


class ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.runs = self.work / "strata-runs"
        for name in ("run-a", "run-b"):
            (self.runs / name).mkdir(parents=True)
            (self.runs / name / "summary.csv").write_text("ctx,status\n1024,ok\n")
        self.dataset = self.root / "dataset"
        script = self.dataset / "modelsquantdb" / "scripts" / "import_strata_runs.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import pathlib, sys\n"
            "marker = pathlib.Path(__file__).with_name('ran.txt')\n"
            "if '--dry-run' not in sys.argv:\n"
            "    marker.write_text(' '.join(sys.argv[1:]))\n"
            "print('fake importer', ' '.join(sys.argv[1:]))\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.marker = script.with_name("ran.txt")

    def tearDown(self):
        self.tmp.cleanup()

    def benchctl(self, *args):
        env = dict(os.environ)
        env["BENCH_WORK_ROOT"] = str(self.work)
        env["BENCH_CONFIG"] = str(BENCH / "config.json")
        return subprocess.run([str(BENCHCTL), "import", "--dataset", str(self.dataset),
                               "--json"] + list(args),
                              env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)

    def test_dry_run_detects_dirs_and_does_not_import(self):
        proc = self.benchctl("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(len(data["plan"]["new_dirs"]), 2)
        self.assertIn("--dry-run", data["argv"])
        self.assertFalse(self.marker.exists())

    def test_real_import_updates_state_and_second_run_is_empty(self):
        proc = self.benchctl()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertTrue(data["state_updated"])
        self.assertTrue(self.marker.exists())
        state_file = self.work / "bench" / "state" / "import.json"
        self.assertTrue(state_file.exists())
        state = json.loads(state_file.read_text())
        self.assertEqual(len(state["imported"]), 2)
        again = self.benchctl("--dry-run")
        self.assertEqual(again.returncode, 0, again.stderr)
        data = json.loads(again.stdout)
        self.assertEqual(data["plan"]["new_dirs"], [])
        forced = self.benchctl("--dry-run", "--force")
        self.assertEqual(len(json.loads(forced.stdout)["plan"]["new_dirs"]), 2)

    def test_missing_importer_is_a_clean_error(self):
        env = dict(os.environ)
        env["BENCH_WORK_ROOT"] = str(self.work)
        proc = subprocess.run([str(BENCHCTL), "import", "--dataset",
                               str(self.root / "nope"), "--dry-run", "--json"],
                              env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("dataset importer not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
