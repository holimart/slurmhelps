"""doctor self-consistency: stable check ids, valid statuses, fixes for warns."""
from __future__ import annotations

import json
import subprocess
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "llm" / "bench"
BENCHCTL = BENCH / "benchctl"


def run_doctor(*extra, timeout=120):
    start = time.time()
    proc = subprocess.run([str(BENCHCTL), "doctor", "--offline", "--json"] + list(extra),
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
    return proc, time.time() - start


class DoctorTest(unittest.TestCase):
    def test_doctor_json_is_self_consistent(self):
        proc, elapsed = run_doctor()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertLess(elapsed, 90, "doctor must not block on the engine CLI")
        data = json.loads(proc.stdout)
        for key in ("generated_at", "backends", "config", "summary", "checks", "ok"):
            self.assertIn(key, data)
        self.assertTrue(data["config"]["config_loaded"])
        ids = [c["id"] for c in data["checks"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate check id")
        valid = {"ok", "warn", "fail", "info", "skip"}
        counts = {"ok": 0, "warn": 0, "fail": 0, "info": 0, "skip": 0}
        for check in data["checks"]:
            self.assertIn(check["status"], valid)
            self.assertTrue(check["group"])
            self.assertTrue(check["detail"])
            counts[check["status"]] += 1
            if check["status"] in ("warn", "fail"):
                self.assertTrue(check.get("fix"), "no fix for %s" % check["id"])
        self.assertEqual(counts, data["summary"])
        self.assertEqual(data["ok"], counts["fail"] == 0)

    def test_required_checks_are_present(self):
        proc, _ = run_doctor()
        data = json.loads(proc.stdout)
        ids = {c["id"] for c in data["checks"]}
        for required in ("host.login_node", "repo.config", "engine.strata",
                         "libs.glibcxx.system", "models.strata.iq3s",
                         "footgun.probe", "footgun.mtp_persisted",
                         "storage.work", "storage.tmp", "cloud.vast_client"):
            self.assertIn(required, ids)
        probe = next(c for c in data["checks"] if c["id"] == "footgun.probe")
        self.assertEqual(probe["status"], "ok", probe)

    def test_backend_filter_limits_the_checks(self):
        proc, _ = run_doctor("--backend", "cloud")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        ids = {c["id"] for c in data["checks"]}
        self.assertIn("cloud.vast_client", ids)
        self.assertNotIn("models.strata.iq3s", ids)
        self.assertNotIn("engine.strata", ids)
        self.assertEqual(data["backends"], ["cloud"])

    def test_text_output_has_no_failed_checks(self):
        proc = subprocess.run([str(BENCHCTL), "doctor", "--offline", "--strict"],
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("summary:", proc.stdout)
        self.assertNotIn("[FAIL]", proc.stdout)


if __name__ == "__main__":
    unittest.main()
