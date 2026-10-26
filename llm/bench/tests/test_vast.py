"""Vast adapter: the offer filter and every mutation's dry-run (no network)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "llm" / "bench"
VAST = ROOT / "llm" / "cloud" / "vast" / "vast.py"

OFFERS = [
    {"id": 1, "gpu_name": "RTX 3090", "num_gpus": 1, "rentable": True,
     "cpu_ram": 128000, "disk_space": 200, "driver_version": "610.57.04",
     "cuda_max_good": "13.0", "internet_down_cost_per_tb": 2.67,
     "reliability2": 0.98, "dph_total": 0.14, "geolocation": "CZ",
     "inet_down": 495, "machine_id": 11, "verified": True},
    # The Oman trap: $53.33/TB download cost that dwarfed the hourly price.
    {"id": 2, "gpu_name": "RTX 3090", "num_gpus": 1, "rentable": True,
     "cpu_ram": 160000, "disk_space": 400, "driver_version": "580.126.09",
     "cuda_max_good": "13.0", "internet_down_cost_per_tb": 53.33,
     "reliability2": 0.99, "dph_total": 0.2196, "geolocation": "OM",
     "inet_down": 500, "machine_id": 12},
    # Below both driver floors (570 cannot run the cuda12 engine either).
    {"id": 3, "gpu_name": "RTX 3090", "num_gpus": 1, "rentable": True,
     "cpu_ram": 128000, "disk_space": 200, "driver_version": "570.181",
     "cuda_max_good": "12.0", "internet_down_cost_per_tb": 0.1,
     "reliability2": 0.99, "dph_total": 0.12, "geolocation": "CA",
     "inet_down": 900, "machine_id": 13},
    {"id": 4, "gpu_name": "RTX 3090", "num_gpus": 1, "rentable": False,
     "cpu_ram": 128000, "disk_space": 200, "driver_version": "610.0",
     "cuda_max_good": "13.0", "internet_down_cost_per_tb": 1.0,
     "reliability2": 0.99, "dph_total": 0.10, "geolocation": "US",
     "inet_down": 900, "machine_id": 14},
]


class VastAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.tmp.name) / "offers.json"
        cls.fixture.write_text(json.dumps({"offers": OFFERS}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_vast(self, *args):
        proc = subprocess.run([sys.executable, str(VAST)] + list(args),
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=60)
        return proc

    def filters(self):
        return ["--offers-file", str(self.fixture), "--gpu-name", "RTX 3090",
                "--num-gpus", "1", "--min-ram-gb", "120", "--min-disk-gb", "150",
                "--min-driver", "580.65", "--min-cuda", "13.0",
                "--max-down-cost-tb", "5", "--min-reliability", "0.97"]

    def test_offer_filter_rejects_bandwidth_and_driver_traps(self):
        proc = self.run_vast("--dry-run", "--json", "offers", *self.filters())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual([o["id"] for o in data["eligible"]], [1])
        self.assertEqual(data["rejected"], 3)
        reasons = " ".join(" ".join(x["reasons"]) for x in data["rejected_examples"])
        self.assertIn("download $53.33/TB", reasons)
        self.assertIn("driver 570.181", reasons)
        self.assertIn("not rentable", reasons)

    def test_plan_reports_gpu_and_download_cost(self):
        proc = self.run_vast("--dry-run", "--json", "plan", *self.filters(),
                             "--model-gb", "113.2", "--hours", "2.0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        best = data["best"]
        self.assertEqual(best["id"], 1)
        self.assertAlmostEqual(best["gpu_cost_usd"], 0.28, places=4)
        self.assertAlmostEqual(best["download_cost_usd"], 0.3022, places=4)
        self.assertAlmostEqual(best["total_usd"], 0.5822, places=4)

    def test_plan_from_recipe(self):
        proc = self.run_vast("--dry-run", "--json", "plan",
                             "--offers-file", str(self.fixture),
                             "--gpu-name", "RTX 3090", "--num-gpus", "1",
                             "--recipe-file", str(BENCH / "recipes" / "vast-3090.json"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["recipe"], "vast-3090")
        self.assertEqual(data["model_gb"], 113.2)
        self.assertEqual(data["hours"], 2.0)

    def test_no_eligible_offer_is_a_clean_failure(self):
        proc = self.run_vast("--dry-run", "--json", "plan",
                             "--offers-file", str(self.fixture),
                             "--gpu-name", "Titan RTX", "--num-gpus", "2")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no offer", proc.stdout)

    def test_rent_dry_run_prints_the_request_without_creating(self):
        proc = self.run_vast("--dry-run", "--json", "rent",
                             "--offers-file", str(self.fixture), "--offer-id", "1",
                             "--label", "test-label", "--disk", "160",
                             "--price", "0.14")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = json.loads(proc.stdout)
        self.assertEqual(body["method"], "PUT")
        self.assertEqual(body["path"], "/asks/1/")
        self.assertEqual(body["body"]["label"], "test-label")
        self.assertEqual(body["body"]["disk"], 160)
        self.assertEqual(body["body"]["price"], 0.14)
        self.assertIn("cuda-13.0.1-auto", body["body"]["image"])

    def test_destroy_requires_yes_unless_dry_run(self):
        dry = self.run_vast("--dry-run", "destroy", "--instance", "7")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("DELETE", dry.stdout)
        refused = self.run_vast("destroy", "--instance", "7")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("--yes", refused.stderr)

    def test_stage_dry_run_uses_the_3090_setup_script(self):
        proc = self.run_vast("--dry-run", "stage", "--ssh", "ssh7.vast.ai:20748",
                             "--recipe-file",
                             str(BENCH / "recipes" / "vast-3090.json"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("setup3090.sh", proc.stdout)
        self.assertIn("rsync", proc.stdout)
        self.assertIn("PACKS_TO_FETCH", proc.stdout)


if __name__ == "__main__":
    unittest.main()
