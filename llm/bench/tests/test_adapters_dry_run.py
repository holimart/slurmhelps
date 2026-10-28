"""Every adapter has a no-submit --dry-run; a real strata submit is intercepted.

The fake sbatch records its argv and prints a job id, so the submit path is
exercised end-to-end without touching Slurm.
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
ADAPTERS = BENCH / "adapters"


def _fake_bin(tmp: Path):
    bindir = tmp / "fakebin"
    bindir.mkdir()
    log = tmp / "sbatch.log"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> ' + str(log) + "\n"
        "echo 424242\n")
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    return bindir, log


def _env(tmp: Path):
    env = dict(os.environ)
    env["BENCH_WORK_ROOT"] = str(tmp / "work")
    env["PATH"] = str(tmp / "fakebin") + ":" + env.get("PATH", "")
    env.pop("VAST_API_KEY", None)
    return env


class AdapterDryRunTest(unittest.TestCase):
    def _run(self, argv, tmp):
        return subprocess.run(argv, env=_env(tmp), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_strata_dry_run_prints_sbatch_without_running_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bindir, log = _fake_bin(tmp)
            proc = self._run([
                str(ADAPTERS / "strata.sh"), "--dry-run", "--runner", "matrix",
                "--model", "iq3s", "--cells", "1024:1,4096:4", "--gpus", "0",
                "--outdir", str(tmp / "out"), "--budget-min", "60",
                "--partition", "gpu-ffa", "--gres", "gpu:1"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("sbatch --nice=100", proc.stdout)
            self.assertIn("strata-bench.sbatch", proc.stdout)
            self.assertFalse(log.exists(), "dry-run executed sbatch")
            self.assertFalse((tmp / "out").exists(), "dry-run created the outdir")

    def test_adapters_cover_strata_vllm_glm_cloud(self):
        cases = [
            (["strata.sh", "--dry-run", "--runner", "knobs", "--model", "iq3s",
              "--node", "uk-g6-07", "--contexts", "1024,4096",
              "--batches", "1,4", "--variants", "all",
              "--max-context", "131072", "--outdir", "/tmp/x"],
             "knob-sweep.sbatch"),
            (["strata.sh", "--dry-run", "--runner", "warm", "--model", "iq3s",
              "--cells", "1024:1", "--warm-repeats", "3",
              "--max-context", "131072", "--outdir", "/tmp/x"],
             "warm-cache.sbatch"),
            (["strata.sh", "--dry-run", "--runner", "chain", "--model", "iq3s",
              "--stops", "1024,100000,200000,262144",
              "--tokens-per-turn", "512", "--chain-last-repeats", "1",
              "--max-context", "262144", "--outdir", "/tmp/x"],
             "chain-cache.sbatch"),
            (["vllm.sh", "--dry-run", "--runner", "measure", "--flavour", "qwen38",
              "--outdir", "/tmp/x"], "measure.sh"),
            (["vllm.sh", "--dry-run", "--runner", "measure",
              "--flavour", "qwen35-122b", "--outdir", "/tmp/x"],
             "serve_smoke.py"),
            (["glm53.sh", "--dry-run", "--runner", "context",
              "--engine", "llamacpp", "--variant", "iq4_xs",
              "--contexts", "4096:8 16384:4", "--outdir", "/tmp/x"],
             "context-sweep.sh"),
            (["glm53.sh", "--dry-run", "--runner", "cache",
              "--engine", "llamacpp", "--variant", "iq4_xs",
              "--prefixes", "100000 200000", "--outdir", "/tmp/x"],
             "cache-sweep.sh"),
            (["cloud_vast.sh", "--dry-run", "plan",
              "--recipe-file", str(BENCH / "recipes" / "vast-titan.json")],
             "vast.py --dry-run plan"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            for argv, needle in cases:
                proc = self._run([str(ADAPTERS / argv[0])] + argv[1:], tmp)
                self.assertEqual(proc.returncode, 0,
                                 "%s: %s" % (argv[0], proc.stderr))
                self.assertIn(needle, proc.stdout, argv)
            self.assertFalse((tmp / "sbatch.log").exists())

    def test_benchctl_run_dry_run_never_submits_for_every_recipe(self):
        recipes = sorted(p.stem for p in (BENCH / "recipes").glob("*.json"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            for name in recipes:
                proc = self._run([
                    str(BENCH / "benchctl"), "run", "--recipe", name,
                    "--dry-run", "--stamp", "TEST"], tmp)
                self.assertEqual(proc.returncode, 0,
                                 "%s: %s\n%s" % (name, proc.stderr, proc.stdout))
                self.assertIn("submit", proc.stdout)
            self.assertFalse((tmp / "sbatch.log").exists(),
                             "a dry-run submitted a job")

    def test_real_submit_path_is_exercised_with_fake_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bindir, log = _fake_bin(tmp)
            proc = self._run([
                str(BENCH / "benchctl"), "run", "--recipe", "strata-l40x1",
                "--stamp", "TEST"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(log.exists(), "the fake sbatch was never called")
            args = log.read_text()
            self.assertIn("--nice=100", args)
            self.assertIn("strata-bench.sbatch", args)
            self.assertIn("--model iq3s", args)
            self.assertIn("--cells", args)
            self.assertIn("--partition gpu-ffa", args)
            outdir = tmp / "work" / "strata-runs" / "strata-l40x1-TEST"
            self.assertTrue(outdir.is_dir(), "run must create the output root")

    def test_vast_run_without_yes_only_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            proc = self._run([
                str(BENCH / "benchctl"), "run", "--recipe", "vast-titan",
                "--stamp", "TEST"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("not renting", proc.stdout)
            self.assertFalse((tmp / "sbatch.log").exists())

    def test_glm_client_run_is_blocked_on_the_login_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            proc = self._run([
                str(BENCH / "benchctl"), "run", "--recipe",
                "glm53-context-sweep", "--stamp", "TEST"], tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("login node", proc.stderr)
            self.assertFalse((tmp / "sbatch.log").exists())

    def test_prepare_cloud_dry_run_stages_the_recipe_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            proc = self._run([
                str(BENCH / "benchctl"), "prepare", "--backend", "cloud",
                "--model", "iq3s", "--recipe", "vast-titan",
                "--instance", "123", "--dry-run", "--json"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            step = data["steps"][0]
            self.assertEqual(step["id"], "cloud.stage")
            self.assertEqual(step["status"], "blocked")
            self.assertIn("vast.py stage", step["command"])
            self.assertIn("vast-titan.json", step["command"])
            self.assertIn("--instance 123", step["command"])

    def test_strata_prepare_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _fake_bin(tmp)
            proc = self._run([
                str(BENCH / "benchctl"), "prepare", "--backend", "strata",
                "--model", "iq3s", "--dry-run", "--json"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            ids = [step["id"] for step in data["steps"]]
            self.assertIn("strata.install-engine", ids)
            self.assertIn("strata.fetch.iq3s", ids)
            self.assertFalse((tmp / "sbatch.log").exists())

    def test_strata_matrix_self_test_through_the_adapter(self):
        """Orchestrator -> adapter -> unchanged bench-matrix.sh --self-test."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = _env(tmp)
            env["SELFTEST_DIR"] = str(tmp / "selftest")
            env["OUTDIR"] = str(tmp / "selftest-out")
            proc = subprocess.run([
                str(ADAPTERS / "strata.sh"), "--runner", "matrix",
                "--model", "iq3s", "--self-test"],
                env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stdout[-4000:])
            self.assertIn("SELF-TEST PASS", proc.stdout)
            self.assertIn("JSONL contract", proc.stdout)
            found = list(tmp.rglob("cells.jsonl"))
            self.assertTrue(found, "self-test wrote no cells.jsonl")
            cells = found[0]
            rows = [json.loads(line) for line in cells.read_text().splitlines() if line]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["error"] is None for row in rows))

    def test_strata_chain_self_test_through_the_adapter(self):
        """Orchestrator -> adapter -> chain-cache.sh --self-test (stub only)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = _env(tmp)
            env["SELFTEST_DIR"] = str(tmp / "chain-selftest")
            proc = subprocess.run([
                str(ADAPTERS / "strata.sh"), "--runner", "chain",
                "--model", "iq3s", "--self-test"],
                env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stdout[-4000:])
            self.assertIn("SELF-TEST PASS", proc.stdout)
            self.assertIn("prefix growth", proc.stdout)
            found = list(tmp.rglob("cells.jsonl"))
            self.assertTrue(found, "chain self-test wrote no cells.jsonl")
            rows = [json.loads(line) for line in found[0].read_text().splitlines() if line]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["mode"] == "chain" for row in rows))
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertGreater(rows[-1]["reused_tokens"], rows[0]["reused_tokens"])


if __name__ == "__main__":
    unittest.main()
