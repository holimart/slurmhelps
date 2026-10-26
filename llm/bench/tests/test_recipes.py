"""Recipe schema and semantic validation (no network, no Slurm, no GPU)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "llm" / "bench"
sys.path.insert(0, str(BENCH))

from benchlib import config as config_mod  # noqa: E402
from benchlib import plan as plan_mod  # noqa: E402
from benchlib import recipes as recipes_mod  # noqa: E402

EXPECTED = {
    "strata-l40x1", "strata-l40x2", "strata-v100x1",
    "strata-v100-resident-2x", "strata-v100-resident-4x",
    "strata-knobs", "strata-warm", "strata-sustained-512",
    "strata-chain-l40", "strata-chain-v100-4x",
    "qwen38-bw-fp4", "qwen35-ampere02",
    "glm53-cache-sweep", "glm53-context-sweep",
    "vast-titan", "vast-3090",
}


class RecipeValidationTest(unittest.TestCase):
    def setUp(self):
        self.schema = recipes_mod.load_schema()

    def errors(self, raw):
        errors = []
        errors.extend(recipes_mod.validate(raw, self.schema, path="$"))
        recipes_mod._semantic_check(raw, errors)
        return errors

    def test_all_expected_recipes_exist_and_validate(self):
        names = set(recipes_mod.recipe_names())
        self.assertEqual(names, EXPECTED)
        for name in sorted(names):
            recipe = recipes_mod.load_recipe(name)
            self.assertEqual(recipe.name, name)
            self.assertTrue(recipe.caveats, name)
            self.assertIn(recipe.backend, ("strata", "vllm", "glm53", "vast"))
            if recipe.backend == "strata" and recipe.runner in ("matrix", "warm"):
                self.assertTrue(recipe.cells, name)

    def test_unknown_property_is_a_typo_error(self):
        raw = json.loads((BENCH / "recipes" / "strata-l40x1.json").read_text())
        raw["hardware"]["nodelst"] = "uk-g6-06"
        errors = self.errors(raw)
        self.assertTrue(any("unknown property" in e and "nodelst" in e for e in errors),
                        errors)

    def test_bad_cell_spelling_fails(self):
        raw = json.loads((BENCH / "recipes" / "strata-l40x1.json").read_text())
        raw["cells"] = ["1024 x 1"]
        errors = self.errors(raw)
        self.assertTrue(any("$.cells" in e for e in errors), errors)

    def test_unknown_backend_fails(self):
        raw = json.loads((BENCH / "recipes" / "strata-l40x1.json").read_text())
        raw["backend"] = "tgi"
        errors = self.errors(raw)
        self.assertTrue(any("backend" in e for e in errors), errors)

    def test_gpu_index_beyond_gres_fails(self):
        raw = json.loads((BENCH / "recipes" / "strata-l40x1.json").read_text())
        raw["hardware"]["gpus"] = "0,1"
        errors = self.errors(raw)
        self.assertTrue(any("beyond" in e for e in errors), errors)

    def test_cell_above_window_fails(self):
        raw = json.loads((BENCH / "recipes" / "strata-v100x1.json").read_text())
        raw["cells"] = raw["cells"] + ["131072:1"]
        errors = self.errors(raw)
        self.assertTrue(any("exceeds probe.max_context" in e for e in errors), errors)

    def test_chain_recipe_requires_increasing_stops(self):
        raw = json.loads((BENCH / "recipes" / "strata-chain-l40.json").read_text())
        del raw["probe"]["stops"]
        errors = self.errors(raw)
        self.assertTrue(any("probe.stops" in e for e in errors), errors)
        raw = json.loads((BENCH / "recipes" / "strata-chain-l40.json").read_text())
        raw["probe"]["stops"] = "100000,1024"
        errors = self.errors(raw)
        self.assertTrue(any("increasing" in e for e in errors), errors)

    def test_cloud_recipe_requires_offer(self):
        raw = json.loads((BENCH / "recipes" / "vast-titan.json").read_text())
        del raw["hardware"]["offer"]
        errors = self.errors(raw)
        self.assertTrue(any("hardware.offer" in e for e in errors), errors)

    def test_outdir_uses_name_and_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_mod.Config(path=BENCH / "config.json", work_root=tmp)
            recipe = recipes_mod.load_recipe("strata-l40x1")
            out = recipe.outdir(cfg, stamp="20260101T000000Z")
            self.assertEqual(out, Path(tmp) / "strata-runs" /
                             "strata-l40x1-20260101T000000Z")


class PlanTest(unittest.TestCase):
    def test_every_recipe_plans_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_mod.Config(path=BENCH / "config.json", work_root=tmp)
            markers = sorted(p.name for p in Path(tmp).rglob("*"))
            for name in sorted(EXPECTED):
                recipe = recipes_mod.load_recipe(name)
                plan = plan_mod.build(recipe, cfg, stamp="20260101T000000Z",
                                      dry_run=True)
                self.assertEqual(plan["name"], name)
                self.assertEqual(plan["adapter_check_rc"], 0,
                                 "%s adapter failed: %s" % (name, plan["submit_line"]))
                self.assertTrue(plan["submit_line"])
                self.assertIn(str(cfg.work_root), plan["outdir"])
                if recipe.backend == "strata":
                    self.assertIn("sbatch --nice=100", plan["submit_line"], name)
                if recipe.backend == "vast":
                    self.assertIn("vast.py", plan["submit_line"])
            self.assertEqual(sorted(p.name for p in Path(tmp).rglob("*")), markers,
                             "plan must not create files")

    def test_plan_json_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, BENCH_WORK_ROOT=tmp)
            proc = subprocess.run(
                [str(BENCH / "benchctl"), "plan", "--recipe", "strata-warm",
                 "--stamp", "X", "--json"],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertEqual(data["name"], "strata-warm")
            self.assertEqual(data["cells"], ["1024:1", "4096:1", "16384:1",
                                             "65536:1", "1024:4", "4096:4"])
            self.assertEqual(data["window"], 131072)
            self.assertEqual(data["cell_count"], 6)
            self.assertIn("command", data)
            self.assertIn("estimate", data)


if __name__ == "__main__":
    unittest.main()
