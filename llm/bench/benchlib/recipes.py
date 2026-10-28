"""Recipe loading, JSON-schema validation and semantic checks.

Every recipe is validated against llm/bench/schema.json before use.  The tiny
validator below implements the subset of JSON Schema the file uses, so the CLI
stays stdlib-only (no `jsonschema` on the cluster's system python).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config as config_mod
from . import util

RECIPE_DIR = config_mod.BENCH_DIR / "recipes"
SCHEMA_PATH = config_mod.BENCH_DIR / "schema.json"

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class RecipeError(ValueError):
    pass


def _type_ok(value, expected):
    if isinstance(expected, list):
        return any(_type_ok(value, t) for t in expected)
    fn = _TYPE_CHECKS.get(expected)
    return bool(fn and fn(value))


def validate(instance, schema, path="$"):
    """Return a list of 'path: message' strings. Empty list means valid."""
    errors = []
    if not isinstance(schema, dict):
        return ["%s: schema node is not an object" % path]

    if "const" in schema and instance != schema["const"]:
        errors.append("%s: expected const %r, got %r" % (path, schema["const"], instance))
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: %r is not one of %s" % (path, instance, schema["enum"]))
    if "type" in schema and not _type_ok(instance, schema["type"]):
        errors.append("%s: expected type %s, got %s"
                      % (path, schema["type"], type(instance).__name__))
        return errors

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required property %r" % (path, key))
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child = "%s.%s" % (path, key)
            if key in props:
                errors.extend(validate(value, props[key], child))
            elif extra is False:
                errors.append("%s: unknown property (typo?); allowed: %s"
                              % (child, ", ".join(sorted(props))))
            elif isinstance(extra, dict):
                errors.extend(validate(value, extra, child))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("%s: needs at least %d item(s)" % (path, schema["minItems"]))
        if schema.get("uniqueItems") and len(instance) != len(set(map(str, instance))):
            errors.append("%s: items must be unique" % path)
        item_schema = schema.get("items")
        if item_schema:
            for i, value in enumerate(instance):
                errors.extend(validate(value, item_schema, "%s[%d]" % (path, i)))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("%s: shorter than %d characters" % (path, schema["minLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append("%s: %r does not match %s" % (path, instance, schema["pattern"]))

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: %r < minimum %r" % (path, instance, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: %r > maximum %r" % (path, instance, schema["maximum"]))
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append("%s: %r <= exclusiveMinimum %r"
                          % (path, instance, schema["exclusiveMinimum"]))

    return errors


def load_schema(path=None):
    with open(str(path or SCHEMA_PATH), "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_cells(cells):
    out = []
    for item in cells or []:
        try:
            ctx, conc = item.split(":", 1)
            out.append((int(ctx), int(conc)))
        except (ValueError, AttributeError):
            raise RecipeError("bad cell %r (want 'context:concurrency', e.g. '4096:4')" % item)
    return out


def _semantic_check(raw, errors):
    backend = raw.get("backend")
    runner = raw.get("runner")
    hw = raw.get("hardware") or {}
    options = raw.get("options") or {}
    probe = raw.get("probe") or {}

    if backend == "strata":
        if runner not in ("matrix", "knobs", "warm", "chain"):
            errors.append("$.runner: strata supports matrix|knobs|warm|chain, got %r" % runner)
        if runner in ("matrix", "warm") and not raw.get("cells"):
            errors.append("$.cells: required for a strata %s recipe" % runner)
        if runner == "knobs" and not options.get("contexts"):
            errors.append("$.options.contexts: required for a strata knobs recipe")
        if runner in ("knobs", "warm", "chain") and not probe.get("max_context"):
            errors.append("$.probe.max_context: required for a strata %s recipe" % runner)
        if runner == "chain":
            if not probe.get("stops"):
                errors.append("$.probe.stops: required for a strata chain recipe")
            if not probe.get("tokens_per_turn"):
                errors.append("$.probe.tokens_per_turn: required for a strata chain recipe")
            stops = []
            for item in str(probe.get("stops") or "").split(","):
                try:
                    stops.append(int(item))
                except ValueError:
                    errors.append("$.probe.stops: %r is not an integer" % item)
            if stops != sorted(stops) or len(stops) != len(set(stops)):
                errors.append("$.probe.stops: must be strictly increasing")
            else:
                max_ctx = probe.get("max_context")
                if max_ctx and stops and stops[-1] > max_ctx:
                    errors.append("$.probe.stops: last stop %d exceeds "
                                  "probe.max_context=%d" % (stops[-1], max_ctx))
            if int(probe.get("tokens_per_turn") or 0) >= int(probe.get("max_context") or 0):
                errors.append("$.probe.tokens_per_turn: must be smaller than max_context")
        if not hw.get("gres"):
            errors.append("$.hardware.gres: required for strata (e.g. 'gpu:1')")
        if hw.get("gpus") and hw.get("gres"):
            try:
                cards = int(str(hw["gres"]).split(":", 1)[1])
                used = [int(x) for x in str(hw["gpus"]).split(",")]
            except (ValueError, IndexError):
                cards, used = 1, [0]
            if max(used) >= cards:
                errors.append("$.hardware.gpus: %r indexes a card beyond %s"
                              % (hw["gpus"], hw["gres"]))

    elif backend == "vllm":
        if runner not in ("serve", "measure", "tasks"):
            errors.append("$.runner: vllm supports serve|measure|tasks, got %r" % runner)
        if not (options.get("flavour") or raw.get("model")):
            errors.append("$.options.flavour: needed to pick the qwen38|qwen35 adapter path")

    elif backend == "glm53":
        if runner not in ("submit", "bench", "context", "cache"):
            errors.append("$.runner: glm53 supports submit|bench|context|cache, got %r" % runner)
        if runner == "submit" and options.get("engine") in (None, ""):
            errors.append("$.options.engine: required for a glm53 submit recipe")

    elif backend == "vast":
        if runner != "cloud":
            errors.append("$.runner: vast recipes use runner 'cloud', got %r" % runner)
        if hw.get("kind") != "cloud":
            errors.append("$.hardware.kind: vast recipes must be 'cloud'")
        if not hw.get("offer"):
            errors.append("$.hardware.offer: required for a vast recipe")

    if hw.get("kind") == "cluster" and not hw.get("partition"):
        errors.append("$.hardware.partition: required for a cluster recipe")
    if hw.get("kind") == "cloud" and backend != "vast":
        errors.append("$.hardware.kind: only the vast backend may be 'cloud'")

    for cell in (raw.get("cells") or []) + (raw.get("warm_cells") or []):
        try:
            ctx, conc = (int(x) for x in str(cell).split(":", 1))
        except ValueError:
            continue
        if conc < 1:
            errors.append("$.cells: concurrency must be >= 1 in %r" % cell)
        max_ctx = probe.get("max_context")
        if max_ctx and ctx > max_ctx:
            errors.append("$.cells: %r exceeds probe.max_context=%d" % (cell, max_ctx))


class Recipe:
    def __init__(self, raw, path):
        self.raw = raw
        self.path = Path(path)

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def name(self):
        return self.raw["name"]

    @property
    def backend(self):
        return self.raw["backend"]

    @property
    def runner(self):
        return self.raw["runner"]

    @property
    def model(self):
        return self.raw["model"]

    @property
    def cells(self):
        return parse_cells(self.raw.get("cells"))

    @property
    def warm_cells(self):
        return parse_cells(self.raw.get("warm_cells"))

    @property
    def probe(self):
        return dict(self.raw.get("probe") or {})

    @property
    def options(self):
        return dict(self.raw.get("options") or {})

    @property
    def hardware(self):
        return dict(self.raw.get("hardware") or {})

    @property
    def budget_min(self):
        return self.raw.get("budget_min")

    @property
    def caveats(self):
        return list(self.raw.get("caveats") or [])

    def window(self):
        """Server context window: probe.max_context, else the largest cell."""
        if self.probe.get("max_context"):
            return int(self.probe["max_context"])
        cells = self.cells + self.warm_cells
        return max([c for c, _ in cells], default=None)

    def outdir(self, cfg, stamp=None):
        out = self.raw.get("output") or {}
        root = cfg.output_root(out["root"])
        subdir = str(out["subdir"]).format(
            name=self.name,
            stamp=stamp or util.utc_stamp(),
            model=self.model,
            runner=self.runner,
        )
        return root / subdir

    def hardware_summary(self):
        hw = self.hardware
        if hw.get("kind") == "cloud":
            offer = hw.get("offer") or {}
            return "%dx %s (Vast.ai), %s h, model %.1f GB" % (
                int(offer.get("num_gpus", 1)), offer.get("gpu_name", "?"),
                offer.get("hours", "?"), float(offer.get("model_gb", 0)))
        bits = []
        if hw.get("gpu_type"):
            bits.append(str(hw["gpu_type"]))
        bits.append(str(hw.get("gres", "")))
        where = hw.get("nodelist") or hw.get("node")
        if where:
            bits.append("@%s" % where)
        if hw.get("partition"):
            bits.append(hw["partition"])
        if hw.get("cpus"):
            bits.append("%s cpus" % hw["cpus"])
        if hw.get("mem"):
            bits.append(str(hw["mem"]))
        if hw.get("time"):
            bits.append(str(hw["time"]))
        return " ".join(b for b in bits if b)

    def evidence_paths(self):
        return list(self.raw.get("evidence") or [])


def load_recipe(path_or_name):
    """Load by file path, by name in recipes/, or by 'name' without .json."""
    ref = str(path_or_name)
    candidates = []
    p = Path(ref).expanduser()
    if p.is_file():
        candidates.append(p)
    else:
        if not ref.endswith(".json"):
            ref_json = ref + ".json"
        else:
            ref_json = ref
        candidates.append(RECIPE_DIR / ref_json)
        candidates.append(RECIPE_DIR / Path(ref_json).name)
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        known = ", ".join(sorted(x.stem for x in RECIPE_DIR.glob("*.json")))
        raise RecipeError("recipe not found: %s (known: %s)" % (path_or_name, known))
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except ValueError as exc:
        raise RecipeError("%s: invalid JSON: %s" % (path, exc))
    errors = validate(raw, load_schema(), path="$")
    _semantic_check(raw, errors)
    if errors:
        raise RecipeError("%s:\n  %s" % (path, "\n  ".join(errors)))
    return Recipe(raw, path)


def list_recipes():
    out = []
    for path in sorted(RECIPE_DIR.glob("*.json")):
        try:
            out.append(load_recipe(path))
        except RecipeError as exc:
            out.append(exc)
    return out


def recipe_names():
    return sorted(p.stem for p in RECIPE_DIR.glob("*.json"))
