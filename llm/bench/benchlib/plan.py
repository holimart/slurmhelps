"""Side-effect-free plan rendering (text and JSON)."""
from __future__ import annotations

import os
from collections import OrderedDict

from . import adapter
from . import config as config_mod
from . import util

PACK_INFO = {
    "iq3s": "Qwen3.8-Flash-Next IQ3_S (54.8 GB experts + 28.8 GB shared PLE)",
    "coder": "Qwen3.8-Flash-Next Coder IQ1_M (29.6 GB experts + 28.8 GB shared PLE)",
    "q2_0": "Qwen3.8-Flash-Next Q2_0 (37.6 GB experts + 28.8 GB shared PLE)",
    "qwen38-flash-next-fp4-local": "Qwen3.8-Flash-Next FP4 local (Blackwell, vLLM container)",
    "qwen35-122b-a10b": "Qwen3.5-122B-A10B AWQ-4bit (2x A100)",
    "qwen35-35b-a3b": "Qwen3.5-35B-A3B AWQ-4bit (1x A100)",
    "glm53-reap50": "GLM-5.3-Flash REAP50 (SGLang/llama.cpp)",
    "strata-packs": "Strata GGUF packs on Vast.ai",
}


def _cells_by_context(cells):
    grouped = OrderedDict()
    for ctx, conc in cells:
        grouped.setdefault(ctx, []).append(conc)
    return grouped


def estimate(recipe, cfg, outdir):
    declared = recipe.raw.get("estimate") or {}
    hw = recipe.hardware
    est = {
        "wall_minutes": declared.get("wall_minutes"),
        "wall": util.human_minutes(declared.get("wall_minutes")),
        "basis": declared.get("basis"),
        "cost_usd": declared.get("cost_usd"),
        "cost_note": "Slurm allocation; no direct charge (see bin/hpc_minutes.py)",
    }
    if recipe.backend == "vast":
        offer = hw.get("offer") or {}
        down_tb = float(offer.get("max_down_cost_per_tb", cfg.vast.get("max_down_cost_per_tb", 5.0)))
        model_gb = float(offer.get("model_gb", 0))
        hours = float(offer.get("hours", cfg.vast.get("default_hours", 2.0)))
        download_cost = round(model_gb / 1000.0 * down_tb, 4)
        est["download_cost_usd"] = download_cost
        est["download_basis"] = "%.1f GB x $%.2f/TB (upper bound at the offer filter)" % (model_gb, down_tb)
        est["hours"] = hours
        est["cost_note"] = ("GPU cost = dph_total x %.2f h from the live offer; "
                            "download ceiling shown separately" % hours)
        if est.get("cost_usd") is None:
            est["cost_usd"] = None
    est["outdir_exists"] = outdir.exists()
    return est


def build(recipe, cfg, stamp=None, dry_run=True):
    outdir = recipe.outdir(cfg, stamp=stamp)
    cells = recipe.cells
    warm = recipe.warm_cells
    argv = adapter.adapter_argv(recipe, cfg, outdir, dry_run=dry_run)
    grouped = _cells_by_context(cells)
    # The probe always uses the adapter's --dry-run, even when the returned
    # command is the real one: planning must never have side effects.
    probe_argv = adapter.adapter_argv(recipe, cfg, outdir, dry_run=True)
    adapter_probe = util.run(probe_argv, env=adapter.adapter_env(cfg), timeout=30)
    submit_line = adapter_probe["out"].strip() or util.shell_join(argv)
    plan = {
        "name": recipe.name,
        "description": recipe.get("description"),
        "recipe_file": str(recipe.path),
        "experiment": recipe.get("experiment"),
        "backend": recipe.backend,
        "runner": recipe.runner,
        "model": recipe.model,
        "model_info": PACK_INFO.get(recipe.model, recipe.model),
        "mode": recipe.get("mode"),
        "hardware": recipe.hardware,
        "hardware_summary": recipe.hardware_summary(),
        "cells": recipe.raw.get("cells") or [],
        "cell_count": len(cells),
        "cells_by_context": [{"context": c, "concurrency": sorted(v)}
                             for c, v in grouped.items()],
        "warm_cells": recipe.raw.get("warm_cells") or [],
        "probe": recipe.probe,
        "window": recipe.window(),
        "options": recipe.options,
        "budget_min": recipe.budget_min,
        "outdir": str(outdir),
        "output_root": recipe["output"]["root"],
        "adapter": str(adapter.adapter_path(recipe.backend)),
        "command": argv,
        "command_line": util.shell_join(argv),
        "submit_line": submit_line,
        "adapter_check_rc": adapter_probe["rc"],
        "env": {k: v for k, v in adapter.adapter_env(cfg).items() if k != "PATH"},
        "estimate": estimate(recipe, cfg, outdir),
        "caveats": recipe.caveats,
        "evidence": recipe.evidence_paths(),
        "warnings": [],
    }
    if outdir.exists():
        plan["warnings"].append("output directory already exists: %s" % outdir)
    if recipe.backend != "vast" and not config_mod.is_login_node(cfg):
        plan["warnings"].append("not on the configured login node; check where this is running")
    if recipe.backend == "vast":
        plan["warnings"].append("renting starts billing; `run` without --yes only plans")
    if not plan["warnings"]:
        plan["warnings"] = []
    return plan


def _fmt_cells(plan):
    parts = []
    for row in plan["cells_by_context"]:
        parts.append("%d x C%s" % (row["context"], ",".join(str(c) for c in row["concurrency"])))
    return "; ".join(parts) if parts else "(none)"


def render_text(plan):
    est = plan["estimate"]
    lines = []
    lines.append("recipe      %s  (%s/%s)" % (plan["name"], plan["backend"], plan["runner"]))
    if plan.get("experiment"):
        lines.append("experiment  %s" % plan["experiment"])
    lines.append("model       %s — %s" % (plan["model"], plan["model_info"]))
    lines.append("hardware    %s" % plan["hardware_summary"])
    lines.append("mode        %s" % plan["mode"])
    lines.append("cells       %d: %s" % (plan["cell_count"], _fmt_cells(plan)))
    if plan["warm_cells"]:
        lines.append("warm cells  %s" % ", ".join(plan["warm_cells"]))
    probe = plan["probe"]
    probe_bits = []
    for key in ("max_context", "max_tokens", "requests_per_slot", "pad_safety",
                "warm_repeats", "temperature", "timeout_s"):
        if key in probe:
            probe_bits.append("%s=%s" % (key, probe[key]))
    lines.append("probe       %s" % (" ".join(probe_bits) or "backend defaults"))
    lines.append("window      %s" % (plan["window"] or "derived by the runner"))
    lines.append("outdir      %s" % plan["outdir"])
    if plan.get("budget_min"):
        lines.append("budget      %d min" % plan["budget_min"])
    lines.append("estimate    wall %s%s" % (
        est["wall"], (" (%s)" % est["basis"]) if est.get("basis") else ""))
    if plan["backend"] == "vast":
        lines.append("cost        GPU = dph_total x %s h (live offer needed); download <= $%.2f (%s)"
                     % (est.get("hours"), est.get("download_cost_usd", 0.0),
                        est.get("download_basis", "")))
    else:
        lines.append("cost        %s" % est["cost_note"])
    lines.append("submit      %s" % plan["submit_line"])
    lines.append("adapter     %s" % plan["command_line"])
    envbits = ["%s=%s" % (k, v) for k, v in sorted(plan["env"].items())
               if k in ("STRATA_ROOT", "MODELS_ROOT", "DATA_DIR", "RESULTS_ROOT", "WORK_ROOT")]
    if envbits:
        lines.append("env         %s" % " ".join(envbits))
    for warning in plan["warnings"]:
        lines.append("WARN        %s" % warning)
    lines.append("caveats")
    for caveat in plan["caveats"]:
        lines.append("  - %s" % caveat)
    if plan["evidence"]:
        lines.append("evidence    %s" % ", ".join(plan["evidence"]))
    return "\n".join(lines)


def validate_output_root(recipe, cfg):
    root = recipe["output"]["root"]
    if root not in cfg.paths:
        raise ValueError("output.root %r is not a configured path key" % root)
    return cfg.output_root(root)
