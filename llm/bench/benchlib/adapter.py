"""Translate a validated recipe into the argv of the thin shell adapters.

The adapters (llm/bench/adapters/*.sh) own the final translation to the
existing backend scripts.  benchctl never re-implements a runner; it only
passes flags.  Both `plan` and `run` use the same function, so the printed
command and the executed command cannot drift.
"""
from __future__ import annotations

from . import config as config_mod

ADAPTERS = {
    "strata": "strata.sh",
    "vllm": "vllm.sh",
    "glm53": "glm53.sh",
    "vast": "cloud_vast.sh",
}


def adapter_path(backend):
    return config_mod.BENCH_DIR / "adapters" / ADAPTERS[backend]


def _add(argv, flag, value):
    if value is None or value == "":
        return
    argv.extend([flag, str(value)])


def _add_flag(argv, flag, enabled):
    if enabled:
        argv.append(flag)


def _cluster_resources(argv, recipe, cfg):
    hw = recipe.hardware
    _add(argv, "--partition", hw.get("partition"))
    _add(argv, "--nodelist", hw.get("nodelist"))
    _add(argv, "--gres", hw.get("gres"))
    _add(argv, "--cpus", hw.get("cpus"))
    _add(argv, "--mem", hw.get("mem"))
    _add(argv, "--time", hw.get("time"))
    if cfg.account:
        _add(argv, "--account", cfg.account)


def _strata_args(recipe, cfg, outdir):
    probe = recipe.probe
    opts = recipe.options
    hw = recipe.hardware
    argv = ["--runner", recipe.runner, "--model", recipe.model,
            "--outdir", str(outdir)]
    _add(argv, "--gpus", hw.get("gpus"))
    _add(argv, "--budget-min", recipe.budget_min)
    _add(argv, "--max-tokens", probe.get("max_tokens"))

    if recipe.runner in ("matrix", "warm"):
        _add(argv, "--cells", ",".join(recipe.raw.get("cells") or []))
    if recipe.runner == "matrix":
        _add(argv, "--requests-per-slot", probe.get("requests_per_slot"))
        if recipe.raw.get("warm_cells"):
            _add(argv, "--warm-cells", ",".join(recipe.raw["warm_cells"]))
            _add(argv, "--warm-repeats", probe.get("warm_repeats"))
    if recipe.runner == "knobs":
        _add(argv, "--node", hw.get("nodelist") or hw.get("node"))
        _add(argv, "--contexts", opts.get("contexts"))
        _add(argv, "--batches", opts.get("batches"))
        _add(argv, "--variants", opts.get("variants"))
        _add(argv, "--max-context", probe.get("max_context"))
    if recipe.runner == "warm":
        _add(argv, "--node", hw.get("nodelist") or hw.get("node"))
        _add(argv, "--warm-repeats", probe.get("warm_repeats"))
        _add(argv, "--max-context", probe.get("max_context"))
    if opts.get("self_test"):
        argv.append("--self-test")

    _cluster_resources(argv, recipe, cfg)
    return argv


def _vllm_args(recipe, cfg, outdir):
    opts = recipe.options
    flavour = opts.get("flavour")
    if not flavour:
        name = recipe.model.lower()
        flavour = "qwen38" if "qwen38" in name else (
            "qwen35-35b" if "35b" in name else "qwen35-122b")
    argv = ["--runner", recipe.runner, "--flavour", flavour,
            "--outdir", str(outdir)]
    _add(argv, "--model", recipe.model)
    _add(argv, "--base-url", opts.get("base_url"))
    _add(argv, "--served-model", opts.get("served_model"))
    _add(argv, "--contexts", opts.get("contexts"))
    _add(argv, "--concurrency", opts.get("concurrency"))
    _add(argv, "--max-model-len", opts.get("max_model_len"))
    _add(argv, "--partition", recipe.hardware.get("partition"))
    _add(argv, "--nodelist", recipe.hardware.get("nodelist"))
    _add(argv, "--gpus", recipe.hardware.get("gpus"))
    _add(argv, "--account", cfg.account)
    if opts.get("self_test"):
        argv.append("--self-test")
    return argv


def _glm53_args(recipe, cfg, outdir):
    opts = recipe.options
    argv = ["--runner", recipe.runner, "--outdir", str(outdir)]
    _add(argv, "--engine", opts.get("engine"))
    _add(argv, "--variant", opts.get("variant"))
    _add(argv, "--concurrency", opts.get("concurrency"))
    _add(argv, "--isl", opts.get("isl"))
    _add(argv, "--osl", opts.get("osl"))
    _add(argv, "--runs", opts.get("runs"))
    _add(argv, "--contexts", opts.get("contexts"))
    _add(argv, "--prefixes", opts.get("prefixes"))
    _add(argv, "--suffix", opts.get("suffix"))
    _add(argv, "--ctx", opts.get("ctx"))
    _add(argv, "--base-url", opts.get("base_url"))
    _add(argv, "--served-model", opts.get("served_model"))
    _add(argv, "--budget-min", recipe.budget_min)
    if recipe.runner == "submit":
        if recipe.get("mode") == "context":
            argv.append("--context-sweep")
        if recipe.get("mode") == "cache":
            argv.append("--cache-sweep")
    _add(argv, "--node", recipe.hardware.get("nodelist"))
    _add(argv, "--partition", recipe.hardware.get("partition"))
    _add(argv, "--gres", recipe.hardware.get("gres"))
    _add(argv, "--cpus", recipe.hardware.get("cpus"))
    _add(argv, "--mem", recipe.hardware.get("mem"))
    _add(argv, "--time", recipe.hardware.get("time"))
    _add(argv, "--account", cfg.account)
    return argv


def _vast_args(recipe, cfg, outdir):
    flow = recipe.options.get("flow") or "plan"
    argv = [flow, "--recipe-file", str(recipe.path)]
    if flow == "rent":
        _add(argv, "--outdir", str(outdir))
    return argv


def adapter_argv(recipe, cfg, outdir, dry_run=True, yes=False):
    """Full argv for the adapter. `--dry-run` is always first when dry_run."""
    path = str(adapter_path(recipe.backend))
    if recipe.backend == "strata":
        tail = _strata_args(recipe, cfg, outdir)
    elif recipe.backend == "vllm":
        tail = _vllm_args(recipe, cfg, outdir)
    elif recipe.backend == "glm53":
        tail = _glm53_args(recipe, cfg, outdir)
    else:
        tail = _vast_args(recipe, cfg, outdir)
    argv = [path]
    if dry_run:
        argv.append("--dry-run")
    if yes:
        argv.append("--yes")
    return argv + tail


def adapter_env(cfg):
    """Environment that keeps the backends pointed at this site's paths."""
    return {
        "WORK_ROOT": cfg.work_root,
        "STRATA_ROOT": cfg.path("strata_engine"),
        "MODELS_ROOT": cfg.path("strata_models"),
        "DATA_DIR": cfg.path("strata_data"),
        "RESULTS_ROOT": cfg.path("results"),
        "CXX_LIB_DIR": cfg.path("cxx_lib"),
        "BENCH_WORK_ROOT": cfg.work_root,
        "BENCH_CONFIG": str(cfg.config_path),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
