"""`benchctl prepare`: engine + model staging, delegated to existing scripts.

Steps are idempotent: presence checks run first, and an existing engine/model
is left alone unless --force.  Heavy steps (image build, compiled install) are
refused on the login node unless --allow-heavy, exactly like the backend docs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config as config_mod
from . import doctor as doctor_mod
from . import util

HEAVY_ON_LOGIN = {"vllm.pull", "glm53.install", "cloud.setup"}


def _step(sid, argv, heavy=False, note=None, env=None, cwd=None):
    return {"id": sid, "argv": argv, "heavy": heavy, "note": note,
            "env": env or {}, "cwd": cwd}


def _strata_steps(cfg, model, force, submit):
    steps = []
    engine = Path(cfg.path("strata_engine"))
    comps = list((engine / "components" / "engine").glob("*cuda-*"))
    if force or not (engine / "bin" / "strata").exists() or len(comps) < 2:
        steps.append(_step("strata.install-engine",
                           [str(config_mod.REPO_ROOT / "llm" / "strata" / "install-engine.sh")],
                           note="installs the Terminal and both cuda13/cuda12 components"))
    else:
        steps.append(_step("strata.install-engine", None,
                           note="already installed: %s" % ", ".join(p.name for p in comps)))
    packs = ["iq3s", "coder", "q2_0"] if model in ("all", "strata-packs") else [model]
    for pack in packs:
        if pack not in doctor_mod.MODEL_PACKS:
            steps.append(_step("strata.fetch.%s" % pack, None,
                               note="unknown pack %r (known: %s)"
                               % (pack, ", ".join(doctor_mod.MODEL_PACKS))))
            continue
        complete = _pack_complete(cfg, pack)
        if complete and not force:
            steps.append(_step("strata.fetch.%s" % pack, None,
                               note="pack %s already complete (size check)" % pack))
            continue
        argv = [str(config_mod.REPO_ROOT / "llm" / "strata" / "fetch-models.sh"),
                "--only", pack]
        if submit:
            argv.append("--submit")
        steps.append(_step("strata.fetch.%s" % pack, argv,
                           note="resumable hf download; both shards are required"))
    return steps


def _pack_complete(cfg, pack):
    root = Path(cfg.path("strata_models"))
    for sub, name, size, _sha in doctor_mod.MODEL_PACKS[pack]:
        if util.file_size(root / sub / name) != size:
            return False
    return True


def _vllm_steps(cfg, model, force):
    steps = []
    venv = Path(cfg.path("venvs")) / "vllm032py" / "bin" / "vllm"
    steps.append(_step("vllm.venv", None if venv.is_file() else ["echo", "missing vLLM 0.32 venv"],
                       note=str(venv)))
    if "qwen38" in model or model == "all":
        sif = Path(cfg.path("qwen38_sif"))
        if force or not sif.is_file():
            steps.append(_step("vllm.pull", 
                               [str(config_mod.REPO_ROOT / "llm" / "qwen38" / "pull-container.sh")],
                               heavy=True,
                               note="builds the 11 GB Blackwell SIF; run inside a Slurm job"))
        else:
            steps.append(_step("vllm.pull", None,
                               note="SIF present: %s" % sif))
        qmodel = Path(cfg.path("qwen38_model"))
        steps.append(_step("vllm.qwen38-model",
                           None if (qmodel / "config.json").is_file() else
                           ["echo", "missing checkpoint %s; see llm/qwen38/README.md" % qmodel],
                           note="local FP4 checkpoint"))
    if "qwen35" in model or model == "all":
        for label, key, repo in (
                ("qwen35-122b", "qwen35_122b_model", "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit"),
                ("qwen35-35b", "qwen35_35b_model", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")):
            target = Path(cfg.path(key))
            if (target / "config.json").is_file() and not force:
                steps.append(_step("vllm.%s-model" % label, None,
                                   note="checkpoint present: %s" % target))
            else:
                steps.append(_step("vllm.%s-model" % label,
                                   ["hf", "download", repo, "--local-dir", str(target)],
                                   note="HF download (network), ~23/75 GiB"))
    return steps


def _glm_steps(cfg, model, force, engine, variant):
    install = str(config_mod.REPO_ROOT / "llm" / "glm53-reap50" / "install.sh")
    argv = [install, "--engine", engine, "--variant", variant]
    if force:
        argv.append("--force")
    model_root = Path(cfg.path("glm_model"))
    present = model_root.is_dir() and any(model_root.iterdir())
    if present and not force:
        return [_step("glm53.install", None,
                      note="model root already populated: %s" % model_root)]
    return [_step("glm53.install", argv, heavy=True,
                  note="pulls the image/builds llama.cpp and downloads the checkpoint; "
                       "run inside a Slurm job, not the login node")]


def build(cfg, backend, model, force=False, submit=False, engine=None, variant=None,
          allow_heavy=False):
    if backend == "strata":
        steps = _strata_steps(cfg, model, force, submit)
    elif backend == "vllm":
        steps = _vllm_steps(cfg, model, force)
    elif backend == "glm53":
        known = ("nvfp4", "fp8", "q4_k_s", "q4_k_m", "iq4_xs", "iq3_m")
        v = variant or (model if model in known else "iq4_xs")
        e = engine or ("llamacpp" if v.startswith(("iq", "q4")) else "sglang")
        steps = _glm_steps(cfg, model, force, e, v)
    elif backend == "cloud":
        vast_py = str(config_mod.REPO_ROOT / "llm" / "cloud" / "vast" / "vast.py")
        steps = [_step("cloud.stage", [sys.executable, vast_py, "stage",
                                       "--recipe-file", model],
                       heavy=True,
                       note="upload llm/cloud/vast/{setup,install-engine,strata_probe}.sh "
                            "and run setup on the rented instance")]
    else:
        raise ValueError("unknown backend %r (strata|vllm|glm53|cloud)" % backend)
    login = config_mod.is_login_node(cfg)
    for step in steps:
        step["blocked"] = bool(step["heavy"] and login and not allow_heavy
                               and step["argv"] is not None)
    return steps


def render_text(steps, dry_run):
    lines = []
    for step in steps:
        if step["argv"] is None:
            lines.append("[skip] %-28s %s" % (step["id"], step["note"] or ""))
            continue
        if step.get("blocked"):
            lines.append("[hold] %-28s %s" % (step["id"], step["note"] or ""))
            lines.append("       command: %s" % util.shell_join(step["argv"]))
            lines.append("       fix: pass --allow-heavy only if you really mean to run this here")
            continue
        prefix = "[dry] " if dry_run else "[run] "
        lines.append(prefix + ("%-28s %s" % (step["id"], util.shell_join(step["argv"]))))
        if step["note"]:
            lines.append("       %s" % step["note"])
    return "\n".join(lines)
