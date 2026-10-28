"""Site configuration for benchctl.

All cluster-specific values live in llm/bench/config.json (or a file named by
BENCH_CONFIG).  Code only knows logical path names.  BENCH_WORK_ROOT replaces
the work_root prefix, which is what tests and another site need.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import util

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_DIR.parents[1]

DEFAULTS = {
    "schema_version": 1,
    "work_root": str(REPO_ROOT.parent),
    "paths": {
        "results": "{work_root}/results",
        "strata_runs": "{work_root}/strata-runs",
        "logs": "{work_root}/logs",
        "models": "{work_root}/models",
        "strata_models": "{work_root}/models/strata",
        "strata_engine": "{work_root}/strata-engine",
        "strata_data": "{work_root}/models/strata/_strata-data",
        "cxx_lib": "{work_root}/strata-runs/2026-10-25-l40x1/deps/cxx/lib",
        "images": "{work_root}/images",
        "glm_model": "{work_root}/models/glm53-reap50",
        "glm_runs": "{work_root}/runs/glm53-reap50",
        "venvs": "{work_root}/venvs",
        "qwen38_model": "{work_root}/models/qwen38-flash-next-fp4-local",
        "qwen38_sif": "{work_root}/images/qwen38-vllm.sif",
        "qwen35_35b_model": "{work_root}/models/qwen35-35b-a3b-awq",
        "qwen35_122b_model": "{work_root}/models/qwen35-122b-a10b-awq",
        "node_tmp": "/tmp",
        "state": "{work_root}/bench/state",
        "datasets": "~/Projects/opportunityscan",
    },
    "slurm": {
        "account": "ffa",
        "nice": 100,
        "default_partition": "gpu-ffa",
        "login_node_glob": "hpc-head*",
    },
    "vast": {
        "api": "https://console.vast.ai/api/v0",
        "image": "vastai/base-image:cuda-13.0.1-auto",
        "max_down_cost_per_tb": 5.0,
        "min_driver": "580.65",
        "min_reliability": 0.97,
        "runtype": "ssh_direc ssh_proxy",
        "ssh_key": "~/.ssh/id_ed25519",
        "default_hours": 2.0,
    },
}


def _merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    return override


def _expand(value, work_root):
    if not isinstance(value, str):
        return value
    text = value.replace("{work_root}", work_root)
    for _ in range(4):
        expanded = os.path.expandvars(os.path.expanduser(text))
        if expanded == text:
            break
        text = expanded
    return text


class Config:
    def __init__(self, path=None, work_root=None):
        self.config_path = Path(path or os.environ.get("BENCH_CONFIG") or BENCH_DIR / "config.json")
        self.raw = dict(DEFAULTS)
        if self.config_path.is_file():
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self.raw = _merge(self.raw, json.load(fh))
        self.loaded = self.config_path.is_file()
        wr = work_root or os.environ.get("BENCH_WORK_ROOT") or self.raw.get("work_root")
        self.work_root = str(Path(_expand(wr, str(Path.home()))).expanduser())
        self.paths = {}
        for key, value in self.raw.get("paths", {}).items():
            self.paths[key] = _expand(value, self.work_root)
        self.slurm = self.raw.get("slurm", {})
        self.vast = self.raw.get("vast", {})

    def path(self, name, default=None):
        if name in self.paths:
            return self.paths[name]
        if default is not None:
            return _expand(default, self.work_root)
        raise KeyError("unknown benchctl path %r (known: %s)"
                       % (name, ", ".join(sorted(self.paths))))

    def output_root(self, name):
        return Path(self.path(name))

    @property
    def nice(self):
        return int(self.slurm.get("nice", 100))

    @property
    def account(self):
        return str(self.slurm.get("account", ""))

    @property
    def login_glob(self):
        return str(self.slurm.get("login_node_glob", "hpc-head*"))

    def as_dict(self):
        return {
            "config_file": str(self.config_path),
            "config_loaded": self.loaded,
            "work_root": self.work_root,
            "paths": dict(self.paths),
            "slurm": dict(self.slurm),
            "vast": dict(self.vast),
            "repo_root": str(REPO_ROOT),
            "bench_dir": str(BENCH_DIR),
        }

    def state_path(self, name):
        return Path(self.path("state")) / name

    def dataset_importer(self):
        root = Path(self.path("datasets")).expanduser()
        return root / "modelsquantdb" / "scripts" / "import_strata_runs.py"


def load(path=None, work_root=None):
    return Config(path=path, work_root=work_root)


def is_login_node(cfg=None):
    import socket
    import fnmatch
    host = socket.gethostname().split(".")[0]
    glob = (cfg.login_glob if cfg else "hpc-head*")
    return fnmatch.fnmatch(host, glob)


def repo_relative(value):
    """Resolve 'repo:llm/...' evidence entries and absolute/relative paths."""
    if not isinstance(value, str):
        return Path(str(value))
    if value.startswith("repo:"):
        return REPO_ROOT / value[len("repo:"):]
    p = Path(os.path.expanduser(value))
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_evidence(cfg, value):
    """Evidence paths are relative to the work root unless marked repo:."""
    if not isinstance(value, str):
        return Path(str(value))
    if value.startswith("repo:"):
        return REPO_ROOT / value[len("repo:"):]
    p = Path(os.path.expanduser(value))
    if p.is_absolute():
        return p
    work = Path(cfg.work_root) / value
    repo = REPO_ROOT / value
    if work.exists():
        return work
    if repo.exists():
        return repo
    return work


__all__ = ["Config", "load", "is_login_node", "repo_relative", "resolve_evidence", "BENCH_DIR",
           "REPO_ROOT", "DEFAULTS", "util"]
