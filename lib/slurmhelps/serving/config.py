#!/usr/bin/env python3
"""Serving configuration for the preemptible Slurm gateway (stdlib only).

``ServingConfig`` is the typed form of the ``serving`` block in
``docs/serving-gateway.md``.  Defaults are the documented 2x A100 starting
point, not claims of optimality.

Security rules enforced here:

* ``key_env`` is a variable *name*, never a value.  This module does not read
  the environment for the key and never stores it in the dataclass;
* a listener bind must be loopback unless the caller explicitly passes
  ``allow_public=True`` (or sets ``api.allow_public`` in the mapping).
"""

from __future__ import annotations

import copy
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

# The checkpoint's text max_position_embeddings (786432); kept as a named
# constant so the "700K support" contract is testable.
MAX_MODEL_LEN_LIMIT = 1_048_576
MIN_MODEL_LEN = 128
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Documented cold components: the full cold start was measured at 9-10 min
# for the 98 GiB checkpoint, dominated by weights.
COLD_FALLBACK_S = {"weights": 540.0, "kv": 30.0, "warmup": 45.0}

_SERVING_FIELDS = {
    "model",
    "engine",
    "gpus",
    "gpu_type",
    "partition",
    "account",
    "cpus",
    "mem",
    "time",
    "requeue",
    "nice",
    "exclusive",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "startup_timeout_s",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "block_size",
    "host_kv_ram_gib",
    "queue_limit",
    "coalesce_ms",
    "bind",
    "port",
    "key_env",
    "relay_port",
    "relay_bind",
    "via",
    "spill_enabled",
    "idle_seconds",
    "spill_dir",
    "spill_max_gib",
    "spill_mode",
    "workdir",
    "allow_public",
}

_FLAG_ONLY = frozenset({"--requeue", "--exclusive"})


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _as_int(name: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _as_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


@dataclass
class ServingConfig:
    """Validated serving configuration.

    Field names are flat so callers can use ``cfg.max_model_len`` directly;
    the documented nested views are available through :attr:`api`,
    :attr:`tunnel` and :attr:`spill`.
    """

    model: str = "glm53-reap50-nvfp4"
    engine: str = "vllm-sm80"
    gpus: int = 2
    gpu_type: str = "A100"
    partition: str = "ffa-preempt"
    account: str = ""
    cpus: int = 8
    mem: str = "220G"
    time: str = "04:00:00"
    requeue: bool = True
    nice: int = 100
    exclusive: bool = False

    max_model_len: int = 786_432
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8_192
    # Seconds the worker waits for /health before failing the job (weights + kernel prep).
    startup_timeout_s: int = 3_600
    gpu_memory_utilization: float = 0.94
    kv_cache_dtype: str = "bfloat16"
    block_size: int = 128

    host_kv_ram_gib: int = 96
    queue_limit: int = 64
    coalesce_ms: int = 0

    bind: str = "127.0.0.1"
    port: int = 8_000
    key_env: str = "SLURMHELPS_API_KEY"

    via: str = ""
    relay_port: int = 18_000
    relay_bind: str = "127.0.0.1"

    spill_enabled: bool = False
    idle_seconds: int = 600
    spill_dir: str = ""
    spill_max_gib: int = 128
    spill_mode: str = "state-only"

    # Project root used for ``cd`` and relative spill paths.  Not a secret.
    workdir: str = ""
    allow_public: bool = False

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        cluster: Mapping[str, Any] | None = None,
        root: str | os.PathLike[str] | None = None,
        allow_public: bool = False,
    ) -> "ServingConfig":
        """Build a config from a ``serving`` mapping.

        ``mapping`` may be either the serving block itself or a full project
        document containing a ``"serving"`` key.  Nested ``api``, ``tunnel``
        and ``spill`` objects are merged.  ``cluster`` supplies ``account``
        when the serving account is empty (``cluster.account``).
        """
        raw_doc: Mapping[str, Any] = mapping or {}
        nested = raw_doc.get("serving")
        raw: Mapping[str, Any] = nested if isinstance(nested, Mapping) else raw_doc
        if cluster is None and isinstance(raw_doc.get("cluster"), Mapping):
            cluster = raw_doc["cluster"]
        if isinstance(cluster, Mapping) and isinstance(cluster.get("cluster"), Mapping):
            cluster = cluster["cluster"]  # type: ignore[index]

        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"api", "tunnel", "spill"} and isinstance(value, Mapping):
                continue
            if key in _SERVING_FIELDS:
                values[key] = value

        api = raw.get("api")
        if isinstance(api, Mapping):
            for key in ("bind", "port", "key_env", "allow_public"):
                if key in api:
                    values[key] = api[key]

        tunnel = raw.get("tunnel")
        if isinstance(tunnel, Mapping):
            if "bind" in tunnel:
                values["relay_bind"] = tunnel["bind"]
            for key in ("relay_port", "via"):
                if key in tunnel:
                    values[key] = tunnel[key]

        spill = raw.get("spill")
        if isinstance(spill, bool):
            values["spill_enabled"] = spill
        elif isinstance(spill, Mapping):
            if "enabled" in spill:
                values["spill_enabled"] = spill["enabled"]
            if "idle_seconds" in spill:
                values["idle_seconds"] = spill["idle_seconds"]
            if "dir" in spill:
                values["spill_dir"] = spill["dir"]
            if "max_gib" in spill:
                values["spill_max_gib"] = spill["max_gib"]
            if "mode" in spill:
                values["spill_mode"] = spill["mode"]

        defaults = cls()
        data = {f.name: getattr(defaults, f.name) for f in fields(cls)}
        data.update(values)

        # account: empty means "inherit cluster.account" (or the documented
        # symbolic marker, which is also not a literal Slurm account name).
        account = str(data.get("account") or "")
        if account in {"", "from cluster.account", "{cluster.account}"}:
            account = str((cluster or {}).get("account", "") or "")
        data["account"] = account

        # The documented tunnel via defaults to cluster.ssh.
        if not data.get("via"):
            data["via"] = str((cluster or {}).get("ssh", "") or "")

        allow_public = bool(allow_public or data.get("allow_public", False))
        data["allow_public"] = allow_public

        if root is not None:
            root_text = os.fspath(root)
            if not data.get("workdir"):
                data["workdir"] = root_text
            spill_dir = str(data.get("spill_dir") or "")
            if spill_dir:
                spill_dir = spill_dir.replace("{work_root}", root_text).replace("{root}", root_text)
                if spill_dir and not os.path.isabs(spill_dir):
                    spill_dir = os.path.join(root_text, spill_dir)
                data["spill_dir"] = spill_dir

        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate numeric ranges and the loopback policy; raise ``ValueError``."""
        self.model = _as_text("model", self.model)
        self.engine = _as_text("engine", self.engine)
        self.gpu_type = _as_text("gpu_type", self.gpu_type)
        self.partition = _as_text("partition", self.partition)
        self.mem = _as_text("mem", self.mem)
        self.time = _as_text("time", self.time)
        self.kv_cache_dtype = _as_text("kv_cache_dtype", self.kv_cache_dtype)
        self.key_env = _as_text("key_env", self.key_env)
        self.spill_mode = _as_text("spill_mode", self.spill_mode)

        self.gpus = _as_int("gpus", self.gpus, minimum=1)
        self.cpus = _as_int("cpus", self.cpus, minimum=1)
        self.nice = _as_int("nice", self.nice)
        self.max_model_len = _as_int(
            "max_model_len", self.max_model_len, minimum=MIN_MODEL_LEN, maximum=MAX_MODEL_LEN_LIMIT
        )
        self.max_num_seqs = _as_int("max_num_seqs", self.max_num_seqs, minimum=1)
        self.max_num_batched_tokens = _as_int(
            "max_num_batched_tokens", self.max_num_batched_tokens, minimum=1
        )
        self.startup_timeout_s = _as_int("startup_timeout_s", self.startup_timeout_s, minimum=60)
        self.block_size = _as_int("block_size", self.block_size, minimum=1)
        self.host_kv_ram_gib = _as_int("host_kv_ram_gib", self.host_kv_ram_gib, minimum=0)
        self.queue_limit = _as_int("queue_limit", self.queue_limit, minimum=0)
        self.coalesce_ms = _as_int("coalesce_ms", self.coalesce_ms, minimum=0, maximum=1000)
        self.port = _as_int("port", self.port, minimum=1, maximum=65535)
        self.relay_port = _as_int("relay_port", self.relay_port, minimum=1, maximum=65535)
        self.idle_seconds = _as_int("idle_seconds", self.idle_seconds, minimum=0)
        self.spill_max_gib = _as_int("spill_max_gib", self.spill_max_gib, minimum=0)

        if isinstance(self.gpu_memory_utilization, bool) or not isinstance(
            self.gpu_memory_utilization, (int, float)
        ):
            raise ValueError("gpu_memory_utilization must be a number")
        if not 0 < float(self.gpu_memory_utilization) <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.gpu_memory_utilization = float(self.gpu_memory_utilization)

        self.requeue = _as_bool("requeue", self.requeue)
        self.exclusive = _as_bool("exclusive", self.exclusive)
        self.spill_enabled = _as_bool("spill_enabled", self.spill_enabled)
        self.allow_public = _as_bool("allow_public", self.allow_public)

        self.bind = _as_text("bind", self.bind)
        self.relay_bind = _as_text("relay_bind", self.relay_bind)
        if not self.allow_public:
            if not _loopback(self.bind):
                raise ValueError("bind must be 127.0.0.1 or ::1 unless allow_public=True")
            if not _loopback(self.relay_bind):
                raise ValueError("relay bind must be loopback unless allow_public=True")

        self.account = str(self.account or "")
        self.via = str(self.via or "")
        self.spill_dir = str(self.spill_dir or "")
        self.workdir = str(self.workdir or "")

    # ------------------------------------------------------------------
    # Documented nested views
    # ------------------------------------------------------------------
    @property
    def api(self) -> dict[str, Any]:
        return {"bind": self.bind, "port": self.port, "key_env": self.key_env}

    @property
    def tunnel(self) -> dict[str, Any]:
        return {"via": self.via, "relay_port": self.relay_port, "bind": self.relay_bind}

    @property
    def spill(self) -> dict[str, Any]:
        return {
            "enabled": self.spill_enabled,
            "idle_seconds": self.idle_seconds,
            "dir": self.spill_dir,
            "max_gib": self.spill_max_gib,
            "mode": self.spill_mode,
        }

    # ``spill.dir`` / ``spill.max_gib`` aliases for callers that model the
    # documented JSON keys rather than the flat dataclass.
    @property
    def dir(self) -> str:
        return self.spill_dir

    @property
    def max_gib(self) -> int:
        return self.spill_max_gib

    # ------------------------------------------------------------------
    # Slurm / engine argv
    # ------------------------------------------------------------------
    def slurm_args(self) -> list[str]:
        """``sbatch`` options in the documented preemptible-by-default shape."""
        args = ["--partition", self.partition]
        if self.account:
            args += ["--account", self.account]
        args += [
            f"--gres=gpu:{self.gpu_type}:{self.gpus}",
            "--cpus-per-task",
            str(self.cpus),
            "--mem",
            self.mem,
            "--time",
            self.time,
        ]
        if self.requeue:
            args.append("--requeue")
        args += ["--nice", str(self.nice)]
        if self.exclusive:
            args.append("--exclusive")
        return args

    def backend_args(self) -> list[str]:
        """vLLM flags; the model path is supplied by the worker, not config."""
        return [
            "--tensor-parallel-size",
            str(self.gpus),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--block-size",
            str(self.block_size),
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--served-model-name",
            self.model,
        ]

    def render_sbatch(self, worker_command: list[str] | str) -> str:
        """Render the job script.  It contains no secret and no API-key value."""
        if isinstance(worker_command, str):
            command = worker_command.strip()
        else:
            command = shlex.join([str(x) for x in worker_command])
        if not command:
            raise ValueError("worker_command is empty")

        lines = ["#!/bin/bash"]
        args = self.slurm_args()
        i = 0
        while i < len(args):
            token = args[i]
            if token in _FLAG_ONLY:
                lines.append(f"#SBATCH {token}")
                i += 1
            elif token.startswith("--") and "=" in token:
                # Combined form such as --gres=gpu:a100:2.
                lines.append(f"#SBATCH {token}")
                i += 1
            elif token.startswith("-") and i + 1 < len(args):
                lines.append(f"#SBATCH {token}={args[i + 1]}")
                i += 2
            else:  # defensive: never index past the end
                lines.append(f"#SBATCH {token}")
                i += 1
        lines += ["", "set -euo pipefail"]
        if self.workdir:
            lines.append(f"cd {shlex.quote(self.workdir)}")
        else:
            lines.append('cd "$(dirname "$0")"')
        lines.append(f"exec {command}")
        return "\n".join(lines) + "\n"


__all__ = [
    "COLD_FALLBACK_S",
    "MAX_MODEL_LEN_LIMIT",
    "MIN_MODEL_LEN",
    "ServingConfig",
]
