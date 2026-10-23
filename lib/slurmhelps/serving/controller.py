#!/usr/bin/env python3
"""Preemptible model-serving controller for Slurm (stdlib only).

The controller is deliberately thin and injectable: tests pass fake
``runner``/``ssh_runner`` callables, clocks and sleeps, so nothing here needs a
GPU, a real ``sbatch``, or an SSH connection.

Lifecycle::

    plan() -> submit() -> poll()/status() -> wait_ready() -> endpoint()
                                             \\-> down()

Slurm is the source of truth for allocation state only.  It can never make the
service ``ready``; only the worker's heartbeat/state file can do that.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ServingConfig
from .state import (
    DEFAULT_HEARTBEAT_TIMEOUT_S,
    StateStore,
    default_state,
    heartbeat_age_s,
    is_stale,
    utc_now,
)
from .tunnel import tunnel_argv

ACTIVE_STATES = frozenset(
    {"queued", "starting", "loading", "warming", "ready", "draining", "preempted", "requeued"}
)

# Slurm state -> service state.  ``RUNNING`` is intentionally absent: RUNNING
# says the allocation exists, not that the engine is loaded.
_SLURM_MAP = {
    "PENDING": "queued",
    "CONFIGURING": "starting",
    "STARTING": "starting",
    "RESIZING": "starting",
    "COMPLETING": "draining",
    "COMPLETED": "stopped",
    "CANCELLED": "stopped",
    "TIMEOUT": "stopped",
    "FAILED": "failed",
    "OOM": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "PREEMPTED": "preempted",
    "REQUEUED": "requeued",
    "REQUEUE_HOLD": "requeued",
}

DEFAULT_WORKER_COMMAND = ["python3", "-m", "slurmhelps.serving.worker"]


def _default_runner(argv: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _result(res: Any) -> tuple[int, str, str]:
    """Normalize a fake/real process result into ``(rc, stdout, stderr)``."""
    if res is None:
        return 0, "", ""
    if isinstance(res, str):
        return 0, res, ""
    if isinstance(res, tuple):
        parts = list(res) + ["", ""]
        return int(parts[0] or 0), str(parts[1] or ""), str(parts[2] or "")
    rc = int(getattr(res, "returncode", 0) or 0)
    out = getattr(res, "stdout", "") or ""
    err = getattr(res, "stderr", "") or ""
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    return rc, str(out), str(err)


def _first(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


class Controller:
    """Submit and follow one preemptible serving job.

    ``store`` may be a :class:`StateStore` or a directory path.  ``runner`` is
    called as ``runner(argv, timeout=...)``; ``ssh_runner`` receives the full
    ``ssh ...`` argv when ``ssh`` is configured.
    """

    def __init__(
        self,
        config: ServingConfig,
        store: StateStore | str | Path | None = None,
        *,
        directory: str | Path | None = None,
        runner: Callable[..., Any] | None = None,
        ssh_runner: Callable[..., Any] | None = None,
        ssh: str = "",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        command_timeout: float = 60.0,
    ) -> None:
        self.config = config
        if isinstance(store, StateStore):
            self.store = store
        else:
            if store is not None:
                directory = store
            self.store = StateStore(directory or Path(".slurmhelps-serve"), clock=clock)
        self.runner = runner or _default_runner
        self.ssh_runner = ssh_runner
        self.ssh = ssh
        self.clock = clock
        self.sleep = sleep
        self.command_timeout = command_timeout

    # ------------------------------------------------------------------
    # Process plumbing
    # ------------------------------------------------------------------
    def _exec(self, argv: Sequence[str], *, timeout: float | None = None) -> tuple[int, str, str]:
        """Run locally, or on the login host when ``ssh`` is configured."""
        argv = [str(a) for a in argv]
        timeout = self.command_timeout if timeout is None else timeout
        try:
            if self.ssh:
                remote = " ".join(shlex.quote(a) for a in argv)
                ssh_argv = [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    self.ssh,
                    remote,
                ]
                fn = self.ssh_runner or self.runner
                return _result(fn(ssh_argv, timeout=timeout))
            return _result(self.runner(argv, timeout=timeout))
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, "", str(exc)

    def _now(self) -> str:
        return utc_now(self.clock)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, worker_command: list[str] | str | None = None) -> dict[str, Any]:
        """Return the config-derived submission plan without side effects."""
        command = worker_command if worker_command is not None else DEFAULT_WORKER_COMMAND
        return {
            "model": self.config.model,
            "engine": self.config.engine,
            "partition": self.config.partition,
            "account": self.config.account,
            "gres": f"gpu:{self.config.gpu_type}:{self.config.gpus}",
            "gpus": self.config.gpus,
            "cpus": self.config.cpus,
            "mem": self.config.mem,
            "time": self.config.time,
            "slurm_args": self.config.slurm_args(),
            "backend_args": self.config.backend_args(),
            "sbatch": self.config.render_sbatch(command),
        }

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(
        self,
        script_path: str | Path | list[str] | tuple[str, ...] | None = None,
        command: list[str] | str | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Submit one job with ``sbatch --parsable`` and write ``state=queued``.

        Pass a rendered script path, or a worker command (list or string) to
        wrap.  With neither, the default worker module is used.
        """
        if isinstance(script_path, (list, tuple)) and command is None:
            command = list(script_path)
            script_path = None

        previous = self.store.read()
        if previous and not force and self._is_live(previous):
            raise RuntimeError(
                f"a live serving job already exists (job {previous.get('job_id')}, "
                f"state {previous.get('state')}); use force=True to submit anyway"
            )

        if script_path is None:
            if command is None:
                command = DEFAULT_WORKER_COMMAND
            script = self.config.render_sbatch(command)
            self.store.dir.mkdir(parents=True, exist_ok=True)
            script_path = self.store.dir / "serve.sbatch"
            Path(script_path).write_text(script, encoding="utf-8")

        rc, out, err = self._exec(["sbatch", "--parsable", str(script_path)])
        job_id = self._parse_job_id(out)
        if rc != 0 or not job_id:
            detail = err.strip() or out.strip() or f"exit {rc}"
            raise RuntimeError(f"sbatch failed: {detail}")

        attempt = int((previous or {}).get("attempt") or 0) + 1
        state = default_state()
        state.update(
            {
                "job_id": job_id,
                "state": "queued",
                "phase": "submitted",
                "attempt": attempt,
                "last_heartbeat": self._now(),
                "updated_at": self._now(),
                "error": None,
                "endpoint": None,
                "script": str(script_path),
            }
        )
        return self.store.write(state)

    @staticmethod
    def _is_live(state: Mapping[str, Any]) -> bool:
        return bool(state.get("job_id")) and state.get("state") in ACTIVE_STATES

    @staticmethod
    def _parse_job_id(text: str) -> str | None:
        line = _first(text)
        if not line:
            return None
        token = line.split(";", 1)[0].strip()
        if token.isdigit():
            return token
        match = re.search(r"Submitted batch job\s+(\d+)", line)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Polling / status
    # ------------------------------------------------------------------
    def poll(self) -> dict[str, Any] | None:
        """Refresh from ``squeue`` (``sacct`` fallback); never marks ready."""
        state = self.store.read()
        if state is None:
            return None
        job_id = str(state.get("job_id") or "")
        if not job_id:
            return state

        slurm_state = ""
        node = ""
        rc, out, _ = self._exec(["squeue", "-h", "-j", job_id, "-o", "%T|%N|%M"])
        if rc == 0 and out.strip():
            fields = _first(out).split("|")
            slurm_state = fields[0].strip()
            node = fields[1].strip() if len(fields) > 1 else ""
        else:
            rc, out, _ = self._exec(["sacct", "-n", "-j", job_id, "-o", "State"])
            if rc == 0 and out.strip():
                slurm_state = _first(out).split()[0].strip().rstrip("+")

        if (
            node in {"", "(null)", "None", "n/a", "N/A"}
            or node.startswith("(")
            or node.startswith("None assigned")
        ):
            node = ""

        if slurm_state:
            state = self._apply_slurm(state, slurm_state, node)
            state["slurm_state"] = slurm_state
            state["updated_at"] = self._now()
            state = self.store.write(state)
        return state

    def _apply_slurm(
        self, state: Mapping[str, Any], slurm_state: str, node: str
    ) -> dict[str, Any]:
        updated = dict(state)
        if node:
            updated["node"] = node
        current = updated.get("state")

        if slurm_state == "RUNNING":
            # The allocation exists, not the service.  Keep the worker's
            # state as-is (in particular: RUNNING alone never means ready).
            return updated

        target = _SLURM_MAP.get(slurm_state)
        if target is None:
            return updated
        if target == current:
            return updated
        if target == "failed":
            updated["error"] = f"slurm:{slurm_state}"
        # Terminal Slurm states collapse directly (COMPLETED -> stopped); the
        # service state machine is still enforced for worker-driven changes.
        updated["state"] = target
        return updated

    def status(self, *, heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S) -> dict[str, Any]:
        """State file + live Slurm view, usable while HTTP is not up yet."""
        state = self.poll()
        if state is None:
            return {
                "schema": 1,
                "state": None,
                "job_id": None,
                "slurm_state": None,
                "phase": None,
                "endpoint": None,
                "heartbeat_age_s": None,
                "stale": True,
                "load_estimate_s": None,
                "load_estimate_source": "no-state",
            }

        out = dict(state)
        age = heartbeat_age_s(state.get("last_heartbeat"), self.clock())
        out["heartbeat_age_s"] = age
        out["stale"] = is_stale(state.get("last_heartbeat"), self.clock(), heartbeat_timeout_s)
        out["slurm_state"] = state.get("slurm_state")

        # A stale endpoint from an old allocation is never advertised.
        if out.get("state") != "ready":
            out["endpoint"] = None
        else:
            out["endpoint"] = self.endpoint()

        if out.get("state") in {"starting", "loading", "warming"}:
            estimate = self.load_estimate(out.get("phase"))
            out["load_estimate"] = estimate
            if out.get("load_estimate_s") is None:
                out["load_estimate_s"] = estimate.get("p50")
            if not out.get("load_estimate_source"):
                out["load_estimate_source"] = estimate.get("source")
        return out

    def wait_ready(self, timeout: float = 600.0, poll_s: float = 5.0) -> dict[str, Any]:
        """Poll until the worker says ``ready``; raise ``TimeoutError``."""
        start = self.clock()
        while True:
            status = self.status()
            if status.get("state") == "ready":
                return status
            if self.clock() - start >= timeout:
                raise TimeoutError(
                    f"serving job not ready after {timeout}s "
                    f"(state={status.get('state')}, phase={status.get('phase')})"
                )
            self.sleep(poll_s)

    # ------------------------------------------------------------------
    # Endpoint / teardown
    # ------------------------------------------------------------------
    def endpoint(self) -> str | None:
        """The endpoint only when the worker state is ready and it is present."""
        state = self.store.read()
        if not state or state.get("state") != "ready":
            return None
        endpoint = state.get("endpoint")
        if not endpoint:
            return None
        if isinstance(endpoint, Mapping):
            # The worker publishes {"host", "port", "url"}; the client-facing
            # URL is the loopback endpoint reached through ssh -L.
            if endpoint.get("url"):
                return str(endpoint["url"])
            port = endpoint.get("port") or self.config.port
            return f"http://127.0.0.1:{port}/v1"
        return str(endpoint)

    def endpoint_hint(self) -> str | None:
        """The ssh ``-L`` command to use while no ready endpoint is available."""
        state = self.store.read() or {}
        node = state.get("node")
        if not node:
            return None
        argv = tunnel_argv(
            str(node),
            self.config.port,
            self.config.port,
            via=self.config.via or None,
        )
        return " ".join(shlex.quote(part) for part in argv)

    def endpoint_info(self) -> dict[str, Any]:
        endpoint = self.endpoint()
        return {
            "endpoint": endpoint,
            "hint": None if endpoint else self.endpoint_hint(),
        }

    def down(self) -> dict[str, Any]:
        """Cancel the job and transition the service to ``stopped``."""
        state = self.store.read()
        if not state or not state.get("job_id"):
            raise RuntimeError("no serving job is recorded")
        job_id = str(state["job_id"])
        rc, out, err = self._exec(["scancel", job_id])
        if rc != 0:
            raise RuntimeError(f"scancel failed: {err.strip() or out.strip() or f'exit {rc}'}")

        if state.get("state") == "ready":
            state = self.store.transition("draining", phase="draining")
        try:
            return self.store.transition("stopped", phase="stopped", endpoint=None)
        except ValueError:
            # Terminal/non-adjacent states still need an explicit stop.
            state["state"] = "stopped"
            state["phase"] = "stopped"
            state["endpoint"] = None
            state["updated_at"] = self._now()
            return self.store.write(state)

    def load_estimate(self, phase: str | None = None) -> dict[str, Any]:
        """Load estimate from :class:`StateStore` history, with honest fallback."""
        if phase is None:
            state = self.store.read() or {}
            phase = state.get("phase") or "loading:weights"
        return self.store.estimate(str(phase))


__all__ = ["ACTIVE_STATES", "Controller", "DEFAULT_WORKER_COMMAND"]
