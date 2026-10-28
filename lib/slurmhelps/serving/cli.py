#!/usr/bin/env python3
"""CLI for the preemptible serving gateway.

Examples (login node)::

    slurmhelps-serve up --yes
    slurmhelps-serve status --json
    slurmhelps-serve wait --timeout 3600
    slurmhelps-serve endpoint
    slurmhelps-serve tunnel --local-port 8000 --print

The CLI never reads or writes an API key.  It passes the *name* of the key
environment variable to the gateway; the value stays in the operator's
environment or a 0600 file outside this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import ServingConfig
from .controller import Controller
from .gateway import GatewayServer, load_bad_words
from .rewrite import Rewriter, default_rules, load_rules
from .mock import MockOpenAIServer
from .spill import SpillManager
from .state import StateStore
from .tunnel import Relay, tunnel_argv
from .worker import WorkerSpec, run_worker

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_SUBDIR = ".slurmhelps-serve"
DEFAULT_SERVER_HELPER = ROOT / "llm" / "glm53-reap50" / "serve-server.sh"


def project_config() -> dict[str, Any]:
    try:
        from slurmhelps import config as project
        return project.load()
    except Exception:  # noqa: BLE001 - CLI still works with defaults
        return {}


def load_config(path: str | None = None) -> ServingConfig:
    doc = project_config()
    if path:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    serving = doc.get("serving", doc)
    cluster = doc.get("cluster") if isinstance(doc, dict) else None
    root = doc.get("_root") if isinstance(doc, dict) else None
    return ServingConfig.from_mapping(serving, cluster=cluster, root=root)


def work_root(cfg: ServingConfig) -> str:
    env = os.environ.get("WORK_ROOT")
    if env:
        return env
    doc = project_config()
    site = doc.get("site") if isinstance(doc, dict) else None
    if isinstance(site, dict) and site.get("work_root"):
        return str(site["work_root"])
    # The cluster checkout is normally <work_root>/slurmhelps-fleet.
    if cfg.workdir:
        return str(Path(cfg.workdir).resolve().parent)
    return ""


def state_dir(cfg: ServingConfig, override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("SLURMHELPS_SERVE_DIR")
    if env:
        return Path(env)
    base = Path(cfg.workdir or os.getcwd())
    return base / DEFAULT_STATE_SUBDIR / cfg.model


def controller_for(cfg: ServingConfig, directory: Path) -> Controller:
    return Controller(cfg, directory, ssh=cfg.via or "")


def _print_status(status: dict[str, Any]) -> None:
    phase = status.get("phase") or "-"
    estimate = status.get("load_estimate_s")
    eta = f" eta~{estimate:.0f}s" if isinstance(estimate, (int, float)) else ""
    print(
        f"{status.get('state') or 'unknown':10s} phase={phase:16s} "
        f"job={status.get('job_id') or '-'} node={status.get('node') or '-'}"
        f"{eta}"
    )
    if status.get("slurm_state"):
        print(f"slurm={status['slurm_state']} heartbeat_age={status.get('heartbeat_age_s')}s "
              f"stale={status.get('stale')}")
    if status.get("error"):
        print(f"error={status['error']}")
    endpoint = status.get("endpoint")
    if endpoint:
        print(f"endpoint={endpoint}")


def cmd_up(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    backend = list(args.backend_command or [])
    if backend and backend[0] == "--":
        backend = backend[1:]
    if not backend:
        backend = ["bash", str(DEFAULT_SERVER_HELPER)]
    if backend[0] in {"bash", "sh", "python3", "python"}:
        if len(backend) > 1 and not Path(backend[1]).exists():
            raise SystemExit(f"backend command not found: {backend[1]}")
    elif not Path(backend[0]).exists():
        raise SystemExit(f"backend command not found: {backend[0]}")
    wr = work_root(cfg)
    env_parts = [
        f"PYTHONPATH={ROOT / 'lib'}",
        f"PORT={cfg.port}",
        f"MAX_MODEL_LEN={cfg.max_model_len}",
        f"MAX_NUM_SEQS={cfg.max_num_seqs}",
        f"MAX_NUM_BATCHED_TOKENS={cfg.max_num_batched_tokens}",
        f"GPU_MEM_UTIL={cfg.gpu_memory_utilization}",
        f"KV_DTYPE={cfg.kv_cache_dtype}",
        f"BLOCK_SIZE={cfg.block_size}",
        "ENABLE_CHUNKED_PREFILL=1",
    ]
    if wr:
        env_parts.append(f"WORK_ROOT={wr}")
    worker = [
        "env",
        *env_parts,
        "python3",
        "-m",
        "slurmhelps.serving.cli",
        "worker",
        "--state-dir",
        str(directory),
        "--health-url",
        f"http://127.0.0.1:{cfg.port}/health",
        "--port",
        str(cfg.port),
        "--max-model-len",
        str(cfg.max_model_len),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--max-num-batched-tokens",
        str(cfg.max_num_batched_tokens),
        "--startup-timeout",
        str(cfg.startup_timeout_s),
        "--",
        *backend,
    ]
    ctl = controller_for(cfg, directory)
    plan = ctl.plan(worker)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.yes:
        print("refusing to submit without --yes; plan:", file=sys.stderr)
        print(json.dumps(plan, indent=2), file=sys.stderr)
        return 2
    state = ctl.submit(command=worker, force=args.force)
    print(json.dumps(state, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    ctl = controller_for(cfg, directory)
    status = ctl.status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_status(status)
    code = {"ready": 0, "loading": 3, "warming": 3, "starting": 3,
            "queued": 4, "requeued": 4, "failed": 5, "stopped": 5,
            "preempted": 5}.get(str(status.get("state")), 3)
    return code


def cmd_wait(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    ctl = controller_for(cfg, directory)
    try:
        status = ctl.wait_ready(timeout=args.timeout, poll_s=args.poll_s)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_status(status)
    return 0


def cmd_endpoint(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    state = StateStore(directory).read() or {}
    endpoint = state.get("endpoint")
    if isinstance(endpoint, dict):
        host = endpoint.get("host") or state.get("node")
        port = endpoint.get("port") or cfg.port
        endpoint = f"http://127.0.0.1:{port}/v1"
        if host:
            state["endpoint"] = endpoint
    if not endpoint or state.get("state") != "ready":
        node = state.get("node")
        hint = None
        if node:
            hint = " ".join(shlex.quote(x) for x in tunnel_argv(
                str(node), cfg.port, cfg.port, via=cfg.via or None))
        out = {"endpoint": None, "state": state.get("state"), "hint": hint}
        print(json.dumps(out, indent=2) if args.json else hint or "not ready")
        return 3
    if args.json:
        print(json.dumps({"endpoint": endpoint, "state": state.get("state"),
                          "node": state.get("node"), "port": cfg.port}, indent=2))
    else:
        print(endpoint)
    return 0


def cmd_tunnel(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    state = StateStore(directory).read() or {}
    node = args.node or state.get("node")
    if not node:
        print("no node: service is not running", file=sys.stderr)
        return 3
    via = args.via if args.via is not None else (cfg.via or None)
    argv = tunnel_argv(str(node), args.remote_port or cfg.port, args.local_port, via=via)
    print(" ".join(shlex.quote(x) for x in argv))
    if args.print:
        return 0
    os.execvp(argv[0], argv)
    return 0


def cmd_relay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    state = StateStore(directory).read() or {}
    node = args.target_host or state.get("node")
    if not node:
        print("no node: service is not running", file=sys.stderr)
        return 3
    target_port = args.target_port or cfg.port
    relay = Relay(str(node), target_port, listen_host=args.listen_host,
                  listen_port=args.listen_port)
    host, port = relay.start()
    print(f"relay {host}:{port} -> {node}:{target_port}", flush=True)
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        relay.stop()
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    ctl = controller_for(cfg, directory)
    try:
        state = ctl.down()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    print(json.dumps(state, indent=2))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = ["bash", str(DEFAULT_SERVER_HELPER)]
    spec = WorkerSpec(
        command=command,
        state_dir=Path(args.state_dir),
        health_url=args.health_url,
        job_id=args.job_id or os.environ.get("SLURM_JOB_ID", ""),
        node=args.node or os.environ.get("SLURMD_NODENAME", ""),
        port=args.port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        startup_timeout_s=args.startup_timeout,
    )
    state = run_worker(spec, run_until_ready=args.until_ready)
    print(json.dumps(state, indent=2))
    return 0


def cmd_mock(args: argparse.Namespace) -> int:
    server = MockOpenAIServer(
        port=args.port,
        model=args.model,
        max_seqs=args.max_seqs,
        max_model_len=args.max_model_len,
        coalesce_ms=args.coalesce_ms,
    ).start()
    print(server.base_url, flush=True)
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def cmd_spill(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    state = StateStore(directory).read() or {}
    target = Path(args.spill_dir or cfg.spill_dir or (directory / "spill"))
    spill = SpillManager(target, idle_seconds=args.idle_seconds if args.idle_seconds is not None else cfg.idle_seconds,
                         max_gib=args.max_gib if args.max_gib is not None else cfg.spill_max_gib)
    if args.activity_file:
        try:
            spill.note_activity(Path(args.activity_file).stat().st_mtime)
        except OSError:
            pass
    elif args.force:
        spill.note_activity(spill.clock() - spill.idle_seconds)
    path = spill.snapshot(
        {"state": state.get("state"), "model": cfg.model,
         "capacity": state.get("capacity"), "endpoint": state.get("endpoint")},
        active_requests=int(state.get("active_requests") or 0),
        when=None if args.force else spill.clock(),
    )
    if path is None:
        print("not idle: no spill written")
        return 3
    print(str(path))
    return 0


def cmd_gateway(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    state = StateStore(directory).read() or {}
    upstream = args.upstream
    if not upstream:
        node = state.get("node")
        if not node:
            print("no node: service is not running", file=sys.stderr)
            return 3
        upstream = f"http://{node}:{cfg.port}/v1"
    bad_words = list(args.bad_word or [])
    if args.bad_words_file:
        bad_words += load_bad_words(args.bad_words_file)
    rewriter = None
    if args.rewrite_rules:
        rules = default_rules() if args.rewrite_rules == "default" else load_rules(args.rewrite_rules)
        rewriter = Rewriter(rules, log_path=args.rewrite_log or None, history_mode=args.history_mode,
                            shellguard=not args.no_shellguard,
                            privguard=not args.no_privilege_guard, allow_hosts=args.allow_host or ())
    max_active = None if args.max_active < 0 else (args.max_active or cfg.max_num_seqs)
    gateway = GatewayServer(
        upstream=upstream,
        host=args.listen_host,
        port=args.listen_port,
        key_env=args.key_env or cfg.key_env,
        max_active=max_active,
        coalesce_ms=args.coalesce_ms if args.coalesce_ms is not None else cfg.coalesce_ms,
        timeout=args.timeout,
        bad_words=bad_words,
        rewriter=rewriter,
    ).start()
    print(gateway.base_url, flush=True)
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        gateway.stop()
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    script = ROOT / "tests" / "serve_smoke.py"
    argv = [sys.executable, str(script), "--base-url", args.base_url, "--model", args.model]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    if args.auth_required:
        argv.append("--auth-required")
    if args.contexts:
        argv += ["--contexts", args.contexts]
    if args.concurrency:
        argv += ["--concurrency", args.concurrency]
    if args.max_active:
        argv += ["--max-active", str(args.max_active)]
    if args.output:
        argv += ["--output", args.output]
    if args.only:
        argv += ["--only", args.only]
    return subprocess.call(argv)


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    directory = state_dir(cfg, args.state_dir)
    out = {
        "config": cfg.__dict__,
        "state_dir": str(directory),
        "state": StateStore(directory).read(),
        "server_helper": str(DEFAULT_SERVER_HELPER),
        "server_helper_exists": DEFAULT_SERVER_HELPER.exists(),
        "python": sys.executable,
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slurmhelps-serve", description=__doc__)
    parser.add_argument("--config", default="", help="JSON config path (default: project slurmhelps.json)")
    parser.add_argument("--state-dir", default="", help="override shared state directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("up", help="submit the serving job")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--backend-command", nargs=argparse.REMAINDER, default=None)
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("status", help="state + Slurm status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("wait", help="wait until ready")
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("endpoint", help="print the API URL or SSH hint")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_endpoint)

    p = sub.add_parser("tunnel", help="open ssh -L to the compute node")
    p.add_argument("--local-port", type=int, default=8000)
    p.add_argument("--remote-port", type=int, default=0)
    p.add_argument("--node", default="")
    p.add_argument("--via", default=None)
    p.add_argument("--print", action="store_true", dest="print_")
    p.set_defaults(func=cmd_tunnel)

    p = sub.add_parser("relay", help="loopback TCP relay on the login node")
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=18000)
    p.add_argument("--target-host", default="")
    p.add_argument("--target-port", type=int, default=0)
    p.set_defaults(func=cmd_relay)

    p = sub.add_parser("down", help="cancel the job")
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("worker", help="compute-side worker (run inside the job)")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--job-id", default="")
    p.add_argument("--node", default="")
    p.add_argument("--max-model-len", type=int, default=786432)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--startup-timeout", type=float, default=3600.0)
    p.add_argument("--until-ready", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("mock", help="run the OpenAI-compatible mock backend")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--model", default="glm53-reap50")
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=786432)
    p.add_argument("--coalesce-ms", type=float, default=0.0)
    p.set_defaults(func=cmd_mock)

    p = sub.add_parser("spill", help="write an idle state-only manifest")
    p.add_argument("--spill-dir", default="")
    p.add_argument("--idle-seconds", type=float, default=None)
    p.add_argument("--max-gib", type=float, default=None)
    p.add_argument("--activity-file", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_spill)

    p = sub.add_parser("gateway", help="loopback OpenAI proxy with auth/admission")
    p.add_argument("--upstream", default="")
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=18001)
    p.add_argument("--key-env", default="")
    p.add_argument("--max-active", type=int, default=0,
                   help="0 = config max_num_seqs, -1 = no limit")
    p.add_argument("--coalesce-ms", type=float, default=None)
    p.add_argument("--timeout", type=float, default=600.0,
                   help="upstream socket timeout in seconds")
    p.add_argument("--bad-word", action="append", default=[],
                   help="string the model may not emit (repeatable)")
    p.add_argument("--bad-words-file", default="",
                   help="JSON list of banned strings merged into every completion request")
    p.add_argument("--rewrite-rules", default="",
                   help="'default' or a JSON rules file: scrub history and rewrite output")
    p.add_argument("--rewrite-log", default="", help="JSONL log of every rewrite (original text)")
    p.add_argument("--history-mode", choices=("drop", "swap"), default="drop",
                   help="denied tool-call URLs in the history: drop the call and result, or swap to the intended URL")
    p.add_argument("--no-shellguard", action="store_true",
                   help="do not remove off-task uploads and metadata requests from command tool calls")
    p.add_argument("--no-privilege-guard", action="store_true",
                   help="do not replace privilege workarounds in command tool calls with a sandbox notice")
    p.add_argument("--allow-host", action="append", default=[],
                   help="host that command tool calls may upload to (repeatable; subdomains match)")
    p.set_defaults(func=cmd_gateway)

    p = sub.add_parser("smoke", help="run the HTTP smoke matrix")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="glm53-reap50")
    p.add_argument("--api-key", default="")
    p.add_argument("--auth-required", action="store_true")
    p.add_argument("--contexts", default="500 800")
    p.add_argument("--concurrency", default="1 2 4 8")
    p.add_argument("--max-active", type=int, default=0)
    p.add_argument("--only", default="")
    p.add_argument("--output", default="")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("doctor", help="print config, state and helper paths")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse uses dest print_ because print is a builtin.
    if hasattr(args, "print_"):
        args.print = args.print_
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
