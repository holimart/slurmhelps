#!/usr/bin/env python3
"""Loopback HTTP router over every slurmhelps serving deployment (stdlib only).

The router runs on the login node and gives clients one stable port.

Discovery
    Every ``<serve_root>/<name>/state.json`` is a deployment (``<name>`` is the
    config's ``serving.model``, i.e. the state directory ``up`` created).
    ``llm/models.json`` adds metadata (display name, GPUs, configured port,
    served model name, guards); a deployment without a catalogue entry is
    still discovered. A deployment is *ready* when its state is ``ready`` with
    a fresh heartbeat and a node. Its port comes from the state's endpoint
    (written by the worker when ready), else from the catalogued config.
    The models it serves are read from its ``/v1/models`` (cached per job).

Routing
    A request naming a ``model`` (JSON body) goes to the first ready
    deployment serving that model, in preference order (``prefer`` names
    first, then catalogue order, then alphabetical); ``aliases`` map extra
    names to a served name. Requests without a model go to the first ready
    deployment. ``/v1/models`` is the de-duplicated union over ready
    deployments; ``/health`` and ``/router/status`` are answered locally.
    A deployment that refuses a connection (or fails before any response
    byte reached the client) is put on a cooldown and the next candidate is
    tried. With nothing ready the answer is ``503 "no ready deployment"``.

Guards
    Per served model (``models.json`` ``guards``: ``bad_words_file``,
    ``rewrite_rules``), completion requests get the node gateway's request
    guard (:func:`gateway.guard_request_body`: bad_words merge + history
    scrub with shellguard/privguard) and their responses the gateway's
    JSON/SSE rewrite (``_GatewayHandler._relay_rewritten``), so a deployment
    running without its node gateway is still guarded. The guards are
    idempotent, so they are applied whether or not a node gateway also runs
    (defence in depth; nothing to detect or trust on the node side).

Headers, including ``Authorization``, are passed through and never logged;
the router adds no key. The listener binds loopback unless
``allow_public=True`` (same rule as :class:`tunnel.Relay`).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .config import ServingConfig
from .gateway import HOP_BY_HOP, _GatewayHandler, guard_request_body, load_bad_words
from .rewrite import Rewriter, load_rules
from .state import DEFAULT_HEARTBEAT_TIMEOUT_S, heartbeat_age_s, read_json
from .tunnel import _check_bind

DEFAULT_ROUTER_PORT = 18_000
BALANCE_MODES = ("prefer", "least_inflight")


@dataclass(frozen=True)
class CatalogEntry:
    """``models.json`` metadata for the deployment in state dir ``name``."""

    name: str
    model_id: str
    display_name: str = ""
    gpus: str = ""
    port: int | None = None
    served_model_name: str = ""


@dataclass
class Guard:
    """Model-specific guards, the same objects the node gateway uses."""

    model: str
    bad_words: list[str] = field(default_factory=list)
    rewriter: Rewriter | None = None
    sources: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "bad_words": len(self.bad_words),
            "rewrite_rules": len(self.rewriter.rules) if self.rewriter else 0,
            "shellguard": bool(self.rewriter and self.rewriter.shellguard),
            "privguard": bool(self.rewriter and self.rewriter.privguard),
            "sources": dict(self.sources),
        }


def load_catalog(
    models_json: str | Path, root: str | Path, *, rewrite_log_dir: str | Path | None = None
) -> tuple[list[CatalogEntry], dict[str, Guard]]:
    """Read ``models.json``: deployment metadata and per-model guards.

    Entries whose ``config`` is a JSON serving config map to the state
    directory named by its ``serving.model``. ``guards`` paths are relative to
    ``root`` (the checkout). The first entry naming a served model defines its
    guards.
    """
    root = Path(root)
    doc = json.loads(Path(models_json).read_text(encoding="utf-8"))
    entries: list[CatalogEntry] = []
    guards: dict[str, Guard] = {}
    for model_id, meta in (doc.get("models") or {}).items():
        if not isinstance(meta, dict):
            continue
        config = meta.get("config")
        name, port = model_id, None
        if isinstance(config, str) and config.endswith(".json") and (root / config).is_file():
            cfg = ServingConfig.from_mapping(json.loads((root / config).read_text(encoding="utf-8")))
            name, port = cfg.model, cfg.port
        served = str(meta.get("served_model_name") or "")
        entries.append(CatalogEntry(name=name, model_id=model_id,
                                    display_name=str(meta.get("display_name") or ""),
                                    gpus=str(meta.get("gpus") or ""), port=port,
                                    served_model_name=served))
        spec = meta.get("guards")
        if isinstance(spec, dict) and served and served not in guards:
            guards[served] = build_guard(served, spec, root, rewrite_log_dir)
    return entries, guards


def build_guard(model: str, spec: dict[str, Any], root: str | Path,
                rewrite_log_dir: str | Path | None = None) -> Guard:
    """Guards as ``serve_with_gateway.sh`` configures them on the node."""
    root = Path(root)
    guard = Guard(model=model)
    if spec.get("bad_words_file"):
        path = root / str(spec["bad_words_file"])
        guard.bad_words = load_bad_words(str(path))
        guard.sources["bad_words_file"] = str(path)
    if spec.get("rewrite_rules"):
        path = root / str(spec["rewrite_rules"])
        log = None
        if rewrite_log_dir:
            Path(rewrite_log_dir).mkdir(parents=True, exist_ok=True)
            log = str(Path(rewrite_log_dir) / f"rewrites-{model}.jsonl")
        guard.rewriter = Rewriter(
            load_rules(str(path)), log_path=log,
            history_mode=str(spec.get("history_mode") or "drop"),
            shellguard=bool(spec.get("shellguard", True)),
            privguard=bool(spec.get("privguard", True)),
            allow_hosts=tuple(spec.get("allow_hosts") or ()),
        )
        guard.sources["rewrite_rules"] = str(path)
    return guard


class Router:
    """Discovery, readiness, model routing and failover bookkeeping."""

    def __init__(
        self,
        serve_root: str | Path,
        *,
        catalog: list[CatalogEntry] | None = None,
        guards: dict[str, Guard] | None = None,
        prefer: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        cooldown_s: float = 30.0,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 3600.0,
        state_ttl_s: float = 2.0,
        models_ttl_s: float = 60.0,
        balance: str = "prefer",
        clock: Callable[[], float] = time.time,
    ) -> None:
        if balance not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {BALANCE_MODES}, got {balance!r}")
        self.balance = balance
        self.serve_root = Path(serve_root)
        self.catalog = {entry.name: entry for entry in (catalog or [])}
        self._catalog_order = [entry.name for entry in (catalog or [])]
        self.guards = dict(guards or {})
        self.prefer = list(prefer or [])
        self.aliases = dict(aliases or {})
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.cooldown_s = float(cooldown_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self.state_ttl_s = float(state_ttl_s)
        self.models_ttl_s = float(models_ttl_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        self._state_cache: tuple[float, dict[str, dict[str, Any]]] | None = None
        self._models_cache: dict[str, tuple[float, tuple[Any, ...], list[dict[str, Any]]]] = {}
        self.counters: dict[str, int] = {}
        self._inflight: dict[str, int] = {}
        self._refreshing: set[str] = set()
        self.failovers = 0

    # -- discovery -------------------------------------------------------------
    def _rank(self, name: str) -> tuple[int, int, str]:
        p = self.prefer.index(name) if name in self.prefer else len(self.prefer)
        c = self._catalog_order.index(name) if name in self._catalog_order else len(self._catalog_order)
        return (p, c, name)

    def states(self) -> dict[str, dict[str, Any]]:
        """``{name: state}`` for every state dir, in preference order."""
        now = self.clock()
        with self._lock:
            cached = self._state_cache
            if cached is not None and now - cached[0] < self.state_ttl_s:
                return cached[1]
        found: dict[str, dict[str, Any]] = {}
        if self.serve_root.is_dir():
            for path in self.serve_root.iterdir():
                if (path / "state.json").is_file():
                    found[path.name] = read_json(path / "state.json") or {}
        ordered = {name: found[name] for name in sorted(found, key=self._rank)}
        with self._lock:
            self._state_cache = (now, ordered)
        return ordered

    def _port(self, name: str, state: dict[str, Any]) -> int | None:
        endpoint = state.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            try:
                port = urlsplit(endpoint).port
            except ValueError:
                port = None
            if port:
                return port
        entry = self.catalog.get(name)
        return entry.port if entry else None

    def describe(self, name: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        if state is None:
            state = self.states().get(name) or {}
        now = self.clock()
        node = str(state.get("node") or "")
        port = self._port(name, state)
        age = heartbeat_age_s(state.get("last_heartbeat"), now)
        entry = self.catalog.get(name)
        info: dict[str, Any] = {
            "name": name,
            "catalog": entry.model_id if entry else None,
            "display_name": entry.display_name if entry else None,
            "gpus": entry.gpus if entry else None,
            "state": state.get("state"),
            "job_id": state.get("job_id"),
            "node": node or None,
            "port": port,
            "heartbeat_age_s": None if age is None else round(age, 1),
            "ready": False,
            "reason": "",
            "models": [],
        }
        cooldown = self._cooldown_until.get(name, 0.0) - now
        if not state:
            info["reason"] = "no state file"
        elif state.get("state") != "ready":
            info["reason"] = f"state={state.get('state')}"
        elif not node:
            info["reason"] = "no node"
        elif not port:
            info["reason"] = "no port"
        elif age is None or age > self.heartbeat_timeout_s:
            info["reason"] = "stale heartbeat"
        elif cooldown > 0:
            info["reason"] = f"connection failed; cooldown {cooldown:.0f}s"
        else:
            info["ready"] = True
            info["models"] = [m["id"] for m in self.served_models(name, node, port, state)]
        return info

    def served_models(self, name: str, node: str, port: int, state: dict[str, Any]) -> list[dict[str, Any]]:
        """The deployment's ``/v1/models`` entries (cached per job/node/port)."""
        key = (state.get("job_id"), node, port)
        now = self.clock()
        cached = self._models_cache.get(name)
        if cached is not None and cached[1] == key and now - cached[0] < self.models_ttl_s:
            return cached[2]
        if cached is not None and cached[1] == key:
            # Stale but for the same job/node/port: answer from it and refresh in the
            # background, so /health and request routing never wait on a busy backend.
            self._refresh_models_async(name, key, node, port)
            return cached[2]
        data = self._probe_models(node, port)
        if data:
            self._models_cache[name] = (now, key, data)
            return data
        entry = self.catalog.get(name)
        fallback = (entry.served_model_name if entry else "") or name
        return [{"id": fallback, "object": "model", "owned_by": "slurmhelps"}]

    def _refresh_models_async(self, name: str, key: tuple[Any, ...], node: str, port: int) -> None:
        with self._lock:
            if name in self._refreshing:
                return
            self._refreshing.add(name)

        def work() -> None:
            try:
                data = self._probe_models(node, port)
                if data:
                    self._models_cache[name] = (self.clock(), key, data)
                else:  # keep serving the old list; retry on the next lookup after a short pause
                    old = self._models_cache.get(name)
                    if old is not None:
                        self._models_cache[name] = (self.clock() - self.models_ttl_s + 5.0, old[1], old[2])
            finally:
                with self._lock:
                    self._refreshing.discard(name)

        threading.Thread(target=work, name=f"models-refresh-{name}", daemon=True).start()

    def _probe_models(self, node: str, port: int) -> list[dict[str, Any]] | None:
        conn = http.client.HTTPConnection(node, port, timeout=min(self.connect_timeout_s, 5.0))
        try:
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            if resp.status != 200:
                return None
            doc = json.loads(resp.read())
        except (OSError, http.client.HTTPException, ValueError):
            return None
        finally:
            conn.close()
        data = doc.get("data") if isinstance(doc, dict) else None
        if not isinstance(data, list):
            return None
        return [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]

    def ready(self) -> list[dict[str, Any]]:
        return [info for info in (self.describe(n, s) for n, s in self.states().items()) if info["ready"]]

    def resolve(self, model: str | None) -> str | None:
        return self.aliases.get(model, model) if model else model

    def candidates(self, model: str | None) -> list[dict[str, Any]]:
        """Ready deployments (preference order) serving ``model`` (any, if None)."""
        ready = self.ready()
        if model is None:
            return ready
        return [info for info in ready if model in info["models"]]

    def guard_for(self, model: str | None, info: dict[str, Any]) -> Guard | None:
        """The guard of the requested model, else of the deployment's first guarded model."""
        if model and model in self.guards:
            return self.guards[model]
        for served in info.get("models", []):
            if served in self.guards:
                return self.guards[served]
        return None

    def order(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Try order for one request. ``prefer``: preference order (first ready
        wins, the rest are failover). ``least_inflight``: fewest requests in
        flight first, preference order breaking ties -- every ready deployment
        takes load, and failover still walks the rest of the list."""
        if self.balance != "least_inflight":
            return list(candidates)
        with self._lock:
            load = dict(self._inflight)
        ranked = sorted(enumerate(candidates), key=lambda pair: (load.get(pair[1]["name"], 0), pair[0]))
        return [info for _, info in ranked]

    def acquire(self, name: str) -> None:
        with self._lock:
            self._inflight[name] = self._inflight.get(name, 0) + 1

    def release(self, name: str) -> None:
        with self._lock:
            left = self._inflight.get(name, 0) - 1
            if left > 0:
                self._inflight[name] = left
            else:
                self._inflight.pop(name, None)

    def mark_failed(self, name: str) -> None:
        with self._lock:
            self._cooldown_until[name] = self.clock() + self.cooldown_s
            self.failovers += 1

    def mark_ok(self, name: str) -> None:
        with self._lock:
            self._cooldown_until.pop(name, None)
            self.counters[name] = self.counters.get(name, 0) + 1

    def models_union(self) -> list[dict[str, Any]]:
        union: dict[str, dict[str, Any]] = {}
        states = self.states()
        for info in self.ready():
            for model in self.served_models(info["name"], info["node"], info["port"],
                                            states.get(info["name"]) or {}):
                entry = union.setdefault(model["id"], {**model, "x_slurmhelps_deployments": []})
                entry["x_slurmhelps_deployments"].append(info["name"])
        return list(union.values())

    def status(self) -> dict[str, Any]:
        infos = [self.describe(n, s) for n, s in self.states().items()]
        ready = [i for i in infos if i["ready"]]
        return {
            "router": "slurmhelps-router",
            "serve_root": str(self.serve_root),
            "status": "ok" if ready else "no ready deployment",
            "active": ready[0]["name"] if ready else None,
            "models": sorted({m for i in ready for m in i["models"]}),
            "deployments": infos,
            "guards": {model: guard.describe() for model, guard in self.guards.items()},
            "aliases": dict(self.aliases),
            "prefer": list(self.prefer),
            "balance": self.balance,
            "inflight": dict(self._inflight),
            "requests": dict(self.counters),
            "failovers": self.failovers,
        }


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # a fleet of agents connects at once


class RouterServer:
    """Threaded loopback HTTP server around a :class:`Router`."""

    def __init__(
        self,
        router: Router,
        *,
        listen_host: str = "127.0.0.1",
        listen_port: int = DEFAULT_ROUTER_PORT,
        allow_public: bool = False,
    ) -> None:
        self.router = router
        self.listen_host = _check_bind(listen_host, allow_public)
        self.listen_port = int(listen_port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self.listen_port
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.listen_host}:{self.port}"

    def start(self) -> "RouterServer":
        if self._server is not None:
            return self
        server = _HTTPServer((self.listen_host, self.listen_port), _RouterHandler)
        server.router = self.router  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="slurmhelps-router", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "RouterServer":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


class _UpstreamFailed(Exception):
    """The upstream failed before anything was sent to the client."""


class _RouterHandler(_GatewayHandler):
    """Reuses the gateway handler's response relays (plain, JSON and SSE
    rewrite) and its no-header-logging policy; only dispatch differs."""

    _deployment: str = ""

    @property
    def router(self) -> Router:
        return getattr(self.server, "router")

    def end_headers(self) -> None:
        if self._deployment:
            self.send_header("X-Slurmhelps-Deployment", self._deployment)
        super().end_headers()

    def _handle(self) -> None:
        router = self.router
        self._headers_sent = False
        self._deployment = ""
        request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/router/status":
            self._send_json(200, router.status(), request_id)
            return
        if path == "/health":
            status = router.status()
            self._send_json(200 if status["active"] else 503,
                            {"status": status["status"], "models": status["models"],
                             "ready": [d["name"] for d in status["deployments"] if d["ready"]]},
                            request_id)
            return
        body = self._read_body()
        if path in ("/v1/models", "/models") and self.command == "GET":
            data = router.models_union()
            if not data:
                self._unavailable(request_id, "no ready deployment")
                return
            self._send_json(200, {"object": "list", "data": data}, request_id)
            return

        requested = _body_model(body)
        model = router.resolve(requested)
        if model is not None and model != requested:
            body = _set_body_model(body, model)
        candidates = router.candidates(model)
        if not candidates:
            if model is not None and router.ready():
                self._send_json(404, {"error": {
                    "message": f"model {requested!r} is not served by any ready deployment",
                    "type": "invalid_request_error", "code": "model_not_found",
                    "available": router.status()["models"]}}, request_id)
            else:
                self._unavailable(request_id, "no ready deployment")
            return
        completion = self.command == "POST" and path.endswith("/completions")
        tried: list[str] = []
        for info in router.order(candidates):
            tried.append(info["name"])
            guard = router.guard_for(model, info) if completion else None
            out = body
            if guard is not None:
                out, _ = guard_request_body(body, guard.bad_words, guard.rewriter, request_id)
            router.acquire(info["name"])
            try:
                self._proxy_to(info, out, request_id, guard)
            except _UpstreamFailed:
                router.mark_failed(info["name"])
                continue
            finally:
                router.release(info["name"])
            router.mark_ok(info["name"])
            return
        self._unavailable(request_id, f"no ready deployment answered (tried {', '.join(tried)})")

    def _unavailable(self, request_id: str, message: str) -> None:
        self._send_json(503, {"status": "unavailable",
                              "error": {"message": message, "type": "router_error", "code": 503},
                              "deployments": self.router.status()["deployments"]},
                        request_id, extra_headers={"Retry-After": "30"})

    def _proxy_to(self, info: dict[str, Any], body: bytes, request_id: str, guard: Guard | None) -> None:
        router = self.router
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP or lower in ("host", "content-length", "expect", "x-request-id"):
                continue
            headers[key] = value
        headers["X-Request-Id"] = request_id
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(info["node"], info["port"], timeout=router.connect_timeout_s)
        upstream: http.client.HTTPResponse | None = None
        try:
            try:
                conn.connect()
                if conn.sock is not None:
                    conn.sock.settimeout(router.read_timeout_s)
                conn.request(self.command, self.path, body=body or None, headers=headers)
                upstream = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise _UpstreamFailed(str(exc)) from exc
            # Response headers go out from here on: no failover past this point.
            self._deployment = info["name"]
            try:
                if guard is not None and guard.rewriter is not None:
                    self._relay_rewritten(upstream, request_id, guard.rewriter)
                else:
                    self._relay_response(upstream, request_id, body_allowed=self.command != "HEAD")
            except (OSError, http.client.HTTPException):
                self.close_connection = True
        finally:
            if upstream is not None:
                upstream.close()
            conn.close()


def _body_model(body: bytes) -> str | None:
    if not body:
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    model = doc.get("model") if isinstance(doc, dict) else None
    return model if isinstance(model, str) and model else None


def _set_body_model(body: bytes, model: str) -> bytes:
    doc = json.loads(body)
    doc["model"] = model
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


__all__ = [
    "CatalogEntry",
    "DEFAULT_ROUTER_PORT",
    "Guard",
    "Router",
    "RouterServer",
    "build_guard",
    "load_catalog",
]
