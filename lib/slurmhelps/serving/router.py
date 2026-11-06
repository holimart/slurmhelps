#!/usr/bin/env python3
"""Loopback HTTP router over several serving deployments (stdlib only).

The router runs on the login node and gives clients one stable port.  Each
request goes to the first deployment, in preference order, whose slurmhelps
state file says ``ready`` with a fresh heartbeat and a node; the request is
proxied to ``http://<node>:<port>``.  A deployment that refuses a connection
(or fails before any response byte reached the client) is put on a short
cooldown and the next ready deployment is tried.

* Streaming (SSE) responses are relayed chunk by chunk.
* ``/health``, ``/v1/models`` and every other path are forwarded unchanged.
  With no ready deployment the router answers ``503`` with
  ``"no ready deployment"`` instead of hanging.
* ``/router/status`` is answered by the router itself.
* Headers, including ``Authorization``, are passed through; the router adds
  no key of its own and never logs headers.
* The listener binds loopback unless ``allow_public=True`` is explicit (same
  rule as :class:`slurmhelps.serving.tunnel.Relay`).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .state import DEFAULT_HEARTBEAT_TIMEOUT_S, heartbeat_age_s, read_json
from .tunnel import _check_bind

DEFAULT_ROUTER_PORT = 18_000

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass(frozen=True)
class Deployment:
    """One serving deployment: its state directory and node-side port."""

    name: str
    state_dir: Path
    port: int
    host: str = ""  # fixed host override (tests); default is the state's node

    def read_state(self) -> dict[str, Any] | None:
        return read_json(Path(self.state_dir) / "state.json")


class Router:
    """Pick-and-proxy logic, independent of the HTTP server."""

    def __init__(
        self,
        deployments: list[Deployment],
        *,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        cooldown_s: float = 30.0,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 3600.0,
        state_ttl_s: float = 2.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not deployments:
            raise ValueError("router needs at least one deployment")
        names = [d.name for d in deployments]
        if len(set(names)) != len(names):
            raise ValueError("deployment names must be unique")
        self.deployments = list(deployments)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.cooldown_s = float(cooldown_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self.state_ttl_s = float(state_ttl_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self.counters: dict[str, int] = {d.name: 0 for d in deployments}
        self.failovers = 0

    # -- state ---------------------------------------------------------------
    def _state(self, dep: Deployment) -> dict[str, Any] | None:
        now = self.clock()
        with self._lock:
            cached = self._cache.get(dep.name)
            if cached is not None and now - cached[0] < self.state_ttl_s:
                return cached[1]
        state = dep.read_state()
        with self._lock:
            self._cache[dep.name] = (now, state)
        return state

    def describe(self, dep: Deployment) -> dict[str, Any]:
        """Readiness verdict for one deployment, with the reason when not ready."""
        state = self._state(dep) or {}
        now = self.clock()
        node = dep.host or state.get("node") or ""
        age = heartbeat_age_s(state.get("last_heartbeat"), now)
        info: dict[str, Any] = {
            "name": dep.name,
            "state": state.get("state"),
            "job_id": state.get("job_id"),
            "node": node or None,
            "port": dep.port,
            "heartbeat_age_s": None if age is None else round(age, 1),
            "ready": False,
            "reason": "",
        }
        cooldown = self._cooldown_until.get(dep.name, 0.0) - now
        if not state:
            info["reason"] = "no state file"
        elif state.get("state") != "ready":
            info["reason"] = f"state={state.get('state')}"
        elif not node:
            info["reason"] = "no node"
        elif age is None or age > self.heartbeat_timeout_s:
            info["reason"] = "stale heartbeat"
        elif cooldown > 0:
            info["reason"] = f"connection failed; cooldown {cooldown:.0f}s"
        else:
            info["ready"] = True
        return info

    def candidates(self) -> list[tuple[Deployment, str]]:
        """Ready deployments in preference order as ``(deployment, host)``."""
        out = []
        for dep in self.deployments:
            info = self.describe(dep)
            if info["ready"]:
                out.append((dep, str(info["node"])))
        return out

    def mark_failed(self, dep: Deployment) -> None:
        with self._lock:
            self._cooldown_until[dep.name] = self.clock() + self.cooldown_s
            self.failovers += 1

    def mark_ok(self, dep: Deployment) -> None:
        with self._lock:
            self._cooldown_until.pop(dep.name, None)
            self.counters[dep.name] = self.counters.get(dep.name, 0) + 1

    def status(self) -> dict[str, Any]:
        infos = [self.describe(dep) for dep in self.deployments]
        active = next((i["name"] for i in infos if i["ready"]), None)
        return {
            "router": "slurmhelps-router",
            "active": active,
            "status": "ok" if active else "no ready deployment",
            "deployments": infos,
            "requests": dict(self.counters),
            "failovers": self.failovers,
        }


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
        server = ThreadingHTTPServer((self.listen_host, self.listen_port), _RouterHandler)
        server.daemon_threads = True
        server.router = self.router  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="slurmhelps-router", daemon=True
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.start()
        assert self._thread is not None
        self._thread.join()

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


class _RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def router(self) -> Router:
        return getattr(self.server, "router")

    def log_message(self, *_args: Any) -> None:
        # Never log headers: Authorization must not reach logs.
        return

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        router = self.router
        request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/router/status":
            self._send_json(200, router.status(), request_id)
            return
        body = self._read_body()
        tried: list[str] = []
        for dep, host in router.candidates():
            tried.append(dep.name)
            try:
                self._proxy(dep, host, body, request_id)
            except _UpstreamFailed:
                router.mark_failed(dep)
                continue
            router.mark_ok(dep)
            return
        status = router.status()
        message = (
            "no ready deployment"
            if not tried
            else f"no ready deployment answered (tried {', '.join(tried)})"
        )
        self._send_json(
            503,
            {
                "status": "unavailable",
                "error": {"message": message, "type": "router_error", "code": 503},
                "deployments": status["deployments"],
            },
            request_id,
            extra_headers={"Retry-After": "30"},
        )

    def _proxy(self, dep: Deployment, host: str, body: bytes, request_id: str) -> None:
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
        conn = http.client.HTTPConnection(host, dep.port, timeout=router.connect_timeout_s)
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
            self._relay(upstream, dep, request_id)
        finally:
            if upstream is not None:
                upstream.close()
            conn.close()

    def _relay(self, upstream: http.client.HTTPResponse, dep: Deployment, request_id: str) -> None:
        content_length = upstream.getheader("Content-Length")
        self.send_response(upstream.status)
        for key, value in upstream.getheaders():
            lower = key.lower()
            if lower in HOP_BY_HOP or lower in ("content-length", "server", "date", "x-request-id"):
                continue
            self.send_header(key, value)
        if content_length is not None:
            self.send_header("Content-Length", content_length)
        self.send_header("X-Request-Id", request_id)
        self.send_header("X-Slurmhelps-Deployment", dep.name)
        if content_length is None:
            # Unframed (typically SSE): EOF ends the response.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command == "HEAD":
            return
        # Response headers are out: a failure from here on cannot fail over.
        try:
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            self.close_connection = True

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        request_id: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", request_id)
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


__all__ = ["DEFAULT_ROUTER_PORT", "Deployment", "Router", "RouterServer"]
