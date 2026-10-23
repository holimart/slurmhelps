#!/usr/bin/env python3
"""SSH tunnel argv and a loopback-only TCP relay (stdlib only).

Two ways to reach a compute-node server without opening a public port:

* :func:`tunnel_argv` builds ``ssh -N -L local:node:remote node`` (optionally
  with ``-J login``);
* :class:`Relay` is a dumb byte relay for compute nodes reachable only from
  the login node.  It binds loopback unless ``allow_public=True`` is explicit.
"""

from __future__ import annotations

import ipaddress
import select
import socket
import socketserver
import threading
from typing import Any

LOOPBACK = "127.0.0.1"


def _valid_port(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be an integer in 1..65535")
    return value


def _check_bind(host: str, allow_public: bool) -> str:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("listen host must be non-empty")
    host = host.strip()
    if allow_public:
        return host
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError("relay must bind loopback unless allow_public=True")


def _destination(node: str, ssh_user_host: str) -> str:
    """Destination host for ssh.

    ``ssh_user_host`` is an optional ``user`` prefix (``alice`` ->
    ``alice@node``) or a complete ``user@host`` override, which is useful when
    the compute node is only resolvable from the login host.
    """
    if not ssh_user_host:
        return node
    if "@" in ssh_user_host:
        return ssh_user_host
    return f"{ssh_user_host}@{node}"


def tunnel_argv(
    node: str,
    remote_port: int,
    local_port: int,
    via: str | None = None,
    ssh_user_host: str = "",
) -> list[str]:
    """Build the local ``ssh -L`` tunnel command.

    No public listener is created: the local side is loopback by default and
    the remote side is the compute node's loopback-bound server.
    """
    if not isinstance(node, str) or not node.strip():
        raise ValueError("node must be non-empty")
    remote_port = _valid_port("remote_port", remote_port)
    local_port = _valid_port("local_port", local_port)

    argv = ["ssh", "-N"]
    if via:
        argv += ["-J", str(via)]
    argv += ["-L", f"{local_port}:{node}:{remote_port}"]
    argv += [
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        _destination(node, ssh_user_host),
    ]
    return argv


class _RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        target: tuple[str, int],
        stop_event: threading.Event,
    ) -> None:
        self.target = target
        self.stop_event = stop_event
        self.connections = 0
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _RelayHandler)


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: _RelayServer = self.server  # type: ignore[assignment]
        if server.stop_event.is_set():
            return
        try:
            upstream = socket.create_connection(server.target, timeout=10)
        except OSError:
            return
        server.connections += 1
        client = self.request
        client.settimeout(0.5)
        upstream.settimeout(0.5)
        try:
            while not server.stop_event.is_set():
                try:
                    readable, _, _ = select.select([client, upstream], [], [], 0.25)
                except (OSError, ValueError):
                    return
                for sock in readable:
                    try:
                        data = sock.recv(65536)
                    except (OSError, socket.timeout):
                        return
                    if not data:
                        return
                    other = upstream if sock is client else client
                    try:
                        other.sendall(data)
                    except OSError:
                        return
        finally:
            try:
                upstream.close()
            except OSError:
                pass


class Relay:
    """Threaded raw-TCP relay, loopback-only by default.

    ``Relay(target_host, target_port, listen_host="127.0.0.1",
    listen_port=18000)``.  :meth:`start` returns the bound ``(host, port)``;
    passing ``listen_port=0`` asks the kernel for a free test port.
    """

    def __init__(
        self,
        target_host: str,
        target_port: int,
        *,
        listen_host: str = LOOPBACK,
        listen_port: int = 18_000,
        allow_public: bool = False,
    ) -> None:
        if not isinstance(target_host, str) or not target_host.strip():
            raise ValueError("target_host must be non-empty")
        self.target_host = target_host.strip()
        self.target_port = _valid_port("target_port", target_port)
        self.listen_host = _check_bind(listen_host, allow_public)
        self.listen_port = int(listen_port)
        if not 0 <= self.listen_port <= 65535:
            raise ValueError("listen_port must be in 0..65535")
        self.allow_public = bool(allow_public)
        self._stop = threading.Event()
        self._server: _RelayServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return (self.listen_host, self.listen_port)
        host, port = self._server.server_address[:2]
        return (str(host), int(port))

    @property
    def port(self) -> int:
        return self.address[1]

    @property
    def connections(self) -> int:
        return self._server.connections if self._server is not None else 0

    def start(self) -> tuple[str, int]:
        """Bind and start serving on a daemon thread (idempotent)."""
        if self._server is not None:
            return self.address
        self._stop.clear()
        server = _RelayServer(
            (self.listen_host, self.listen_port),
            (self.target_host, self.target_port),
            self._stop,
        )
        self._server = server
        self.listen_port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            name=f"slurmhelps-relay-{self.listen_port}",
            daemon=True,
        )
        self._thread.start()
        return self.address

    def stop(self) -> None:
        """Stop accepting and drop active relays (idempotent)."""
        self._stop.set()
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "Relay":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_relay(
    target_host: str,
    target_port: int,
    listen_host: str = LOOPBACK,
    listen_port: int = 18_000,
    *,
    allow_public: bool = False,
) -> Relay:
    """Create and start a :class:`Relay` in one call."""
    relay = Relay(
        target_host,
        target_port,
        listen_host=listen_host,
        listen_port=listen_port,
        allow_public=allow_public,
    )
    relay.start()
    return relay


# Conventional alias for callers that name the class by its server type.
TCPRelay = Relay

__all__ = ["LOOPBACK", "Relay", "TCPRelay", "start_relay", "tunnel_argv"]
