"""Tunnel argv and loopback relay tests (no real ssh, no public listener)."""

from __future__ import annotations

import socketserver
import threading

import pytest

from slurmhelps.serving.tunnel import Relay, start_relay, tunnel_argv


def test_direct_tunnel_argv_is_loopback_and_noninteractive():
    argv = tunnel_argv("ampere02", 8000, 8100)
    assert argv[0] == "ssh"
    assert "-N" in argv
    assert "BatchMode=yes" in argv
    assert "ExitOnForwardFailure=yes" in argv
    assert "ServerAliveInterval=15" in argv
    index = argv.index("-L")
    assert argv[index + 1] == "8100:ampere02:8000"
    assert argv[-1] == "ampere02"
    assert "-J" not in argv


def test_tunnel_with_proxyjump_and_user():
    argv = tunnel_argv("ampere02", 8000, 8000, via="me@login")
    jump = argv.index("-J")
    assert argv[jump + 1] == "me@login"
    forward = argv.index("-L")
    assert argv[forward + 1] == "8000:ampere02:8000"
    assert argv[-1] == "ampere02"

    user = tunnel_argv("ampere02", 8000, 8000, ssh_user_host="alice")
    assert user[-1] == "alice@ampere02"

    override = tunnel_argv("ampere02", 8000, 8000, ssh_user_host="alice@login")
    assert override[-1] == "alice@login"


@pytest.mark.parametrize(
    "node,remote,local",
    [("", 8000, 8000), ("n", 0, 8000), ("n", 8000, 0), ("n", 70000, 8000)],
)
def test_tunnel_rejects_bad_arguments(node, remote, local):
    with pytest.raises(ValueError):
        tunnel_argv(node, remote, local)


def test_relay_defaults_to_loopback_and_rejects_wildcard():
    relay = Relay("127.0.0.1", 8000, listen_port=0)
    assert relay.listen_host == "127.0.0.1"
    with pytest.raises(ValueError):
        Relay("127.0.0.1", 8000, listen_host="0.0.0.0")
    with pytest.raises(ValueError):
        Relay("127.0.0.1", 8000, listen_host="203.0.113.5")

    public = Relay("127.0.0.1", 8000, listen_host="0.0.0.0", allow_public=True)
    assert public.listen_host == "0.0.0.0"


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            data = self.request.recv(4096)
            if not data:
                return
            self.request.sendall(data)


class _EchoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture()
def echo_server():
    server = _EchoServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _recv_exact(sock, size: int) -> bytes:
    chunks = []
    while sum(len(chunk) for chunk in chunks) < size:
        chunk = sock.recv(size - sum(len(chunk) for chunk in chunks))
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def test_relay_forwards_raw_tcp_bytes(echo_server):
    echo_host, echo_port = echo_server
    relay = start_relay(echo_host, echo_port, listen_port=0)
    try:
        host, port = relay.address
        assert host == "127.0.0.1"
        assert port != 0
        client = __import__("socket").create_connection((host, port), timeout=5)
        try:
            payload = b"hello relay\x00\xff"
            client.sendall(payload)
            assert _recv_exact(client, len(payload)) == payload
            client.sendall(b"second")
            assert _recv_exact(client, 6) == b"second"
        finally:
            client.close()
        assert relay.connections >= 1
    finally:
        relay.stop()
    relay.stop()  # idempotent


def test_relay_context_manager_and_stop_closes_listener(echo_server):
    echo_host, echo_port = echo_server
    relay = Relay(echo_host, echo_port, listen_port=0)
    with relay as running:
        assert running.address[1] > 0
    # After stop the port is no longer accepting.
    import socket as socket_mod

    with pytest.raises(OSError):
        socket_mod.create_connection(running.address, timeout=0.5)
