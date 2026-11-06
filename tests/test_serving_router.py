"""Login-node router tests: mock backends, fake state files, loopback only."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from slurmhelps.serving import cli
from slurmhelps.serving.mock import MockOpenAIServer
from slurmhelps.serving.router import Deployment, Router, RouterServer
from slurmhelps.serving.state import atomic_write_json, utc_now


def _write_state(directory: Path, state: str, node: str = "127.0.0.1", **extra: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    doc = {"state": state, "node": node, "job_id": "1", "last_heartbeat": utc_now()}
    doc.update(extra)
    atomic_write_json(directory / "state.json", doc)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


@pytest.fixture
def backends():
    servers: list[MockOpenAIServer] = []

    def make(model: str) -> MockOpenAIServer:
        server = MockOpenAIServer(model=model, prefill_delay=0.0, token_delay=0.0).start()
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def _router(tmp_path: Path, ports: dict[str, int], **kwargs: Any) -> tuple[Router, RouterServer]:
    deps = [Deployment(name, tmp_path / name, port) for name, port in ports.items()]
    router = Router(deps, state_ttl_s=0.0, **kwargs)
    server = RouterServer(router, listen_port=0).start()
    return router, server


def test_no_ready_deployment_is_a_clean_503(tmp_path):
    _write_state(tmp_path / "bw", "queued")
    router, server = _router(tmp_path, {"bw": _free_port(), "hopper": _free_port()})
    try:
        status, headers, body = _request(server.base_url + "/health")
        assert status == 503
        doc = json.loads(body)
        assert doc["error"]["message"] == "no ready deployment"
        reasons = {d["name"]: d["reason"] for d in doc["deployments"]}
        assert reasons == {"bw": "state=queued", "hopper": "no state file"}
        assert headers.get("Retry-After") == "30"

        status, _, body = _request(server.base_url + "/router/status")
        assert status == 200
        assert json.loads(body)["status"] == "no ready deployment"
    finally:
        server.stop()


def test_prefers_first_ready_and_forwards_models_health_chat(tmp_path, backends):
    bw, hopper = backends("bw-model"), backends("hopper-model")
    _write_state(tmp_path / "bw", "ready")
    _write_state(tmp_path / "hopper", "ready")
    _, server = _router(tmp_path, {"bw": bw.port, "hopper": hopper.port})
    try:
        status, headers, body = _request(server.base_url + "/v1/models")
        assert status == 200
        assert headers["X-Slurmhelps-Deployment"] == "bw"
        assert json.loads(body)["data"][0]["id"] == "bw-model"

        status, headers, _ = _request(server.base_url + "/health")
        assert status == 200 and headers["X-Slurmhelps-Deployment"] == "bw"

        status, headers, body = _request(
            server.base_url + "/v1/chat/completions",
            {"model": "bw-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            headers={"Authorization": "Bearer passthrough"},
        )
        assert status == 200
        assert json.loads(body)["choices"]
    finally:
        server.stop()


def test_falls_back_when_preferred_not_ready_or_stale(tmp_path, backends):
    hopper = backends("hopper-model")
    _write_state(tmp_path / "bw", "ready", last_heartbeat="2020-01-01T00:00:00Z")
    _write_state(tmp_path / "hopper", "ready")
    _, server = _router(tmp_path, {"bw": _free_port(), "hopper": hopper.port})
    try:
        status, headers, _ = _request(server.base_url + "/v1/models")
        assert status == 200 and headers["X-Slurmhelps-Deployment"] == "hopper"
        _, _, body = _request(server.base_url + "/router/status")
        bw = json.loads(body)["deployments"][0]
        assert bw["reason"] == "stale heartbeat"
    finally:
        server.stop()


def test_connection_failure_fails_over_and_cools_down(tmp_path, backends):
    hopper = backends("hopper-model")
    _write_state(tmp_path / "bw", "ready")  # claims ready, but nothing listens
    _write_state(tmp_path / "hopper", "ready")
    router, server = _router(tmp_path, {"bw": _free_port(), "hopper": hopper.port}, cooldown_s=60)
    try:
        status, headers, _ = _request(server.base_url + "/v1/models")
        assert status == 200 and headers["X-Slurmhelps-Deployment"] == "hopper"
        assert router.failovers == 1
        info = router.describe(router.deployments[0])
        assert not info["ready"] and info["reason"].startswith("connection failed")
        # Cooldown: the next request goes straight to hopper, no new failover.
        _request(server.base_url + "/v1/models")
        assert router.failovers == 1
        assert router.counters == {"bw": 0, "hopper": 2}
    finally:
        server.stop()


def test_failover_follows_state_changes(tmp_path, backends):
    bw, hopper = backends("bw-model"), backends("hopper-model")
    _write_state(tmp_path / "bw", "ready")
    _write_state(tmp_path / "hopper", "ready")
    _, server = _router(tmp_path, {"bw": bw.port, "hopper": hopper.port})
    try:
        assert _request(server.base_url + "/v1/models")[1]["X-Slurmhelps-Deployment"] == "bw"
        _write_state(tmp_path / "bw", "preempted")
        assert _request(server.base_url + "/v1/models")[1]["X-Slurmhelps-Deployment"] == "hopper"
        _write_state(tmp_path / "hopper", "draining")
        assert _request(server.base_url + "/v1/models")[0] == 503
        _write_state(tmp_path / "bw", "ready")
        assert _request(server.base_url + "/v1/models")[1]["X-Slurmhelps-Deployment"] == "bw"
    finally:
        server.stop()


def test_streaming_sse_is_relayed(tmp_path, backends):
    bw = backends("bw-model")
    _write_state(tmp_path / "bw", "ready")
    _, server = _router(tmp_path, {"bw": bw.port})
    try:
        status, headers, body = _request(
            server.base_url + "/v1/chat/completions",
            {"model": "bw-model", "stream": True, "max_tokens": 8,
             "messages": [{"role": "user", "content": "stream please"}]},
        )
        assert status == 200
        assert headers["Content-Type"].startswith("text/event-stream")
        events = [line for line in body.decode().splitlines() if line.startswith("data: ")]
        assert len(events) >= 2
        assert events[-1] == "data: [DONE]"
        assert json.loads(events[0][len("data: "):])["object"] == "chat.completion.chunk"
    finally:
        server.stop()


def test_router_refuses_public_bind(tmp_path):
    router = Router([Deployment("bw", tmp_path / "bw", 1)])
    with pytest.raises(ValueError):
        RouterServer(router, listen_host="0.0.0.0")
    RouterServer(router, listen_host="0.0.0.0", allow_public=True)  # explicit opt-in only


def test_router_rejects_bad_deployment_lists(tmp_path):
    with pytest.raises(ValueError):
        Router([])
    with pytest.raises(ValueError):
        Router([Deployment("a", tmp_path, 1), Deployment("a", tmp_path, 2)])


def test_cli_default_deployments_are_bw_then_hopper(tmp_path):
    deps = cli.router_deployments([], tmp_path)
    assert [(d.name, d.port) for d in deps] == [("bw", 8010), ("hopper", 8020)]
    assert deps[0].state_dir == tmp_path / "qwen38-flash-next"
    assert deps[1].state_dir == tmp_path / "qwen38-flash-next-hopper"


def test_cli_deployment_specs_set_preference_order(tmp_path):
    root = Path(cli.ROOT) / "llm" / "qwen38"
    deps = cli.router_deployments(
        [f"hopper={root / 'config-hopper.json'}", f"bw={root / 'config-bw.json'}"], tmp_path
    )
    assert [d.name for d in deps] == ["hopper", "bw"]
    with pytest.raises(SystemExit):
        cli.router_deployments(["nonsense"], tmp_path)


def test_cli_router_status_reports_unreachable(capsys):
    rc = cli.main(["router", "--status", "--listen-port", str(_free_port())])
    assert rc == 3
    assert "router not answering" in capsys.readouterr().err
