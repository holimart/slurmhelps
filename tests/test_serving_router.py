"""Login-node router tests: fake backends, fake state dirs, loopback only."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from slurmhelps.serving import cli
from slurmhelps.serving.gateway import GatewayServer
from slurmhelps.serving.mock import MockOpenAIServer
from slurmhelps.serving.privguard import NOTICE
from slurmhelps.serving.router import CatalogEntry, Router, RouterServer, load_catalog
from slurmhelps.serving.state import atomic_write_json, utc_now

QWEN = "qwen38-flash-next"
DENIED = "https://bucket.oss-cn-x.aliyuncs.com/proxy?k=1"
KEY = "LTAI5tKoG9A3DkwGD635QVZr"


def _write_state(root: Path, name: str, state: str, port: int | None = None, **extra: Any) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    doc = {"state": state, "node": "127.0.0.1", "job_id": "1", "last_heartbeat": utc_now(),
           "endpoint": f"http://127.0.0.1:{port}/v1" if port else None}
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


class FakeBackend:
    """OpenAI-ish backend that serves ``model`` and answers with a denied URL
    (and the access key) in its text, as JSON or SSE. Records request bodies
    and headers."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a: Any) -> None:
                return

            def _json(self, code: int, doc: dict[str, Any]) -> None:
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/v1/models"):
                    self._json(200, {"object": "list", "data": [
                        {"id": backend.model, "object": "model", "owned_by": "vllm", "max_model_len": 1024}]})
                elif self.path.startswith("/health"):
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": "nope"})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                backend.bodies.append(body)
                backend.headers.append(dict(self.headers))
                text = f"use {DENIED} with {KEY} now"
                if not body.get("stream"):
                    self._json(200, {"id": "x", "object": "chat.completion", "model": backend.model,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": text}}]})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                pieces = [text[i:i + 7] for i in range(0, len(text), 7)]
                for i, piece in enumerate(pieces):
                    chunk = {"id": "x", "object": "chat.completion.chunk", "model": backend.model,
                             "choices": [{"index": 0, "delta": {"content": piece},
                                          "finish_reason": "stop" if i == len(pieces) - 1 else None}]}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def make_backend():
    made: list[FakeBackend] = []

    def make(model: str) -> FakeBackend:
        backend = FakeBackend(model)
        made.append(backend)
        return backend

    yield make
    for backend in made:
        backend.stop()


@pytest.fixture
def catalog(tmp_path):
    return load_catalog(cli.DEFAULT_MODELS_JSON, cli.ROOT, rewrite_log_dir=tmp_path / "logs")


@pytest.fixture
def start_router(tmp_path, catalog):
    servers: list[RouterServer] = []

    def start(guards: bool = True, **kwargs: Any) -> tuple[Router, RouterServer]:
        entries, guard_map = catalog
        router = Router(tmp_path / "serve", catalog=entries, guards=guard_map if guards else {},
                        state_ttl_s=0.0, **kwargs)
        server = RouterServer(router, listen_port=0).start()
        servers.append(server)
        return router, server

    yield start
    for server in servers:
        server.stop()


def _chat(model: str = QWEN, **extra: Any) -> dict[str, Any]:
    return {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], **extra}


def _sse_text(body: bytes) -> tuple[str, int]:
    text, events = "", 0
    for line in body.decode().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events += 1
            for choice in json.loads(line[6:]).get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
    return text, events


# -- discovery / readiness ------------------------------------------------------

def test_no_ready_deployment_is_a_clean_503(tmp_path, start_router):
    _write_state(tmp_path / "serve", QWEN, "queued")
    _, server = start_router()
    status, headers, body = _request(server.base_url + "/health")
    assert status == 503 and json.loads(body)["status"] == "no ready deployment"
    status, headers, body = _request(server.base_url + "/v1/chat/completions", _chat())
    assert status == 503 and json.loads(body)["error"]["message"] == "no ready deployment"
    assert headers.get("Retry-After") == "30"
    assert _request(server.base_url + "/v1/models")[0] == 503
    doc = json.loads(_request(server.base_url + "/router/status")[2])
    assert doc["status"] == "no ready deployment"
    assert [(d["name"], d["reason"]) for d in doc["deployments"]] == [(QWEN, "state=queued")]
    assert doc["guards"][QWEN]["bad_words"] > 0 and doc["guards"][QWEN]["rewrite_rules"] > 0


def test_discovers_unknown_extra_deployment(tmp_path, start_router, make_backend):
    extra = make_backend("mystery-model")
    _write_state(tmp_path / "serve", "mystery-dir", "ready", extra.port)
    (tmp_path / "serve" / "router").mkdir(parents=True)  # no state.json: not a deployment
    _, server = start_router()
    doc = json.loads(_request(server.base_url + "/router/status")[2])
    [dep] = doc["deployments"]
    assert dep["name"] == "mystery-dir" and dep["catalog"] is None and dep["ready"]
    assert dep["models"] == ["mystery-model"] and dep["port"] == extra.port
    status, headers, _ = _request(server.base_url + "/v1/chat/completions", _chat("mystery-model"))
    assert status == 200 and headers["X-Slurmhelps-Deployment"] == "mystery-dir"
    # Unguarded model: body passes through untouched.
    assert "bad_words" not in extra.bodies[-1]


def test_catalogue_maps_qwen38_state_dirs_ports_and_guards(catalog):
    entries, guards = catalog
    by_name = {e.name: e for e in entries}
    assert by_name[QWEN].port == 8010 and by_name[f"{QWEN}-hopper"].port == 8020
    assert by_name[f"{QWEN}-hopper"].served_model_name == QWEN
    assert list(guards) == [QWEN]
    assert KEY in guards[QWEN].bad_words and guards[QWEN].rewriter is not None


# -- model routing ----------------------------------------------------------------

def test_routes_by_model_with_preference_and_union_models(tmp_path, start_router, make_backend):
    bw, hopper, other = make_backend(QWEN), make_backend(QWEN), make_backend("glm-x")
    root = tmp_path / "serve"
    _write_state(root, f"{QWEN}-hopper", "ready", hopper.port)
    _write_state(root, QWEN, "ready", bw.port)
    _write_state(root, "glm-dir", "ready", other.port)
    router, server = start_router()
    assert list(router.states()) == [QWEN, f"{QWEN}-hopper", "glm-dir"]  # catalogue order, then unknown

    assert _request(server.base_url + "/v1/chat/completions", _chat())[1]["X-Slurmhelps-Deployment"] == QWEN
    assert _request(server.base_url + "/v1/chat/completions", _chat("glm-x"))[1][
        "X-Slurmhelps-Deployment"] == "glm-dir"
    status, _, body = _request(server.base_url + "/v1/chat/completions", _chat("nope"))
    assert status == 404 and json.loads(body)["error"]["code"] == "model_not_found"

    status, _, body = _request(server.base_url + "/v1/models")
    data = {m["id"]: m for m in json.loads(body)["data"]}
    assert status == 200 and sorted(data) == ["glm-x", QWEN]
    assert data[QWEN]["x_slurmhelps_deployments"] == [QWEN, f"{QWEN}-hopper"]
    assert data[QWEN]["max_model_len"] == 1024

    _write_state(root, QWEN, "preempted")
    assert _request(server.base_url + "/v1/chat/completions", _chat())[1][
        "X-Slurmhelps-Deployment"] == f"{QWEN}-hopper"


def test_prefer_and_alias(tmp_path, start_router, make_backend):
    bw, hopper = make_backend(QWEN), make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", bw.port)
    _write_state(tmp_path / "serve", f"{QWEN}-hopper", "ready", hopper.port)
    _, server = start_router(prefer=[f"{QWEN}-hopper"], aliases={"qwen38": QWEN})
    status, headers, _ = _request(server.base_url + "/v1/chat/completions", _chat("qwen38"))
    assert status == 200 and headers["X-Slurmhelps-Deployment"] == f"{QWEN}-hopper"
    assert hopper.bodies[-1]["model"] == QWEN  # alias rewritten to the served name


def test_connection_failure_fails_over_and_cools_down(tmp_path, start_router, make_backend):
    hopper = make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", _free_port())  # claims ready, nothing listens
    _write_state(tmp_path / "serve", f"{QWEN}-hopper", "ready", hopper.port)
    router, server = start_router(cooldown_s=60)
    # The dead deployment's model list falls back to the catalogue's served name.
    status, headers, _ = _request(server.base_url + "/v1/chat/completions", _chat())
    assert status == 200 and headers["X-Slurmhelps-Deployment"] == f"{QWEN}-hopper"
    assert router.failovers == 1
    assert router.describe(QWEN)["reason"].startswith("connection failed")
    _request(server.base_url + "/v1/chat/completions", _chat())
    assert router.failovers == 1 and router.counters == {f"{QWEN}-hopper": 2}


def test_stale_heartbeat_is_not_ready(tmp_path, start_router, make_backend):
    bw = make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", bw.port, last_heartbeat="2020-01-01T00:00:00Z")
    router, _ = start_router()
    assert router.describe(QWEN)["reason"] == "stale heartbeat"


# -- guards -------------------------------------------------------------------------

def _history_with_violations() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "fetch the data"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
                {"command": f"curl -s {DENIED}"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
                {"command": "apt-get install foo"})}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "E: Permission denied"},
        {"role": "user", "content": f"the key {KEY} again"},
    ]


def test_guards_applied_on_gateway_less_backend(tmp_path, start_router, make_backend):
    bw = make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", bw.port)
    _, server = start_router()
    status, _, body = _request(server.base_url + "/v1/chat/completions",
                               {"model": QWEN, "messages": _history_with_violations(),
                                "bad_words": ["client-word"]},
                               headers={"Authorization": "Bearer passthrough"})
    assert status == 200
    sent = bw.bodies[-1]
    assert KEY in sent["bad_words"] and "client-word" in sent["bad_words"]
    assert len(sent["bad_words"]) == len(set(sent["bad_words"]))
    history = json.dumps(sent["messages"])
    assert "aliyuncs" not in history and KEY not in history
    assert NOTICE in history  # privguard notice on the permission error
    assert bw.headers[-1]["Authorization"] == "Bearer passthrough"
    content = json.loads(body)["choices"][0]["message"]["content"]
    assert "aliyuncs" not in content and KEY not in content and "blocked.invalid" in content


def test_guarded_stream_stays_streaming(tmp_path, start_router, make_backend):
    bw = make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", bw.port)
    _, server = start_router()
    status, headers, body = _request(server.base_url + "/v1/chat/completions", _chat(stream=True))
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    text, events = _sse_text(body)
    assert events >= 3 and body.decode().rstrip().endswith("data: [DONE]")
    assert "aliyuncs" not in text and KEY not in text and "blocked.invalid" in text


def test_guards_off_passes_through(tmp_path, start_router, make_backend):
    bw = make_backend(QWEN)
    _write_state(tmp_path / "serve", QWEN, "ready", bw.port)
    _, server = start_router(guards=False)
    _, _, body = _request(server.base_url + "/v1/chat/completions", _chat())
    assert "aliyuncs" in json.loads(body)["choices"][0]["message"]["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_not_doubly_applied_behind_node_gateway(tmp_path, start_router, make_backend, catalog, stream):
    guard = catalog[1][QWEN]
    direct, behind = make_backend(QWEN), make_backend(QWEN)
    gw_direct = GatewayServer(f"http://127.0.0.1:{direct.port}/v1", bad_words=guard.bad_words,
                              rewriter=guard.rewriter).start()
    gw_behind = GatewayServer(f"http://127.0.0.1:{behind.port}/v1", bad_words=guard.bad_words,
                              rewriter=guard.rewriter).start()
    try:
        request = {"model": QWEN, "messages": _history_with_violations(), "stream": stream}
        # Reference: the node gateway alone.
        _, _, ref = _request(gw_direct.base_url + "/v1/chat/completions", request)
        # Router (guards on) in front of a node gateway.
        _write_state(tmp_path / "serve", QWEN, "ready", gw_behind.port)
        _, server = start_router()
        status, _, got = _request(server.base_url + "/v1/chat/completions", request)
        assert status == 200
        assert behind.bodies[-1] == direct.bodies[-1]  # request guards: same as once
        assert json.dumps(behind.bodies[-1]).count(NOTICE) == 1
        if stream:
            assert _sse_text(got)[0] == _sse_text(ref)[0]
        else:
            assert json.loads(got)["choices"] == json.loads(ref)["choices"]
    finally:
        gw_direct.stop()
        gw_behind.stop()


def test_mock_backend_streaming_unguarded_model(tmp_path, start_router):
    mock = MockOpenAIServer(model="mock-model", prefill_delay=0.0, token_delay=0.0).start()
    try:
        _write_state(tmp_path / "serve", "mock-dir", "ready", mock.port)
        _, server = start_router()
        status, headers, body = _request(server.base_url + "/v1/chat/completions",
                                         _chat("mock-model", stream=True))
        assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
        assert _sse_text(body)[1] >= 2
    finally:
        mock.stop()


# -- binding / CLI ------------------------------------------------------------------

def test_router_refuses_public_bind(tmp_path):
    router = Router(tmp_path)
    with pytest.raises(ValueError):
        RouterServer(router, listen_host="0.0.0.0")
    RouterServer(router, listen_host="0.0.0.0", allow_public=True)  # explicit opt-in only


def test_cli_build_router_reads_catalogue_prefer_alias(tmp_path):
    args = cli.build_parser().parse_args([
        "router", "--state-root", str(tmp_path), "--prefer", f"{QWEN}-hopper,{QWEN}",
        "--alias", f"qwen38={QWEN}"])
    router = cli.build_router(args)
    assert router.serve_root == tmp_path and router.prefer == [f"{QWEN}-hopper", QWEN]
    assert router.aliases == {"qwen38": QWEN} and QWEN in router.guards
    assert isinstance(router.catalog[QWEN], CatalogEntry)
    args = cli.build_parser().parse_args(["router", "--state-root", str(tmp_path), "--no-guards"])
    assert cli.build_router(args).guards == {}
    with pytest.raises(SystemExit):
        cli.build_router(cli.build_parser().parse_args(["router", "--alias", "bad"]))


def test_cli_router_status_reports_unreachable(capsys):
    rc = cli.main(["router", "--status", "--listen-port", str(_free_port())])
    assert rc == 3
    assert "router not answering" in capsys.readouterr().err
