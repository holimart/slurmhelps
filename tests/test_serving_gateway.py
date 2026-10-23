"""Tests for the OpenAI-compatible serving gateway (no GPU, loopback only)."""

from __future__ import annotations

import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

import pytest

from slurmhelps.serving.gateway import GatewayServer
from slurmhelps.serving.mock import MockOpenAIServer


def _headers(items: list[tuple[str, str]]) -> dict[str, str]:
    return {key.lower(): value for key, value in items}


def request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], bytes]:
    split = urlsplit(base_url)
    conn = http.client.HTTPConnection(split.hostname, split.port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    sent = {"Content-Type": "application/json", **(headers or {})}
    try:
        conn.request(method, path, body=body, headers=sent)
        response = conn.getresponse()
        raw = response.read()
        return response.status, _headers(response.getheaders()), raw
    finally:
        conn.close()


def json_body(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


@pytest.fixture()
def mock_factory():
    servers: list[MockOpenAIServer] = []

    def make(**kwargs: Any) -> MockOpenAIServer:
        server = MockOpenAIServer(**kwargs)
        server.start()
        servers.append(server)
        return server

    try:
        yield make
    finally:
        for server in servers:
            server.stop()


@pytest.fixture()
def mock(mock_factory):
    return mock_factory(
        model="gateway-mock",
        max_seqs=8,
        max_model_len=8192,
        max_tokens_cap=64,
        prefill_delay=0.02,
        token_delay=0.001,
    )


@pytest.fixture()
def gateway_factory():
    servers: list[GatewayServer] = []

    def make(upstream: str, **kwargs: Any) -> GatewayServer:
        gateway = GatewayServer(upstream, **kwargs)
        gateway.start()
        servers.append(gateway)
        return gateway

    try:
        yield make
    finally:
        for gateway in servers:
            gateway.stop()


def test_nonstream_proxy_forwards_headers_and_request_id(mock, gateway_factory):
    gateway = gateway_factory(mock.base_url)
    status, headers, raw = request(
        gateway.base_url,
        "POST",
        "/v1/completions?trace=1",
        {
            "model": "gateway-mock",
            "prompt": "alpha beta gamma",
            "max_tokens": 6,
        },
        headers={"X-Test-Header": "forward-me", "X-Request-Id": "rid-42"},
    )
    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert headers["x-request-id"] == "rid-42"

    payload = json_body(raw)
    assert payload["usage"]["completion_tokens"] == 6
    assert payload["usage"]["prompt_tokens"] == 3

    record = mock.requests[-1]
    assert record.headers["x-test-header"] == "forward-me"
    assert record.headers["x-request-id"] == "rid-42"
    assert "trace=1" in record.path

    # A caller-supplied request id is used; a missing one is generated.
    status2, headers2, _ = request(
        gateway.base_url,
        "POST",
        "/v1/completions",
        {"prompt": "generated id", "max_tokens": 2},
    )
    assert status2 == 200
    generated = headers2["x-request-id"]
    assert generated
    assert mock.requests[-1].headers["x-request-id"] == generated


def test_stream_proxy_relays_sse_and_usage(mock, gateway_factory):
    gateway = gateway_factory(mock.base_url)
    split = urlsplit(gateway.base_url)
    conn = http.client.HTTPConnection(split.hostname, split.port, timeout=10.0)
    try:
        body = json.dumps(
            {
                "model": "gateway-mock",
                "prompt": "one two three four",
                "max_tokens": 7,
                "stream": True,
            }
        ).encode("utf-8")
        conn.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": "stream-rid",
            },
        )
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/event-stream")
        assert response.getheader("X-Request-Id") == "stream-rid"
        lines = []
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8").strip()
            if text:
                lines.append(text)
    finally:
        conn.close()

    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[5:].strip()) for line in lines[:-1] if line.startswith("data:")]
    streamed = ""
    usage = None
    for event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            streamed += choice.get("text", "")
    assert len(streamed.split()) == 7
    assert usage is not None
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 11
    assert mock.requests[-1].stream is True


def test_auth_is_optional_and_health_stays_public(mock, gateway_factory):
    gateway = gateway_factory(mock.base_url, api_key="test-key")

    health_status, _, health_raw = request(gateway.base_url, "GET", "/health")
    assert health_status == 200
    assert json_body(health_raw)["status"] == "ok"

    missing_status, missing_headers, _ = request(gateway.base_url, "GET", "/v1/models")
    assert missing_status == 401
    assert missing_headers["www-authenticate"] == "Bearer"

    wrong_status, _, _ = request(
        gateway.base_url,
        "GET",
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert wrong_status == 401

    ok_status, _, ok_raw = request(
        gateway.base_url,
        "GET",
        "/v1/models",
        headers={"Authorization": "Bearer test-key"},
    )
    assert ok_status == 200
    assert json_body(ok_raw)["data"][0]["id"] == "gateway-mock"

    # /status is a local, unauthenticated monitoring endpoint.
    status_code, status_headers, status_raw = request(
        gateway.base_url, "GET", "/status", headers={"X-Request-Id": "status-rid"}
    )
    assert status_code == 200
    assert status_headers["x-request-id"] == "status-rid"
    status = json_body(status_raw)
    assert status["request_id"] == "status-rid"
    assert status["active"] == 0
    assert status["queue"] == 0
    assert status["auth"] == "bearer"


def test_key_env_and_absent_key(mock, gateway_factory, monkeypatch):
    monkeypatch.setenv("GW_TEST_API_KEY", "env-key")
    gateway = gateway_factory(mock.base_url, key_env="GW_TEST_API_KEY")
    no_auth_status, _, _ = request(gateway.base_url, "GET", "/v1/models")
    assert no_auth_status == 401
    ok_status, _, _ = request(
        gateway.base_url,
        "GET",
        "/v1/models",
        headers={"Authorization": "Bearer env-key"},
    )
    assert ok_status == 200

    # Unset key_env -> loopback/SSH mode: no application key required.
    monkeypatch.delenv("GW_TEST_API_KEY")
    open_gateway = gateway_factory(mock.base_url, key_env="GW_TEST_API_KEY")
    open_status, _, _ = request(open_gateway.base_url, "GET", "/v1/models")
    assert open_status == 200


def test_max_active_returns_429_and_tracks_active(mock_factory, gateway_factory):
    slow_mock = mock_factory(
        max_seqs=8,
        max_tokens_cap=64,
        prefill_delay=0.5,
        token_delay=0.0,
    )
    gateway = gateway_factory(slow_mock.base_url, max_active=1)
    payload = {"prompt": "hold the slot", "max_tokens": 1}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, gateway.base_url, "POST", "/v1/completions", payload)
        deadline = time.monotonic() + 5.0
        while gateway.active < 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert gateway.active == 1, "first proxied request never became active"

        second_status, second_headers, second_raw = request(
            gateway.base_url, "POST", "/v1/completions", payload
        )
        assert second_status == 429
        assert second_headers["retry-after"] == "1"
        assert second_headers["x-request-id"]
        assert "max_active" in json_body(second_raw)["error"]["message"]

        first_status, _, _ = first.result(timeout=10.0)
    assert first_status == 200
    assert gateway.active == 0
    assert gateway.max_active_seen == 1


def test_max_queue_alias_and_metrics_passthrough(mock, gateway_factory):
    gateway = gateway_factory(mock.base_url, max_queue=2)
    assert gateway.max_active == 2

    status, headers, raw = request(gateway.base_url, "GET", "/metrics")
    assert status == 200
    assert headers["content-type"].startswith("text/plain")
    metrics = raw.decode("utf-8")
    assert "mock_requests_total" in metrics
    assert "mock_cache_hits_total" in metrics


def test_gateway_stop_is_idempotent_and_base_url_uses_ephemeral_port():
    gateway = GatewayServer("http://127.0.0.1:1")
    gateway.start()
    assert gateway.port > 0
    assert gateway.base_url.startswith("http://127.0.0.1:")
    gateway.stop()
    gateway.stop()
