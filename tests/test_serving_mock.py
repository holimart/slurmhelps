"""Tests for the stdlib OpenAI-compatible mock backend (no GPU, loopback only)."""

from __future__ import annotations

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

import pytest

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
    """Minimal one-shot HTTP client (loopback only)."""
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
        model="mock-test",
        max_seqs=8,
        max_model_len=8192,
        prefill_delay=0.01,
        token_delay=0.001,
    )


def test_nonstream_completion_usage_and_request_id(mock: MockOpenAIServer):
    status, headers, raw = request(
        mock.base_url,
        "POST",
        "/v1/completions",
        {
            "model": "mock-test",
            "prompt": "alpha beta gamma delta",
            "max_tokens": 8,
            "temperature": 0.9,
        },
        headers={"X-Request-Id": "mock-rid-1"},
    )
    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert headers["x-request-id"] == "mock-rid-1"

    payload = json_body(raw)
    choice = payload["choices"][0]
    assert len(choice["text"].split()) == 8
    usage = payload["usage"]
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 12
    assert usage["cached_tokens"] == 0

    # temperature is ignored: same prompt/model produces the same deterministic text
    status2, _, raw2 = request(
        mock.base_url,
        "POST",
        "/v1/completions",
        {
            "model": "mock-test",
            "prompt": "alpha beta gamma delta",
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert status2 == 200
    assert json_body(raw2)["choices"][0]["text"] == choice["text"]


def test_chat_completion(mock: MockOpenAIServer):
    status, _, raw = request(
        mock.base_url,
        "POST",
        "/v1/chat/completions",
        {
            "model": "mock-test",
            "messages": [{"role": "user", "content": "hello world from chat"}],
            "max_tokens": 5,
        },
    )
    assert status == 200
    payload = json_body(raw)
    message = payload["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert len(message["content"].split()) == 5
    assert payload["usage"]["prompt_tokens"] == 4
    assert payload["usage"]["completion_tokens"] == 5


def test_stream_sse_has_chunks_usage_and_done(mock: MockOpenAIServer):
    split = urlsplit(mock.base_url)
    conn = http.client.HTTPConnection(split.hostname, split.port, timeout=10.0)
    try:
        body = json.dumps(
            {
                "model": "mock-test",
                "prompt": "one two three four five",
                "max_tokens": 8,
                "stream": True,
            }
        ).encode("utf-8")
        conn.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/event-stream")
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
    assert len(streamed.split()) == 8
    assert usage is not None
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 13
    assert mock.requests[-1].stream is True
    assert mock.requests[-1].completion_tokens == 8


def test_cache_hit_reports_cached_tokens(mock_factory):
    server = mock_factory(max_tokens_cap=64, prefill_delay=0.0, token_delay=0.0)
    body = {
        "model": "mock-test",
        "prompt": "one two three",
        "max_tokens": 4,
        "cache_salt": "warm-run",
    }
    first_status, _, first_raw = request(server.base_url, "POST", "/v1/completions", body)
    second_status, _, second_raw = request(server.base_url, "POST", "/v1/completions", body)
    assert first_status == second_status == 200
    first = json_body(first_raw)
    second = json_body(second_raw)
    assert first["usage"]["cached_tokens"] == 0
    assert second["usage"]["cached_tokens"] == 3
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert second["choices"][0]["text"] == first["choices"][0]["text"]

    # cached_tokens is capped at 384 even for a much longer prompt
    long_body = {
        "model": "mock-test",
        "prompt": " ".join(f"word{i}" for i in range(500)),
        "max_tokens": 1,
        "cache_salt": "long",
    }
    request(server.base_url, "POST", "/v1/completions", long_body)
    _, _, long_raw = request(server.base_url, "POST", "/v1/completions", long_body)
    assert json_body(long_raw)["usage"]["cached_tokens"] == 384

    status = json_body(request(server.base_url, "GET", "/status")[2])
    assert status["cache_hits"] == 2
    _, metrics_headers, metrics_raw = request(server.base_url, "GET", "/metrics")
    assert metrics_headers["content-type"].startswith("text/plain")
    metrics = metrics_raw.decode("utf-8")
    assert "mock_cache_hits_total 2" in metrics
    assert "mock_active_requests 0" in metrics


def test_max_model_len_rejects_400(mock_factory):
    server = mock_factory(max_model_len=20, prefill_delay=0.0, token_delay=0.0)
    status, _, raw = request(
        server.base_url,
        "POST",
        "/v1/completions",
        {"prompt": " ".join(f"w{i}" for i in range(20)), "max_tokens": 5},
    )
    assert status == 400
    error = json_body(raw)["error"]
    assert "max_model_len" in error["message"]
    assert server.requests[-1].status == 400

    ok_status, _, _ = request(
        server.base_url,
        "POST",
        "/v1/completions",
        {"prompt": "short prompt", "max_tokens": 5},
    )
    assert ok_status == 200


def test_max_seqs_rejects_429(mock_factory):
    server = mock_factory(max_seqs=1, prefill_delay=0.3, token_delay=0.0)
    payload = {"prompt": "single slot", "max_tokens": 1}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, server.base_url, "POST", "/v1/completions", payload)
        deadline = time.monotonic() + 5.0
        while server.active < 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert server.active == 1, "first request never became active"
        status, headers, raw = request(
            server.base_url, "POST", "/v1/completions", payload
        )
        assert status == 429
        assert headers["retry-after"] == "1"
        assert "max_seqs" in json_body(raw)["error"]["message"]
        first_status, _, _ = first.result(timeout=10.0)
    assert first_status == 200

    # the slot is released after the first request completes
    retry_status, _, _ = request(server.base_url, "POST", "/v1/completions", payload)
    assert retry_status == 200


def test_concurrent_requests_observe_batch_size(mock_factory):
    server = mock_factory(max_seqs=4, prefill_delay=0.2, token_delay=0.0)
    barrier = threading.Barrier(4)

    def one(index: int):
        barrier.wait(timeout=5.0)
        return request(
            server.base_url,
            "POST",
            "/v1/completions",
            {"prompt": f"concurrent prompt {index}", "max_tokens": 1},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [future.result(timeout=10.0) for future in [pool.submit(one, i) for i in range(4)]]

    assert [status for status, _, _ in results] == [200, 200, 200, 200]
    assert server.max_active_seen >= 2
    assert max(record.batch_size for record in server.requests) >= 2
    assert server.batch_size >= 2


def test_health_status_models(mock: MockOpenAIServer):
    request(
        mock.base_url,
        "POST",
        "/v1/completions",
        {"prompt": "one two", "max_tokens": 2},
    )
    health_status, _, health_raw = request(mock.base_url, "GET", "/health")
    assert health_status == 200
    assert json_body(health_raw)["status"] == "ok"

    status_code, _, status_raw = request(mock.base_url, "GET", "/status")
    assert status_code == 200
    status = json_body(status_raw)
    assert status["active"] == 0
    assert status["completed"] == 1
    assert status["requests"] == 1
    assert status["max_seqs"] == 8

    models_status, _, models_raw = request(mock.base_url, "GET", "/v1/models")
    assert models_status == 200
    assert json_body(models_raw)["data"][0]["id"] == "mock-test"


def test_stop_is_idempotent_and_base_url_uses_ephemeral_port():
    server = MockOpenAIServer(prefill_delay=0.0, token_delay=0.0)
    server.start()
    assert server.port > 0
    assert server.base_url.startswith("http://127.0.0.1:")
    server.stop()
    server.stop()
