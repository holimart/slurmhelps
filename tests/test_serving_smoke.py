"""Ten-check HTTP smoke matrix against the mock backend + gateway.

This is the fast CI/smoke path: a fake GPU/engine answers OpenAI requests while
the gateway exercises auth, admission, streaming, cache accounting and
concurrency.  The real-GPU smoke uses ``tests/serve_smoke.py`` with the same
checks against the Slurm endpoint.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from slurmhelps.serving.gateway import GatewayServer
from slurmhelps.serving.mock import MockOpenAIServer

MODEL = "glm53-reap50"
MAX_LEN = 786_432


@pytest.fixture(scope="module")
def mock():
    server = MockOpenAIServer(
        model=MODEL, max_seqs=32, max_model_len=MAX_LEN,
        prefill_delay=0.01, token_delay=0.001,
    ).start()
    gateway = GatewayServer(upstream=server.base_url, max_active=32).start()
    try:
        yield server, gateway
    finally:
        gateway.stop()
        server.stop()


def _post(url: str, body: dict, key: str = "", timeout: float = 60.0):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
        except ValueError:
            payload = {}
        return exc.code, payload


def _get(url: str, key: str = ""):
    headers = {"Authorization": "Bearer " + key} if key else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def _body(tokens: int = 500, out: int = 16, salt: str = "", stream: bool = False) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": ("word " * tokens).strip()}],
        "max_tokens": out,
        "temperature": 0,
        "stream": stream,
        "cache_salt": salt,
    }


def test_01_health_and_status(mock):
    _server, gateway = mock
    assert _get(gateway.base_url + "/health")[0] == 200
    code, status = _get(gateway.base_url + "/status")
    assert code == 200
    assert status["status"] == "ok"
    assert status["auth"] == "none"


def test_02_models(mock):
    _server, gateway = mock
    code, obj = _get(gateway.base_url + "/v1/models")
    assert code == 200
    assert any(m.get("id") == MODEL for m in obj["data"])


def test_03_opencode_compatible_chat_nonstream(mock):
    _server, gateway = mock
    code, obj = _post(gateway.base_url + "/v1/chat/completions", _body(500, 16))
    assert code == 200
    assert obj["choices"][0]["message"]["content"]
    assert obj["usage"]["prompt_tokens"] > 0


def test_04_streaming_sse(mock):
    _server, gateway = mock
    req = urllib.request.Request(
        gateway.base_url + "/v1/chat/completions",
        data=json.dumps(_body(500, 16, stream=True)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = 0
    done = False
    with urllib.request.urlopen(req, timeout=60) as response:
        assert response.status == 200
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done = True
                break
            if line.startswith("data:"):
                chunks += 1
    assert chunks > 0 and done


def test_05_cache_hit_accounting(mock):
    _server, gateway = mock
    _post(gateway.base_url + "/v1/chat/completions", _body(500, 4, salt="cache-test"))
    code, obj = _post(gateway.base_url + "/v1/chat/completions", _body(500, 4, salt="cache-test"))
    assert code == 200
    assert obj["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_06_shared_prefix_concurrent_short_finishes_early(mock):
    _server, gateway = mock
    prefix = "shared prefix " * 80
    results: list[tuple[int, float, int]] = []
    lock = threading.Lock()

    def one(i: int, out: int):
        body = _body(500, out, salt="shared")
        body["messages"][0]["content"] = prefix + f" unique-tail-{i} " + ("x " * 500)
        started = time.perf_counter()
        code, _obj = _post(gateway.base_url + "/v1/chat/completions", body)
        with lock:
            results.append((i, time.perf_counter() - started, code))

    threads = [threading.Thread(target=one, args=(i, out))
               for i, out in enumerate((8, 32, 64, 128, 256))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 5 and all(code == 200 for _, _, code in results)
    by_i = {i: elapsed for i, elapsed, _ in results}
    assert by_i[0] <= by_i[4] * 1.5  # timer noise tolerant; mock has no barrier


def test_07_concurrency_sweep_500_800(mock):
    _server, gateway = mock
    for ctx in (500, 800):
        for conc in (1, 2, 4, 8):
            codes: list[int] = []
            lock = threading.Lock()

            def one(i: int):
                code, _obj = _post(gateway.base_url + "/v1/chat/completions",
                                   _body(ctx, 8, salt=f"{ctx}-{conc}-{i}"))
                with lock:
                    codes.append(code)

            threads = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            assert sorted(codes) == [200] * conc, (ctx, conc, codes)


def test_08_max_model_len_rejected(mock):
    _server, gateway = mock
    code, _obj = _post(gateway.base_url + "/v1/chat/completions",
                       _body(10, MAX_LEN + 1))
    assert code == 400


def test_09_auth_required():
    server = MockOpenAIServer(model=MODEL).start()
    gateway = GatewayServer(upstream=server.base_url, api_key="test-key").start()
    try:
        assert _post(gateway.base_url + "/v1/chat/completions", _body(8, 4))[0] == 401
        assert _post(gateway.base_url + "/v1/chat/completions", _body(8, 4), key="test-key")[0] == 200
        # Local health is not authenticated (used by probes before a tunnel).
        assert _get(gateway.base_url + "/health")[0] == 200
    finally:
        gateway.stop()
        server.stop()


def test_10_queue_limit_returns_429():
    server = MockOpenAIServer(model=MODEL, max_seqs=8, token_delay=0.01).start()
    gateway = GatewayServer(upstream=server.base_url, max_active=1).start()
    try:
        codes: list[int] = []
        lock = threading.Lock()

        def one():
            code, _obj = _post(gateway.base_url + "/v1/chat/completions", _body(64, 32))
            with lock:
                codes.append(code)

        threads = [threading.Thread(target=one) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert 429 in codes, codes
    finally:
        gateway.stop()
        server.stop()
