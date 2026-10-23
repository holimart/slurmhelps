#!/usr/bin/env python3
"""Deterministic, stdlib-only OpenAI-compatible mock backend.

The server exists to exercise the serving gateway and the benchmark client
without a GPU, a model or a network: it binds loopback only, generates output
from a prompt/seed hash, and keeps enough per-request timing data to assert
continuous-batching and admission behaviour.

Endpoints::

    GET  /health
    GET  /status
    GET  /v1/models
    POST /v1/completions
    POST /v1/chat/completions
    GET  /metrics

The mock deliberately simulates work (``prefill_delay`` + ``token_delay`` per
generated token) so concurrent clients overlap and ``batch_size`` is
observable. A request with a ``cache_salt`` (or any repeated model/prompt pair)
reports ``cached_tokens = min(prompt_tokens, 384)``.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega one two three four "
    "five six seven eight nine ten red green blue amber violet silver golden "
    "run jump walk read write build test measure serve cache batch token layer "
    "model server client request response latency throughput memory compute "
    "prefix suffix stream decode prefill sample route expert attention sparse "
    "dense weight matrix vector tensor kernel graph node edge path queue slot"
).split()

CACHED_TOKENS_CAP = 384
DEFAULT_MAX_TOKENS_CAP = 512

_MISSING = object()


@dataclass
class RequestRecord:
    """One completion request, including the active batch size at finish."""

    request_id: str
    path: str
    model: str
    stream: bool
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    cached_tokens: int
    arrival: float
    finish: float = 0.0
    batch_size: int = 0
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def arrival_time(self) -> float:
        """Alias for callers that spell out the timestamp."""
        return self.arrival

    @property
    def finish_time(self) -> float:
        """Alias for callers that spell out the timestamp."""
        return self.finish

    @property
    def latency(self) -> float:
        return max(0.0, self.finish - self.arrival)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access, so records work for both dataclass and mapping users."""
        aliases = {"arrival_time": "arrival", "finish_time": "finish"}
        return getattr(self, aliases.get(key, key), default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value


class MockOpenAIServer:
    """Thread-safe OpenAI-compatible mock on 127.0.0.1:0 by default."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        model: str = "mock-model",
        max_seqs: int = 8,
        max_model_len: int = 327680,
        max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP,
        prefill_delay: float = 0.05,
        token_delay: float = 0.002,
        seed: int = 0,
        coalesce_ms: float = 0.0,
        max_num_seqs: int | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.max_seqs = int(max_num_seqs if max_num_seqs is not None else max_seqs)
        self.max_model_len = int(max_model_len)
        self.max_tokens_cap = max(1, int(max_tokens_cap))
        self.prefill_delay = max(0.0, float(prefill_delay))
        self.token_delay = max(0.0, float(token_delay))
        self.seed = seed
        self.coalesce_ms = max(0.0, float(coalesce_ms))

        self.requests: list[RequestRecord] = []
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._cache: set[tuple[str, str, str]] = set()
        self._active = 0
        self.max_active_seen = 0
        self._counters: dict[str, int] = {
            "completed": 0,
            "cache_hits": 0,
            "rejected": 0,
            "failed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "MockOpenAIServer":
        with self._start_lock:
            if self._httpd is not None:
                return self
            httpd = ThreadingHTTPServer((self.host, self.port), _MockHandler)
            httpd.daemon_threads = True
            setattr(httpd, "mock", self)
            self._httpd = httpd
            self.port = int(httpd.server_address[1])
            self._thread = threading.Thread(
                target=httpd.serve_forever,
                name="mock-openai",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._start_lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def __enter__(self) -> "MockOpenAIServer":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("", "0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def batch_size(self) -> int:
        """Largest batch size observed at a request completion."""
        with self._lock:
            return max((r.batch_size for r in self.requests), default=0)

    @property
    def last_batch_size(self) -> int:
        with self._lock:
            return self.requests[-1].batch_size if self.requests else 0

    # -- admission / accounting (called from handler threads) --------------
    def admit(
        self,
        cache_key: tuple[str, str, str],
        prompt_tokens: int,
    ) -> tuple[bool, bool, int]:
        """Reserve an active slot and report (admitted, cache_hit, cached_tokens)."""
        with self._lock:
            if self._active >= self.max_seqs:
                return False, False, 0
            self._active += 1
            self.max_active_seen = max(self.max_active_seen, self._active)
            hit = cache_key in self._cache
            if hit:
                self._counters["cache_hits"] += 1
            self._cache.add(cache_key)
            cached = min(prompt_tokens, CACHED_TOKENS_CAP) if hit else 0
            return True, hit, cached

    def complete(self, record: RequestRecord, status: int = 200) -> None:
        """Close an admitted request, snapshotting the active batch size."""
        with self._lock:
            record.finish = time.monotonic()
            record.batch_size = self._active
            record.status = status
            self.requests.append(record)
            self._active = max(0, self._active - 1)
            if status == 200:
                self._counters["completed"] += 1
                self._counters["prompt_tokens"] += record.prompt_tokens
                self._counters["completion_tokens"] += record.completion_tokens
            else:
                self._counters["failed"] += 1

    def reject(self, record: RequestRecord, status: int) -> None:
        """Record a request rejected before admission (400/429)."""
        with self._lock:
            record.finish = time.monotonic()
            record.batch_size = self._active
            record.status = status
            self.requests.append(record)
            self._counters["rejected"] += 1

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "model": self.model,
                "requests": len(self.requests),
                "requests_total": len(self.requests),
                "completed": self._counters["completed"],
                "failed": self._counters["failed"],
                "rejected": self._counters["rejected"],
                "active": self._active,
                "active_requests": self._active,
                "max_active_seen": self.max_active_seen,
                "batch_size": max((r.batch_size for r in self.requests), default=0),
                "last_batch_size": self.requests[-1].batch_size if self.requests else 0,
                "cache_hits": self._counters["cache_hits"],
                "cache_hits_total": self._counters["cache_hits"],
                "prompt_tokens": self._counters["prompt_tokens"],
                "completion_tokens": self._counters["completion_tokens"],
                "max_seqs": self.max_seqs,
                "max_model_len": self.max_model_len,
            }

    def metrics(self) -> str:
        with self._lock:
            requests_total = len(self.requests)
            active = self._active
            batch = max((r.batch_size for r in self.requests), default=0)
            counters = dict(self._counters)
        lines = [
            "# HELP mock_requests_total Completion requests seen by the mock.",
            "# TYPE mock_requests_total counter",
            f"mock_requests_total {requests_total}",
            "# HELP mock_completed_total Admitted requests that returned 200.",
            "# TYPE mock_completed_total counter",
            f"mock_completed_total {counters['completed']}",
            "# HELP mock_cache_hits_total Requests that found a cached prompt.",
            "# TYPE mock_cache_hits_total counter",
            f"mock_cache_hits_total {counters['cache_hits']}",
            "# HELP mock_rejected_total Requests rejected by admission or validation.",
            "# TYPE mock_rejected_total counter",
            f"mock_rejected_total {counters['rejected']}",
            "# HELP mock_active_requests Requests currently in flight.",
            "# TYPE mock_active_requests gauge",
            f"mock_active_requests {active}",
            "# HELP mock_batch_size Largest active batch observed at completion.",
            "# TYPE mock_batch_size gauge",
            f"mock_batch_size {batch}",
            "# HELP mock_prompt_tokens_total Prompt tokens accounted for.",
            "# TYPE mock_prompt_tokens_total counter",
            f"mock_prompt_tokens_total {counters['prompt_tokens']}",
            "# HELP mock_completion_tokens_total Generated tokens accounted for.",
            "# TYPE mock_completion_tokens_total counter",
            f"mock_completion_tokens_total {counters['completion_tokens']}",
        ]
        return "\n".join(lines) + "\n"


def _count_tokens(text: str) -> int:
    return len(text.split())


def _extract_prompt(body: dict[str, Any], chat: bool) -> str:
    if chat:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return ""
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
        return "\n".join(parts)
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        if all(isinstance(item, str) for item in prompt):
            return "\n".join(prompt)
        # A list of token ids: one placeholder word per token keeps the count.
        return " ".join("tok" for _ in prompt)
    if prompt is None:
        return ""
    return str(prompt)


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def _completion_id(prefix: str, model: str, prompt: str, seed: Any) -> str:
    raw = f"{model}|{prompt}|{seed}".encode("utf-8", "replace")
    return prefix + hashlib.sha256(raw).hexdigest()[:24]


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def mock(self) -> MockOpenAIServer:
        return getattr(self.server, "mock")

    def log_message(self, *_args: Any) -> None:
        return

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex
        if path == "/health":
            self._send_json(200, {"status": "ok", "model": self.mock.model}, request_id)
            return
        if path == "/status":
            self._send_json(200, self.mock.status(), request_id)
            return
        if path == "/v1/models":
            created = int(time.time())
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.mock.model,
                            "object": "model",
                            "created": created,
                            "owned_by": "slurmhelps-mock",
                        }
                    ],
                },
                request_id,
            )
            return
        if path == "/metrics":
            body = self.mock.metrics().encode("utf-8")
            self._send_bytes(200, "text/plain; version=0.0.4", body, request_id)
            return
        self._send_error(404, f"unknown path: {path}", request_id=request_id)

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex
        if path in ("/v1/completions", "/v1/chat/completions"):
            self._handle_completion(chat=path.endswith("/chat/completions"), request_id=request_id)
            return
        if path in ("/flush_cache", "/reset_prefix_cache"):
            try:
                self._read_json()  # consume the body; an invalid body is irrelevant here
            except ValueError:
                pass
            self.mock.clear_cache()
            self._send_json(200, {"status": "ok", "cache": "cleared"}, request_id)
            return
        self._send_error(404, f"unknown path: {path}", request_id=request_id)

    # -- completion --------------------------------------------------------
    def _handle_completion(self, *, chat: bool, request_id: str) -> None:
        server = self.mock
        arrival = time.monotonic()
        headers = {
            key.lower(): value
            for key, value in self.headers.items()
            if key.lower() != "authorization"
        }
        try:
            body = self._read_json()
        except ValueError as exc:
            record = RequestRecord(
                request_id=request_id,
                path=self.path,
                model=server.model,
                stream=False,
                prompt_tokens=0,
                completion_tokens=0,
                cache_hit=False,
                cached_tokens=0,
                arrival=arrival,
                headers=headers,
            )
            server.reject(record, 400)
            self._send_error(400, str(exc), request_id=request_id)
            return

        model = str(body.get("model") or server.model)
        prompt = _extract_prompt(body, chat)
        prompt_tokens = _count_tokens(prompt)
        stream = bool(body.get("stream", False))
        salt = str(body.get("cache_salt") or "")

        raw_max = body.get("max_tokens", body.get("max_completion_tokens", 16))
        try:
            requested_max = int(raw_max)
        except (TypeError, ValueError):
            requested_max = -1
        if requested_max < 0:
            record = RequestRecord(
                request_id=request_id,
                path=self.path,
                model=model,
                stream=stream,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                cache_hit=False,
                cached_tokens=0,
                arrival=arrival,
                headers=headers,
            )
            server.reject(record, 400)
            self._send_error(400, "max_tokens must be a non-negative integer", request_id=request_id)
            return

        # The literal request is checked, not the capped output: a caller asking
        # for more than the model window is invalid even if the mock would cap it.
        if prompt_tokens + requested_max > server.max_model_len:
            record = RequestRecord(
                request_id=request_id,
                path=self.path,
                model=model,
                stream=stream,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                cache_hit=False,
                cached_tokens=0,
                arrival=arrival,
                headers=headers,
            )
            server.reject(record, 400)
            self._send_error(
                400,
                f"prompt_tokens ({prompt_tokens}) + max_tokens ({requested_max}) "
                f"exceeds max_model_len ({server.max_model_len})",
                request_id=request_id,
            )
            return

        completion_tokens = min(requested_max, server.max_tokens_cap)
        if server.coalesce_ms:
            time.sleep(server.coalesce_ms / 1000.0)

        cache_key = (model, prompt, salt)
        admitted, cache_hit, cached_tokens = server.admit(cache_key, prompt_tokens)
        record = RequestRecord(
            request_id=request_id,
            path=self.path,
            model=model,
            stream=stream,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if admitted else 0,
            cache_hit=cache_hit,
            cached_tokens=cached_tokens,
            arrival=arrival,
            headers=headers,
        )
        if not admitted:
            server.reject(record, 429)
            self._send_error(
                429,
                f"max_seqs ({server.max_seqs}) reached",
                request_id=request_id,
                retry_after=1,
            )
            return

        seed = body.get("seed", server.seed)
        rng = random.Random(hashlib.sha256(
            f"{seed}|{model}|{prompt}".encode("utf-8", "replace")
        ).digest())
        words = [rng.choice(WORDS) for _ in range(completion_tokens)]
        usage = _usage(prompt_tokens, completion_tokens, cached_tokens)
        created = int(time.time())
        completion_id = _completion_id(
            "chatcmpl-" if chat else "cmpl-", model, prompt, seed
        )

        status = 200
        try:
            if server.prefill_delay:
                time.sleep(server.prefill_delay)
            if stream:
                self._stream_completion(
                    chat=chat,
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    words=words,
                    usage=usage,
                    request_id=request_id,
                )
            else:
                text = " ".join(words)
                choice: dict[str, Any] = {
                    "index": 0,
                    "finish_reason": "length",
                }
                if chat:
                    choice["message"] = {"role": "assistant", "content": text}
                else:
                    choice["text"] = text
                    choice["logprobs"] = None
                payload = {
                    "id": completion_id,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [choice],
                    "usage": usage,
                }
                self._send_json(status, payload, request_id)
        except (BrokenPipeError, ConnectionResetError, OSError):
            status = 499
        finally:
            server.complete(record, status)

    def _stream_completion(
        self,
        *,
        chat: bool,
        completion_id: str,
        created: int,
        model: str,
        words: list[str],
        usage: dict[str, Any],
        request_id: str,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.send_header("X-Request-Id", request_id)
        self.end_headers()
        self.close_connection = True

        object_name = "chat.completion.chunk" if chat else "text_completion"
        if chat:
            role_chunk = {
                "id": completion_id,
                "object": object_name,
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            self._write_sse(role_chunk)

        for index, word in enumerate(words):
            content = (" " if index else "") + word
            if chat:
                choice = {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            else:
                choice = {
                    "index": 0,
                    "text": content,
                    "logprobs": None,
                    "finish_reason": None,
                }
            self._write_sse({
                "id": completion_id,
                "object": object_name,
                "created": created,
                "model": model,
                "choices": [choice],
            })
            if self.mock.token_delay:
                time.sleep(self.mock.token_delay)

        self._write_sse({
            "id": completion_id,
            "object": object_name,
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        })
        self._write_sse({
            "id": completion_id,
            "object": object_name,
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- response helpers --------------------------------------------------
    def _write_sse(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        request_id: str,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, "application/json", body, request_id)

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        request_id: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self,
        status: int,
        message: str,
        *,
        request_id: str,
        retry_after: int | None = None,
    ) -> None:
        if status == 429:
            error_type = "rate_limit_error"
        elif status == 400 or status == 404:
            error_type = "invalid_request_error"
        else:
            error_type = "server_error"
        body = json.dumps({
            "error": {
                "message": message,
                "type": error_type,
                "code": status,
            }
        }, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", request_id)
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(body)
