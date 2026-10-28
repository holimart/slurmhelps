#!/usr/bin/env python3
"""tests/stub_server.py — stdlib-only fake OpenAI-compatible chat server.

Used by `bench-matrix.sh --self-test` to prove the harness plumbing (launch
wait, probe, CSV/JSONL contract) without Slurm, a GPU, or the Strata engine.
Binds 127.0.0.1 only and is meant to be killed right after the test.

Endpoints:
  GET  /health              -> {"status":"ok"}
  GET  /v1/models           -> one model, id "strata-selftest"
  GET  /metrics             -> vLLM-style JSON (some keys the probe keeps)
  POST /v1/chat/completions -> SSE stream (or JSON when stream=false):
      content deltas, a finish chunk carrying llama.cpp-style `timings`
      {prompt_ms, predicted_ms, predicted_per_second}, and a final
      usage-only chunk (honors stream_options.include_usage).

The synthetic latencies are deterministic so the self-test can assert the
engine-reported decode rate: completion tokens at exactly TOKEN_DELAY_MS
each -> decode_tok_s_p50 ~ 1000/TOKEN_DELAY_MS.

`--cache-reuse` (off by default) simulates the engine's prompt cache.  A
byte-identical prompt seen before reports `reused = prompt_tokens-1` and its
prefill time is multiplied by `--warm-prefill-factor` (the historical warm
behaviour).  A prefix extension (a new prompt that starts with an earlier one)
reports `reused = prefix_tokens-1` and prefills only the delta, so a chain-mode
probe test sees both growing reuse and a shorter TTFT on later turns.

GET /debug/state exposes the last few full request bodies (seq/chars/text) so
tests can prove that consecutive chain prompts are prefix extensions.

Usage: stub_server.py [--port N] [--token-delay-ms 10] [--prefill-ms-per-token 0.025]
                      [--cache-reuse] [--warm-prefill-factor 0.2]
Prints "STUB PORT <n>" once listening (port 0 -> OS-chosen).
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "strata-selftest"
CONFIG = {"token_delay_ms": 10.0, "prefill_ms_per_token": 0.025,
          "cache_reuse": False, "warm_prefill_factor": 0.2}
STATS = {"requests": 0, "reused_requests": 0, "reused_tokens": 0,
         "prompt_tokens": 0, "last_reused": 0, "last_prompt": 0,
         "cache": {}, "cache_text": [], "request_log": [], "log": []}
STATS_LOCK = threading.Lock()


def sse(handler, obj):
    handler.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
    handler.wfile.flush()


def chunk(cid, model, delta, finish=None, usage=None, timings=None):
    obj = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
           "model": model,
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        obj["usage"] = usage
    if timings is not None:
        obj["timings"] = timings
    return obj


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # quieter than the default
        sys.stderr.write("[stub] %s\n" % (fmt % args))

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_ID})
        elif path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "strata-selftest"}]})
        elif path == "/debug/state":
            with STATS_LOCK:
                requests = [{"seq": i + 1, "chars": len(text), "text": text}
                            for i, text in enumerate(STATS["request_log"])]
            self._send_json(200, {"requests": requests})
        elif path == "/metrics":
            with STATS_LOCK:
                n = STATS["requests"]
                cc = {
                    "enabled": bool(CONFIG["cache_reuse"]),
                    "budget_mib": 8192,
                    "slots": 4,
                    "parked": min(4, len(STATS["cache"])),
                    "bytes": 0,
                    "evictions": 0,
                    "parks": len(STATS["cache"]),
                    "restores": STATS["reused_requests"],
                    "last_event": "restored" if STATS["reused_requests"] else "parked",
                    "last_tokens": STATS["last_prompt"],
                    "requests": n,
                    "requests_reused": STATS["reused_requests"],
                    "reused_tokens": STATS["reused_tokens"],
                    "prompt_tokens": STATS["prompt_tokens"],
                    "last_reused": STATS["last_reused"],
                    "last_prompt": STATS["last_prompt"],
                }
                log = list(STATS["log"][:32])
            self._send_json(200, {
                "vllm:num_requests_running": 0,
                "vllm:num_requests_waiting": 0,
                "vllm:gpu_cache_usage": 0.05,
                "vllm:gpu_prefix_cache_hits_total": 2 * n,
                "vllm:gpu_prefix_cache_queries_total": 50 * n,
                "expert_cache_hits": 3 * n,
                "expert_cache_misses": 7 * n,
                "kv_cache_tokens": 1024 * n,
                "gpu_utilization": 0.7,
                "ram_used_bytes": 12 * 1024 ** 3,
                "conversation_cache": cc,
                "requests": log,
            })
        else:
            self._send_json(404, {"error": {"message": "no route %s" % path}})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "no route %s" % path}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8", "replace"))
            content = ""
            for m in body.get("messages") or []:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    content = m["content"]
            prompt_tokens = max(1, len(content) // 4)
            completion_tokens = int(body.get("max_tokens") or 16)
            completion_tokens = max(1, min(completion_tokens, 1024))
            stream = bool(body.get("stream"))
            model = body.get("model") or MODEL_ID
        except Exception as exc:  # noqa: BLE001 - answer 400 like a real server
            self._send_json(400, {"error": {"message": "bad request: %s" % exc}})
            return

        with STATS_LOCK:
            STATS["requests"] += 1
            n = STATS["requests"]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            reused = 0
            exact_repeat = False
            if CONFIG["cache_reuse"]:
                exact_repeat = STATS["cache"].get(digest) is not None
                if exact_repeat:
                    reused = max(1, prompt_tokens - 1)
                else:
                    # Longest common prefix with any cached prompt (one turn
                    # earlier in a chain).  Tokens are len//4, like prompt_tokens.
                    prefix_chars = 0
                    for old in STATS["cache_text"]:
                        limit = min(len(old), len(content))
                        i = 0
                        while i < limit and old[i] == content[i]:
                            i += 1
                        if i > prefix_chars:
                            prefix_chars = i
                    if prefix_chars:
                        reused = max(0, prefix_chars // 4 - 1)
                if exact_repeat:
                    STATS["cache"][digest] += 1
                else:
                    STATS["cache"][digest] = 1
                STATS["cache_text"].append(content)
                del STATS["cache_text"][:-8]
                if reused:
                    STATS["reused_requests"] += 1
                    STATS["reused_tokens"] += reused
            STATS["prompt_tokens"] += prompt_tokens
            STATS["last_reused"] = reused
            STATS["last_prompt"] = prompt_tokens
            STATS["request_log"].append(content)
            del STATS["request_log"][:-32]
        cid = "chatcmpl-stub-%d" % n
        if exact_repeat:
            # Historical warm shape: full prompt x warm factor.
            prompt_ms = prompt_tokens * CONFIG["prefill_ms_per_token"]
            prompt_ms *= CONFIG["warm_prefill_factor"]
        elif reused:
            # Prefix extension: only the delta is prefilled.
            prompt_ms = max(1, prompt_tokens - reused) * CONFIG["prefill_ms_per_token"]
        else:
            prompt_ms = prompt_tokens * CONFIG["prefill_ms_per_token"]
        predicted_ms = completion_tokens * CONFIG["token_delay_ms"]
        usage = {"prompt_tokens": prompt_tokens,
                 "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        with STATS_LOCK:
            # newest first, like the real engine's `requests` list
            STATS["log"].insert(0, {
                "id": "stubreq-%d" % n, "time": time.time(),
                "duration_s": 0.0, "finish": "length",
                "prompt_tokens": prompt_tokens, "reused": reused,
                "output_tokens": completion_tokens,
                "prompt_ms": prompt_ms,
                "decode_ms": predicted_ms,
                "decode_tok_s": 1000.0 / CONFIG["token_delay_ms"],
            })
            del STATS["log"][32:]

        if not stream:
            time.sleep(min(prompt_ms, 500) / 1000.0)
            self._send_json(200, {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": "stub " * completion_tokens},
                             "finish_reason": "stop"}],
                "usage": usage,
                "timings": {"prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                            "predicted_per_second": 1000.0 / CONFIG["token_delay_ms"]}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            time.sleep(min(prompt_ms, 500) / 1000.0)  # simulated prefill
            for i in range(completion_tokens):
                sse(self, chunk(cid, model, {"content": "tok%04d " % i}))
                time.sleep(CONFIG["token_delay_ms"] / 1000.0)
            sse(self, chunk(cid, model, {}, finish="stop",
                            timings={"prompt_ms": prompt_ms,
                                     "predicted_ms": predicted_ms,
                                     "predicted_per_second":
                                         1000.0 / CONFIG["token_delay_ms"]}))
            sse(self, chunk(cid, model, {}, usage=usage))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass


def main():
    ap = argparse.ArgumentParser(description="stdlib OpenAI-compatible test stub")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1",
                    help="only 127.0.0.1 is intended; do not expose this stub")
    ap.add_argument("--token-delay-ms", type=float, default=10.0, dest="token_delay_ms")
    ap.add_argument("--prefill-ms-per-token", type=float, default=0.025,
                    dest="prefill_ms_per_token")
    ap.add_argument("--cache-reuse", action="store_true",
                    help="simulate a prompt cache: a repeated prompt has "
                         "reused=prompt_tokens-1 and prefill reduced by "
                         "--warm-prefill-factor (off by default)")
    ap.add_argument("--warm-prefill-factor", type=float, default=0.2,
                    dest="warm_prefill_factor")
    args = ap.parse_args()
    CONFIG["token_delay_ms"] = args.token_delay_ms
    CONFIG["prefill_ms_per_token"] = args.prefill_ms_per_token
    CONFIG["cache_reuse"] = bool(args.cache_reuse)
    CONFIG["warm_prefill_factor"] = args.warm_prefill_factor

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print("STUB PORT %d" % httpd.server_address[1], flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
