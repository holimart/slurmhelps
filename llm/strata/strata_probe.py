#!/usr/bin/env python3
"""strata_probe.py — OpenAI-compatible streaming benchmark probe (stdlib only).

One "cell" = C concurrent clients, each sending R sequential streaming chat
completions with a prompt padded to ~N tokens. Measures, over the samples:

  - TTFT (client monotonic time from dispatch to first content/reasoning/
    tool-call delta — the definition in Strata's docs/BENCHMARK_PROTOCOL.md)
  - per-stream decode tok/s   (completion_tokens / time-after-first-token)
  - per-request prefill tok/s (prompt_tokens / engine timings.prompt_ms when
    the server reports them — llama.cpp-style `timings` on the final chunk —
    else prompt_tokens / ttft)
  - aggregate tok/s           (total completion tokens of ok requests / wall
    time from first dispatch to last termination, protocol definition)
  - p50/p95 over the per-request samples

Pulls the vLLM-style /metrics before and after for expert-cache and GPU/RAM
context (Strata serves vLLM metric names on /metrics, docs/DETAILS.md).

Appends ONE JSON line to --json-out (fields: ts, model, context, concurrency,
prefill_tok_s, decode_tok_s_p50, decode_tok_s_p95, aggregate_tok_s,
ttft_s_p50, ttft_s_p95, completion_tokens, prompt_tokens, engine_metrics,
error), even when the cell failed.

Usage:
  strata_probe.py --base-url http://127.0.0.1:8080 --model NAME \
      --context 4096 --concurrency 4 --max-tokens 128 \
      --requests-per-slot 3 --json-out cell.jsonl

Patterned after /work/ffa/holecem/scratch-claude/qwen_smoke.py (read-only).
"""
import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

FILLER_WORDS = ("alpha beta gamma delta epsilon zeta eta theta "
                "iota kappa lambda mu nu xi omicron pi rho sigma")


def build_prompt(context, salt):
    """Deterministic filler padded to ~context tokens at ~4 chars/token.

    A unique salt line leads every prompt so repeated cells never read the
    user payload from the engine's conversation/prompt cache (which would
    fake up the prefill rate)."""
    target_chars = int(context) * 4
    parts = ["[strata-probe salt=%s ctx=%d]\n" % (salt, context)]
    n = 0
    have = len(parts[0])
    while have < target_chars:
        line = "record%06d: %s.\n" % (n, FILLER_WORDS)
        parts.append(line)
        have += len(line)
        n += 1
    return "".join(parts)


def percentile(values, q):
    """Nearest-rank percentile; None on empty."""
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(math.ceil(q * len(s))) - 1)
    return s[idx]


def api_get_json(url, timeout):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(4 * 1024 * 1024)
    return json.loads(body.decode("utf-8", "replace"))


def summarize(value, depth=0):
    """Bound arbitrary metric values so one JSONL line stays readable."""
    if isinstance(value, dict):
        if depth >= 2:
            return "<dict %d keys>" % len(value)
        return {str(k): summarize(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [summarize(v, depth + 1) for v in value[:6]]
    if isinstance(value, str):
        return value[:200]
    if isinstance(value, float):
        return round(value, 4)
    return value


METRIC_KEEP = ("expert", "vllm", "strata", "gpu", "ram", "cache", "slot",
               "token", "request", "context", "queue", "admission", "load",
               "kv", "memory")


def metrics_digest(base_root, timeout):
    """GET /metrics (Strata answers JSON; vLLM names under vllm:*)."""
    try:
        raw = api_get_json(base_root.rstrip("/") + "/metrics", timeout)
    except Exception as exc:  # noqa: BLE001 - metrics are best-effort
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(raw, dict):
        return {"error": "unexpected /metrics payload %s" % type(raw).__name__}
    out = {"_keys": sorted(str(k) for k in raw)[:60]}
    for key, val in raw.items():
        low = str(key).lower()
        if any(t in low for t in METRIC_KEEP):
            out[str(key)] = summarize(val)
    return out


def chat_root(base_url):
    """Normalize --base-url to the /v1 root of the server."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def host_root(base_url):
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-len("/v1")]
    return b


def stream_one(base, model, prompt, max_tokens, temperature, timeout):
    """One streaming chat completion. Returns a result dict, never raises."""
    rec = {"ok": False, "error": None, "prompt_tokens": None,
           "completion_tokens": None, "ttft_s": None, "total_s": None,
           "decode_tok_s": None, "prefill_tok_s": None,
           "timing_source": None, "finish_reason": None}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"})
    t0 = time.monotonic()
    t_first = None
    content_chunks = 0
    usage = None
    timings = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if isinstance(chunk.get("timings"), dict):
                    timings = chunk["timings"]
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning"))
                    has_tool = bool(delta.get("tool_calls"))
                    if (piece or has_tool) and t_first is None:
                        t_first = time.monotonic() - t0
                    if piece:
                        content_chunks += 1
    except Exception as exc:  # noqa: BLE001 - report, do not crash the cell
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["total_s"] = time.monotonic() - t0
        return rec

    t_done = time.monotonic() - t0
    rec["total_s"] = t_done
    rec["finish_reason"] = finish
    prompt_tokens = None
    completion_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = content_chunks  # last resort; counted chunks
    rec["prompt_tokens"] = prompt_tokens
    rec["completion_tokens"] = completion_tokens
    rec["ttft_s"] = t_first

    # Decode rate: engine predicted_ms when the server reports timings
    # (llama.cpp names, BENCHMARK_PROTOCOL), else the client decode window.
    pred_ms = timings.get("predicted_ms") if isinstance(timings, dict) else None
    if pred_ms and completion_tokens:
        rec["decode_tok_s"] = completion_tokens / (pred_ms / 1000.0)
        rec["timing_source"] = "engine.timings.predicted_ms"
    elif t_first is not None and completion_tokens and t_done > t_first:
        rec["decode_tok_s"] = completion_tokens / (t_done - t_first)
        rec["timing_source"] = "client.stream-window"

    prompt_ms = timings.get("prompt_ms") if isinstance(timings, dict) else None
    if prompt_ms and prompt_tokens:
        rec["prefill_tok_s"] = prompt_tokens / (prompt_ms / 1000.0)
    elif t_first and prompt_tokens:
        rec["prefill_tok_s"] = prompt_tokens / t_first

    if t_first is None or not completion_tokens:
        rec["error"] = rec["error"] or "no content delta or no completion tokens"
    else:
        rec["ok"] = True
    return rec


def worker(out, lock, base, model, context, max_tokens, temperature,
           timeout, count, slot):
    for i in range(count):
        salt = "%s-%d-%d" % (uuid.uuid4().hex[:8], slot, i)
        prompt = build_prompt(context, salt)
        rec = stream_one(base, model, prompt, max_tokens, temperature, timeout)
        rec["slot"] = slot
        rec["seq"] = i
        with lock:
            out.append(rec)


def main():
    ap = argparse.ArgumentParser(description="OpenAI streaming benchmark probe")
    ap.add_argument("--base-url", required=True,
                    help="server root, e.g. http://127.0.0.1:8080 (…/v1 also accepted)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, required=True,
                    help="prompt size in tokens (~4 chars/token filler)")
    ap.add_argument("--concurrency", type=int, required=True, help="parallel clients C")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests-per-slot", type=int, default=3, dest="rps")
    ap.add_argument("--json-out", required=True, help="JSONL results file to append")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-request wall-clock timeout (protocol default 1800 s)")
    ap.add_argument("--metrics-timeout", type=float, default=15.0)
    args = ap.parse_args()

    base = chat_root(args.base_url)
    root = host_root(args.base_url)

    # Confirm the server answers and the model id exists (warn, do not fail:
    # bench-matrix treats any produced line with `error` as the record).
    model_warn = None
    try:
        models = api_get_json(base + "/models", min(args.metrics_timeout, 30))
        ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        if ids and args.model not in ids:
            model_warn = "model %r not in /v1/models ids %s" % (args.model, ids)
    except Exception as exc:  # noqa: BLE001
        model_warn = "models probe failed: %s: %s" % (type(exc).__name__, exc)

    metrics_before = metrics_digest(root, args.metrics_timeout)

    results = []
    lock = threading.Lock()
    threads = []
    t_start = time.monotonic()
    for slot in range(args.concurrency):
        th = threading.Thread(target=worker,
                              args=(results, lock, base, args.model, args.context,
                                    args.max_tokens, args.temperature,
                                    args.timeout, args.rps, slot))
        th.daemon = True
        threads.append(th)
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t_end = time.monotonic()
    # dispatch instant of the very first request approximates t_start; use the
    # earliest per-request start we can derive (total_s back-calculation).
    first_dispatch = t_start

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    decode_samples = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    prefill_samples = [r["prefill_tok_s"] for r in ok if r["prefill_tok_s"]]
    ttft_samples = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    total_completion = sum(r["completion_tokens"] or 0 for r in ok)
    total_prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    wall = t_end - first_dispatch

    err = None
    if model_warn and not ok:
        err = model_warn
    elif not ok:
        first_err = failed[0]["error"] if failed else "no samples"
        err = "all %d requests failed (first: %s)" % (len(failed), first_err)
    elif failed:
        err = None  # partial success is still a valid cell; counts are kept

    line = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "context": args.context,
        "concurrency": args.concurrency,
        "prefill_tok_s": percentile(prefill_samples, 0.50),
        "decode_tok_s_p50": percentile(decode_samples, 0.50),
        "decode_tok_s_p95": percentile(decode_samples, 0.95),
        "aggregate_tok_s": (total_completion / wall) if wall > 0 and total_completion else None,
        "ttft_s_p50": percentile(ttft_samples, 0.50),
        "ttft_s_p95": percentile(ttft_samples, 0.95),
        "completion_tokens": total_completion,
        "prompt_tokens": total_prompt,
        "engine_metrics": {"before": metrics_before, "after": metrics_digest(root, args.metrics_timeout)},
        "error": err,
        # extras (diagnostics beyond the agreed cell contract):
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "samples": len(ok),
        "wall_s": round(wall, 3),
        "max_tokens": args.max_tokens,
        "requests_per_slot": args.rps,
    }

    with open(args.json_out, "a") as fh:
        fh.write(json.dumps(line) + "\n")

    print("[strata-probe] ctx=%d C=%d ok=%d/%d decode p50=%s agg=%s ttft p50=%s -> %s"
          % (args.context, args.concurrency, len(ok), len(results),
             "%.1f" % line["decode_tok_s_p50"] if line["decode_tok_s_p50"] else "-",
             "%.1f" % line["aggregate_tok_s"] if line["aggregate_tok_s"] else "-",
             "%.3f" % line["ttft_s_p50"] if line["ttft_s_p50"] else "-",
             args.json_out))
    if err:
        print("[strata-probe] error: %s" % err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
