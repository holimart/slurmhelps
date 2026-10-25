#!/usr/bin/env python3
"""strata_probe.py — OpenAI-compatible streaming benchmark probe (stdlib only).

One "cell" = C concurrent clients, each sending R sequential streaming chat
completions with a prompt calibrated to ~N real tokens (two tiny include_usage
requests measure the chars/token slope and chat-template overhead first;
nominal N is clamped to max_context - max_tokens - pad_safety). Measures,
over the samples:

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


DEFAULT_CHARS_PER_TOKEN = 4.0
DEFAULT_PAD_SAFETY = 128


def build_filler(target_chars, salt, context):
    """Deterministic filler of at least target_chars characters.

    A unique salt line leads every prompt so repeated cells never read the
    user payload from the engine's conversation/prompt cache (which would
    fake up the prefill rate)."""
    target_chars = max(1, int(target_chars))
    parts = ["[strata-probe salt=%s ctx=%d]\n" % (salt, context)]
    n = 0
    have = len(parts[0])
    while have < target_chars:
        line = "record%06d: %s.\n" % (n, FILLER_WORDS)
        parts.append(line)
        have += len(line)
        n += 1
    return "".join(parts)


def build_prompt(context, salt, chars_per_token=DEFAULT_CHARS_PER_TOKEN,
                 intercept_tokens=0.0):
    """Filler scaled so the server tokenizer sees about `context` tokens.

    `chars_per_token` and `intercept_tokens` are measured from real
    include_usage requests (see calibrate_chars_per_token); a zero intercept
    is the last-resort fallback."""
    chars = (float(context) - float(intercept_tokens)) * float(chars_per_token)
    return build_filler(max(1.0, chars), salt, context)


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


def _calibration_point(base, model, temperature, timeout, chars):
    prompt = build_filler(chars, "cal-" + uuid.uuid4().hex[:12], 0)
    rec = stream_one(base, model, prompt, 1, temperature,
                     min(timeout, 120.0))
    return len(prompt), rec.get("prompt_tokens"), rec.get("ok")


def calibrate_chars_per_token(base, model, temperature, timeout, max_context):
    """Measure the tokenizer's chars/token slope and fixed template overhead.

    The old hard-coded 4.0 chars/token overshot by ~14% at large contexts,
    which made nominal 131072-token cells 400s.  A single small prompt is
    biased by the chat-template tokens (it makes the ratio look smaller), so
    measure two small points and fit
        prompt_tokens = intercept + prompt_chars / chars_per_token.
    Both calibration prompts are kept well under the server window.  Returns
    a dict recorded in every JSONL cell.
    """
    points_chars = [2048, 8192]
    if max_context:
        cap = max(512, int(max_context) * 2)
        points_chars = [min(2048, cap // 2), min(8192, cap)]
        if points_chars[1] <= points_chars[0]:
            points_chars = [max(256, cap // 4), max(512, cap // 2)]
    measured = []
    for chars in points_chars:
        got_chars, got_tokens, ok = _calibration_point(
            base, model, temperature, timeout, chars)
        measured.append({"prompt_chars": got_chars,
                         "prompt_tokens": got_tokens, "ok": bool(ok)})
    out = {"source": "include_usage+2point", "points": measured,
           "prompt_chars": measured[0]["prompt_chars"],
           "prompt_tokens": measured[0]["prompt_tokens"],
           "chars_per_token": DEFAULT_CHARS_PER_TOKEN,
           "tokens_per_char": 1.0 / DEFAULT_CHARS_PER_TOKEN,
           "intercept_tokens": 0.0}
    good = [p for p in measured
            if p["ok"] and isinstance(p["prompt_tokens"], int)
            and p["prompt_tokens"] > 0]
    if len(good) >= 2:
        (c1, t1), (c2, t2) = ((good[0]["prompt_chars"], good[0]["prompt_tokens"]),
                              (good[1]["prompt_chars"], good[1]["prompt_tokens"]))
        if c2 != c1 and t2 != t1:
            slope = (c2 - c1) / float(t2 - t1)          # chars per token
            intercept = t1 - (c1 / slope)               # template tokens
            if 1.5 <= slope <= 8.0 and -2048.0 <= intercept <= 2048.0:
                out.update({"chars_per_token": slope,
                            "tokens_per_char": 1.0 / slope,
                            "intercept_tokens": intercept})
                return out
    if good:
        ratio = good[0]["prompt_chars"] / float(good[0]["prompt_tokens"])
        if 1.5 <= ratio <= 8.0:
            out.update({"source": "include_usage+1point",
                        "chars_per_token": ratio,
                        "tokens_per_char": 1.0 / ratio,
                        "intercept_tokens": 0.0})
            return out
    out["source"] = "fallback-4.0"
    out["error"] = "no usable include_usage calibration points"
    return out


def prompt_budget(nominal_context, max_context, max_tokens, safety):
    """Clamp the requested prompt so prompt + answer + safety fits the window."""
    if not max_context:
        return int(nominal_context), False
    budget = int(max_context) - int(max_tokens) - int(safety)
    if budget < 1:
        return 1, True
    if int(nominal_context) > budget:
        return budget, True
    return int(nominal_context), False


def worker(out, lock, base, model, context, prompt_tokens, chars_per_token,
           intercept_tokens, max_tokens, temperature, timeout, count, slot):
    for i in range(count):
        salt = "%s-%d-%d" % (uuid.uuid4().hex[:8], slot, i)
        prompt = build_prompt(prompt_tokens, salt, chars_per_token,
                              intercept_tokens)
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
                    help="nominal prompt size in tokens (clamped to the window)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="server context window; clamps prompt to "
                         "max_context - max_tokens - pad_safety")
    ap.add_argument("--pad-safety", type=int, default=DEFAULT_PAD_SAFETY,
                    help="tokens kept free beyond max_tokens (default 128)")
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

    calibration = calibrate_chars_per_token(base, args.model, args.temperature,
                                            args.timeout, args.max_context)
    prompt_tokens, clamped = prompt_budget(args.context, args.max_context,
                                           args.max_tokens, args.pad_safety)
    if clamped:
        print("[strata-probe] nominal ctx=%d clamped to %d prompt tokens "
              "(max_context=%d max_tokens=%d safety=%d)"
              % (args.context, prompt_tokens, args.max_context, args.max_tokens,
                 args.pad_safety), file=sys.stderr)

    # Calibration runs before this snapshot, so it does not pollute the cell's
    # expert-cache / request metrics.
    metrics_before = metrics_digest(root, args.metrics_timeout)

    results = []
    lock = threading.Lock()
    threads = []
    t_start = time.monotonic()
    for slot in range(args.concurrency):
        th = threading.Thread(target=worker,
                              args=(results, lock, base, args.model, args.context,
                                    prompt_tokens, calibration["chars_per_token"],
                                    calibration.get("intercept_tokens", 0.0),
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
        # prompt calibration / window clamp (harness fix 2026-10-25):
        "prompt_target_tokens": prompt_tokens,
        "prompt_clamped": clamped,
        "max_context": args.max_context,
        "pad_safety": args.pad_safety,
        "calibration": calibration,
    }

    with open(args.json_out, "a") as fh:
        fh.write(json.dumps(line) + "\n")

    print("[strata-probe] ctx=%d target=%d%s C=%d ok=%d/%d decode p50=%s agg=%s "
          "ttft p50=%s cal=%.3f chars/tok+%.0f -> %s"
          % (args.context, prompt_tokens, " (clamped)" if clamped else "",
             args.concurrency, len(ok), len(results),
             "%.1f" % line["decode_tok_s_p50"] if line["decode_tok_s_p50"] else "-",
             "%.1f" % line["aggregate_tok_s"] if line["aggregate_tok_s"] else "-",
             "%.3f" % line["ttft_s_p50"] if line["ttft_s_p50"] else "-",
             calibration["chars_per_token"],
             calibration.get("intercept_tokens", 0.0), args.json_out))
    if err:
        print("[strata-probe] error: %s" % err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
