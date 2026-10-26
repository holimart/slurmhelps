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

Warm-cache mode (opt-in; cold stays the default and its request shape is
unchanged):
  strata_probe.py ... --mode warm --warm-repeats 3 --json-out cell.jsonl

In warm mode each slot sends one saltless prompt, then repeats the SAME
prompt --warm-repeats times.  The per-request lines record TTFT and the
engine's `reused` prompt tokens; the cell adds ttft_cold_s,
ttft_warm_s_p50/p95, reused_tokens_p50, ttft speedup and the engine's
conversation_cache block/budget.  Repeat 1 can itself hit a checkpoint left
by a previous cell with the same context (the prompt is deterministic), so
its reuse is reported rather than assumed zero.

Chain (checkpoint / prefix-growth) mode:
  strata_probe.py ... --mode chain --stops 1024,100000,200000,262144 \\
      --tokens-per-turn 512 --max-context 262144 --json-out turns.jsonl

One fixed salt is chosen per invocation; every turn's prompt is the previous
turn's prompt plus deterministically appended filler lines, so each stop is an
exact prefix extension of the previous one.  The requests are sequential in
ONE conversation slot.  Each turn appends its own JSON line (mode="chain")
with the nominal stop, actual prompt/reused tokens, TTFT, engine prefill time,
delta-prefill tokens, the strict client stream-window decode rate
(decode_tok_s_strict = (N-1)/(t_done-t_first)), TPOT, and the engine's
`reused` count as proof that the checkpoint path engaged.

All modes also record per-request strict timing (decode_s, tpot_ms,
decode_tok_s_strict) and the cell summary carries their p50/p95.  Existing
fields keep their meaning; historical JSONL is untouched.

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


def build_warm_filler(target_chars, context, slot):
    """Saltless filler for the warm mode.

    The prompt carries only a deterministic slot marker, never a random salt:
    every repeat of a slot therefore sends byte-identical messages, which is
    what the engine's prompt/conversation cache keys on.  The slot number
    keeps concurrent slots distinct (with C>1 an identical prompt in four
    slots would be read once and reused by the others, destroying the cold
    reference); a C=1 cell reuses the slot-0 prompt of an earlier C=1 cell,
    which is intentional and reported as a possible first-repeat hit."""
    target_chars = max(1, int(target_chars))
    head = "[strata-probe warm slot=%d ctx=%d]\n" % (slot, context)
    parts = [head]
    n = 0
    have = len(head)
    while have < target_chars:
        line = "record%06d: %s.\n" % (n, FILLER_WORDS)
        parts.append(line)
        have += len(line)
        n += 1
    return "".join(parts)


def build_warm_prompt(context, slot, chars_per_token=DEFAULT_CHARS_PER_TOKEN,
                      intercept_tokens=0.0):
    chars = (float(context) - float(intercept_tokens)) * float(chars_per_token)
    return build_warm_filler(max(1.0, chars), context, slot)


def build_chain_filler(target_chars, salt):
    """Filler for one chain turn; a larger target is a prefix extension.

    The head carries only the ONE salt chosen for the whole chain -- never the
    stop number or a per-turn marker.  Appending `record000000...` lines in
    order therefore makes prompt(stop_i) an exact prefix of prompt(stop_j) for
    every i < j, which is what the engine's checkpoint lookup keys on."""
    target_chars = max(1, int(target_chars))
    # Salt first, so a fresh chain shares no content prefix with calibration
    # prompts or with a chain from an earlier invocation; within this chain the
    # head is byte-identical for every turn.
    head = "%s [chain]\n" % salt
    parts = [head]
    n = 0
    have = len(head)
    while have < target_chars:
        line = "record%06d: %s.\n" % (n, FILLER_WORDS)
        parts.append(line)
        have += len(line)
        n += 1
    return "".join(parts)


def build_chain_prompt(context, salt, chars_per_token=DEFAULT_CHARS_PER_TOKEN,
                       intercept_tokens=0.0):
    chars = (float(context) - float(intercept_tokens)) * float(chars_per_token)
    return build_chain_filler(max(1.0, chars), salt)


def parse_stops(spec):
    """'1024,100000,262144' -> [1024, 100000, 262144] (strictly increasing)."""
    out = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            raise ValueError("stop %r is not an integer" % item)
        if value < 1:
            raise ValueError("stop %d must be >= 1" % value)
        if out and value <= out[-1]:
            raise ValueError("stops must be strictly increasing (%d after %d)"
                             % (value, out[-1]))
        out.append(value)
    if not out:
        raise ValueError("--stops needs at least one size")
    return out


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


def metrics_raw(base_root, timeout):
    """GET /metrics and return the raw dict, or None (best effort)."""
    try:
        raw = api_get_json(base_root.rstrip("/") + "/metrics", timeout)
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return None
    return raw if isinstance(raw, dict) else None


def metrics_digest_from_raw(raw):
    if not isinstance(raw, dict):
        return {"error": "unexpected /metrics payload %s" % type(raw).__name__}
    out = {"_keys": sorted(str(k) for k in raw)[:60]}
    for key, val in raw.items():
        low = str(key).lower()
        if any(t in low for t in METRIC_KEEP):
            out[str(key)] = summarize(val)
    return out


def metrics_digest(base_root, timeout):
    """GET /metrics (Strata answers JSON; vLLM names under vllm:*)."""
    try:
        raw = api_get_json(base_root.rstrip("/") + "/metrics", timeout)
    except Exception as exc:  # noqa: BLE001 - metrics are best-effort
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(raw, dict):
        return {"error": "unexpected /metrics payload %s" % type(raw).__name__}
    return metrics_digest_from_raw(raw)


def conversation_cache_of(raw):
    """The engine's conversation_cache block (scalars only), or None."""
    if not isinstance(raw, dict):
        return None
    cc = raw.get("conversation_cache")
    if not isinstance(cc, dict):
        return None
    return summarize(cc, 0)


def metric_int(block, key):
    if not isinstance(block, dict):
        return None
    val = block.get(key)
    return val if isinstance(val, int) else None


def claim_engine_request(raw, seen_times, lock):
    """Claim one not-yet-seen entry from the engine's `requests` list.

    Strata appends each finished request with `time`, `prompt_tokens`,
    `reused` and `prompt_ms`.  At C=1 the newest unclaimed entry is exactly
    the request that just finished.  With concurrent slots the entries can be
    claimed out of order; the cell-level conversation_cache counters are then
    the authoritative reuse number and the per-request value is best effort.
    """
    if not isinstance(raw, dict):
        return None
    entries = raw.get("requests")
    if not isinstance(entries, list):
        return None
    with lock:
        fresh = [e for e in entries
                 if isinstance(e, dict) and e.get("time") not in seen_times]
        if not fresh:
            return None
        entry = max(fresh, key=lambda e: e.get("time") or 0)
        seen_times.add(entry.get("time"))
    return {"reused": entry.get("reused"),
            "prompt_tokens": entry.get("prompt_tokens"),
            "prompt_ms": entry.get("prompt_ms"),
            "time": entry.get("time")}


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
           "decode_s": None, "tpot_ms": None, "decode_tok_s_strict": None,
           "engine_prompt_ms": None, "engine_predicted_ms": None,
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

    # Strict client-side decode window: first content delta to stream close.
    # TPOT excludes the first token by definition, so N-1 intervals.
    if t_first is not None:
        decode_s = t_done - t_first
        rec["decode_s"] = decode_s
        if completion_tokens and completion_tokens >= 2 and decode_s > 0:
            rec["tpot_ms"] = (decode_s / float(completion_tokens - 1)) * 1000.0
            rec["decode_tok_s_strict"] = (completion_tokens - 1) / decode_s

    # Decode rate: engine predicted_ms when the server reports timings
    # (llama.cpp names, BENCHMARK_PROTOCOL), else the client decode window.
    pred_ms = timings.get("predicted_ms") if isinstance(timings, dict) else None
    rec["engine_predicted_ms"] = pred_ms
    if pred_ms and completion_tokens:
        rec["decode_tok_s"] = completion_tokens / (pred_ms / 1000.0)
        rec["timing_source"] = "engine.timings.predicted_ms"
    elif t_first is not None and completion_tokens and t_done > t_first:
        rec["decode_tok_s"] = completion_tokens / (t_done - t_first)
        rec["timing_source"] = "client.stream-window"
    if rec["decode_tok_s"] is None and rec["decode_tok_s_strict"] is not None:
        rec["decode_tok_s"] = rec["decode_tok_s_strict"]
        rec["timing_source"] = "client.stream-window-strict"

    prompt_ms = timings.get("prompt_ms") if isinstance(timings, dict) else None
    rec["engine_prompt_ms"] = prompt_ms
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


def warm_worker(out, lock, base, model, root, context, prompt_tokens,
                chars_per_token, intercept_tokens, max_tokens, temperature,
                timeout, warm_repeats, metrics_timeout, slot, seen_times):
    """One cold (saltless) request, then `warm_repeats` identical ones.

    Per-request engine reuse is read from /metrics after each stream.  The
    prompt has no random salt, so repeat 2..N hit whatever prefix the engine
    kept from repeat 1 (prompt/conversation cache)."""
    prompt = build_warm_prompt(prompt_tokens, slot, chars_per_token,
                               intercept_tokens)
    for i in range(1 + warm_repeats):
        rec = stream_one(base, model, prompt, max_tokens, temperature, timeout)
        rec["slot"] = slot
        rec["seq"] = i
        rec["request_index"] = i + 1      # 1 = initial cold, 2.. = warm
        rec["repeat"] = i + 1
        rec["warm"] = i > 0
        rec["saltless"] = True
        raw = metrics_raw(root, metrics_timeout)
        claim = claim_engine_request(raw, seen_times, lock) if raw else None
        if claim:
            rec["reused_tokens"] = claim.get("reused")
            rec["engine_prompt_tokens"] = claim.get("prompt_tokens")
            rec["engine_prompt_ms"] = claim.get("prompt_ms")
        else:
            rec["reused_tokens"] = None
            rec["engine_prompt_tokens"] = None
            rec["engine_prompt_ms"] = None
        with lock:
            out.append(rec)


def warm_cell_fields(results, ok, cc_before, cc_after, warm_repeats):
    """Extra JSONL fields for --mode warm (never emitted in cold mode)."""
    def nums(rows, key):
        return [r[key] for r in rows
                if isinstance(r.get(key), (int, float))]

    cold_rows = [r for r in ok if r.get("request_index") == 1]
    warm_rows = [r for r in ok if r.get("request_index", 0) >= 2]
    cold_ttft = nums(cold_rows, "ttft_s")
    warm_ttft = nums(warm_rows, "ttft_s")
    warm_reuse = nums(warm_rows, "reused_tokens")

    by_repeat = []
    for idx in range(1, 1 + warm_repeats + 1):
        rows = [r for r in ok if r.get("request_index") == idx]
        by_repeat.append({
            "repeat": idx,
            "role": "cold" if idx == 1 else "warm",
            "requests": len(rows),
            "ttft_s_p50": percentile(nums(rows, "ttft_s"), 0.50),
            "ttft_s_p95": percentile(nums(rows, "ttft_s"), 0.95),
            "reused_tokens_p50": percentile(nums(rows, "reused_tokens"), 0.50),
        })

    before_reused = metric_int(cc_before, "reused_tokens")
    after_reused = metric_int(cc_after, "reused_tokens")
    before_reqs = metric_int(cc_before, "requests_reused")
    after_reqs = metric_int(cc_after, "requests_reused")
    reused_total = None
    if before_reused is not None and after_reused is not None:
        reused_total = after_reused - before_reused
    requests_reused_total = None
    if before_reqs is not None and after_reqs is not None:
        requests_reused_total = after_reqs - before_reqs

    cold_p50 = percentile(cold_ttft, 0.50)
    warm_p50 = percentile(warm_ttft, 0.50)
    speedup = None
    if cold_p50 and warm_p50:
        speedup = cold_p50 / warm_p50

    per_request = []
    for r in sorted(results, key=lambda x: (x.get("slot", 0),
                                            x.get("request_index", 0))):
        per_request.append({
            "request_index": r.get("request_index"),
            "slot": r.get("slot"),
            "ttft_s": r.get("ttft_s"),
            "prompt_tokens": r.get("prompt_tokens"),
            "reused_tokens": r.get("reused_tokens"),
            "engine_prompt_tokens": r.get("engine_prompt_tokens"),
            "engine_prompt_ms": r.get("engine_prompt_ms"),
            "prefill_tok_s": r.get("prefill_tok_s"),
            "ok": r.get("ok"),
            "error": r.get("error"),
        })

    return {
        "mode": "warm",
        "warm_repeats": warm_repeats,
        "total_requests_per_slot": 1 + warm_repeats,
        "ttft_cold_s": cold_p50,
        "ttft_cold_s_p95": percentile(cold_ttft, 0.95),
        "ttft_warm_s_p50": warm_p50,
        "ttft_warm_s_p95": percentile(warm_ttft, 0.95),
        "warm_ttft_speedup": speedup,
        "reused_tokens_p50": percentile(warm_reuse, 0.50),
        "reused_tokens_total": reused_total,
        "requests_reused_total": requests_reused_total,
        "cache_reuse_engaged": bool((reused_total or 0) > 0
                                    or (requests_reused_total or 0) > 0),
        "conversation_cache_before": cc_before,
        "conversation_cache_after": cc_after,
        "conversation_cache_budget_mib": metric_int(cc_after, "budget_mib"),
        "conversation_cache_slots": metric_int(cc_after, "slots"),
        "repeat_breakdown": by_repeat,
        "per_request": per_request,
    }


def chain_schedule(stops, last_repeats):
    """(stop, repeat, turn) triples: one turn per stop, optional tail repeats."""
    schedule = [(stop, 1, idx + 1) for idx, stop in enumerate(stops)]
    for rep in range(int(last_repeats)):
        schedule.append((stops[-1], rep + 2, len(stops)))
    return schedule


def chain_turn_record(rec, claim, stop, prev_stop, turn, repeat,
                      prev_prompt_tokens, gen_tokens, max_context, safety,
                      target_tokens, clamped, calibration, salt, cc_now):
    """One JSONL line for a chain turn (the engine's `reused` is the proof)."""
    prompt_tokens = rec.get("prompt_tokens")
    engine_prompt_tokens = claim.get("prompt_tokens") if claim else None
    if prompt_tokens is None:
        prompt_tokens = engine_prompt_tokens
    reused = claim.get("reused") if claim else None

    reuse_fraction = None
    delta_tokens = None
    if prompt_tokens is not None:
        if reused is not None:
            reuse_fraction = (float(reused) / float(prompt_tokens)
                              if prompt_tokens else None)
            delta_tokens = max(0, int(prompt_tokens) - int(reused))
        else:
            delta_tokens = int(prompt_tokens)

    prefill_s = None
    timing_source = None
    claim_ms = claim.get("prompt_ms") if claim else None
    if claim_ms:
        prefill_s = float(claim_ms) / 1000.0
        timing_source = "engine.metrics.prompt_ms"
    elif rec.get("engine_prompt_ms"):
        prefill_s = float(rec["engine_prompt_ms"]) / 1000.0
        timing_source = "engine.timings.prompt_ms"
    elif rec.get("ttft_s") is not None:
        prefill_s = rec["ttft_s"]
        timing_source = "client.ttft"

    delta_tok_s = None
    if delta_tokens is not None and prefill_s and prefill_s > 0:
        delta_tok_s = delta_tokens / prefill_s

    completion_tokens = rec.get("completion_tokens")
    total_s = rec.get("total_s")
    aggregate = None
    if completion_tokens and total_s and total_s > 0:
        aggregate = completion_tokens / total_s

    strict = rec.get("decode_tok_s_strict")
    decode_tok_s = rec.get("decode_tok_s")
    if decode_tok_s is None:
        decode_tok_s = strict

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "chain",
        "model": None,                      # filled by the caller
        "stop": stop,
        "prev_stop": prev_stop,
        "turn": turn,
        "repeat": repeat,
        "prompt_tokens": prompt_tokens,
        "prev_prompt_tokens": prev_prompt_tokens,
        "reused_tokens": reused,
        "reuse_fraction": reuse_fraction,
        "ttft_s": rec.get("ttft_s"),
        "prefill_s": prefill_s,
        "prefill_delta_tokens": delta_tokens,
        "prefill_delta_tok_s": delta_tok_s,
        "decode_s": rec.get("decode_s"),
        "tpot_ms": rec.get("tpot_ms"),
        "decode_tok_s_strict": strict,
        "decode_tok_s": decode_tok_s,
        "aggregate_tok_s": aggregate,
        "completion_tokens": completion_tokens,
        "timing_source": timing_source,
        "status": "ok" if rec.get("ok") else "fail",
        "error": rec.get("error"),
        "context": stop,
        "concurrency": 1,
        "max_tokens": gen_tokens,
        "engine_prompt_tokens": engine_prompt_tokens,
        "engine_prompt_ms": claim_ms,
        "prompt_target_tokens": target_tokens,
        "prompt_clamped": clamped,
        "max_context": max_context,
        "pad_safety": safety,
        "calibration": calibration,
        "chain_salt": salt,
        "conversation_cache": cc_now,
    }


def chain_worker(out, lock, base, model, root, stops, chars_per_token,
                 intercept_tokens, gen_tokens, temperature, timeout,
                 metrics_timeout, max_context, safety, last_repeats,
                 seen_times, salt, calibration):
    """Sequential requests in ONE conversation slot; never concurrent."""
    prev_stop = None
    prev_prompt_tokens = None
    for stop, repeat, turn in chain_schedule(stops, last_repeats):
        target_tokens, clamped = prompt_budget(stop, max_context, gen_tokens,
                                               safety)
        prompt = build_chain_prompt(target_tokens, salt, chars_per_token,
                                    intercept_tokens)
        rec = stream_one(base, model, prompt, gen_tokens, temperature, timeout)
        raw = metrics_raw(root, metrics_timeout)
        claim = claim_engine_request(raw, seen_times, lock) if raw else None
        record = chain_turn_record(
            rec, claim, stop, prev_stop, turn, repeat, prev_prompt_tokens,
            gen_tokens, max_context, safety, target_tokens, clamped,
            calibration, salt, conversation_cache_of(raw))
        record["model"] = model
        with lock:
            out.append(record)
            prev_prompt_tokens = record.get("prompt_tokens") or prev_prompt_tokens
            prev_stop = stop


def run_chain_mode(args, base, root, calibration, model_warn):
    """--mode chain: append one JSON line per turn, then a stderr summary."""
    stops = parse_stops(args.stops)
    if args.concurrency != 1:
        print("[strata-probe] chain mode is one conversation slot; "
              "--concurrency must be 1", file=sys.stderr)
        return 2
    if args.tokens_per_turn < 2:
        print("[strata-probe] --tokens-per-turn must be >= 2 for TPOT",
              file=sys.stderr)
        return 2
    if args.max_context and stops[-1] > args.max_context:
        print("[strata-probe] chain last stop %d exceeds --max-context %d; "
              "it will be clamped per turn" % (stops[-1], args.max_context),
              file=sys.stderr)

    raw_before = metrics_raw(root, args.metrics_timeout)
    cc_before = conversation_cache_of(raw_before)
    seen_times = set()
    if raw_before is not None:
        for entry in raw_before.get("requests") or []:
            if isinstance(entry, dict):
                seen_times.add(entry.get("time"))

    salt = uuid.uuid4().hex[:12]
    out = []
    lock = threading.Lock()
    t0 = time.monotonic()
    chain_worker(out, lock, base, args.model, root, stops,
                 calibration["chars_per_token"],
                 calibration.get("intercept_tokens", 0.0),
                 args.tokens_per_turn, args.temperature, args.timeout,
                 args.metrics_timeout, args.max_context, args.pad_safety,
                 args.chain_last_repeats, seen_times, salt, calibration)
    wall = time.monotonic() - t0

    with open(args.json_out, "a") as fh:
        for record in out:
            fh.write(json.dumps(record) + "\n")

    print("[strata-probe] chain salt=%s stops=%s tokens/turn=%d turns=%d "
          "wall=%.1fs -> %s"
          % (salt, ",".join(str(s) for s in stops), args.tokens_per_turn,
             len(out), wall, args.json_out))
    for record in out:
        print("[strata-probe]   turn %d stop=%d%s prompt=%s reused=%s "
              "ttft=%s prefill=%s delta=%s strict=%s tpot=%s agg=%s %s"
              % (record["turn"], record["stop"],
                 " r%d" % record["repeat"] if record["repeat"] > 1 else "",
                 record["prompt_tokens"], record["reused_tokens"],
                 "%.3f" % record["ttft_s"] if record["ttft_s"] is not None else "-",
                 "%.3f" % record["prefill_s"] if record["prefill_s"] is not None else "-",
                 record["prefill_delta_tokens"],
                 "%.1f" % record["decode_tok_s_strict"]
                 if record["decode_tok_s_strict"] is not None else "-",
                 "%.2f" % record["tpot_ms"]
                 if record["tpot_ms"] is not None else "-",
                 "%.2f" % record["aggregate_tok_s"]
                 if record["aggregate_tok_s"] is not None else "-",
                 record["status"]))

    failed = [r for r in out if r["status"] != "ok"]
    if model_warn and not out:
        print("[strata-probe] error: %s" % model_warn, file=sys.stderr)
        return 1
    if failed:
        print("[strata-probe] error: %d/%d chain turns failed (first: %s)"
              % (len(failed), len(out), failed[0].get("error")),
              file=sys.stderr)
        return 1
    if not any((r.get("reused_tokens") or 0) > 0 for r in out[1:]):
        print("[strata-probe] warning: no checkpoint reuse reported on any "
              "growth turn (reused_tokens=0 or unavailable)", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description="OpenAI streaming benchmark probe")
    ap.add_argument("--base-url", required=True,
                    help="server root, e.g. http://127.0.0.1:8080 (…/v1 also accepted)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=None,
                    help="nominal prompt size in tokens (clamped to the window); "
                         "not used by --mode chain (see --stops)")
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
    ap.add_argument("--mode", choices=("cold", "warm", "chain"), default="cold",
                    help="cold (default): random-salt prompts, every request a "
                         "full prefill; warm: one saltless cold request followed "
                         "by --warm-repeats identical ones so prompt/conversation "
                         "cache reuse is measured; chain: one conversation whose "
                         "prompt grows through --stops (prefix growth)")
    ap.add_argument("--warm-repeats", type=int, default=3,
                    help="warm repeats after the initial saltless request "
                         "(default 3; total requests per slot = 1 + N)")
    ap.add_argument("--stops", default="1000,100000,200000,262144",
                    help="chain mode: comma-separated prompt sizes reached by "
                         "prefix extension (default 1000,100000,200000,262144)")
    ap.add_argument("--tokens-per-turn", type=int, default=512,
                    dest="tokens_per_turn",
                    help="chain mode: completion tokens generated on every turn "
                         "(default 512)")
    ap.add_argument("--chain-last-repeats", type=int, default=0,
                    dest="chain_last_repeats",
                    help="chain mode: extra repeats of the last stop (default 0)")
    args = ap.parse_args()
    if args.mode != "chain" and args.context is None:
        ap.error("--context is required unless --mode chain")
    if args.warm_repeats < 1:
        ap.error("--warm-repeats must be >= 1")
    if args.tokens_per_turn < 1:
        ap.error("--tokens-per-turn must be >= 1")
    if args.chain_last_repeats < 0:
        ap.error("--chain-last-repeats must be >= 0")

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

    if args.mode == "chain":
        try:
            return run_chain_mode(args, base, root, calibration, model_warn)
        except ValueError as exc:
            ap.error(str(exc))

    prompt_tokens, clamped = prompt_budget(args.context, args.max_context,
                                           args.max_tokens, args.pad_safety)
    if clamped:
        print("[strata-probe] nominal ctx=%d clamped to %d prompt tokens "
              "(max_context=%d max_tokens=%d safety=%d)"
              % (args.context, prompt_tokens, args.max_context, args.max_tokens,
                 args.pad_safety), file=sys.stderr)

    # Calibration runs before this snapshot, so it does not pollute the cell's
    # expert-cache / request metrics.
    warm = args.mode == "warm"
    seen_times = set()
    cc_before = None
    if warm:
        raw_before = metrics_raw(root, args.metrics_timeout)
        metrics_before = (metrics_digest_from_raw(raw_before) if raw_before is not None
                          else {"error": "metrics unavailable"})
        cc_before = conversation_cache_of(raw_before)
        if raw_before is not None:
            for entry in raw_before.get("requests") or []:
                if isinstance(entry, dict):
                    seen_times.add(entry.get("time"))
    else:
        metrics_before = metrics_digest(root, args.metrics_timeout)

    results = []
    lock = threading.Lock()
    threads = []
    t_start = time.monotonic()
    for slot in range(args.concurrency):
        if warm:
            th = threading.Thread(target=warm_worker,
                                  args=(results, lock, base, args.model, root, args.context,
                                        prompt_tokens, calibration["chars_per_token"],
                                        calibration.get("intercept_tokens", 0.0),
                                        args.max_tokens, args.temperature,
                                        args.timeout, args.warm_repeats,
                                        args.metrics_timeout, slot, seen_times))
        else:
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
    if warm and ok and not any(r.get("request_index", 0) >= 2 for r in ok):
        err = "no warm repeat succeeded"

    if warm:
        raw_after = metrics_raw(root, args.metrics_timeout)
        metrics_after = (metrics_digest_from_raw(raw_after) if raw_after is not None
                         else {"error": "metrics unavailable"})
        cc_after = conversation_cache_of(raw_after)
    else:
        metrics_after = metrics_digest(root, args.metrics_timeout)
        cc_after = None

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
        "engine_metrics": {"before": metrics_before, "after": metrics_after},
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
    # Strict client stream-window metrics (added 2026-10-26).  These are
    # additive: decode_tok_s / aggregate_tok_s above keep their old meaning.
    decode_s_samples = [r["decode_s"] for r in ok
                        if isinstance(r.get("decode_s"), (int, float))]
    tpot_samples = [r["tpot_ms"] for r in ok
                    if isinstance(r.get("tpot_ms"), (int, float))]
    strict_samples = [r["decode_tok_s_strict"] for r in ok
                      if isinstance(r.get("decode_tok_s_strict"), (int, float))]
    line.update({
        "decode_s_p50": percentile(decode_s_samples, 0.50),
        "decode_s_p95": percentile(decode_s_samples, 0.95),
        "tpot_ms_p50": percentile(tpot_samples, 0.50),
        "tpot_ms_p95": percentile(tpot_samples, 0.95),
        "decode_tok_s_strict_p50": percentile(strict_samples, 0.50),
        "decode_tok_s_strict_p95": percentile(strict_samples, 0.95),
    })
    if warm:
        line.update(warm_cell_fields(results, ok, cc_before, cc_after,
                                     args.warm_repeats))

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
    if warm:
        print("[strata-probe] warm ctx=%d C=%d cold=%s warm p50=%s p95=%s "
              "reused p50=%s engaged=%s -> %s"
              % (args.context, args.concurrency,
                 "%.3f" % line["ttft_cold_s"] if line.get("ttft_cold_s") else "-",
                 "%.3f" % line["ttft_warm_s_p50"] if line.get("ttft_warm_s_p50") else "-",
                 "%.3f" % line["ttft_warm_s_p95"] if line.get("ttft_warm_s_p95") else "-",
                 line.get("reused_tokens_p50"),
                 line.get("cache_reuse_engaged"), args.json_out))
    if err:
        print("[strata-probe] error: %s" % err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
