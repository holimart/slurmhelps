#!/usr/bin/env python3
"""stability.py — R3: long-run 262k stability loop on the canonical probe.

One-off loop, but built from the same contract pieces as every other cell in
this campaign:

  * `strata_probe.calibrate_chars_per_token()` — the same two-point
    chars/token fit (the previous soak's HTTP 400s came from a probe that
    guessed 4 chars/token and overran the window);
  * `strata_probe.build_prompt()` for the **cold** turn (fresh random salt,
    so it is a real full prefill every iteration);
  * `strata_probe.build_warm_prompt()` for the **warm** turn (byte-identical
    slot-0 prompt, so the engine's prompt/conversation cache can reuse it);
  * `strata_probe.stream_one()` for the SSE/TTFT client;
  * `strata_probe.metrics_raw()` + `claim_engine_request()` for the engine's
    `reused` value on every request.

Each request is one JSONL line (ttft_s, prompt_tokens, completion_tokens,
decode_tok_s, reused_tokens, ok, error), so an HTTP 400 is visible with its
exact body instead of being collapsed into a cell-level counter.  After each
iteration one `kind=sample` line records MemAvailable and per-GPU VRAM.

Usage:
  stability.py --base-url http://127.0.0.1:8080 --model strata \
      --context 262144 --max-context 262144 --max-tokens 128 \
      --duration-min 45 --json-out cells.jsonl --summary-out summary.json
"""
import argparse
import json
import signal
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "strata"))
sys.path.insert(0, STRATA)
import strata_probe as sp  # noqa: E402


def meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable", "MemFree"):
                    out[key] = int(rest.split()[0]) // 1024  # MiB
    except OSError:
        pass
    return out


def gpu_memory():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        vals = []
        for line in raw.strip().splitlines():
            used, _, power = line.partition(",")
            vals.append({"used_mib": int(used.strip()),
                         "power_w": float(power.strip()) if power.strip() else None})
        return vals
    except Exception:  # noqa: BLE001
        return []


def request(base, model, prompt, max_tokens, temperature, timeout,
            seen, lock, metrics_timeout):
    rec = sp.stream_one(base, model, prompt, max_tokens, temperature, timeout)
    raw = sp.metrics_raw(sp.host_root(base), metrics_timeout)
    claim = sp.claim_engine_request(raw, seen, lock) if raw else None
    rec["reused_tokens"] = claim.get("reused") if claim else None
    rec["engine_prompt_tokens"] = claim.get("prompt_tokens") if claim else None
    rec["engine_prompt_ms"] = claim.get("prompt_ms") if claim else None
    rec["engine_request_time"] = claim.get("time") if claim else None
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--pad-safety", type=int, default=128)
    ap.add_argument("--warm-repeats", type=int, default=2,
                    help="warm turns after the first saltless turn (default 2)")
    ap.add_argument("--duration-min", type=float, default=45.0)
    ap.add_argument("--max-iterations", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--metrics-timeout", type=float, default=15.0)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    base = sp.chat_root(args.base_url)
    cal = sp.calibrate_chars_per_token(base, args.model, args.temperature,
                                       args.timeout, args.max_context)
    cpt = cal.get("chars_per_token") or 4.0
    intercept = cal.get("intercept_tokens") or 0.0
    prompt_tokens, clamped = sp.prompt_budget(args.context, args.max_context,
                                              args.max_tokens, args.pad_safety)
    print("[stability] calibration chars/token=%.3f intercept=%.0f "
          "target=%d clamped=%s" % (cpt, intercept, prompt_tokens, clamped),
          file=sys.stderr)

    # one prompt builder per turn kind; both are calibrated (no 4.0 guess)
    cold_prompt = lambda: sp.build_prompt(prompt_tokens, uuid.uuid4().hex[:12],
                                          cpt, intercept)
    warm_prompt = sp.build_warm_prompt(prompt_tokens, 0, cpt, intercept)

    import threading
    lock = threading.Lock()
    seen = set()
    deadline = time.monotonic() + args.duration_min * 60.0
    started = time.monotonic()
    stop = {"now": False}

    def on_term(signum, frame):  # Slurm's TERM on the wall clock
        stop["now"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    rows = []
    iteration = 0
    consecutive_failed_iterations = 0
    aborted = None

    def emit(row):
        row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(row, sort_keys=False) + "\n")
        rows.append(row)

    while time.monotonic() < deadline and not stop["now"]:
        if args.max_iterations and iteration >= args.max_iterations:
            break
        iteration += 1
        # --- cold turn: unique salt => full 262k prefill -------------------
        rec = request(base, args.model, cold_prompt(), args.max_tokens,
                      args.temperature, args.timeout, seen, lock,
                      args.metrics_timeout)
        emit({"kind": "cold", "iteration": iteration, **rec})
        # --- warm turn: identical prompt, one initial + N repeats ----------
        for i in range(1 + args.warm_repeats):
            if time.monotonic() >= deadline or stop["now"]:
                break
            rec = request(base, args.model, warm_prompt, args.max_tokens,
                          args.temperature, args.timeout, seen, lock,
                          args.metrics_timeout)
            emit({"kind": "warm" if i else "warm-cold",
                  "iteration": iteration, "request_index": i + 1, **rec})
        mi = meminfo()
        emit({"kind": "sample", "iteration": iteration,
              "mem_total_mib": mi.get("MemTotal"),
              "mem_available_mib": mi.get("MemAvailable"),
              "gpus": gpu_memory()})
        iteration_requests = [r for r in rows
                              if r.get("iteration") == iteration
                              and r.get("kind") in ("cold", "warm", "warm-cold")]
        iteration_ok = [r for r in iteration_requests if r.get("ok")]
        consecutive_failed_iterations = (0 if iteration_ok
                                         else consecutive_failed_iterations + 1)
        # A probe/window bug must not burn the whole allocation spinning on
        # instant HTTP 400s (job 38249448 did: 19,420 rejections in 45 min).
        if consecutive_failed_iterations >= 3:
            aborted = ("3 consecutive iterations with no successful request; "
                       "first error: %s" % (iteration_requests[0].get("error") or "")[:300])
            print("[stability] ABORT: " + aborted, file=sys.stderr)
            break
        cold = [r for r in rows if r["kind"] == "cold" and r.get("ok")]
        if iteration % 10 == 0 or not iteration_ok:
            print("[stability] iter=%d ok=%d/%d ttft=%s dec=%s mem_avail=%s"
                  % (iteration, len(iteration_ok), len(iteration_requests),
                     "%.1f" % cold[-1]["ttft_s"] if cold and cold[-1].get("ttft_s") else "-",
                     "%.1f" % cold[-1]["decode_tok_s"] if cold and cold[-1].get("decode_tok_s") else "-",
                     mi.get("MemAvailable")), file=sys.stderr)

    # ---- summary ---------------------------------------------------------
    requests = [r for r in rows if r.get("kind") in ("cold", "warm", "warm-cold")]
    errors = [r for r in requests if not r.get("ok")]
    cold = [r for r in requests if r["kind"] == "cold" and r.get("ok")]
    warm = [r for r in requests if r["kind"] == "warm" and r.get("ok")]
    cold_ttft = [r["ttft_s"] for r in cold if r.get("ttft_s")]
    warm_ttft = [r["ttft_s"] for r in warm if r.get("ttft_s")]
    samples = [r for r in rows if r["kind"] == "sample"]

    def med(vals):
        return statistics.median(vals) if vals else None

    def head_tail(vals, n=3):
        if len(vals) < n:
            return None, None
        return med(vals[:n]), med(vals[-n:])

    first, last = head_tail(cold_ttft)
    drift_pct = ((last - first) / first * 100.0) if first and last else None
    # least-squares slope of cold TTFT (ms per iteration)
    slope_ms = None
    if len(cold_ttft) >= 3:
        xs = list(range(len(cold_ttft)))
        xm, ym = sum(xs) / len(xs), sum(cold_ttft) / len(cold_ttft)
        den = sum((x - xm) ** 2 for x in xs)
        if den:
            slope_ms = (sum((x - xm) * (y - ym) for x, y in zip(xs, cold_ttft))
                        / den) * 1000.0
    summary = {
        "iteration_count": iteration,
        "aborted": aborted,
        "requests": len(requests),
        "requests_ok": len(requests) - len(errors),
        "errors": len(errors),
        "error_examples": [{"kind": r.get("kind"), "iteration": r.get("iteration"),
                            "error": (r.get("error") or "")[:300]}
                           for r in errors[:10]],
        "cold_ttft_s_p50": med(cold_ttft),
        "cold_ttft_first3_s": first,
        "cold_ttft_last3_s": last,
        "cold_ttft_drift_pct": drift_pct,
        "cold_ttft_slope_ms_per_iter": slope_ms,
        "cold_decode_tok_s_p50": med([r["decode_tok_s"] for r in cold
                                      if r.get("decode_tok_s")]),
        "warm_ttft_s_p50": med(warm_ttft),
        "warm_reused_tokens_p50": med([r["reused_tokens"] for r in warm
                                       if isinstance(r.get("reused_tokens"), int)]),
        "prompt_tokens_p50": med([r["prompt_tokens"] for r in requests
                                  if isinstance(r.get("prompt_tokens"), int)]),
        "mem_available_min_mib": min(
            [s["mem_available_mib"] for s in samples
             if s.get("mem_available_mib")], default=None),
        "vram_used_max_mib": [max((g["used_mib"] for g in s.get("gpus") or []),
                                  default=None) for s in samples][-1:] or None,
        "power_w_max": max((g["power_w"] for s in samples
                            for g in (s.get("gpus") or [])
                            if g.get("power_w")), default=None),
        "duration_s": round(args.duration_min * 60.0, 1),
        "elapsed_s": round(time.monotonic() - started, 1),
        "calibration": cal,
    }
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)
    print("[stability] %d requests, %d errors, cold TTFT p50=%s drift=%s%%"
          % (summary["requests"], summary["errors"],
             "%.1f" % summary["cold_ttft_s_p50"] if summary["cold_ttft_s_p50"] else "-",
             "%.1f" % drift_pct if drift_pct is not None else "-"),
          file=sys.stderr)


if __name__ == "__main__":
    main()
