#!/usr/bin/env python3
"""multiconv.py — interleaved multi-conversation warm cache for Strata (H3).

The canonical warm runner measures ONE conversation per slot.  This probe
keeps several *distinct* conversations (different salts/headers) alive and
interleaves their cold turn and warm repeats, so the 8 GiB / 4-slot
conversation cache is exercised the way an agent serving several users would.

Inherited from the canonical warm probe (strata_probe.py):
  * build_warm_prompt()-style calibrated, saltless-after-first-turn prompts
    (here each conversation has a fixed distinct header; its repeats are
    byte-identical);
  * stream_one() for the request, TTFT, usage and engine timings;
  * metrics_raw()/claim_engine_request() for per-request `reused` tokens;
  * conversation_cache_of() for budget/slots/parked/evictions;
  * unique-salt cold rule: the first turn of each conversation is cold, and
    the prompt is never shared between conversations.

Order: one cold request per conversation, then warm round 1 for every
conversation, then round 2, ... (interleaved).  One JSON line per request.
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "strata"))
sys.path.insert(0, STRATA)
import strata_probe as sp  # noqa: E402

CORPUS = ("The quarterly report describes the deployment of the service in the "
          "western region. The team reviewed capacity, latency and cost. ")
FILLER = ("The archive contains routine notes from the operations team. Each "
          "entry lists the date, the shift and the systems that were checked. ")


def corpus(nchars):
    reps = nchars // len(FILLER) + 1
    return (FILLER * reps)[:nchars]


def build_prompt(chars, conv):
    head = ("[conversation %d | workspace=%s]\n\n"
            % (conv, "alpha" if conv % 2 == 0 else "beta"))
    return head + corpus(max(1, chars - len(head)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--warm-repeats", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--pad-safety", type=int, default=128)
    ap.add_argument("--overhead-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--metrics-timeout", type=float, default=15.0)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--cache-out", required=True)
    args = ap.parse_args()

    base = sp.chat_root(args.base_url)
    root = sp.host_root(args.base_url)
    cal = sp.calibrate_chars_per_token(base, args.model, args.temperature,
                                       args.timeout, args.max_context)
    cpt = cal.get("chars_per_token") or 4.0
    intercept = cal.get("intercept_tokens") or 0.0
    budget = min(args.context, args.max_context) - args.max_tokens - args.pad_safety
    chars = max(1024, int((budget - args.overhead_tokens - intercept) * cpt))
    prompts = [build_prompt(chars, i) for i in range(args.conversations)]

    raw = sp.metrics_raw(root, args.metrics_timeout)
    cc_before = sp.conversation_cache_of(raw)
    seen = set()
    lock = threading.Lock()
    if raw:
        for e in raw.get("requests") or []:
            if isinstance(e, dict):
                seen.add(e.get("time"))

    rows = []
    with open(args.json_out, "a") as fh:
        # round 0 = cold first turn; rounds 1..N = byte-identical warm repeats
        for rnd in range(0, args.warm_repeats + 1):
            for conv in range(args.conversations):
                rec = sp.stream_one(base, args.model, prompts[conv],
                                    args.max_tokens, args.temperature, args.timeout)
                raw = sp.metrics_raw(root, args.metrics_timeout)
                claim = sp.claim_engine_request(raw, seen, lock) if raw else None
                row = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": args.model, "context": args.context,
                    "conversation": conv, "round": rnd,
                    "warm": rnd > 0,
                    "prompt_tokens": rec.get("prompt_tokens"),
                    "completion_tokens": rec.get("completion_tokens"),
                    "ttft_s": rec.get("ttft_s"),
                    "total_s": rec.get("total_s"),
                    "decode_tok_s": rec.get("decode_tok_s"),
                    "reused_tokens": (claim or {}).get("reused"),
                    "engine_prompt_ms": (claim or {}).get("prompt_ms"),
                    "ok": rec.get("ok"), "error": rec.get("error"),
                    "conversation_cache": sp.conversation_cache_of(raw),
                }
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rows.append(row)
                print("[multiconv] ctx=%d conv=%d round=%d ok=%s ttft=%s reused=%s"
                      % (args.context, conv, rnd, row["ok"],
                         "%.3f" % row["ttft_s"] if row["ttft_s"] else "-",
                         row["reused_tokens"]), file=sys.stderr)

    raw = sp.metrics_raw(root, args.metrics_timeout)
    cc_after = sp.conversation_cache_of(raw)
    with open(args.cache_out, "w") as cf:
        json.dump({"context": args.context, "conversations": args.conversations,
                   "warm_repeats": args.warm_repeats,
                   "before": cc_before, "after": cc_after}, cf, indent=1)

    rounds = {}
    for r in rows:
        rounds.setdefault((r["round"], r["warm"]), []).append(r)
    with open(args.summary_out, "w") as sf:
        sf.write("ctx,round,warm,requests,ttft_s_p50,ttft_s_p95,"
                 "reused_tokens_p50,reused_tokens_total,errors\n")
        for (rnd, warm), rs in sorted(rounds.items()):
            ok = [r for r in rs if r["ok"]]
            ttfts = [r["ttft_s"] for r in ok if isinstance(r["ttft_s"], (int, float))]
            reused = [r["reused_tokens"] for r in ok if isinstance(r["reused_tokens"], (int, float))]
            sf.write("%d,%d,%s,%d,%.3f,%.3f,%s,%d,%d\n" % (
                args.context, rnd, "yes" if warm else "no", len(rs),
                sp.percentile(ttfts, 0.50) or 0, sp.percentile(ttfts, 0.95) or 0,
                "%.0f" % (sp.percentile(reused, 0.50) or 0),
                int(sum(reused)), len(rs) - len(ok)))
    print("[multiconv] cache before=%s after=%s" % (cc_before, cc_after),
          file=sys.stderr)


if __name__ == "__main__":
    main()
