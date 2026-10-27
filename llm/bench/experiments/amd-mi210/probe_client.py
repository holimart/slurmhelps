#!/usr/bin/env python3
"""probe_client.py — stdlib OpenAI/llama.cpp client for the MI210 campaign.

One JSON line per request.  It talks to a running llama-server and uses the
server's own timings block, so TTFT and decode are measured on the same clock
as llama.cpp:

  ttft_ms      = prompt_ms + predicted_per_token_ms   (prefill + first token)
  decode_tps   = 1000 / predicted_per_token_ms
  prompt_n     = the actual prompt token count (never the requested one)

The prompt is built from natural filler and calibrated against /tokenize until
it is within ~0.5 % of the requested token count.  A unique salt is always
included, so no prompt cache can serve the request.

usage:
  probe_client.py --url http://127.0.0.1:8099 --ctx 4096 --n-predict 128 \
                  --out cells.jsonl [--repeats 2] [--parallel 1] [--label ...]
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = (
    "The finite element model discretises the elliptic boundary value problem on an "
    "unstructured tetrahedral mesh. Each substructure is solved independently with a "
    "sparse direct factorisation, and the interface problem is assembled from the "
    "Schur complements. The conditioning of the coarse space depends on the number of "
    "subdomains and on the coefficient contrast across the material interfaces. "
    "Adaptive refinement is driven by a dual weighted residual estimator, which "
    "requires the solution of an adjoint problem on the same mesh. The resulting "
    "linear systems are distributed across the ranks of the MPI communicator, and the "
    "preconditioner applies a small number of smoothing steps before the coarse grid "
    "correction. In the transient setting the time step is chosen from a CFL condition "
    "and the nonlinear residual is reduced by a Newton iteration with a line search. "
    "The implementation stores the element matrices in a compressed row format and "
    "reorders the unknowns with a nested dissection ordering to limit fill in. "
    "Performance is dominated by the sparse matrix-vector product and by the latency "
    "of the global reduction, so the communication pattern is overlapped with local "
    "work wherever the dependency graph allows it. "
)
QUESTION = (
    "\n\nUsing only the document above, answer in one short sentence: what dominates "
    "the runtime of the distributed solver, and which two operations are named?"
)


def _post(url, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(base, text, timeout=600):
    try:
        out = _post(base + "/tokenize", {"content": text}, timeout=timeout)
        return len(out["tokens"])
    except urllib.error.HTTPError:
        return None


def build_prompt(base, target_tokens, salt, timeout=600):
    """Return (text, actual_tokens).  Calibrate chars/token, then trim/expand."""
    header = "Document %s. Read the whole document before answering.\n\n" % salt
    footer = QUESTION
    fixed = tokenize(base, header + footer, timeout) or 0
    body_target = max(64, target_tokens - fixed)
    # calibrate on a modest sample, then scale; three refinement rounds are enough
    sample = FILLER * max(1, 8192 // len(FILLER))
    n_sample = tokenize(base, sample, timeout) or 0
    cpt = len(sample) / max(1, n_sample)
    chars = int(body_target * cpt)
    text = header + (FILLER * (chars // len(FILLER) + 1))[:chars] + footer
    for _ in range(5):
        n = tokenize(base, text, timeout=timeout)
        if n is None:
            break
        if abs(n - target_tokens) <= max(8, target_tokens // 200):
            break
        body = text[len(header):-len(footer)]
        # linear correction on the body length, floor at one filler block
        want = max(len(FILLER), int(len(body) * (target_tokens - fixed) / max(1, n - fixed)))
        text = header + (body * (want // len(body) + 1))[:want] + footer

    # b11785's QSA selection misbehaves when the indexer cache ends exactly on
    # a block boundary (n_prompt % kpool == 0), which showed up as a GPU memory
    # access fault at a 4096-token depth.  Force an incomplete tail block.
    n = tokenize(base, text, timeout=timeout) or -1
    for suffix in (" x", " y", " z", " .", " q"):
        if n < 0 or n % 4 != 0:
            break
        cand = text + suffix
        nc = tokenize(base, cand, timeout=timeout) or -1
        if nc > 0 and nc % 4 != 0:
            text, n = cand, nc
            break
    return text, n


def one_request(base, prompt, n_predict, timeout):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": False,
        "n_probs": 0,
    }
    t0 = time.time()
    out = _post(base + "/completion", payload, timeout)
    t1 = time.time()
    tim = out.get("timings", {}) or {}
    pn = tim.get("prompt_n", out.get("tokens_evaluated"))
    pm = tim.get("prompt_ms")
    dn = tim.get("predicted_n", out.get("tokens_predicted"))
    dm = tim.get("predicted_ms")
    tpot = tim.get("predicted_per_token_ms")
    if tpot is None and dm and dn:
        tpot = dm / dn
    if pm is None:
        pm = (t1 - t0) * 1000.0
    ttft = pm + (tpot or 0.0)
    return {
        "wall_start": t0,
        "wall_end": t1,
        "prompt_n": pn,
        "prompt_ms": pm,
        "predicted_n": dn,
        "predicted_ms": dm,
        "tpot_ms": tpot,
        "decode_tps": (1000.0 / tpot) if tpot else None,
        "ttft_ms": ttft,
        "wall_ms": (t1 - t0) * 1000.0,
        "finish_reason": out.get("stop_type") or out.get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--ctx", type=int, required=True, help="target prompt tokens")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1, help="simultaneous slots")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    # leave room for the generated tokens and the template
    target = max(256, args.ctx - args.n_predict - 16)
    text, actual = build_prompt(base, target, "salt-%d-%d" % (int(time.time()), args.ctx),
                                timeout=min(args.timeout, 600))

    for rep in range(args.repeats):
        results = []
        if args.parallel <= 1:
            r = one_request(base, text, args.n_predict, args.timeout)
            results.append(r)
            agg_tps = r["decode_tps"]
            window_ms = r["wall_ms"]
        else:
            barrier = threading.Barrier(args.parallel)

            def worker(k):
                barrier.wait()
                # distinct salt per slot, same target length
                t, _ = build_prompt(base, target, "salt-%d-%d-%d" % (int(time.time()), args.ctx, k),
                                    timeout=min(args.timeout, 600))
                results.append(one_request(base, t, args.n_predict, args.timeout))

            threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.parallel)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            t0 = min(r["wall_start"] for r in results)
            t1 = max(r["wall_end"] for r in results)
            window_ms = (t1 - t0) * 1000.0
            tok = sum(r["predicted_n"] or 0 for r in results)
            agg_tps = tok / (window_ms / 1000.0) if window_ms else None

        for slot, r in enumerate(results):
            r.update({
                "ctx_target": args.ctx,
                "actual_prompt_tokens": actual,
                "n_predict": args.n_predict,
                "parallel": args.parallel,
                "slot": slot,
                "repeat": rep,
                "label": args.label,
                "aggregate_decode_tps": agg_tps,
                "concurrency_window_ms": window_ms,
            })
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"ctx": args.ctx, "parallel": args.parallel, "repeat": rep,
                          "prompt_n": results[0]["prompt_n"],
                          "ttft_ms": round(results[0]["ttft_ms"], 1),
                          "decode_tps": None if results[0]["decode_tps"] is None else round(results[0]["decode_tps"], 2),
                          "aggregate": None if agg_tps is None else round(agg_tps, 2)}),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
