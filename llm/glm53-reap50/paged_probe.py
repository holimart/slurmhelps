#!/usr/bin/env python3
"""Prefix-cache probe for a paged-KV OpenAI server (vLLM/SGLang).

Fills a prefix ladder, then measures a P+128 request and an immediate repeat.
No cache-miss requests are issued; miss TTFT is inferred from cumulative fill.
The engine's own /metrics prefix-cache counters are recorded as a delta.

usage:
  paged_probe.py --base-url http://127.0.0.1:8000 --model glm53 \
      --prefixes "100000 200000 300000" --suffix 128 --osl 128
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import Config, cache_stats, make_prompt, request_once, scrape_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="glm53-reap50")
    p.add_argument("--prefixes", default="100000 200000 300000")
    p.add_argument("--suffix", type=int, default=128)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="paged-probe")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    prefixes = sorted({int(x) for x in args.prefixes.replace(",", " ").split() if x.strip()})
    max_prefix = max(prefixes)
    rng = random.Random(args.seed)
    words = make_prompt(max_prefix + args.suffix + 1024, rng).split()
    suffix = " ".join(make_prompt(args.suffix, random.Random(args.seed + 999)).split())
    outdir = Path(args.output_dir)
    cfg = Config(
        base_url=args.base_url,
        model=args.model,
        isl=max_prefix,
        osl=args.osl,
        timeout=args.timeout,
        metrics_url=args.base_url.rstrip("/") + "/metrics",
        flush_cache=False,
        seed=args.seed,
    )

    def send(prompt: str, osl: int):
        return request_once(Config(**{**cfg.__dict__, "osl": osl}), prompt)

    def prefix_text(n: int) -> str:
        return " ".join(words[:n])

    if args.dry_run:
        print(f"prefixes={prefixes} suffix={args.suffix} osl={args.osl} max_context={max_prefix+args.suffix+256}")
        return 0

    rows: list[dict] = []
    cumulative = 0.0
    prev = 0
    for P in prefixes:
        prompt = prefix_text(P)
        before = scrape_metrics(cfg.metrics_url)
        fill = send(prompt, 1)
        if not fill.ok:
            print(f"fill P={P} failed: {fill.error}", file=sys.stderr)
            rows.append({"prefix_tokens": P, "status": "fill_failed", "error": fill.error})
            break
        cumulative += fill.ttft_s
        hit_samples = []
        for _ in range(max(1, args.runs)):
            hit = send(prompt + " " + suffix, args.osl)
            if not hit.ok:
                print(f"hit P={P} failed: {hit.error}", file=sys.stderr)
                break
            hit_samples.append(hit)
        if not hit_samples:
            rows.append({"prefix_tokens": P, "status": "hit_failed"})
            break
        hit = hit_samples[0]
        repeat = send(prompt + " " + suffix, args.osl)
        after = scrape_metrics(cfg.metrics_url)
        pc = cache_stats(before, after)
        expected_new = P - prev if prev else P
        row = {
            "prefix_tokens": P,
            "status": "ok",
            "suffix_tokens": args.suffix,
            "fill_ttft_s": round(fill.ttft_s, 4),
            "expected_new_tokens": expected_new,
            "incremental_prefill_tok_s": round(expected_new / max(fill.ttft_s, 1e-9), 2),
            "cumulative_fill_s": round(cumulative, 4),
            "hit_ttft_s": round(hit.ttft_s, 4),
            "hit_reuse_ratio": round(fill.ttft_s / max(hit.ttft_s, 1e-9), 2),
            "hit_suffix_prefill_tok_s": round(args.suffix / max(hit.ttft_s, 1e-9), 2),
            "hit_decode_tok_s": round(hit.out_tokens / max(hit.e2e_s - hit.ttft_s, 1e-9), 2),
            "hit_output_tokens": hit.out_tokens,
            "repeat_ttft_s": round(repeat.ttft_s, 4) if repeat.ok else None,
            "repeat_reuse_ratio": round(hit.ttft_s / max(repeat.ttft_s, 1e-9), 2) if repeat.ok else None,
            "inferred_miss_ttft_s": round(cumulative + hit.ttft_s, 4),
            "prefix_cache": pc,
        }
        rows.append(row)
        print(f"P={P:>7} fill={fill.ttft_s:8.2f}s hit={hit.ttft_s:7.3f}s "
              f"repeat={row['repeat_ttft_s']} decode={row['hit_decode_tok_s']:6.1f} tok/s "
              f"cache={pc}")
        prev = P

    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "paged-openai",
        "base_url": args.base_url,
        "model": args.model,
        "prefixes": prefixes,
        "suffix_tokens": args.suffix,
        "osl": args.osl,
        "miss": "inferred; no cold miss requests",
        "rows": rows,
    }
    (outdir / "paged-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["prefix_tokens", "status", "suffix_tokens", "fill_ttft_s", "expected_new_tokens",
              "incremental_prefill_tok_s", "cumulative_fill_s", "hit_ttft_s", "hit_reuse_ratio",
              "hit_suffix_prefill_tok_s", "hit_decode_tok_s", "hit_output_tokens",
              "repeat_ttft_s", "repeat_reuse_ratio", "inferred_miss_ttft_s"]
    with (outdir / "paged-probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outdir}/paged-probe.json and .csv")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
