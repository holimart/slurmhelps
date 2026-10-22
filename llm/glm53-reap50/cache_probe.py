#!/usr/bin/env python3
"""Cache-reuse probe for llama.cpp: fill a prefix ladder, then measure hit TTFT.

This deliberately does **not** issue cache-miss requests. A cold miss at P is
reported as the cumulative fill time to P plus the suffix prefill; the script
records that inference and prints the unmeasured gaps.

It also deliberately labels the repeat measurement as **one active sequence**:
llama.cpp has one active KV slot per sequence and a host prompt cache, not a
GPU-resident multi-prefix tree. Multi-branch GPU reuse and a true RAM-vs-GPU KV
tier are unmeasured.

usage:
  cache-probe.py --base-url http://127.0.0.1:30000 --model glm53 \
      --prefixes "100000 200000 ... 950000" --suffix 128 --osl 128
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import Config, make_prompt, request_once, percentile  # noqa: E402

CAVEATS = {
    "miss": "not measured; inferred as cumulative fill-to-P + suffix TTFT",
    "gpu_reuse": "one sequence only (active llama.cpp slot); not a multi-branch GPU tree",
    "gpu_prefix_tree": "unmeasured: llama.cpp has no GPU-resident multi-prefix tree",
    "ram_kv_tier": "unmeasured: llama.cpp prompt cache is host state restore, not a paged KV tier",
    "multi_agent_gpu_capacity": "unmeasured",
    "cache_tier_attribution": "unmeasured: no per-tier KV metrics in llama.cpp",
}


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]



def slot_n_past(base_url: str) -> int | None:
    """Ask llama-server for the active slot's n_past, if the endpoint exists."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/slots", timeout=5) as r:
            data = json.load(r)
        vals = [int(s["n_past"]) for s in data if isinstance(s, dict) and "n_past" in s]
        return max(vals) if vals else None
    except Exception:
        return None


def log_prompt_state(path: str) -> tuple[int, int] | None:
    """Return (task.n_tokens, n_past) from the last 'new prompt' block."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    last = None
    for i, line in enumerate(lines):
        if "| new prompt," not in line:
            continue
        m = re.search(r"task\.n_tokens = (\d+)", line)
        if not m:
            continue
        n_tokens = int(m.group(1))
        n_past = None
        for nxt in lines[i + 1:i + 4]:
            m2 = re.search(r"n_past = (\d+)", nxt)
            if m2:
                n_past = int(m2.group(1))
                break
        if n_past is not None:
            last = (n_tokens, n_past)
    return last


def observed_state(base_url: str, log_path: str) -> tuple[int | None, int | None]:
    """Prefer /slots; fall back to the server log. Returns (n_tokens, n_past)."""
    npast = slot_n_past(base_url)
    logged = log_prompt_state(log_path) if log_path else None
    if logged:
        return logged[0], npast if npast is not None else logged[1]
    return (None, npast) if npast is not None else (None, None)


def fmt(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="glm53-reap50")
    p.add_argument("--prefixes", default="100000 200000 300000 400000 500000 600000 700000 800000 900000 950000")
    p.add_argument("--suffix", type=int, default=128)
    p.add_argument("--osl", type=int, default=128, help="output tokens for the hit measurement")
    p.add_argument("--runs", type=int, default=1, help="repeat the hit measurement N times")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--preflight", type=int, default=16384,
                   help="prefix length used to verify prompt-cache reuse before the ladder")
    p.add_argument("--kv-cells", type=int, default=0)
    p.add_argument("--kv-mib", type=float, default=0.0)
    p.add_argument("--host-cache-mib", type=float, default=0.0)
    p.add_argument("--output-dir", default="cache-probe")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    prefixes = sorted(set(parse_ints(args.prefixes)))
    if not prefixes:
        print("no prefixes", file=sys.stderr)
        return 2
    max_prefix = max(prefixes)
    total_words = max_prefix + args.suffix + 1024
    rng = random.Random(args.seed)
    base = make_prompt(total_words, rng)
    words = base.split()
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
        c = Config(**{**cfg.__dict__, "osl": osl})
        return request_once(c, prompt)

    def prefix_text(n: int) -> str:
        return " ".join(words[:n])

    if args.dry_run:
        print(f"preflight={args.preflight} prefixes={prefixes} suffix={args.suffix} osl={args.osl}")
        print(f"max_context={max_prefix + args.suffix + 256} timeout={args.timeout}")
        return 0

    # Preflight: prove the prompt cache is actually reused at a small prefix.
    pre = min(args.preflight, prefixes[0])
    pre_prompt = prefix_text(pre)
    pre_fill = send(pre_prompt, 1)
    if not pre_fill.ok:
        print(f"preflight fill failed: {pre_fill.error}", file=sys.stderr)
        return 1
    pre_hit = send(pre_prompt + " " + suffix, args.osl)
    if not pre_hit.ok:
        print(f"preflight hit failed: {pre_hit.error}", file=sys.stderr)
        return 1
    pre_ratio = pre_fill.ttft_s / max(pre_hit.ttft_s, 1e-9)
    print(f"[preflight] P={pre} fill_ttft={pre_fill.ttft_s:.3f}s "
          f"hit_ttft={pre_hit.ttft_s:.3f}s reuse={pre_ratio:.1f}x")
    if pre_hit.ttft_s > 0.5 * pre_fill.ttft_s and pre_hit.ttft_s > 5.0:
        print("[preflight] prompt cache did not visibly reuse the prefix; aborting "
              "before spending the 1M fill. See caveats.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    cumulative = 0.0
    prev = 0
    for P in prefixes:
        prompt = prefix_text(P)
        fill = send(prompt, 1)
        if not fill.ok:
            print(f"fill P={P} failed: {fill.error}", file=sys.stderr)
            rows.append({"prefix_tokens": P, "status": "fill_failed", "error": fill.error})
            break
        cumulative += fill.ttft_s
        expected_new = P - prev if prev else P
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
        suffix_prefill = args.suffix / max(hit.ttft_s, 1e-9)
        decode = hit.out_tokens / max(hit.e2e_s - hit.ttft_s, 1e-9)
        inferred_miss = cumulative + hit.ttft_s
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
            "hit_suffix_prefill_tok_s": round(suffix_prefill, 2),
            "hit_decode_tok_s": round(decode, 2),
            "hit_output_tokens": hit.out_tokens,
            "repeat_ttft_s": round(repeat.ttft_s, 4) if repeat.ok else None,
            "repeat_reuse_ratio": round(hit.ttft_s / max(repeat.ttft_s, 1e-9), 2) if repeat.ok else None,
            "inferred_miss_ttft_s": round(inferred_miss, 4),
            "cache_reuse_verified": hit.ttft_s < 0.5 * fill.ttft_s or hit.ttft_s < 5.0,
            "gpu_reuse_measurement": "single active sequence only",
        }
        rows.append(row)
        print(f"P={P:>7} fill={fill.ttft_s:8.2f}s cumulative={cumulative:8.1f}s "
              f"hit={hit.ttft_s:7.3f}s reuse={row['hit_reuse_ratio']:6.1f}x "
              f"decode={decode:6.1f} tok/s repeat={row['repeat_ttft_s']}")
        prev = P

    capacity = {
        "kv_cells": args.kv_cells or None,
        "kv_mib": args.kv_mib or None,
        "host_prompt_cache_mib": args.host_cache_mib or None,
        "active_gpu_max_prefix_tokens": args.kv_cells or None,
        "host_cache_branches_at_max_prefix": (
            round((args.host_cache_mib * 1024 * 1024) / (max_prefix * 6.0 * 1024), 2)
            if args.host_cache_mib and max_prefix else None
        ),
        "note": "host cache is serialized prompt state, not a paged GPU tree",
    }
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "llamacpp",
        "base_url": args.base_url,
        "model": args.model,
        "suffix_tokens": args.suffix,
        "osl": args.osl,
        "prefixes": prefixes,
        "caveats": CAVEATS,
        "capacity": capacity,
        "rows": rows,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cache-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["prefix_tokens", "status", "suffix_tokens", "fill_ttft_s",
              "incremental_prefill_tok_s", "cumulative_fill_s", "hit_ttft_s",
              "hit_reuse_ratio", "hit_suffix_prefill_tok_s", "hit_decode_tok_s",
              "hit_output_tokens", "repeat_ttft_s", "repeat_reuse_ratio",
              "inferred_miss_ttft_s", "cache_reuse_verified", "error"]
    with (outdir / "cache-probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"caveats": CAVEATS, "capacity": capacity}, indent=2))
    print(f"wrote {outdir}/cache-probe.json and .csv")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
