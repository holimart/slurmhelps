#!/usr/bin/env python3
"""summarize.py — flatten the MI210 campaign cells into summary.csv.

Inputs
  cells/bench-<pack>-pp<ctx>.json   llama-bench -o json (prompt processing)
  cells/bench-<pack>-tg<ctx>.json   llama-bench -o json (text generation at depth)
  cells/server-<pack>-f16.jsonl     probe_client.py (C1, f16 KV)
  cells/server-<pack>-q8kv.jsonl   probe_client.py (C1, q8_0 KV)
  cells/server-<pack>-np2.jsonl    probe_client.py (C2, f16 KV)

Output columns:
  pack,engine,source,ctx,parallel,prompt_tokens,pp_tps,tg_tps,ttft_ms,tpot_ms,samples
Medians are over repeats; the server rows keep the actual prompt token count.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


def bench_rows(path):
    pack = re.search(r"bench-([a-z0-9_]+)-", os.path.basename(path)).group(1)
    try:
        data = json.load(open(path))
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    rows = []
    for d in data:
        n_prompt = d.get("n_prompt") or 0
        n_gen = d.get("n_gen") or 0
        depth = d.get("n_depth") or 0
        ts = d.get("avg_ts")
        if n_prompt and not n_gen:
            rows.append(dict(pack=pack, engine="llama.cpp-b11785-hip", source="llama-bench",
                             ctx=depth or n_prompt, parallel=1, prompt_tokens=n_prompt,
                             pp_tps=ts, tg_tps="", ttft_ms="", tpot_ms="", samples=1))
        elif n_gen:
            rows.append(dict(pack=pack, engine="llama.cpp-b11785-hip", source="llama-bench",
                             ctx=depth or n_prompt, parallel=1, prompt_tokens=n_prompt,
                             pp_tps="", tg_tps=ts, ttft_ms="", tpot_ms="",
                             samples=1))
    return rows


def server_rows(path):
    base = os.path.basename(path)
    m = re.match(r"server-([a-z0-9_]+)-(f16|q8kv|np2)\.jsonl", base)
    if not m:
        return []
    pack, variant = m.group(1), m.group(2)
    groups = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        key = (d.get("ctx_target"), d.get("parallel", 1),
               d.get("label") or variant)
        groups[key].append(d)
    rows = []
    for (ctx, par, label), ds in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        tps = [x["decode_tps"] for x in ds if x.get("decode_tps")]
        ttf = [x["ttft_ms"] for x in ds if x.get("ttft_ms") is not None]
        tpot = [x["tpot_ms"] for x in ds if x.get("tpot_ms")]
        # concurrency aggregate: sum of tokens over the overlap window
        agg = [x.get("aggregate_decode_tps") for x in ds if x.get("aggregate_decode_tps")]
        rows.append(dict(
            pack=pack, engine="llama.cpp-b11785-hip",
            source="server-" + variant, ctx=ctx, parallel=par,
            prompt_tokens=statistics.median([x["prompt_n"] for x in ds if x.get("prompt_n") is not None] or [0]),
            pp_tps="",
            tg_tps=statistics.median(tps) if tps else "",
            ttft_ms=statistics.median(ttf) if ttf else "",
            tpot_ms=statistics.median(tpot) if tpot else "",
            samples=len(ds),
            aggregate_tps=statistics.median(agg) if agg else "",
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--power", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.cells, "bench-*.json"))):
        rows += bench_rows(p)
    for p in sorted(glob.glob(os.path.join(args.cells, "server-*.jsonl"))):
        rows += server_rows(p)

    cols = ["pack", "engine", "source", "ctx", "parallel", "prompt_tokens", "pp_tps",
            "tg_tps", "ttft_ms", "tpot_ms", "samples", "aggregate_tps"]
    with open(args.out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    sys.exit(main())
