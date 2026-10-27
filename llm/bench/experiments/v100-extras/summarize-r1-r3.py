#!/usr/bin/env python3
"""summarize-r1-r3.py — turn a 2026-10-28-v100-remaining tree into tables.

Reads only the small published artifacts (never engine logs or packs):

  r1-quality/<label>/cells*.jsonl      niah.py rows
  r2-engine/*/cells.jsonl             strata_probe cell rows
  r2-hardware/p2p-bench.json          p2pbench rows
  r2-engine/trace-*-summary.json      trace-summary.py rows
  r3-stability/cells.jsonl            stability.py rows
  r3-stability/summary.json           stability.py summary

usage: summarize-r1-r3.py RESULTS_DIR [--json-out FILE]
"""
import argparse
import glob
import json
import os
import statistics
import sys


def rows(path):
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def med(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def fmt(v, nd=1):
    return ("%%.%df" % nd) % v if isinstance(v, (int, float)) else "-"


def r1(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "r1-quality", "*", "cells*.jsonl"))):
        label = os.path.basename(os.path.dirname(path))
        for kind in ("needle", "tool"):
            selected = [r for r in rows(path) if r.get("kind") == kind]
            if not selected:
                continue
            ok = [r for r in selected if r.get("ok")]
            out.append({
                "pack": label, "kind": kind, "ctx": selected[0].get("context"),
                "n": len(selected), "correct": sum(1 for r in selected if r.get("correct")),
                "correct_any": sum(1 for r in selected if r.get("correct_any")),
                "prompt_tokens_p50": med([r.get("prompt_tokens") for r in selected]),
                "ttft_s_p50": med([r.get("ttft_s") for r in ok]),
                "decode_tok_s_p50": med([r.get("decode_tok_s") for r in ok]),
                "errors": len([r for r in selected if not r.get("ok")]),
                "error_examples": [r.get("error") for r in selected if r.get("error")][:3],
            })
    return out


def r2_cells(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "r2-engine", "*", "cells.jsonl"))):
        label = os.path.basename(os.path.dirname(path))
        for r in rows(path):
            out.append({
                "run": label, "ctx": r.get("context"),
                "ttft_s_p50": r.get("ttft_s_p50"),
                "decode_tok_s_p50": r.get("decode_tok_s_p50"),
                "aggregate_tok_s": r.get("aggregate_tok_s"),
                "prompt_tokens": r.get("prompt_tokens"),
                "requests_ok": r.get("requests_ok"),
                "requests_failed": r.get("requests_failed"),
                "error": r.get("error"),
            })
    return out


def r3(root):
    path = os.path.join(root, "r3-stability", "cells.jsonl")
    rs = rows(path)
    req = [r for r in rs if r.get("kind") in ("cold", "warm", "warm-cold")]
    errs = [r for r in req if not r.get("ok")]
    by_kind = {}
    for kind in ("cold", "warm-cold", "warm"):
        sel = [r for r in req if r.get("kind") == kind]
        by_kind[kind] = {
            "n": len(sel),
            "ok": len([r for r in sel if r.get("ok")]),
            "ttft_s_p50": med([r.get("ttft_s") for r in sel if r.get("ok")]),
            "decode_tok_s_p50": med([r.get("decode_tok_s") for r in sel if r.get("ok")]),
            "reused_tokens_p50": med([r.get("reused_tokens") for r in sel
                                      if isinstance(r.get("reused_tokens"), int)]),
            "prompt_tokens_p50": med([r.get("prompt_tokens") for r in sel
                                      if isinstance(r.get("prompt_tokens"), int)]),
        }
    cold_ttft = [r["ttft_s"] for r in req
                 if r.get("kind") == "cold" and r.get("ok") and r.get("ttft_s")]
    drift = None
    if len(cold_ttft) >= 6:
        drift = (med(cold_ttft[-3:]) - med(cold_ttft[:3])) / med(cold_ttft[:3]) * 100.0
    summary_path = os.path.join(root, "r3-stability", "summary.json")
    summary = {}
    try:
        summary = json.load(open(summary_path))
    except OSError:
        pass
    return {
        "requests": len(req), "errors": len(errs),
        "error_examples": [{"kind": r.get("kind"), "error": (r.get("error") or "")[:160]}
                           for r in errs[:8]],
        "by_kind": by_kind,
        "cold_ttft_drift_pct": drift,
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    data = {
        "r1": r1(args.results),
        "r2_cells": r2_cells(args.results),
        "r2_hardware": {},
        "r2_trace": {},
        "r3": r3(args.results),
    }
    for path in sorted(glob.glob(os.path.join(args.results, "r2-hardware", "p2p-bench.json"))):
        try:
            data["r2_hardware"] = json.load(open(path))
        except OSError:
            pass
    for path in sorted(glob.glob(os.path.join(args.results, "r2-engine", "trace-*-summary.json"))):
        data["r2_trace"][os.path.basename(path).replace("-summary.json", "")] = \
            json.load(open(path))

    print("## R1 — 262,144 quality\n")
    print("| pack | kind | ctx | correct | prompt tok | TTFT s | decode tok/s | errors |")
    print("|:--|:--|---:|:--|---:|---:|---:|---:|")
    for r in data["r1"]:
        print("| %s | %s | %s | %d/%d | %s | %s | %s | %d |" % (
            r["pack"], r["kind"], r["ctx"], r["correct"], r["n"],
            fmt(r["prompt_tokens_p50"], 0), fmt(r["ttft_s_p50"]),
            fmt(r["decode_tok_s_p50"]), r["errors"]))

    print("\n## R2 — 4x layer-split cells\n")
    print("| run | ctx | TTFT s | decode tok/s | aggregate tok/s | prompt tok | ok/failed |")
    print("|:--|---:|---:|---:|---:|---:|:--|")
    for c in data["r2_cells"]:
        print("| %s | %s | %s | %s | %s | %s | %s/%s |" % (
            c["run"], c["ctx"], fmt(c["ttft_s_p50"]), fmt(c["decode_tok_s_p50"]),
            fmt(c["aggregate_tok_s"]), fmt(c["prompt_tokens"], 0),
            c["requests_ok"], c["requests_failed"]))

    print("\n## R2 — peer bandwidth (p2pbench, 256 MiB)\n")
    print("| test | GB/s |")
    print("|:--|---:|")
    for r in data["r2_hardware"]:
        print("| %s | %.2f |" % (r.get("test"), r.get("gbps") or 0))

    print("\n## R2 — copy trace inside the 32k cell window\n")
    for name, t in data["r2_trace"].items():
        print("- **%s**: %.1f MB copied, peer share %s, d2d %.1f MB, host %.1f MB, default %.1f MB"
              % (name, t.get("total_bytes", 0) / 1e6,
                 fmt((t.get("peer_share") or 0) * 100, 1) + " %",
                 (t.get("by_transport") or {}).get("d2d", 0) / 1e6,
                 (t.get("by_transport") or {}).get("host", 0) / 1e6,
                 (t.get("by_transport") or {}).get("default", 0) / 1e6))
        for call, v in sorted((t.get("by_call") or {}).items()):
            print("  - `%s`: %d calls, %.1f MB" % (call, v["calls"], v["bytes"] / 1e6))

    print("\n## R3 — 262k stability\n")
    d = data["r3"]
    print("- requests: %d, errors: %d" % (d["requests"], d["errors"]))
    print("| kind | n | ok | TTFT s | decode tok/s | reused tok | prompt tok |")
    print("|:--|---:|---:|---:|---:|---:|---:|")
    for kind, v in d["by_kind"].items():
        print("| %s | %d | %d | %s | %s | %s | %s |" % (
            kind, v["n"], v["ok"], fmt(v["ttft_s_p50"]), fmt(v["decode_tok_s_p50"]),
            fmt(v["reused_tokens_p50"], 0), fmt(v["prompt_tokens_p50"], 0)))
    if d["cold_ttft_drift_pct"] is not None:
        print("- cold TTFT drift (first 3 vs last 3): %+.1f %%" % d["cold_ttft_drift_pct"])
    s = d.get("summary") or {}
    for key in ("cold_ttft_slope_ms_per_iter", "warm_ttft_s_p50",
                "mem_available_min_mib", "power_w_max"):
        if s.get(key) is not None:
            print("- %s: %s" % (key, s[key]))
    if d["error_examples"]:
        print("- error examples: %s" % d["error_examples"])

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
