#!/usr/bin/env python3
"""report.py — turn the MI210 cells into the markdown tables used in the doc.

usage: report.py --dir /work/.../2026-10-27-amd-mi210
Reads summary.csv, power/cell-*.json and active-bytes-iq3s.json; prints
markdown sections on stdout.  No numbers are invented: a missing cell prints
as an empty table row with "—".
"""
import argparse
import csv
import json
import os
import sys


def read_summary(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fmt(v, nd=2):
    if v in ("", None):
        return "—"
    try:
        return ("%%.%df" % nd) % float(v)
    except ValueError:
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir
    rows = read_summary(os.path.join(d, "summary.csv"))
    packs = []
    for r in rows:
        if r["pack"] not in packs:
            packs.append(r["pack"])

    for pack in packs:
        print("\n### %s — llama-bench (f16 KV, FA on, -ngl 99)\n" % pack)
        print("| target ctx | test | actual prompt/depth | tok/s |")
        print("|---:|:--|---:|---:|")
        for r in rows:
            if r["pack"] != pack or r["source"] != "llama-bench":
                continue
            if r["pp_tps"]:
                print("| %s | pp | %s | **%s** |" % (r["ctx"], fmt(r["prompt_tokens"], 0), fmt(r["pp_tps"])))
        for r in rows:
            if r["pack"] != pack or r["source"] != "llama-bench":
                continue
            if r["tg_tps"]:
                print("| %s | tg | %s | **%s** |" % (r["ctx"], fmt(r["ctx"], 0), fmt(r["tg_tps"])))

        print("\n#### %s — llama-server C1, f16 KV\n" % pack)
        print("| target ctx | prompt tokens | TTFT (ms) | decode tok/s | TPOT (ms) | n |")
        print("|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["pack"] != pack or r["source"] != "server-f16":
                continue
            print("| %s | %s | %s | **%s** | %s | %s |" % (
                r["ctx"], fmt(r["prompt_tokens"], 0), fmt(r["ttft_ms"], 0),
                fmt(r["tg_tps"]), fmt(r["tpot_ms"]), r["samples"]))

    print("\n### Concurrency (-np 2, f16 KV)\n")
    print("| pack | target ctx | prompt tokens | per-stream tok/s | aggregate tok/s | TTFT (ms) | n |")
    print("|:--|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["source"] != "server-np2":
            continue
        print("| %s | %s | %s | %s | **%s** | %s | %s |" % (
            r["pack"], r["ctx"], fmt(r["prompt_tokens"], 0), fmt(r["tg_tps"]),
            fmt(r.get("aggregate_tps")), fmt(r["ttft_ms"], 0), r["samples"]))

    print("\n### q8_0 KV variant (C1)\n")
    print("| pack | target ctx | prompt tokens | decode tok/s | TTFT (ms) | n |")
    print("|:--|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["source"] != "server-q8kv":
            continue
        print("| %s | %s | %s | %s | %s | %s |" % (
            r["pack"], r["ctx"], fmt(r["prompt_tokens"], 0), fmt(r["tg_tps"]),
            fmt(r["ttft_ms"], 0), r["samples"]))

    print("\n### Power and cost\n")
    for name in ("cell-16k", "cell-262k"):
        p = os.path.join(d, "power", name + ".json")
        if not os.path.exists(p):
            print("- %s: missing" % name)
            continue
        o = json.load(open(p))
        c = o["cell"]
        dec = o.get("decode_only", {})
        print("- **%s**: mean GPU power %.0f W, energy %.1f kJ, %.2f s" % (
            name, c["total_mean_power_w"], c["total_energy_j"] / 1000.0, c["window_s"]))
        if "tokens_per_joule" in o:
            print("  - cell: %.5f tok/J, %.0f tok/kWh, **€%.2f/Mtok**" % (
                o["tokens_per_joule"], o["tokens_per_kwh"], o["eur_per_mtok"]))
        if dec:
            print("  - decode-only: %.4f tok/J, %.0f tok/kWh, **€%.2f/Mtok**" % (
                dec["tokens_per_joule"], dec["tokens_per_kwh"], dec["eur_per_mtok"]))
            if "idle_subtracted" in dec:
                x = dec["idle_subtracted"]
                print("  - idle-subtracted: %.4f tok/J, %.0f tok/kWh, €%.2f/Mtok (inc %.0f W)" % (
                    x["tokens_per_joule"], x["tokens_per_kwh"], x["eur_per_mtok"],
                    x["incremental_power_w"]))
        if "idle_power_w" in o:
            print("  - idle GPU power: %.0f W" % o["idle_power_w"])

    p = os.path.join(d, "active-bytes-iq3s.json")
    if os.path.exists(p):
        a = json.load(open(p))
        print("\n### Roofline (IQ3_S, active bytes from GGUF)\n")
        print("- active weights streamed per token: **%.3f GB** (%.1f B params equivalent at %.2f bpw)"
              % (a["active_bytes_per_token"] / 1e9,
                 a["active_bytes_per_token"] * 8 / a["bpw_shard1"] / 1e9, a["bpw_shard1"]))
        print("- expert bytes %.2f GB, active expert %.2f GB, non-expert %.2f GB (token_embd %.2f GB excluded)"
              % (a["expert_bytes"] / 1e9, a["expert_bytes"] * a["expert_used_count"] / a["expert_count"] / 1e9,
                 a["nonexpert_bytes"] / 1e9, a["token_embd_bytes"] / 1e9))
        print("- roofline @ 1.6 TB/s: **%.0f tok/s**; @ measured 1.336 TB/s: **%.0f tok/s**"
              % (a["roofline_spec_1600_tps"], a["roofline_measured_bw_tps"]))
        print("- dataset 6B/3.38bpw assumption: %.2f GB -> roofline %.0f tok/s (1.6 TB/s)"
              % (a["active_bytes_assumption_6B_3_38bpw"] / 1e9, a["roofline_task_assumption_1600_tps"]))
        print("- eta @ measured decode %.2f tok/s: **%.3f** (spec BW), **%.3f** (measured BW); %.3f on the 6B assumption"
              % (a["measured_decode_tps"], a["eta_spec"], a["eta_measured_bw"],
                 a["eta_task_assumption_spec"]))


if __name__ == "__main__":
    sys.exit(main())
