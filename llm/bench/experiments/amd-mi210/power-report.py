#!/usr/bin/env python3
"""power-report.py — energy and cost from power_sampler.py CSV.

usage:
  power-report.py --csv power.csv --start EPOCH --end EPOCH \
                  [--tokens N] [--decode-tps R] [--price-eur-kwh 0.20] \
                  [--json out.json]

Energy prefers the hwmon cumulative counter delta; if it is absent it
integrates power1_average with the trapezoid rule per GPU.  Power is the sum
over both GPUs (GPU package power, not wall power: CPUs/fans/PSU excluded).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["epoch"]),
                    "gpu": int(r["gpu"]),
                    "p": float(r["power_w"]) if r["power_w"] else None,
                    "sclk": float(r["sclk_mhz"]) if r["sclk_mhz"] else None,
                    "temp": float(r["temp_c"]) if r["temp_c"] else None,
                    "use": float(r["use_pct"]) if r["use_pct"] else None,
                    "e": float(r["energy_j"]) if r["energy_j"] else None,
                })
            except (ValueError, KeyError):
                continue
    return rows


def window_stats(rows, t0, t1):
    sel = [r for r in rows if t0 <= r["t"] <= t1]
    by = defaultdict(list)
    for r in sel:
        by[r["gpu"]].append(r)
    out = {"samples": len(sel), "gpus": {}}
    total_power = 0.0
    total_energy = 0.0
    energy_known = True
    for g, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["t"])
        ps = [r["p"] for r in rs if r["p"] is not None]
        es = [r["e"] for r in rs if r["e"] is not None]
        mean_p = sum(ps) / len(ps) if ps else None
        # trapezoid integration of power
        e_int = 0.0
        for a, b in zip(rs, rs[1:]):
            if a["p"] is not None and b["p"] is not None:
                e_int += 0.5 * (a["p"] + b["p"]) * (b["t"] - a["t"])
        e_ctr = (es[-1] - es[0]) if len(es) >= 2 else None
        if e_ctr is None:
            energy_known = False
            e_use = e_int
        else:
            e_use = e_ctr
        out["gpus"][g] = {
            "mean_power_w": mean_p,
            "energy_j_integrated": e_int,
            "energy_j_counter": e_ctr,
            "energy_j_used": e_use,
            "mean_sclk_mhz": (sum(r["sclk"] for r in rs if r["sclk"] is not None) /
                              max(1, len([r for r in rs if r["sclk"] is not None]))),
            "max_temp_c": max((r["temp"] for r in rs if r["temp"] is not None), default=None),
            "mean_use_pct": (sum(r["use"] for r in rs if r["use"] is not None) /
                             max(1, len([r for r in rs if r["use"] is not None]))),
        }
        if mean_p is not None:
            total_power += mean_p
        total_energy += e_use
    out["total_mean_power_w"] = total_power
    out["total_energy_j"] = total_energy
    out["energy_source"] = "counter" if energy_known and total_energy > 0 else "integrated"
    out["window_s"] = t1 - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--tokens", type=float, default=None)
    ap.add_argument("--decode-tps", type=float, default=None)
    ap.add_argument("--price-eur-kwh", type=float, default=0.20)
    ap.add_argument("--idle-start", type=float, default=None)
    ap.add_argument("--idle-end", type=float, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(args.csv)
    cell = window_stats(rows, args.start, args.end)
    res = {"cell": cell, "price_eur_per_kwh": args.price_eur_kwh}
    if args.idle_start is not None and args.idle_end is not None:
        idle = window_stats(rows, args.idle_start, args.idle_end)
        res["idle"] = idle
        res["idle_power_w"] = idle["total_mean_power_w"]
    if args.tokens is not None and cell["total_energy_j"] > 0:
        tpj = args.tokens / cell["total_energy_j"]
        res["tokens"] = args.tokens
        res["tokens_per_joule"] = tpj
        res["tokens_per_kwh"] = tpj * 3.6e6
        res["eur_per_mtok"] = (1e6 / tpj) / 3.6e6 * args.price_eur_kwh
    if args.decode_tps is not None and cell["total_mean_power_w"] > 0:
        tpj = args.decode_tps / cell["total_mean_power_w"]
        res["decode_only"] = {
            "decode_tps": args.decode_tps,
            "tokens_per_joule": tpj,
            "tokens_per_kwh": tpj * 3.6e6,
            "eur_per_mtok": (1e6 / tpj) / 3.6e6 * args.price_eur_kwh,
        }
        idle_p = res.get("idle_power_w")
        if idle_p:
            inc = cell["total_mean_power_w"] - idle_p
            if inc > 0:
                tpj_inc = args.decode_tps / inc
                res["decode_only"]["idle_subtracted"] = {
                    "incremental_power_w": inc,
                    "tokens_per_joule": tpj_inc,
                    "tokens_per_kwh": tpj_inc * 3.6e6,
                    "eur_per_mtok": (1e6 / tpj_inc) / 3.6e6 * args.price_eur_kwh,
                }
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.json:
        with open(args.json, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
