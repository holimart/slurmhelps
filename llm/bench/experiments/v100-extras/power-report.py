#!/usr/bin/env python3
"""power-report.py — tokens per joule / per euro from power-sampler CSV (H6).

usage:
  power-report.py --samples cell.csv --start EPOCH --end EPOCH \
                  --tokens N --idle idle.csv [--price-eur-kwh 0.20] [--json-out F]

`--tokens` is the completion-token count of the measured cell (sum over all
requests; for C1 x3 that is 3 x max_tokens).  Energy is the trapezoidal
integral of the summed GPU power over the cell bracket.  The idle baseline is
the mean summed power over the idle samples and is reported, not subtracted
(the cell's own energy is what the tokens cost).  Decode-only tokens/J is
estimated from the decode share of the cell (decode_s / wall_s), because the
sampler cannot see kernel boundaries; the cell number is the honest one.
"""
import argparse
import csv
import json
import statistics


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, _, rest = line.partition(",")
            try:
                ts = float(head)
            except ValueError:
                continue
            gpus = rest.split(";")
            power, clocks, temps = [], [], []
            for g in gpus:
                fld = [x.strip() for x in g.split(",")]
                if len(fld) < 5:
                    continue
                try:
                    power.append(float(fld[0]))
                except ValueError:
                    continue
                try:
                    clocks.append(float(fld[1]))
                except ValueError:
                    pass
                try:
                    temps.append(float(fld[2]))
                except ValueError:
                    pass
            if power:
                out.append((ts, sum(power), clocks, temps))
    return out


def energy_j(samples, t0, t1):
    pts = [(t, p) for t, p, _, _ in samples if t0 <= t <= t1]
    if len(pts) < 2:
        return None
    e = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        e += (pa + pb) / 2.0 * (tb - ta)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--idle", default="")
    ap.add_argument("--price-eur-kwh", type=float, default=0.20)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    cell = rows(args.samples)
    e = energy_j(cell, args.start, args.end)
    pts = [(t, p, c, te) for t, p, c, te in cell if args.start <= t <= args.end]
    mean_w = statistics.mean(p for _, p, _, _ in pts) if pts else None
    mean_clk = statistics.mean(c for _, _, cs, _ in pts for c in cs) if pts else None
    mean_temp = statistics.mean(t for _, _, _, ts in pts for t in ts) if pts else None
    idle = rows(args.idle) if args.idle else []
    idle_w = statistics.mean(p for _, p, _, _ in idle) if idle else None

    kwh = (e / 3.6e6) if e is not None else None
    cost = (kwh * args.price_eur_kwh) if kwh is not None else None
    out = {
        "cell_start": args.start, "cell_end": args.end,
        "cell_seconds": round(args.end - args.start, 1) if args.end else None,
        "energy_j": round(e, 1) if e is not None else None,
        "energy_kwh": round(kwh, 6) if kwh is not None else None,
        "mean_power_w": round(mean_w, 1) if mean_w is not None else None,
        "mean_sm_clock_mhz": round(mean_clk, 1) if mean_clk is not None else None,
        "mean_temp_c": round(mean_temp, 1) if mean_temp is not None else None,
        "idle_power_w": round(idle_w, 1) if idle_w is not None else None,
        "completion_tokens": args.tokens,
        "tokens_per_joule": round(args.tokens / e, 6) if e else None,
        "tokens_per_kwh": round(args.tokens / kwh, 1) if kwh else None,
        "eur_per_1m_tokens": round(cost / args.tokens * 1e6, 3) if cost and args.tokens else None,
        "tokens_per_eur": round(args.tokens / cost, 1) if cost else None,
        "price_eur_kwh": args.price_eur_kwh,
        "basis": "trapezoidal sum of per-GPU power.draw over the sampler bracket; "
                 "includes prefill+decode of the cell; idle is reported, not subtracted",
    }
    print(json.dumps(out, indent=1))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
