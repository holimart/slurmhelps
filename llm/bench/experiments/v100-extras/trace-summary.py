#!/usr/bin/env python3
"""trace-summary.py — aggregate memcpy-trace.tsv into per-transport totals.

Reads the TSV written by memcpy-trace.c:
    realtime_ns  fn  kind  bytes  src  dst  rc

and prints (and optionally writes JSON):
  * calls and bytes per (fn, kind) inside an optional [--since, --until] window
    (nanoseconds since the epoch, same clock as `date +%s%N`),
  * the peer share of all copied bytes,
  * every non-zero return code, with the first row that produced it.

The window is how a phase's *load* traffic is separated from the decode/
prefill cell: the orchestrator records `date +%s%N` immediately before and
after the probe request.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--since", type=int, default=0)
    ap.add_argument("--until", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    totals = {}
    errors = []
    rows = 0
    for line in open(args.trace, errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 7:
            continue
        ts, fn, kind, nbytes, src, dst, rc = parts
        ts = int(ts)
        if ts < args.since or (args.until and ts > args.until):
            continue
        rows += 1
        key = (fn, kind)
        slot = totals.setdefault(key, {"calls": 0, "bytes": 0})
        slot["calls"] += 1
        slot["bytes"] += int(nbytes)
        if int(rc) != 0 and len(errors) < 10:
            errors.append({"ts": ts, "fn": fn, "kind": kind,
                           "bytes": int(nbytes), "rc": int(rc)})

    by_transport = {"peer": 0, "d2d": 0, "host": 0, "default": 0}
    for (fn, kind), slot in totals.items():
        if fn == "cudaMemcpyPeerAsync" or fn == "cudaMemcpyPeer":
            by_transport["peer"] += slot["bytes"]
        elif fn == "cudaMemcpyAsync" and kind == "d2d":
            by_transport["d2d"] += slot["bytes"]
        elif fn == "cudaMemcpyAsync" and kind in ("h2d", "d2h"):
            by_transport["host"] += slot["bytes"]
        elif fn == "cudaMemcpyAsync" and kind == "default":
            by_transport["default"] += slot["bytes"]

    total = sum(v["bytes"] for v in totals.values())
    summary = {
        "rows_in_window": rows,
        "total_bytes": total,
        "by_transport": by_transport,
        "peer_share": (by_transport["peer"] / total) if total else None,
        "by_call": {"%s/%s" % k: v for k, v in sorted(totals.items())},
        "nonzero_returns": errors,
    }
    print("[trace] window rows=%d total=%.1f MB" % (rows, total / 1e6))
    for (fn, kind), slot in sorted(totals.items()):
        print("  %-22s %-8s %7d calls %10.1f MB"
              % (fn, kind, slot["calls"], slot["bytes"] / 1e6))
    if total:
        print("  peer share = %.1f %%  (d2d %.1f MB, host %.1f MB, default %.1f MB)"
              % (100.0 * by_transport["peer"] / total,
                 by_transport["d2d"] / 1e6, by_transport["host"] / 1e6,
                 by_transport["default"] / 1e6))
    for e in errors:
        print("  ERROR %s/%s bytes=%d rc=%d" % (e["fn"], e["kind"], e["bytes"], e["rc"]))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
