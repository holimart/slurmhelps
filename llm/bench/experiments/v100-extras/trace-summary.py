#!/usr/bin/env python3
"""trace-summary.py — aggregate memcpy-trace.tsv into per-transport totals.

Reads the TSV written by memcpy-trace.c (7 columns from the first version,
9 columns with src_dev/dst_dev from the second):

    realtime_ns  fn  kind  bytes  src  dst  rc  [src_dev  dst_dev]

and prints (and optionally writes JSON):
  * calls and bytes per (fn, kind) inside an optional [--since, --until] window
    (nanoseconds since the epoch, same clock as `date +%s%N`),
  * cross-device bytes (src_dev != dst_dev, both known) per (fn, kind),
  * peer-API bytes and every non-zero return code.

The window separates a phase's *load* traffic from the decode/prefill cell:
the orchestrator records `date +%s%N` immediately before and after the probe.
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
    cross_total = 0
    peer_bytes = 0
    host_bytes = 0
    for line in open(args.trace, errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) not in (7, 9):
            continue
        ts = int(parts[0])
        if ts < args.since or (args.until and ts > args.until):
            continue
        fn, kind, nbytes, rc = parts[1], parts[2], int(parts[3]), int(parts[6])
        sd = int(parts[7]) if len(parts) == 9 else -99
        dd = int(parts[8]) if len(parts) == 9 else -99
        rows += 1
        slot = totals.setdefault((fn, kind), {"calls": 0, "bytes": 0,
                                             "cross_calls": 0, "cross_bytes": 0})
        slot["calls"] += 1
        slot["bytes"] += nbytes
        cross = sd >= 0 and dd >= 0 and sd != dd
        if cross:
            slot["cross_calls"] += 1
            slot["cross_bytes"] += nbytes
            cross_total += nbytes
        if "Peer" in fn:
            peer_bytes += nbytes
        if fn in ("h2d", "d2h") or "Host" in fn:
            host_bytes += nbytes
        if rc != 0 and len(errors) < 10:
            errors.append({"ts": ts, "fn": fn, "kind": kind,
                           "bytes": nbytes, "rc": rc})

    total = sum(v["bytes"] for v in totals.values())
    summary = {
        "rows_in_window": rows,
        "total_bytes": total,
        "cross_device_bytes": cross_total,
        "peer_api_bytes": peer_bytes,
        "host_direction_bytes": host_bytes,
        "by_call": {"%s/%s" % k: v for k, v in sorted(totals.items())},
        "nonzero_returns": errors,
    }
    print("[trace] window rows=%d total=%.1f MB, cross-device=%.1f MB, "
          "peer-API=%.1f MB, host-direction=%.1f MB"
          % (rows, total / 1e6, cross_total / 1e6, peer_bytes / 1e6,
             host_bytes / 1e6))
    for (fn, kind), v in sorted(totals.items()):
        print("  %-20s %-8s %7d calls %9.1f MB  cross %7.1f MB (%d calls)"
              % (fn, kind, v["calls"], v["bytes"] / 1e6,
                 v["cross_bytes"] / 1e6, v["cross_calls"]))
    for e in errors:
        print("  ERROR %s/%s bytes=%d rc=%d" % (e["fn"], e["kind"], e["bytes"], e["rc"]))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
