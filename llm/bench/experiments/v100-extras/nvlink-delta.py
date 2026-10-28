#!/usr/bin/env python3
"""nvlink-delta.py — traffic seen by the NVLink throughput counters.

`nvidia-smi nvlink -gt d` prints per-GPU, per-link data-payload counters (in
KiB).  On V100 the counters are cumulative until reset, but some driver
versions reset them on read; either way the difference between a sample taken
just before and just after a cell brackets that cell's traffic (if they reset
on read, the "after" sample *is* the cell).

usage: nvlink-delta.py before.txt after.txt [--json out.json]

Prints per-GPU MiB moved and the total, and keeps the raw text as evidence.
"""
import argparse
import json
import re

GPU_RE = re.compile(r"GPU\s+(\d+)")
NUM_RE = re.compile(r"(\d+)\s*KiB")


def parse(path):
    per_gpu = {}
    gpu = None
    try:
        for line in open(path, errors="replace"):
            m = GPU_RE.search(line)
            if m:
                gpu = int(m.group(1))
                per_gpu.setdefault(gpu, 0)
                continue
            for n in NUM_RE.findall(line):
                if gpu is not None:
                    per_gpu[gpu] += int(n)
    except OSError:
        pass
    return per_gpu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    before, after = parse(args.before), parse(args.after)
    gpus = sorted(set(before) | set(after))
    delta = {g: after.get(g, 0) - before.get(g, 0) for g in gpus}
    total = sum(delta.values())
    for g in gpus:
        print("GPU %d: %.3f MiB (before %d KiB, after %d KiB)"
              % (g, delta[g] / 1024.0, before.get(g, 0), after.get(g, 0)))
    print("total: %.3f MiB" % (total / 1024.0))
    if not gpus:
        print("(no per-GPU counter rows parsed; see the raw -gt output)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"before_kib": before, "after_kib": after,
                       "delta_kib": delta, "total_mib": total / 1024.0}, fh,
                      indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
