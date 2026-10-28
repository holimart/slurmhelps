#!/usr/bin/env python3
"""active-bytes.py — per-token active weight traffic from the GGUF tensor table.

The serving docs quote "~6B active", but a decode step also streams the
non-expert trunk (attention, hyper-connections, shared expert, output) every
token.  This script sums what the kernels actually read:

  active = (all non-expert tensor bytes - token_embd)
         + expert tensor bytes * expert_used_count / expert_count
         + PLE gather rows (16 rows x 160 values, IQ4_NL)

Then it prints the active-bytes roofline at the MI210 paper bandwidth
(1.6 TB/s/card), at the measured HBM read bandwidth (hbm-bw.txt), and the
"6B active @ 3.38 bpw" assumption used in the dataset, for comparison.

usage: active-bytes.py --gguf SHARD1 [--shard2 SHARD2] [--measured-gbs 1336]
                       [--measured-tps 37.95]
"""
import argparse
import json
import sys

import gguf
from gguf.constants import GGML_QUANT_SIZES


def tensor_bytes(t):
    n = 1
    for d in t.shape:
        n *= int(d)
    blk_elems, blk_bytes = GGML_QUANT_SIZES[t.tensor_type]
    return n / blk_elems * blk_bytes, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--shard2", default=None)
    ap.add_argument("--measured-gbs", type=float, default=None)
    ap.add_argument("--measured-tps", type=float, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)
    arch = r.fields["general.architecture"].contents()
    n_expert = int(r.fields["%s.expert_count" % arch].contents())
    n_used = int(r.fields["%s.expert_used_count" % arch].contents())
    n_layer = int(r.fields["%s.block_count" % arch].contents())
    n_ngram = int(r.fields["%s.ple.ngram_size" % arch].contents())
    n_heads_ngram = int(r.fields["%s.ple.heads_per_ngram" % arch].contents())
    ple_dim = int(r.fields["%s.embedding_length_per_layer_input" % arch].contents())

    total = 0.0
    params = 0
    expert = 0.0
    token_embd = 0.0
    for t in r.tensors:
        b, n = tensor_bytes(t)
        total += b
        params += n
        if "ffn_gate_exps" in t.name or "ffn_up_exps" in t.name or "ffn_down_exps" in t.name:
            expert += b
        if t.name == "token_embd.weight":
            token_embd += b

    # PLE table lives in shard 2 and is gathered, not streamed.
    ple_rows = (n_ngram - 1) * n_heads_ngram
    ple_gather = ple_rows * ple_dim * (4.5 / 8.0)

    nonexpert = total - expert
    active = (nonexpert - token_embd) + expert * n_used / n_expert + ple_gather
    bpw = total * 8.0 / params

    # shard2 (PLE table) is resident in host RAM, lazily gathered
    ple_total = 0.0
    if args.shard2:
        r2 = gguf.GGUFReader(args.shard2)
        for t in r2.tensors:
            b, _ = tensor_bytes(t)
            ple_total += b

    out = {
        "arch": arch,
        "layers": n_layer,
        "expert_count": n_expert,
        "expert_used_count": n_used,
        "total_bytes_shard1": total,
        "ple_table_bytes_shard2": ple_total,
        "total_params_shard1": params,
        "bpw_shard1": bpw,
        "expert_bytes": expert,
        "nonexpert_bytes": nonexpert,
        "token_embd_bytes": token_embd,
        "ple_gather_bytes_per_token": ple_gather,
        "active_bytes_per_token": active,
        "active_bytes_assumption_6B_3_38bpw": 6e9 * 3.38 / 8.0,
    }
    out["roofline_spec_1600_tps"] = 1600e9 / active
    out["roofline_task_assumption_1600_tps"] = 1600e9 / out["active_bytes_assumption_6B_3_38bpw"]
    if args.measured_gbs:
        out["roofline_measured_bw_tps"] = args.measured_gbs * 1e9 / active
        out["roofline_task_assumption_measured_bw_tps"] = args.measured_gbs * 1e9 / out["active_bytes_assumption_6B_3_38bpw"]
    if args.measured_tps:
        out["measured_decode_tps"] = args.measured_tps
        out["eta_spec"] = args.measured_tps / out["roofline_spec_1600_tps"]
        if args.measured_gbs:
            out["eta_measured_bw"] = args.measured_tps / out["roofline_measured_bw_tps"]
        out["eta_task_assumption_spec"] = args.measured_tps / out["roofline_task_assumption_1600_tps"]

    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
