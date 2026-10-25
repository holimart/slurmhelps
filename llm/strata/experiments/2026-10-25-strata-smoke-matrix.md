# 2026-10-25 — Strata smoke matrix, Qwen3.8-Flash-Next GGUFs on L40 + V100

First campaign for the Strata engine harness (`llm/strata/`). Purpose: prove
the IQ3_S and Coder-IQ1_M GGUF packs serve and stream on both GPU fleets and
produce a first decode/TTFT/concurrency picture before any bigger sweep.

## Engine

Installed by `install-engine.sh` into `/work/ffa/holecem/strata-engine`:
Terminal via the official `install.sh` (STRATA_DATA_DIR/STRATA_BIN_DIR
redirected), then **both** signed engine packages from the release feed
(`strata-feed.json`, sha256-verified): `cuda13` for L40 (sm_89) and `cuda12`
for V100 (sm_70 — "CUDA 13 cannot run Volta", docs/NVIDIA_V100.md). Control
auto-selects per node from `nvidia-smi`; no flag needed. Feed at run time
named engine 0.1.50 for both variants.

## Models

```text
IQ3_S   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
        IQ3_S-00001-of-00002.gguf 54 817 524 224 B sha256 4c1eb2ce…d2aca3
        IQ3_S-00002-of-00002.gguf 28 800 138 432 B sha256 316b46f3…61e113
Coder   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
        IQ1_M-00001-of-00002.gguf 29 608 446 496 B sha256 e11083ba…087fad
        + the same 00002 shard (shared n-gram/PLE table, read at every start)
```

Pinned from `/work/ffa/holecem/src/Strata/setup.py` (revisions 2026-09-29).
Library: `/work/ffa/holecem/models/strata/{iq3s,coder}` (fetch with
`fetch-models.sh`; both shards per pack are mandatory). `--vision off`, so
no mmproj. `--kv int8`. MTP drafter cached in `--data-dir
/work/ffa/holecem/models/strata/_strata-data` (downloaded once by the first
launch; fallback = no drafter, check launch logs).

## Matrix (smoke)

One server per pack, `--context 4096 --parallel 4` (max of the ladder):

```text
contexts:    1024, 4096          (V100 nodes only ever 1024/4096/16384/65536)
concurrency: 1, 4
per cell:    C streams × 3 sequential requests, 128 max_tokens, temp 0,
             salted prompts (no cache inflation), engine timings preferred
```

Planned submissions (always `--nice=100`; job time 11:30 inside MaxTime 12):

```bash
sbatch --nice=100 llm/strata/strata-bench.sbatch                       # L40, both packs
sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta03 \
  --gres=gpu:1 --mem=190G --cpus-per-task=24 \
  llm/strata/strata-bench.sbatch --model coder                          # V100 2×16 GB
```

Results land in `/work/ffa/holecem/results/strata-bench-<job>-<ts>/<model>/`
(`summary.csv` + `cells.jsonl`).

## Results

_To be filled from the runs._ Per pack:

| ctx | C | decode tok/s p50 | aggregate tok/s | TTFT p50 | TTFT p95 | status |
|----:|--:|-----------------:|----------------:|---------:|---------:|--------|
|     |   |                  |                 |          |          |        |

## Harness validation (pre-flight, login node)

`bench-matrix.sh --self-test` — 2026-10-25T14:02Z and 14:03Z, exit 0 both:

```text
cells.jsonl lines: 4 (expected 4)      # ctx 512/1024 × C 1/2, stub server
summary.csv data rows: 4 (expected 4)  # header == contract
  cell ctx=512  C=1 decode=100.0 agg=94.1  ttft=0.014 ok=2/2
  cell ctx=512  C=2 decode=100.0 agg=187.7 ttft=0.014 ok=4/4
  cell ctx=1024 C=1 decode=100.0 agg=90.7  ttft=0.027 ok=2/2
  cell ctx=1024 C=2 decode=100.0 agg=179.9 ttft=0.027 ok=4/4
  JSONL contract + stub decode band (90..110 tok/s): ok
SELF-TEST PASS
```

Stub reports deterministic llama.cpp-style `timings` (predicted_ms), so the
engine-timing path must land exactly on 100 tok/s (band 90–110 asserted);
TTFT doubles with context and aggregate ~doubles with C=1→2. Standalone probe
checks against the same stub: nominal C=3 cell ok (6/6, agg 274 tok/s),
`…/v1` base-url + unknown model id non-fatal, dead server → exit 1 with the
error still recorded as a JSONL line. Stub bound to 127.0.0.1, killed after
(verified no lingering processes). `bash -n` clean on all scripts
(shellcheck not installed on the login node).

## Runner gotchas hit/avoided

- `#SBATCH` does not expand variables → partition stays literal `gpu-ffa`,
  V100 overrides go on the `sbatch` command line.
- `strata update` refuses on the GPU-less login node → installer pulls both
  feed-named tarballs itself (this is by design, see README).
- MTP drafter first-run download needs compute-node network; persistent
  `--data-dir` amortizes it.
- `/tmp` copy needs ~86 GiB for IQ3_S; free-space check aborts early with
  `SCRATCH_ROOT`/`COPY_MODELS` guidance.
- pkill fallbacks match our own `$STRATA_ROOT/components/engine` path and run
  as the job user only — safe on shared Volta boxes.
