# Strata engine benchmarks — Qwen3.8-Flash-Next GGUFs

Target: **Qwen3.8-Flash-Next** quantized GGUFs (IQ3_S dense, Coder IQ1_M)
served by the [Strata](https://github.com/Niko1221/Strata) CUDA engine on
the MFF Troja cluster — **L40 48 GB** (`gpu-ffa`) and **V100** (`ffa-gpulab`
/ `volta*`). The full checkpoint activates ~6B/token from a 125B MoE plus a
51B n-gram/PLE table, which is exactly the split the GGUF packs ship: experts
(RAM) + shared table (SSD mmap) + GPU-resident attention.

## Files

```text
install-engine.sh             login node: Strata Terminal + cuda13 & cuda12 engines
fetch-models.sh               hf-download the pinned GGUF packs + manifest
strata-bench.sbatch           Slurm entry point (execs bench-matrix.sh)
bench-matrix.sh               the matrix runner (server per model, cells, CSV/JSONL)
strata_probe.py               stdlib OpenAI-streaming probe (one JSONL cell per run)
tests/stub_server.py          fake server for --self-test (127.0.0.1 only)
experiments/…                 protocol + results of each campaign
```

## Models

```text
IQ3_S   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
        Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf  54.8 GB
        Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf  28.8 GB (shared table)
Coder   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
        Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf  29.6 GB
        + the same 00002 shard (28.8 GB n-gram/PLE table)
        optional mmproj-Qwen3.8-Flash-Next-BF16.gguf (~0.9 GB; unused, --vision off)
```

Both shards are **required** per pack: Strata reads shard 2 (the n-gram/PLE
table) at every start. Sizes/SHA-256 are pinned from
`/work/ffa/holecem/src/Strata/setup.py`; `fetch-models.sh --verify` re-hashes.
Library layout: `/work/ffa/holecem/models/strata/{iq3s,coder}/` plus an
append-only `manifest.txt`.

## Run

```bash
cd $WORK_ROOT/slurmhelps-fleet

# 0. plumbing check on the login node (stub server, no GPU, a few seconds):
llm/strata/bench-matrix.sh --self-test

# 1. engine (login node, network only):
llm/strata/install-engine.sh

# 2. models (someone runs this; ~112 GB shared, resumable):
llm/strata/fetch-models.sh --only iq3s  --submit
llm/strata/fetch-models.sh --only coder --submit

# 3. matrix (L40 default node shape is baked into the sbatch):
sbatch --nice=100 llm/strata/strata-bench.sbatch
sbatch --nice=100 llm/strata/strata-bench.sbatch --model coder \
  --contexts 1024,4096,16384,65536 --batches 1,2,4,8
```

**Always `--nice=100`** — these jobs must never crowd out the fleet's
reservation work. The default partition here is `ffa-short` (2 h), so never
rely on the default; the sbatch pins `gpu-ffa` literally and you override on
the sbatch command line (see next section).

### V100

V100 (sm_70, Volta) cannot run the CUDA 13 engine; `install-engine.sh`
installs **both** engine variants and Strata's Control picks `cuda12`
automatically from `nvidia-smi` archs — nothing to pass. Use only contexts
`1024,4096,16384,65536` (docs/NVIDIA_V100.md). Volta nodes are shared
(`volta01` 1×16 GB, `volta02/03` 2×16 GB, `volta04` 1×32 GB, `volta05`
4×32 GB), so pick a node that fits the pack and give it memory:

```bash
sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta03 \
  --gres=gpu:1 --mem=190G --cpus-per-task=24 \
  llm/strata/strata-bench.sbatch --model coder
```

`#SBATCH` lines do **not** expand shell variables — partition/gres/mem
overrides belong on the `sbatch` command line, exactly as above.
`--time=11:30:00` stays inside both partitions' 12 h MaxTime; keep
`--budget-min` under it so cells close cleanly.

## What the runner does

Per model pack: copy the shards to node-local `/tmp/strata-$SLURM_JOB_ID`
(free-space-checked; `COPY_MODELS=0` reads Lustre directly), start one
server for the whole matrix (`strata launch --context max(CTXS) --parallel
max(BATCHES) --kv int8 --vision off --access local --port <free> --no-api-key
--data-dir /work/ffa/holecem/models/strata/_strata-data --yes --no-tui
--restart`), wait for `/v1/models`, then probe every (context, concurrency)
cell. Output:

```text
$OUTDIR/<model>/cells.jsonl    one JSON per cell: contract fields
                               ts,model,context,concurrency,prefill_tok_s,
                               decode_tok_s_p50,decode_tok_s_p95,aggregate_tok_s,
                               ttft_s_p50,ttft_s_p95,completion_tokens,
                               prompt_tokens,engine_metrics,error (+diagnostics)
$OUTDIR/<model>/summary.csv    ctx,concurrency,decode_tok_s_p50,aggregate_tok_s,
                               ttft_s_p50,ttft_s_p95,status
                               status = ok | fail | server-dead | skipped-budget
```

A cell whose every request fails costs at most one server restart (max 3 per
matrix) plus one retry; a wall-clock `--budget-min` (default 420) marks the
rest `skipped-budget`. Per-job env isolation (`STRATA_PROFILE_DIR`,
`STRATA_RUNTIME_DIR`, `XDG_STATE_HOME`, `XDG_CONFIG_HOME` under the scratch
dir) keeps concurrent jobs from sharing state; the EXIT trap always stops the
server. Decode/prefill prefer the engine's llama.cpp-style `timings`
(`prompt_ms`/`predicted_ms` on the final chunk, per docs/BENCHMARK_PROTOCOL.md),
falling back to client stream windows. TTFT = dispatch→first content delta.
Prompts carry a unique salt per request so the prompt cache never fakes the
prefill numbers.

## Gotchas

- **MTP drafter**: on first launch Strata downloads the ~5–6 GB MTP draft
  layer from the original Qwen checkpoint into `--data-dir`. Compute nodes
  have network; the persistent `--data-dir` on `/work` makes it a one-time
  cost. Without network the engine runs **without the drafter** — decode
  numbers drop, so check the launch log before trusting a low outlier.
- **Login node**: never start servers or load models there. Only installs and
  downloads (network) plus `--self-test` (stdlib stub on 127.0.0.1) are OK.
- **/tmp size**: IQ3_S needs ~86 GiB scratch; if `/tmp` is smaller set
  `SCRATCH_ROOT` to the node's NVMe or `COPY_MODELS=0`.
- **Engine variants**: one components dir serves both fleets; `strata update`
  itself can't run on the GPU-less login node, which is why `install-engine.sh`
  fetches the two feed-named tarballs (sha256-verified) itself.
- **Parallel cap**: `strata launch --parallel` accepts 1–8; `--batches` is
  clamped to 8.
- **Driver floor**: the cuda13 engine needs driver ≥ 580.65 (cuda12 ≥ 575.51);
  Control falls back to cuda12 automatically if older.

## Self-test evidence

`bench-matrix.sh --self-test` (2026-10-25, login node): stub server +
1 MB dummy gguf, 4 cells (ctx 512/1024 × C 1/2), all `ok`; stub reports
engine `timings` so `decode_tok_s_p50` lands at the stub's exact rate
(100.00 tok/s, asserted 90–110), aggregate doubles from C=1 to C=2, and both
output files match the contract. Exit 0 on success, non-zero otherwise; the
stub is killed by the EXIT trap (verified no lingering process).
