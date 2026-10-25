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
knob-sweep.sbatch             Slurm entry point (execs knob-sweep.sh)
knob-sweep.sh                 one-knob-at-a-time sweep at small contexts
strata_probe.py               stdlib OpenAI-streaming probe (one JSONL cell per run)
tests/stub_server.py          fake server for --self-test (127.0.0.1 only)
tests/test_fetch_models.sh    offline fake-hf regression test for fetch-models.sh
experiments/…                 protocol + results of each campaign
```

## Models

```text
IQ3_S   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf  54.8 GB
        IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf  28.8 GB (shared table)
Coder   ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
        IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf  29.6 GB
        + the same 00002 table (28.8 GB n-gram/PLE table)
        optional mmproj-Qwen3.8-Flash-Next-BF16.gguf (~0.9 GB; unused, --vision off)
```

The **Hub paths are prefixed** (`IQ3_S/`, `IQ1_M/`). A bare filename makes
`hf download` exit 0 without fetching anything, so `fetch-models.sh` passes
the prefixed path and then asserts both existence and exact pinned size.
The 28.8 GB shard 2 is byte-identical in both repos (same SHA-256, different
pack filename); it is downloaded **once** into `models/strata/_shared/` and
hardlinked into both pack dirs (copy fallback).

Both shards are **required** per pack: Strata reads shard 2 (the n-gram/PLE
table) at every start. Sizes/SHA-256 are pinned from
`/work/ffa/holecem/src/Strata/setup.py`; `fetch-models.sh --verify` is a
read-only audit (presence + exact size + sha256; never downloads, never
writes the manifest).
Library layout: `/work/ffa/holecem/models/strata/{iq3s,coder}/` plus an
append-only `manifest.txt`. `bench-matrix.sh` also accepts the download job's
alternate dirs `models/strata/models/{IQ3_S,coder-IQ1_M}` — but only when
every shard there has its exact pinned size (partial downloads never launch).

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

# 2-GPU L40 node (the runner passes --gpu 0,1 and the engine layer-splits):
sbatch --nice=100 --nodelist=gpuffaNN llm/strata/strata-bench.sbatch \
  --gpus 0,1 --model iq3s
# equivalent env form (no commas in the value):
GPUS_PER_NODE=2 sbatch --nice=100 ... llm/strata/strata-bench.sbatch --model iq3s
```

**Always `--nice=100`** — these jobs must never crowd out the fleet's
reservation work. The default partition here is `ffa-short` (2 h), so never
rely on the default; the sbatch pins `gpu-ffa` literally and you override on
the sbatch command line (see next section).

### Real env/CLI interface

`MODEL_NAME=iq3s|coder|both` and `GPUS_PER_NODE=N` are honored now (unknown
values fail fast). `--model` and `--gpus LIST` are the CLI equivalents and win.
`GPUS_PER_NODE=2` is shorthand for `--gpus 0,1`; with one card no `--gpu` flag
is passed and Strata picks it.

**Do not put comma lists in `--export`.** Slurm splits the value at commas, so
`--export=ALL,CTXS=1024,4096` silently runs with `CTXS=1024` and a bogus
`4096` key. Pass matrix knobs as trailing arguments instead:

```bash
sbatch --nice=100 llm/strata/strata-bench.sbatch \
  --model iq3s --contexts 1024,4096,16384,65536 --batches 1,2,4
```

### V100

V100 (sm_70, Volta) cannot run the CUDA 13 engine; `install-engine.sh`
installs **both** engine variants and Strata's Control picks `cuda12`
automatically from `nvidia-smi` archs — nothing to pass. Use only contexts
`1024,4096,16384,65536` (docs/NVIDIA_V100.md); the probe now clamps a nominal
65536 cell to `65536 − max_tokens − 128` real prompt tokens. Volta nodes are shared
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
(free-space-checked; `COPY_MODELS=0` reads Lustre directly), then start one
server for the whole matrix (`strata launch --context max(CTXS) --parallel
max(BATCHES) --gpu <list when >1> --kv int8 --vision off --access local
--port <free> --no-api-key --data-dir <private> --yes --no-tui --restart`),
wait for `/v1/models`, then probe every (context, concurrency) cell.

Each job gets a **private engine data dir** `$DATA_DIR/jobs/<job-id>`, seeded
from the shared `_strata-data` tree: read-only `mtp/` tensors and `packs/` are
hardlinked, the small mutable `mtp/rt/` tree is copied. Two jobs on one node
therefore never write the same `experts.bin`. If the shared tree is not
complete yet, the runner holds a `flock` on `$DATA_DIR/.prepare.lock` across
the first launch instead of racing another builder. The private dir is removed
on exit unless `KEEP_DATA_DIR=1`.

A startup precheck selects cuda13/cuda12 from `nvidia-smi` compute capability,
checks the engine's highest required `GLIBCXX_*`, and either accepts the node
`libstdc++`, prepends `CXX_LIB_DIR` (default the reusable l40x1
`deps/cxx/lib`), or fails immediately with the workaround. Engine launch logs
under the per-job state dir are copied to `OUTDIR/<model>/engine-logs/` before
the EXIT trap deletes scratch.

The probe calibrates prompt padding with two small `include_usage` requests
(a slope plus the fixed chat-template overhead), then clamps the nominal
context to `max_context - max_tokens - pad_safety`
(default safety 128) and records the calibration factor and clamp in every
JSONL cell. Output:

```text
$OUTDIR/<model>/cells.jsonl    one JSON per cell: contract fields
                               ts,model,context,concurrency,prefill_tok_s,
                               decode_tok_s_p50,decode_tok_s_p95,aggregate_tok_s,
                               ttft_s_p50,ttft_s_p95,completion_tokens,
                               prompt_tokens,engine_metrics,error (+diagnostics,
                               prompt_target_tokens,prompt_clamped,calibration)
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

## Knob sweep (small contexts)

The matrix pins one engine recipe. `knob-sweep.sh` varies **one knob per
variant** at 1024/4096/16384 only (never 64k/131k/V100/2-GPU — those exist) so
the deltas are attributable:

```bash
# login-node checks (no GPU, no Slurm):
llm/strata/knob-sweep.sh --plan --model iq3s --node uk-g6-07
llm/strata/knob-sweep.sh --self-test      # stub server, 5 cells, schema asserts

# one sweep job (always --nice=100; pin the node and keep --node in sync):
sbatch --nice=100 --nodelist=uk-g6-07 --time=11:00:00 \
  llm/strata/knob-sweep.sbatch --model iq3s --node uk-g6-07 --gpus 0 \
  --contexts 1024,4096,16384 --batches 1,4 --variants all --budget-min 240
```

Variants: `control` (baseline recipe, one 1024/C1 cell as a host/engine
variance check), `nospec` (`--spec` removed — the engine refuses on a native IQ
pack, so it records `launch-fail` with `it needs --spec T (T >= 2)`),
`mtpoff` (`--mtp off`, the workable no-drafts arm), `spec_pure` (`--spec 4
--spec-min-p 0`), `kvfp16` (`--kv fp16`), `kvres16k`/`kvres64k`
(`--kv-resident` 16384/65536), `batch8` (`--parallel 8` plus C=8 cells),
`groups1` (`--batch-mtp --batch-groups 1`), `mtpserial`
(`STRATA_BATCH_MTP=1 STRATA_BATCH_MTP_SERIAL=1`), `vramreserve`
(`--vram-reserve-mib 1024`, since the auto expert cache otherwise leaves ~430
MiB free and the engine warns about #2065 paging). `kvbf16` and
`expert_cache_max` are recorded as `skipped-flag`: engine 0.1.50 has no
`--kv bf16` (only fp16/int8/q4_0/k8v4) and `--expert-cache` has no `max` mode.

The pack is staged to node-local scratch **once** per job; each variant seeds a
private `--data-dir` from `_strata-data` (hardlinked read-only trees, copied
`mtp/rt`) and starts its own headless server at the baseline server window
(`--max-context 131072`, so `--kv-resident` still streams). Each variant also
gets its own launch profile: launch-level choices (notably `--mtp off`) are
persisted in the model's saved settings and would otherwise leak into the next
variant. Output:

```text
$OUTDIR/knobs.csv                     variant,ctx,concurrency,decode_p50,
                                      aggregate,ttft_p50,ttft_p95,status,error
$OUTDIR/variants.csv                  exact --set/env and outcome per variant
$OUTDIR/<model>/<variant>/summary.csv same columns as knobs.csv
$OUTDIR/<model>/<variant>/cells.jsonl one probe cell per line, `variant` field
$OUTDIR/<model>/<variant>/launch-config.json, engine-logs/, variant-facts.txt
```

Status is `ok|fail|server-dead|launch-fail|skipped-flag|skipped-budget`
(`skipped-done` on `--resume`). A failed launch or dead server records the real
error and the sweep continues with the next variant. Re-running with the same
`--outdir` needs `--resume` (keep complete variants) or `--force`; partial
variant dirs are moved to `*.bak-<ts>` before a retry.

## Gotchas

- **MTP drafter**: on first launch Strata downloads the ~5–6 GB MTP draft
  layer from the original Qwen checkpoint into `--data-dir`. Compute nodes
  have network; the persistent shared `_strata-data` makes it a one-time
  cost, and each job then works from its private seeded copy. Without network
  the engine runs **without the drafter** — decode numbers drop, so check the
  launch log for `draft layer loaded` before trusting a low outlier.
- **Hub paths**: `IQ3_S/` and `IQ1_M/` prefixes are mandatory. A bare
  filename makes `hf download` exit 0 without fetching; the fetch script
  asserts existence + exact size and fails loudly.
- **Slurm `--export`**: never put commas in an export value (`CTXS=1024,4096`
  becomes two keys). Use trailing `--contexts`/`--batches` args.
- **DATA_DIR / concurrency**: the runner now always seeds a private
  `$DATA_DIR/jobs/<job>`; if the shared `mtp/rt` tree is missing it holds
  `$DATA_DIR/.prepare.lock` across the first launch. Do not point two old
  runners at the same `--data-dir` — that is the bug this changed.
- **EL9 libstdc++ floor**: the cuda13 engine needs `GLIBCXX_3.4.30`; the
  gpu-ffa image has 3.4.29. The runner auto-prepends the reusable
  `/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib` when present,
  or exits with the `LD_LIBRARY_PATH`/`CXX_LIB_DIR` workaround. It does not
  rely on Control's driver-only cuda12 fallback.
- **Login node**: never start servers or load models there. Only installs and
  downloads (network) plus `--self-test` (stdlib stub on 127.0.0.1) are OK.
- **/tmp size**: IQ3_S needs ~86 GiB scratch; if `/tmp` is smaller set
  `SCRATCH_ROOT` to the node's NVMe or `COPY_MODELS=0`.
- **Engine variants**: one components dir serves both fleets; `strata update`
  itself can't run on the GPU-less login node, which is why `install-engine.sh`
  fetches the two feed-named tarballs (sha256-verified) itself. Control scans
  `~/.local/share/strata/components` unless `STRATA_COMPONENTS_DIR` is set —
  `bench-matrix.sh` exports it to `$STRATA_ROOT/components` (without it the
  engine packages look absent on compute nodes).
- **Parallel cap**: `strata launch --parallel` accepts 1–8; `--batches` is
  clamped to 8.
- **Driver floor**: the cuda13 engine needs driver ≥ 580.65 (cuda12 ≥ 575.51);
  Control falls back to cuda12 automatically if older.
- **Probe padding**: the probe no longer assumes 4 chars/token. It measures
  the tokenizer with a small request and clamps to the window, recording
  `prompt_target_tokens`/`prompt_clamped`/`calibration` in JSONL. A nominal
  131072 cell now runs at ~130.8k real tokens instead of 400ing at ~149k.
- **Concurrent `--verify`**: several login-node audits of the same ~114 GiB
  library contend badly; run one at a time (or rely on the pinned manifest).

## Self-test evidence

`bench-matrix.sh --self-test` (2026-10-25, login node, after the fixes): stub
server + 1 MB dummy gguf, 4 cells (ctx 512/1024 × C 1/2), all `ok`; stub
reports engine `timings` so `decode_tok_s_p50` lands at the exact rate
(100.00 tok/s, asserted 90–110). The probe records a 4.007 chars/token
calibration factor (two-point fit, zero intercept on the stub); the nominal
1024 cell clamps to 864 prompt tokens
(1024 − 32 max_tokens − 128 safety). The test also asserts the JSONL contract,
that engine logs are archived before scratch cleanup, and that a private data
dir isolates a mutable `rt/` file from the shared staging tree. Exit 0 on
success, non-zero otherwise; the stub is killed by the EXIT trap.

`tests/test_fetch_models.sh` (same day): fake `hf`/`sha256sum` + sparse files,
checks prefixed `IQ3_S/`/`IQ1_M/` paths, exactly one shard-2 download,
hardlinks into both pack dirs, exact sizes, a clean `--verify` audit, and a
loud failure when `hf` exits 0 but creates nothing. No network, no real model
data.
