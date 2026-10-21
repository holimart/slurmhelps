# GLM-5.3-Flash REAP50 — deployment and batched-throughput benchmark

This directory installs, serves and benchmarks the **50%-expert-pruned
GLM-5.3-Flash** family (`321B → ~165B`, 144 of 288 routed experts) on a Slurm
cluster. It is deliberately independent of any one project: partition, account,
work root and paths come from the calling project's `slurmhelps.json` (or from
flags/environment).

It supports **two target experiments**, plus a quality-first option and the V100
fallback:

| Experiment | Hardware | Stack | Question it answers |
|---|---|---|---|
| **A. Max throughput (target)** | 2× 96 GB Blackwell, 192 GB total | SGLang v0.5.20+ + **NVFP4 v2** | How many aggregate tok/s can batching + prefix caching reach? |
| **B. Gentle / rented (less VRAM)** | 2× 48 GB L40S, 96 GB total; or 1× 96 GB Blackwell | llama.cpp + **Q4_K_S** (87.1 GiB) or **IQ4_XS** (82.0 GiB) | What is still usable on a smaller, lower-power rental? |
| Quality-first | 4× H100/H200 or 192 GB Blackwell | SGLang/vLLM + **FP8 v2** (160.6 GiB) | Highest published quality, accepting much less KV headroom |
| V100 / GGUF fallback | 4× V100 32 GB, 128 GB total | llama.cpp + Q4_K_S/IQ4_XS | Runs at all on older cards, no BF16/FP8/FP4 |

Read the caveats at the bottom before trusting a number.

## The two experiments

Both use the same `install.sh` / `serve.sh` / `bench.sh`; only the flags and the
node change. Run `--quick` first on either one to validate the install before
the full concurrency sweep.

### Experiment A — max throughput (target)

| | |
|---|---|
| Node | `bw01`, 2× RTX PRO 6000 Blackwell **96 GB each = 192 GB**, 48 CPUs, 760 GB RAM |
| Slurm | `gpu-long`, `--gres gpu:2 --cpus 32 --mem 256G --time 12:00:00` |
| Model | **REAP50 NVFP4 v2**, 98.2 GiB, 0.8386 top-1 |
| Engine | SGLang, TP2, FP8 KV, RadixAttention, chunked prefill, CUDA graphs |
| Server shape | `--mem-fraction-static 0.80 --max-running-requests 96 --max-prefill-tokens 16384 --chunked-prefill-size 8192 --cuda-graph-max-bs-decode 128 --schedule-policy lpm` |
| Measures | Aggregate output tok/s vs concurrency (1–128), TTFT/TPOT, goodput, prefix-cache hit rate |
| Estimate | **~700–1,200 tok/s aggregate at C16–32**, ~150–200 single-stream (no published REAP50 run; must measure) |
| Why this node | NVFP4 fits with ~90 GB left for KV/activations, so concurrency is not VRAM-starved; Blackwell runs the NVFP4 path natively |

```bash
./submit.sh \
  --node bw01 --partition gpu-long --account ffa \
  --gres gpu:2 --cpus 32 --mem 256G --time 12:00:00 \
  --engine sglang --variant nvfp4
```

### Experiment B — gentle / rented, less GPU memory

| | |
|---|---|
| Node | **2× 48 GB L40S = 96 GB total** (rental-friendly) **or 1× 96 GB Blackwell** |
| Slurm / rental | 2×48: `--gres gpu:2 --cpus 32 --mem 256G --time 12:00:00`; 1×96: `--gres gpu:1` |
| Model | **Q4_K_S** 87.1 GiB (PPL 4.5833) or **IQ4_XS** 82.0 GiB (PPL 4.6002, more KV room) |
| Engine | llama.cpp patched fork, layer split across the cards, `-np 16–32`, q8_0 KV, prompt cache |
| Measures | Single-stream and moderate-concurrency throughput; quality/VRAM/power trade |
| Estimate | **~30–120 tok/s aggregate at C8–16**, ~10–30 single-stream on 2×48 GB; a single 96 GB card avoids PCIe layer-split overhead and should be faster |
| Why this node | 87–82 GiB fits in 96 GB; ~700 W for 2× L40S vs ~1,200 W for 2× 96 GB Blackwell. Llama.cpp has no paged KV/RadixAttention, so this is a “gentle” experiment, not a max-concurrency one |

```bash
# 2x 48 GB L40S (replace <node>/<partition>/<account> with the rental's Slurm values)
./submit.sh \
  --node <l40s-node> --partition <partition> --account <account> \
  --gres gpu:2 --cpus 32 --mem 256G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s

# 1x 96 GB Blackwell: same model, no tensor-parallel overhead
./submit.sh \
  --node <blackwell-node> --partition <partition> --account <account> \
  --gres gpu:1 --cpus 32 --mem 256G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s
```

If the 2×48 GB node is tight on KV, switch to `--variant iq4_xs` (82.0 GiB:
~14 GB left instead of ~9 GB). On a single 80 GB card only `iq3_m` (67.2 GiB)
fits, with ~13 GB for KV and lower quality (PPL 4.6253).

**Fit by total VRAM:**

| Total VRAM | Best variant | KV headroom | Verdict |
|---:|---|---:|---|
| 192 GB | NVFP4 v2 98.2 GiB | ~90 GB | Experiment A; best concurrency |
| 96 GB | Q4_K_S 87.1 GiB / IQ4_XS 82.0 GiB | ~9 / ~14 GB | Experiment B; moderate concurrency |
| 80 GB | IQ3_M 67.2 GiB | ~13 GB | Fits, lower quality |
| 48 GB | IQ3_M with `--n-cpu-moe` | — | Hybrid, single-user territory |
| 2×24 GB | IQ2/IQ3 with CPU offload | — | Not recommended for multi-user |

### Non-Blackwell nodes (2 or 4 GPUs) — what to expect

These are **estimates**, not published REAP50 measurements. They assume
Q4_K_S/IQ4_XS, 2–8K context, llama.cpp layer split, and no prefix-cache
contamination. The helper exists to replace them with real numbers.

| Node | GPUs / total VRAM | Best checkpoint | Single stream | Aggregate C8–32 | GPU power | Role |
|---|---|---|---:|---:|---:|---|
| `uk-g6-05..08` | 2× L40S/L40 48 GB = **96 GB** | **IQ4_XS 82.0 GiB** (safer) or Q4_K_S 87.1 | ~20–50 tok/s | ~60–150 tok/s | ~700 W | **Gentle primary** |
| `ampere02` | 2× A100 80 GB = **160 GB** | Q4_K_S 87.1 GiB | ~40–90 | ~100–250 | ~800 W | Middle ground |
| `hopper01` | 2× H100 80 GB = **160 GB** | Q4_K_S 87.1 GiB | ~60–120 | ~150–350 | ~1.4 kW | Fastest 2-GPU, not gentle |
| `fuuk-13` | 4× L40 48 GB = **192 GB** | Q4_K_S 87.1 GiB | ~30–70 | ~100–250 | ~1.2 kW | Most VRAM; 2× TP2 replicas often beat one 4-way split |
| `fuuk-13` + FP8 | 4× L40 48 GB = 192 GB | FP8 v2 160.6 GiB | ~10–30 (KTransformers) | ~30–100 | ~1.2 kW | Highest quality; SM89 + CPU-expert caveat |
| `volta05` | 4× V100 32 GB = **128 GB** | Q4_K_S 87.1 GiB | ~10–25 | ~30–80 | ~1.2 kW | Legacy only |

Why the differences:

- **2× 48 GB** is the gentle rental: 96 GB holds IQ4_XS with ~14 GB left, or
  Q4_K_S with ~9 GB. That is enough KV for 16–32 slots at 4–8K context because
  GLM-5.3’s MLA/KDA cache is only ~11 KB/token. The limit is weight bandwidth
  and llama.cpp’s lack of paged KV, not KV capacity.
- **2× 80 GB** gives 160 GB, so Q4_K_S fits with ~73 GB KV. H100 has
  ~3.35 TB/s per card and A100 ~2.0 TB/s, versus L40 ~864 GB/s, which is why
  they are faster per token. H100/A100 are not “gentle” on power or price.
- **4× L40** gives 192 GB: Q4_K_S fits with ~105 GB KV, and FP8 v2 (160.6 GiB)
  technically fits with ~30 GB. FP8 is only practical through KTransformers on
  SM89; vLLM/SGLang sparse-MLA on L40 is not validated. For throughput, two
  independent 2× L40S servers usually beat one 4-way layer-split server because
  there is no 4-way PCIe synchronization.
- **4× V100** fits Q4_K_S in 128 GB, but there is no BF16/FP8/FP4 and the
  kernels are older; use it only because it is already there.
- **2× V100 16 GB (32 GB total)** cannot hold Q4_K_S; only IQ2/offload, so it
  is not a useful multi-user experiment.
- **FP8 v2 on 2× 80 GB** does not fit (160.6 GiB weights + no KV), and
  **NVFP4 on Hopper/Ampere** is not native (Marlin/emulation, slower and not
  validated for this checkpoint). Use GGUF on these nodes.

Preflight order:

```bash
# 1) cheapest real datapoint: 2x L40S, IQ4_XS, C1/2/4/8
./submit.sh --node uk-g6-05 --partition gpu-ffa --account ffa \
  --gres gpu:2 --cpus 32 --mem 128G --time 04:00:00 \
  --engine llamacpp --variant iq4_xs --quick

# 2) if that loads and single-stream is usable, moderate sweep
./submit.sh --node uk-g6-05 --partition gpu-ffa --account ffa \
  --gres gpu:2 --cpus 32 --mem 128G --time 04:00:00 \
  --engine llamacpp --variant iq4_xs \
  --concurrency "1 2 4 8 16 32" --runs 2

# 3) 4x L40 for more KV headroom and a 4-way comparison
./submit.sh --node fuuk-13 --partition gpu-ffa --account ffa \
  --gres gpu:4 --cpus 32 --mem 256G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s \
  --concurrency "1 2 4 8 16 32" --runs 2
```

If `--quick` gives less than ~10 tok/s single-stream, the full sweep is not
worth the allocation; check the build, the CUDA arch, and whether the model is
actually fully offloaded (`-ngl 99`) before spending a longer job.

## Context budget

GLM-5.3-Flash is unusually cheap in KV because 34 of its 45 layers are KDA
linear attention (fixed recurrent state, not per-token KV) and only the 11 DSA
layers keep per-token MLA latents: 512 values per token per layer. Llama.cpp's
GGUF path therefore uses roughly:

| KV dtype | Per token | 8K | 32K | 128K | 1M |
|---|---:|---:|---:|---:|---:|
| BF16 | ~11 KB | 90 MB | 0.36 GB | 1.4 GB | 11 GB |
| q8_0 | ~5.6 KB | 45 MB | 0.18 GB | 0.72 GB | 5.6 GB |
| q4_0 | ~2.8 KB | 23 MB | 90 MB | 0.36 GB | 2.8 GB |

**Context is not what needs big VRAM here; the weights are.** After loading:

| Node | Weights | Free VRAM |
|---|---|---:|---:|
| 2×48 GB | Q4_K_S 87.1 / IQ4_XS 82.0 | ~9 / ~14 GB |
| 2×80 GB | Q4_K_S 87.1 | ~73 GB |
| 4×48 GB | Q4_K_S 87.1 | ~105 GB |
| 4×32 GB V100 | Q4_K_S 87.1 | ~41 GB |

Memory-wise, **1M tokens of q8_0 KV is only ~5.6 GB**, but the context also
needs attention/compute buffers. Measured on 2×48 GB: 256K q8_0 fits, 512K
q8_0 fails `cudaMalloc` during compute-buffer allocation, and 1M only starts
with q4_0 KV plus a 512/128 batch. The 82 GiB IQ4_XS weights leave ~10 GB of
headroom, so weights and buffers, not the KV arithmetic, decide the ceiling.
The real limits are compute and engine support:

- **llama.cpp GGUF fork**: DSA is gated off, so the 11 sparse-attention layers
  run dense O(n²). **8K** is the prune's assurance, **32K** is usable but slow
  (the model card measured 236 s at 32K on CPU), and **128K+** fits memory but
  is not practical for interactive/multi-user serving. 1M is not practical.
- **4×L40 + FP8 v2 + a DSA-aware engine** (KTransformers, or SGLang/vLLM if
  SM89 works): 128K–1M is memory-feasible, but prefill is still expensive and
  KTransformers runs selected experts on the CPU.
- **4×V100**: the same KV math fits, but the older compute makes 8–32K the
  practical range.

In llama.cpp `-c` is the **total** context shared by `-np` slots. The default
`CTX_TOTAL=262144` with `-np 96` gives ~2.7K per slot. For one long sequence,
set `MAX_RUNNING=1 CTX_TOTAL=131072`; for 32 slots × 8K, use
`CTX_TOTAL=262144`.

```bash
# one 32K-token request on 4x L40, Q4_K_S
MAX_RUNNING=1 CTX_TOTAL=131072 \
./submit.sh --node fuuk-13 --partition gpu-ffa --account ffa \
  --gres gpu:4 --cpus 32 --mem 256G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s \
  --concurrency 1 --runs 1 --isl 32768 --osl 128 \
  --min-prompts 1 --max-prompts 1
```

### Measured on 2× L40S 48 GB (uk-g6-05, IQ4_XS)

First run, 1 h `gpu-ffa` allocation, 32 CPUs, 256 GiB RAM, `q8_0` KV except
where noted. These are real numbers from `bench.py`; the tokenizer produced
~3–5% more tokens than the word budget, and the 64K point below is the failed
C=2 attempt (C=1 is the right shape for 64K on 96 GB).

| Context | C | Prefill tok/s | Decode tok/s | TTFT | TPOT | Status |
|---:|---:|---:|---:|---:|---:|---|
| 4K | 8 | **1,409** | **60.7** | 20.7 s | 70.0 ms | ok |
| 16K | 4 | **1,614** | **23.8** | 36.9 s | 58.7 ms | ok |
| 64K | 2 | — | — | — | — | failed: compute-buffer OOM/segfault at 45 GB/card; use C=1 |
| 256K | 1 | **518** | **22.5** | 476 s (7.9 min) | 44.8 ms | ok |
| 512K | 1 | — | — | — | — | q8_0 `cudaMalloc` OOM; needs q4_0 + batch 512/128 |
| 1M | 1 | ~378 (30% in 13 min) | — | — | — | server allocates with q4_0 + batch 512/128; prefill needs ~43 min, did not finish in the 22.6 min point timeout |

What this says:

- **4K/16K are fast**: prefill ~1.4–1.6K tok/s and decode 24–61 tok/s
  aggregate. Batching helps decode (C=8 gives 60.7 tok/s; C=4 at 16K gives
  23.8).
- **256K is prefill-bound**: 7.9 min TTFT at 518 tok/s. Decode is still
  22.5 tok/s, but end-to-end output is 0.3 tok/s because prefill dominates.
- **512K/1M are not 1-hour points** with this checkpoint/engine. 512K fails
  q8 compute-buffer allocation; 1M needs q4 KV and still ~43 min of prefill.
  The server fitting at 1M does not mean the context is usable interactively.
- For 64K/512K/1M, run one point per 1-hour job (or use 4×48 GB) and use
  `--timeout-mult 3`; do not mix them with 4K/16K in the same hour.

### Long-context batched sweep (1h short queue)

`context-sweep.sh` measures **prefill tok/s and decode tok/s separately** at
several context sizes. It restarts the server for each point because llama.cpp
shares one `-c` across all `-np` slots, so the slot count has to change with the
context. Default points:

```text
4096:8  16384:4  65536:2  262144:1  524288:1  1048576:1
  C=8      C=4       C=2       C=1        C=1
```

It runs one request per slot, uses `--warmup 1` for short points and no warmup
for 256K/1M, keeps earlier results if a later point times out, and writes
`context-results.json` / `.csv` with `status`, `ttft_s`,
`prefill_tokens_per_s`, `decode_tokens_per_s`, and end-to-end tok/s.

For the gentle 96 GB experiment on the **1-hour short queue**, use two of
`ampere01`'s three L40s:

```bash
./submit.sh --node ampere01 --partition gpu-short --account ffa \
  --gpus 2 --cpus 32 --mem 256G --time 01:00:00 \
  --engine llamacpp --variant iq4_xs --no-install \
  --context-sweep \
  --contexts "4096:8 16384:4 65536:2 262144:1 524288:1 1048576:1" \
  --sweep-osl 128 --budget-min 55
```

`--no-install` assumes `install.sh --engine llamacpp --variant iq4_xs
--cuda-arch 89` has already populated the GGUF and built `llama-server`.
Without that, the first part of the hour is spent downloading/building.

Expect 4K/16K/64K to finish, 256K to be slow, and **1M to time out** on
llama.cpp's dense DSA fallback; a timed-out 1M point still tells you whether the
server could allocate the 1M KV (health succeeded) or not (server_failed). For
usable 128K+ context you need the FP8 v2 checkpoint plus a DSA-aware engine, not
the GGUF fork.

## Experiments

- [2026-10-21 — REAP50 context sweep on 2× L40S 48 GB](experiments/2026-10-21-2x48gb-context-sweep.md) — measured 4K/16K/256K prefill and decode, plus the 64K/512K/1M limits. Raw table: [CSV](experiments/data/2026-10-21-2x48gb-context-results.csv) / [JSON](experiments/data/2026-10-21-2x48gb-context-results.json).
- [2026-10-21 — Faster prefill options](experiments/2026-10-21-faster-prefill-options.md) — why upstream llama.cpp DSA is the only SM89 prefill lever, and the 4×L40 FP8 / 2×H100 alternatives.

## Why this stack

There is **no published multi-user benchmark for the REAP50 checkpoint itself**.
The choice is based on full-model GLM-5.3-Flash evidence and on what fits:

| Evidence | Result |
|---|---|
| 4× B200, SGLang, full NVFP4, TP4/EP4 | 2,420 tok/s at C24; **6,030 tok/s at C96** |
| 2× GB300, SGLang, full NVFP4, TP2 | 1,734 tok/s at C32; **2,100 tok/s at C64** |
| 4× RTX PRO 6000, SGLang, full NVFP4 | 2,277 tok/s at C17–32 |
| 8× H100, vLLM, full FP8, FA3 | 754 tok/s at C8 (default FlashInfer: 230 tok/s) |

SGLang has the strongest published GLM-5.3-Flash numbers and the mature
RadixAttention/prefix-cache path; vLLM is second and has a known SM120 sparse-MLA
blocker for this model. REAP50 halves the weights but still activates 8 experts
per token, so the gain is concurrency/VRAM, not a faster token.

## Checkpoints

All are MIT, from `patrickbdevaney` (the v2 mask; v3/HOPE is not evaluated end
to end and is excluded by `install.sh`).

| `--variant` | Repo / file | Size | Format | Quality |
|---|---|---|---|---|
| `nvfp4` | `GLM-5.3-Flash-REAP50-NVFP4-v2` (root) | 98.2 GiB | compressed-tensors NVFP4, W4A16 | 0.8386 top-1 agreement |
| `fp8` | `GLM-5.3-Flash-REAP50-FP8-v2` (root) | 160.6 GiB | FP8 e4m3, 128×128 | **0.8425 top-1** |
| `q4_k_s` | `GLM-5.3-Flash-REAP50-Q4_K_S.gguf` | 87.1 GiB | GGUF | PPL 4.5833 |
| `q4_k_m` | `GLM-5.3-Flash-REAP50-Q4_K_M.gguf` | 92.5 GiB | GGUF | PPL 4.5959 |
| `iq4_xs` | `GLM-5.3-Flash-REAP50-IQ4_XS.gguf` | 82.0 GiB | GGUF | PPL 4.6002 |
| `iq3_m` | `GLM-5.3-Flash-REAP50-IQ3_M.gguf` | 67.2 GiB | GGUF | PPL 4.6253 |

The GGUF PPL differences are within the noise of the published 8×512-token
sample. The FP8 parent PPL is ~4.449, so the 4-bit GGUFs are roughly +3% PPL on
top of the prune. There is **no published generative/top-1 benchmark for any
GGUF** — treat that as the quality unknown.

## Quick start

On the cluster, from this directory:

```bash
# 1) see what is available
./install.sh --list

# 2) Experiment A: 2x 96 GB Blackwell, SGLang, NVFP4
./submit.sh \
  --node bw01 --partition gpu-long --account ffa \
  --gres gpu:2 --cpus 32 --mem 256G --time 12:00:00 \
  --engine sglang --variant nvfp4

# 3) Experiment B: 2x 48 GB L40S, llama.cpp, Q4_K_S (or IQ4_XS)
./submit.sh \
  --node <l40s-node> --partition <partition> --account <account> \
  --gres gpu:2 --cpus 32 --mem 256G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s

# 4) 4x V100 32 GB, GGUF Q4_K_S, llama.cpp
./submit.sh \
  --node volta05 --partition gpu-ffa-gpulab --account ffa \
  --gres gpu:v100:4 --cpus 16 --mem 128G --time 12:00:00 \
  --engine llamacpp --variant q4_k_s
```

`submit.sh` prints the result directory and the `squeue`/`tail` commands. The
job installs (unless `--no-install`), starts the server, waits for `/health`,
runs the sweep, stops the server and writes `results.json` / `results.csv`.

Run pieces by hand inside an allocation:

```bash
./install.sh --engine sglang --variant nvfp4
./serve.sh   --engine sglang --variant nvfp4 --background --wait
./bench.sh   --base-url http://127.0.0.1:30000 --concurrency "1 2 4 8 16 32 64"
./serve.sh   --stop
```

## What gets installed

| Engine | Install | Why |
|---|---|---|
| `sglang` | Apptainer image `lmsysorg/sglang:glm-5.3-flash`, or `uv` venv with `sglang[all]>=0.5.20` | Highest published GLM-5.3-Flash multi-user throughput; RadixAttention prefix cache, chunked prefill, CUDA graphs, NVFP4 loading |
| `vllm` | `vllm/vllm-openai:glm53-flash`, or `uv` venv with `vllm>=0.30` | Alternative; **SM120 (RTX PRO 6000) currently needs a patched image** for the NoPE sparse-MLA `pe_dim==64` assert |
| `llamacpp` | patched fork `patrickbdevaney/llama.cpp`, tag `glm5-next-reap50-gguf-v1`, built with CUDA | The published GGUFs were built/validated against this fork; upstream b11273+ can load them with KV-key overrides but is not the zero-friction path |

Everything is idempotent. `--skip-model`, `--skip-runtime`, `--force`,
`--method container|venv`, `--cuda-arch 70|89|120` are available.

## Recommended server shape

SGLang on 2× 96 GB Blackwell (`bw01`-class), NVFP4:

```bash
python3 -m sglang.launch_server \
  --model-path .../nvfp4 --tp 2 \
  --served-model-name glm53-reap50 \
  --mem-fraction-static 0.80 \
  --max-running-requests 96 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --cuda-graph-max-bs-decode 128 \
  --schedule-policy lpm \
  --kv-cache-dtype fp8_e4m3 \
  --enable-metrics --enable-cache-report
```

- `--schedule-policy lpm` only pays when requests share prefixes (agent system
  prompt / RAG); use `fcfs` otherwise.
- `--mem-fraction-static` 0.80 leaves room for the 98 GiB weights on 192 GB;
  raise in 0.01 steps until startup leaves ~5–8 GB free.
- NVFP4 is **W4A16**: weights are 4-bit, activations 16-bit. It saves weight
  bandwidth and VRAM, but it does **not** use FP4 tensor cores for compute. A
  W4A4 re-quant is the one experiment that could raise the compute ceiling.
- MTP is present in the v2 checkpoints. Speculative decoding may help at high
  batch; test it, do not assume it (llama.cpp MTP was measured slower at small
  batch).

llama.cpp on 4× V100 32 GB, Q4_K_S (87.1 GiB):

```bash
llama-server -m .../GLM-5.3-Flash-REAP50-Q4_K_S.gguf \
  -ngl 99 --split-mode layer \
  -np 96 -c 262144 \
  -b 4096 -ub 1024 -fa on -ctk q8_0 -ctv q8_0 \
  --cache-prompt --cache-reuse 256 --metrics \
  --override-kv tokenizer.ggml.eot_token_id=int:154827 \
  --override-kv tokenizer.ggml.eom_token_id=int:154829
```

`-c` is the **total** context shared by `-np` slots. The defaults in
`config.env` are 96 slots × ~2.7k tokens; raise `CTX_TOTAL` for longer slots.
The two stop-token overrides are mandatory: the GGUF header carries only
`eos=154820`, and without them the model emits `<|user|>` and keeps talking.

## Benchmark protocol

`bench.py` implements a closed-loop sweep (`request-rate=inf`, `max-concurrency
C`), which is the right shape for a capacity number. For each point:

1. flush the prefix cache (cold regime) or reuse a shared prefix (warm-shared);
2. warm up `max(2*C, 32)` requests, capped at 256;
3. flush again for a cold point;
4. send `max(200, 5*C)` requests with a per-run `cache_salt`;
5. repeat `--runs 3` and report the median.

```bash
./bench.sh \
  --base-url http://127.0.0.1:30000 \
  --concurrency "1 2 4 8 16 32 64 96 128" \
  --isl 2048 --osl 256 --runs 3
```

Outputs: `results.json` (full schema), `results.csv` (one row per point/run).

Headline metrics:

- **output tok/s** — aggregate decode tokens / wall time
- **total tok/s** — input + output; do not compare across different ISL
- **TTFT** / **TPOT** p50/p99
- **goodput** — requests meeting `TTFT ≤ 2 s` and `TPOT ≤ 80 ms`
- **prefix hit rate** — from `/metrics` when the server exposes it

For >3k tok/s aggregate, use `--workers 2..8`; a single CPython process can
become the client bottleneck. For >64 concurrency, AIPerf or several bench
processes are the safer client.

### Expected ballpark

There is **no published multi-user benchmark for REAP50**. The following are
estimates informed by full-model and adjacent measurements; measure, do not
promise:

| Hardware | Experiment / config | Single stream | Aggregate (C16–64) |
|---|---|---:|---:|
| 2× 96 GB Blackwell | **A** SGLang + NVFP4, TP2 | ~150–200 tok/s | ~700–1,200 tok/s |
| 1× 96 GB Blackwell | **B** llama.cpp + Q4_K_S, one card | ~20–60 | ~60–150 (C8–16) |
| 2× 48 GB L40S | **B** llama.cpp + Q4_K_S/IQ4_XS, layer split | ~10–30 | ~30–120 (C8–16) |
| 2× H100 80 GB | vLLM + W4A16, TP2 | ~60–100 | ~250–450 |
| 4× L40 48 GB | llama.cpp Q4 / KTransformers FP8 | ~10–30 | ~30–120 |
| 4× V100 32 GB | llama.cpp Q4_K_S, layer split | ~5–15 | ~20–50 |

The prune halves the **weight footprint**, not the active compute: it still
routes to 8 experts per token. The throughput win comes from fitting in fewer
GPUs and leaving more room for KV/concurrency, not from a 2× faster token.

## Caveats

- **vLLM on SM120**: the NoPE sparse-MLA path asserts `pe_dim == 64`; the
  REAP50 config has `qk_rope_head_dim = 0`. Use SGLang, or a patched vLLM
  image. vLLM is the second choice here.
- **FP8 vs NVFP4**: FP8 is +0.4 pp top-1, but 160.6 GiB leaves only ~20 GB for
  KV/activations on 2× 96 GB (much less concurrency than NVFP4) and does not fit
  2× 80 GB. Use it on 4× H100/H200 or a 192 GB node at shorter context.
- **llama.cpp DSA**: the fork loads the indexer tensors but gates DSA off; the
  11 sparse-attention layers run dense, so long context is compute-bound. The
  published long-context assurance is 8,192 tokens, not the advertised 1M.
- **V100**: no BF16, no FP8, no FP4. Use GGUF only; CUDA 13 dropped `sm_70`, so
  build with `/usr/local/cuda-12.5` and `--cuda-arch 70`, or use Vulkan.
- **v3/HOPE** weights are not evaluated; `install.sh` excludes `pass3-hope/`.
  Do not benchmark v3 and call it v2.
- **NFS**: keep the checkpoint on local scratch when the node has it. On
  `bw01`/`uk-g6` the first load streams from `/work`; 760 GB RAM then holds the
  page cache.
- **Stop tokens**: always pass the two `--override-kv` flags for the GGUFs.

## Layout

| File | What |
|---|---|
| `install.sh` | image/venv, llama.cpp build, checkpoint download, `install-state.json` |
| `serve.sh` | SGLang / vLLM / llama.cpp launch, health wait, pid/log, `--stop` |
| `bench.py` | stdlib-only batched benchmark client (`--self-test` for CI) |
| `bench.sh` | config defaults around `bench.py` |
| `context-sweep.sh` | long-context sweep: restart server per context, prefill/decode tok/s, per-point timeout |
| `experiments/` | dated measurement write-ups and the raw CSV/JSON result tables |
| `run-job.sh` | install → serve → wait → bench → stop, writes `job.json` |
| `glm53-reap50.sbatch` | thin Slurm entry point |
| `submit.sh` | fills partition/account/resources and submits |
| `config.env` | every default, overrideable |

## Sources

- GGUF + patched llama.cpp: <https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF>
- FP8 v2: <https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2>
- NVFP4 v2: <https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2>
- SGLang GLM-5.3-Flash cookbook: <https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash>
- SGLang server args / tuning: <https://docs.sglang.io/docs/advanced_features/server_arguments>
- vLLM GLM-5.3 support / benchmarks: <https://github.com/vllm-project/vllm/issues/56564>
- 4× B200 / 4× RTX PRO 6000 SGLang results: <https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200>
- DGX Station / GB300 benchmarks: <https://github.com/catid/dgx_station_benchmarks/tree/main/glm-5.3-flash>
- Benchmark methodology: <https://docs.vllm.ai/en/stable/cli/bench/serve/>, <https://docs.nvidia.com/aiperf/>
