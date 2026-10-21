# 2026-10-21 — REAP50 context sweep on 2× L40S 48 GB

First measurement of the gentle experiment: the 50%-expert-pruned
GLM-5.3-Flash (REAP50) on **two 48 GB L40S cards (96 GB total)**, with
llama.cpp and IQ4_XS. The point was to find the batched prefill/decode
throughput at increasing context sizes, up to 1M, in one-hour `gpu-ffa`
allocations.

## Setup

| | |
|---|---|
| Node | `uk-g6-05` (2× NVIDIA L40S 48 GB, 96 GB total) |
| CPUs / RAM | 32 CPUs, 256 GiB |
| Slurm | `gpu-ffa`, 1 h, 2 GPUs |
| Checkpoint | `GLM-5.3-Flash-REAP50-IQ4_XS.gguf` — 88,048,752,032 B (82.0 GiB), PPL 4.6002 |
| Engine | patched llama.cpp fork `glm5-next-reap50-gguf-v1` (`2a4a412`), CUDA 12.5, `CMAKE_CUDA_ARCHITECTURES=89` |
| Server | `-ngl 99 --split-mode layer --no-mmap -fa on -ctk q8_0 -ctv q8_0 --no-cache-prompt` |
| Benchmark | `bench.py` closed-loop, one request per slot, OSL 128, cold cache, 90% word budget |

The model was read from `/work` with `--no-mmap`; mmap page-faults on this NFS
mount were ~10 MB/s, while sequential `read()` was fast enough to load 82 GiB
in roughly 8–20 min depending on NFS contention.

## Results

`C` is the number of concurrent requests / server slots. Prefill and decode
are separate aggregates: prefill = input tokens / longest TTFT; decode =
output tokens / longest decode window. End-to-end output rate includes the
prefill wait and is therefore much lower at long context.

| Context | C | Prefill tok/s | Decode tok/s | End-to-end output tok/s | TTFT p50 | TPOT p50 | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4K | 8 | **1,409** | **60.7** | 34.5 | 20.7 s | 70.0 ms | ok |
| 16K | 4 | **1,614** | **23.8** | 11.5 | 36.9 s | 58.7 ms | ok |
| 64K | 2 | — | — | — | — | — | failed: compute-buffer OOM/segfault at ~45 GB/card; use C=1 |
| 256K | 1 | **518** | **22.5** | 0.3 | **476 s (7.9 min)** | 44.8 ms | ok |
| 512K | 1 | — | — | — | — | — | q8_0 `cudaMalloc` OOM during compute-buffer allocation; q4_0 + batch 512/128 is the configured retry |
| 1M | 1 | ~378 (30% in 13 min) | — | — | — | — | server allocates with q4_0 KV + batch 512/128, but prefill needs ~43 min; timed out at 22.6 min |

Raw run: `/work/ffa/holecem/runs/glm53-reap50/20261021T084616Z/`
(the 1M server-fit/timeout observation is in `ctx-1048576-c1/server.log`;
the 512K q8 OOM is in `ctx-524288-c1/server.log`).

A compact copy of the table is in
[`data/2026-10-21-2x48gb-context-results.csv`](data/2026-10-21-2x48gb-context-results.csv)
and [`.json`](data/2026-10-21-2x48gb-context-results.json).

## Interpretation

- **4K/16K are fast.** Prefill is 1.4–1.6K tok/s; 8-way 4K gives 60.7 tok/s
  aggregate decode, 4-way 16K gives 23.8.
- **256K is prefill-bound.** A 246.7K-token prompt takes 7.9 min to prefill
  (518 tok/s). Decode is still 22.5 tok/s, but end-to-end output is only
  0.3 tok/s because every request pays the prefill.
- **512K and 1M are not one-hour points** on this node/checkpoint/engine.
  - 512K q8_0 needs more compute-buffer memory than 96 GB provides. The retry
    uses q4_0 KV and a 512/128 batch.
  - 1M **does allocate** with q4_0 KV + 512/128 batch, but prefill is
    ~378 tok/s, i.e. ~43 min for ~988K tokens, before decode. It did not
    complete within the 22.6 min per-point timeout.
- **Context is not the KV-memory problem.** 1M q8_0 KV is only ~5.6 GB. The
  wall is the dense DSA fallback in this llama.cpp fork plus the attention /
  compute buffers. The same context is much cheaper in an engine that
  implements DSA (see below).
- **64K C=2 is too tight.** The server allocated ~45 GB on each card and then
  crashed on the request. C=1 is the right shape; a clean cold 64K C=1 point
  was not obtained in this run.

## Cache-contamination fix

The first 64K attempt reported a fake **490,781 tok/s prefill** with a 126 ms
TTFT. The warmup request used the same full prompt as the measurement, so
llama.cpp's prompt cache served the measured request. The helper now:

- warms up with a short, distinct `"warmup"` prompt, and
- starts context-sweep servers with `--no-cache-prompt`.

That 64K row is not a valid measurement and is omitted.

## Reproduction

```bash
cd /work/ffa/holecem/slurmhelps-fleet/llm/glm53-reap50

# 4K, 16K, 256K in one hour; 64K C=1 and 512K q4 are separate points
LLAMA_CUDA_HOME=/usr/local/cuda-12.5 ./submit.sh \
  --node uk-g6-05 --partition gpu-ffa --account ffa \
  --gpus 2 --cpus 32 --mem 256G --time 01:00:00 \
  --engine llamacpp --variant iq4_xs --no-install \
  --context-sweep \
  --contexts "4096:8 16384:4 262144:1" \
  --sweep-osl 128 --budget-min 55
```

For the 512K/1M retries, run one point per hour with `--timeout-mult 3` and
the helper’s q4_0 + 512/128 batch path; do not mix them with short-context
points.

## What to try next for faster prefill

See the dedicated research note:
[`2026-10-21-faster-prefill-options.md`](2026-10-21-faster-prefill-options.md).
The short version is below.


The measurements above are with llama.cpp’s **dense DSA fallback**. The
largest expected wins, in order, are:

1. **A DSA-aware engine** (SGLang/vLLM with sparse-MLA kernels, or upstream
   llama.cpp once its DSA path is active). That changes prefill from O(n²)
   to O(n·2048) on the 11 sparse-attention layers.
2. **Four GPUs (4×48 GB = 192 GB)** for the same IQ4_XS checkpoint: more SMs
   for prefill and room for larger batches; expected to cut 512K/1M prefill
   roughly 2–4×, but it adds PCIe layer-split overhead.
3. **A smaller model/quant for 1M** (IQ3_M, 67.2 GiB) to free ~15 GB for
   compute buffers; this trades quality (PPL 4.6253 vs 4.6002) for context.
4. **A node with local scratch or warm page cache** to remove the 8–20 min
   NFS model load from the one-hour budget.
