# 2026-10-24 — Qwen3.8-Flash-Next 4-bit on 2× BW6000 96 GB

The Blackwell FP4-local checkpoint now runs in the required community vLLM
container, and the full context/batch/cache matrix completed.

## Setup

```text
model:  nvidia/Qwen3.8-Flash-Next-FP4-Local (~96 GiB)
GPUs:   2× NVIDIA BW6000 96 GB (192 GB total, SM120)
image:  ghcr.io/local-inference-lab/vllm:release-10152026
TP:     2
KV:     bfloat16 (the config overrode the container default FP8)
MTP:    3 speculative tokens
max_model_len: 262144
max_num_seqs: 32
max_num_batched_tokens: 8192
```

The image was built on a node-local tmpfs (`/tmp` 378 GB) and only the
finished SIF was copied to `/work`; writing the extraction directly on Lustre
was the earlier failure mode.

## Capacity

```text
Available KV cache memory: 41.78 GiB per GPU
GPU KV cache size: 2,642,309 tokens
Max concurrency at 262,144 tokens/request: 10.08×
```

## Startup

- 95.1 GiB weights loaded by InstantTensor at ~140 MB/s.
- Weight loading: **731 s**.
- First B12X JIT/autotune plus CUDA-graph capture made total startup about
  **43 min**.
- The server then passed the 10-check HTTP smoke matrix, including cached-token
  accounting (`cache_hit` now passes, unlike the older fork).

## Batched throughput

32 output tokens per request; cache cold.

| Context | C | Output tok/s | Total tok/s | TTFT p50 | TTFT p99 | TPOT p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1 | 68.2 | 1,182 | 0.321 s | 0.331 s | 4.50 ms |
| 500 | 2 | 73.8 | 1,283 | 0.383 s | 0.683 s | 15.9 ms |
| 500 | 4 | 90.8 | 1,579 | 0.681 s | 1.055 s | 27.7 ms |
| 500 | 8 | 121.0 | 2,098 | 0.707 s | 1.047 s | 45.7 ms |
| **500** | **16** | **208.1** | **3,605** | 0.829 s | 1.293 s | 49.1 ms |
| 500 | 32 | 161.7 | 2,804 | 1.601 s | 1.931 s | 137.9 ms |
| 800 | 1 | 69.2 | 1,880 | 0.321 s | 0.347 s | 4.48 ms |
| 800 | 2 | 75.7 | 2,059 | 0.376 s | 0.672 s | 15.6 ms |
| 800 | 4 | 97.3 | 2,642 | 0.694 s | 1.012 s | 17.0 ms |
| 800 | 8 | 148.6 | 4,035 | 0.718 s | 1.039 s | 29.5 ms |
| **800** | **16** | **281.9** | **7,655** | 1.099 s | 1.175 s | 20.9 ms |
| 800 | 32 | 157.9 | 4,288 | 1.709 s | 2.385 s | 140.4 ms |

C=16 is the peak in this single-run matrix; C=32 lowers per-stream and
aggregate output throughput. This is a batching knee, not a capacity limit.

## Context sweep

Prompt = 90% of context; 128 output tokens; cold cache.

| Context | C | Output tok/s | Prefill tok/s | TTFT p50 | TTFT p99 | TPOT p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 8 | **353.6** | 14,802 | **1.71 s** | 2.08 s | 8.4 ms |
| 16K | 4 | 124.7 | 17,740 | 2.66 s | 3.47 s | 10.5 ms |
| 64K | 2 | 34.0 | 17,348 | 5.63 s | 7.09 s | 14.7 ms |
| **256K** | **1** | **7.9** | **15,469** | **15.95 s** | 15.95 s | 2.37 ms |

The 256K point is a 235,929-token prompt. Prefill is remarkably flat at
~15–18K tok/s across the ladder; the TTFT growth is mostly prompt length, not
a collapse in prefill rate.

## Prefix cache

One sequence, cumulative prefixes.

| Prefix | Fill TTFT | Hit TTFT | Repeat TTFT | Hit reuse | Decode | Cache hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| 50K | 3.06 s | **0.402 s** | 0.075 s | 7.6× | 344 tok/s | 66.6% |
| 100K | 3.22 s | **0.468 s** | 0.123 s | 6.9× | 352 tok/s | 83.3% |
| 200K | 7.38 s | **0.612 s** | 0.200 s | 12.1× | 397 tok/s | 83.3% |

The 200K cumulative fill is only 13.7 s because of prefix reuse between the
ladder points. A cached 200K prefix returns in ~0.6 s, and the active repeat
is ~0.2 s.

## Interpretation

- Qwen3.8-Flash-Next is genuinely usable on **2× BW6000 96 GB** at 4-bit:
  ~2.64M KV tokens and 10× concurrency at 262K.
- It is much faster than the 2× A100 Qwen3.5-122B for short-context prefill:
  ~15K tok/s vs ~3.9K tok/s, and 353 vs 61 output tok/s at 4K/C8.
- The 256K prefill is ~16 s, far better than the A100 GLM 256K path; this is
  the QSA/GDN + Blackwell combination.
- **C=16 is the sweet spot** for 500–800-token batches; C=32 hurts.
- This is native **262K**, not 700K. 1M requires YaRN and is not validated;
  quality degradation beyond ~390K has been reported.

## Caveats

- One run per point, cold cache for batch/context; not a statistical sweep.
- MTP speculative decoding was enabled (3 tokens), so these are optimized
  decode numbers, not a plain decode baseline.
- KV is BF16, not FP8; FP8 could roughly double capacity but was not measured.
- The first-start JIT/autotune is expensive (~43 min total). A warm/reused
  cache will be much faster.
- No OpenCode agent suite was run on this model yet; the HTTP smoke passed.

Raw files: [`data/2026-10-24-qwen38-bw`](data/2026-10-24-qwen38-bw).

## Reproduce

```bash
llm/qwen38/serve.sh up
llm/qwen38/serve.sh wait
BASE_URL=http://bw01:8010 \
OUT=llm/qwen38/experiments/data/2026-10-24-qwen38-bw \
  llm/qwen38/measure.sh
```
