# 2026-10-23 — paged-KV on 2× A100 160 GB (working)

The SM80 vLLM fork + REAP50 NVFP4 v2 path now runs on `ampere02`
(2× A100 80 GB, 160 GB total) and produces real paged-KV numbers.

## What was broken

Two independent issues, both now fixed:

1. **Checkpoint prefix.** The checkpoint serializes text weights as
   `model.language_model.*`; the fork's `Glm5NextForConditionalGeneration`
   expects `language_model.model.*`. The local
   `vllm-patch/glm4v.py` mapping handles that.
2. **Dense-MLP quantization mismatch.** The fork applied the NVFP4 quant
   config to all `Linear` layers, so layers 0–2 wanted
   `mlp.{gate_up,down}_proj.weight_packed`. Those three dense MLPs are plain
   BF16 in this checkpoint (the sparse MoE layers are the packed ones). The
   resulting `KeyError: 'layers.0.mlp.down_proj.weight'` was therefore not a
   prefix problem.

Fix: add a checkpoint-config ignore for the three dense layers, before
`lm_head`/`embed_tokens`:

```json
"re:.*layers\\.(0|1|2)\\.mlp\\..*"
```

With that, vLLM constructs plain BF16 linears for layers 0–2 and the loader
matches the checkpoint. The original config is saved as
`config.json.pre-dense-ignore`.

The earlier debug run also confirmed the loader's local parameter names
(`layers.0.mlp.down_proj.weight`), so no further key alias is needed.

## Configuration

- vLLM SM80 fork image `vllm-glm53-sm80.sif` (vLLM 0.4, `TRITON_MLA_SPARSE`).
- `GLM-5.3-Flash-REAP50-NVFP4-v2`, TP2, BF16 KV, block size 128.
- Prefix caching on, Marlin MoE, eager mode, 4,096 batched tokens,
  8 max sequences, `gpu_memory_utilization=0.94`.

## Capacity

```text
Available KV cache memory: 22.45 GiB per GPU
GPU KV cache size: 1,949,696 tokens
Maximum concurrency at 327,680 tokens/request: 5.95x
```

That is **1.95M KV tokens** on 2× A100, or roughly six full 327K sessions
resident at once. This is the paged-KV capacity result the llama.cpp
slot-cache path on 2× L40S could not provide.

## Guarded cache ladder (single sequence)

One sequence, cumulative prefix fills. No cold-miss request is sent; miss TTFT
is inferred as `cumulative_fill_s + hit_ttft_s`. Reuse is only measured for
this one sequence; shared-prefix multi-session reuse is **not measured**.

| Prefix P | Fill TTFT | Incremental prefill | Cumulative fill | Hit TTFT | Reuse ratio | Repeat TTFT | Decode | Inferred miss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100K | 28.49 s | 3,510 tok/s | 28.49 s | **1.475 s** | 19.31× | 0.285 s | 11.7 tok/s | 29.96 s |
| 200K | 31.77 s | 3,147 tok/s | 60.26 s | **0.599 s** | 53.04× | 1.010 s | 12.5 tok/s | 60.86 s |
| 300K | 35.76 s | 2,797 tok/s | 96.02 s | **0.961 s** | 37.23× | 1.050 s | 12.6 tok/s | 96.98 s |

The cache hit removes essentially all prefix prefill. The active-slot repeat
is 0.29–1.05 s. Decode is attention-bound and stays at 11.7–12.6 tok/s.

Raw: [`data/2026-10-23-paged-cache-2xa100.csv`](data/2026-10-23-paged-cache-2xa100.csv).

## Cold context / concurrency sweep

Prompt is 90% of context; 128 output tokens; one measured run after warmup.
`C` independent one-shot requests (no shared prefix).

| Context | C | Duration | Prefill | Decode | End-to-end total | TTFT p50 | TTFT p99 | TPOT p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 8 | 18.40 s | **3,969 tok/s** | **61.2 tok/s** | **1,731 tok/s** | 5.00 s | 7.74 s | 103 ms |
| 16K | 4 | 26.51 s | **3,922 tok/s** | 25.2 tok/s | 2,349 tok/s | 10.33 s | 15.65 s | 121 ms |
| 64K | 2 | 43.07 s | **3,818 tok/s** | 10.6 tok/s | 2,875 tok/s | 24.50 s | 32.21 s | 137 ms |
| 256K | 1 | 82.57 s | **3,420 tok/s** | 12.3 tok/s | 2,990 tok/s | **72.14 s** | 72.14 s | 82 ms |

The 256K point used a 235,929-token prompt and still had 22.45 GiB of KV
cache; there is enough capacity for a 300K+ prompt at C=1. The 4K/16K/64K
points are cold-cache and therefore pay full prefill; they are not cache-hit
numbers.

Raw: [`data/2026-10-23-paged-context-2xa100.csv`](data/2026-10-23-paged-context-2xa100.csv).

## Comparison with llama.cpp on 2× L40S

| Shape | 2× L40S (llama.cpp IQ4_XS) | 2× A100 (paged vLLM NVFP4) |
|---|---:|---:|
| 4K, C=8 TTFT p50 | 22.1 s | **5.0 s** |
| 16K, C=4 TTFT p50 | 40.4 s | **10.3 s** |
| 64K, C=1/2 TTFT p50 | 62.2 s | **24.5 s** (C=2) |
| 256K TTFT | 476.9 s | **72.1 s** |
| 300K cache-hit TTFT | 1.086 s | **0.961 s** |
| 300K cold fill | 426.9 s | **96.0 s** |

The paged engine is 4–7× faster on cold prefill (and 6.6× at 256K). Cache-hit
latency is similar; the difference is that A100 paged KV holds ~1.95M tokens
versus one 300K llama.cpp slot.

## Caveats / not measured

- **Single-sequence reuse only.** The ladder is one sequence filling
  cumulatively. Multi-session shared-prefix reuse, a branching GPU prefix
  tree, and per-agent resident capacity are not measured here.
- No separate H2D/RAM-tier attribution; this engine is GPU-paged only.
- 4× L40 is a separate note; the 2× H100 comparison is
  [here](2026-10-23-paged-kv-h100.md).
- The fork is vLLM 0.4, not stock vLLM; compare engine features carefully.
- SGLang was not tried on A100: v0.5.22 has no SM80/SM89 GLM-5.3 sparse-DSA
  path (only SM90+/SM100/SM120), so it cannot load this model on A100.
- The 2× A100 job still has 1.95M KV tokens capacity but only ~22.45 GiB/GPU;
  higher concurrency at 327K will evict rather than exceed that capacity.

## Hardware / queue notes

- `ampere02` had both A100 GPUs free; the job ran with two `beckd` CPU jobs
  on the node. No GPU preemption was needed this time.
- `hopper01` (2× H100) was free for the stock-path follow-up.
