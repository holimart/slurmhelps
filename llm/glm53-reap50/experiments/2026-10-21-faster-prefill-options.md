# Faster prefill options for REAP50 — research note (2026-10-21)

Scope: the expert-pruned GLM-5.3-Flash (REAP50) on **non-Blackwell** GPUs, with
the 2× L40S 48 GB (96 GB, SM89) result as the baseline. The baseline measured
**1.4–1.6K tok/s prefill at 4–16K**, **518 tok/s at 256K**, and did not finish
512K/1M in an hour.

## The short answer

There is **no production DSA/top-k sparse-MLA prefill kernel for SM89** in
upstream vLLM, SGLang, FlashMLA, FlashAttention-3, DeepGEMM, TensorRT-LLM or
TokenSpeed. They all start at SM90 and the fast paths are SM100+.

The only upstream SM89 path that accelerates prefill is **llama.cpp's native
DSA implementation**, merged in PR #27773 (build b11279 / v0.7.0). It runs on
any ggml backend including Ada. The catch: the REAP50 GGUFs were built by the
older fork and need four metadata overrides plus **294 tensor-name aliases**
(or a reconversion), not just `--override-kv`.

## Ranked options

| Target | Best option | Why | Expected prefill |
|---|---|---|---|
| **2× L40S 96 GB** (current) | **Upstream llama.cpp v0.7.0+ with native DSA**, IQ3_M/Q3_K_M | Only upstream SM89 sparse-prefill path; fits with headroom | ~1.2–1.5× at 32K, ~2–3× at 128K vs the dense fork; 1M still not realistic on 96 GB |
| **4× L40 192 GB** | **vLLM SM89 fork + FP8 REAP50** (`yhfgyyf/vllm-deepseek-v4-sm89`) | FP8 160.6 GiB fits with ~31 GiB left; the fork demonstrates GLM-5.3-Flash on 8×4090 48 GB with FlashInfer SM89 JIT | ~3–5K tok/s at 8K (inferred; no published 4×L40 number) |
| **4× L40 fallback** | llama.cpp Q4_K_S/IQ4_XS, fully resident | Most reliable; 192 GB leaves ~100 GB for context | Lower than FP8 vLLM, but no fork/JIT risk |
| **2× H100 160 GB** | **SGLang v0.5.22 TileLang DSA** or vLLM v0.32 `flashmla_sparse` | H100/H200 are the validated default; FP8 160.6 GiB does **not** fit, so use NVFP4-W4A16 or a 4-bit GGUF | ~3–5× the SM89 prefill; H200 `flashmla_sparse_q8` adds +5.8–12.1% prefill-only |
| **Any SM89 with 1M context** | KTransformers FP8 (SM89-blessed) | Fits 1M context by streaming CPU experts | **Not faster**: CPU-expert prefill was measured at ~3.6 tok/s on the full model; use only when context capacity matters more than speed |

## 1. llama.cpp upstream — the best SM89 prefill lever

- **Merge:** PR [#27773](https://github.com/ggml-org/llama.cpp/pull/27773),
  commit `649dcb1`, first build **b11279**, first stable **v0.7.0**.
- **DSA is on by default**, with `index_topk=2048` and `kpool=4`, in
  `src/models/glm5-next.cpp`. There is no `--dsa` flag.
- **`-fa on` is required**; without it the sparse mask is still applied but the
  full `n_kv × n_tokens` score matrix is materialised, so there is no compute
  saving.
- **The fork's `GLM5_DSA=1` is decode-only.** `build_attn_dsa()` returns
  `nullptr` for `T != 1`, so prefill stays dense; do not use it expecting a
  prefill speedup.
- **REAP50 compatibility:** the fork GGUF has `glm5-next.attention.hc.*` keys
  and tensor names without `.weight` / with `indexer.kpool_*`. Upstream needs:
  - `--override-kv glm5-next.hyper_connection.count=int:4`
  - `--override-kv glm5-next.hyper_connection.sinkhorn_iterations=int:20`
  - `--override-kv glm5-next.hyper_connection.epsilon=float:1e-06`
  - `--override-kv glm5-next.attention.indexer.kpool=int:4`
  - plus **294 tensor aliases** (`hc_*` → `hc_*.weight`, `indexer.kpool_gate` →
    `indexer_compressor_gate.weight`, etc.).
  A ~15-line loader alias in `get_tensor_meta()` is enough for a first
  measurement; for production, reconvert the REAP50 FP8-v2 parent with the
  upstream v0.7.0 converter and pin `indexer=q8_0`.
- **Measured anchors:** 2× RTX 5090 + CPU experts, UD-IQ3_XXS: **573 tok/s at
  32K, 610 tok/s at 190K**; 1× RTX PRO 6000 96 GB: **458 tok/s at 32K, 375 at
  128K**; MI50 sparse PR: +38%; M5 Ultra MLX: +24.5% at 16K, +29.5% at 64K.
- **Expected on 2× L40S:** ~1.2–1.5× at 32K and ~2–3× at 128K versus the dense
  fork, because the indexer still scores every query against every pool
  (`O(n²/4)`); 1M remains a memory/prefill problem. Treat 1M as a 4×48 GB or
  H100 job, not 2×48 GB.

## 2. vLLM SM89 fork — the 4×L40 option

- Upstream vLLM v0.30+ supports `Glm5NextForConditionalGeneration`, but its
  sparse-MLA backends are gated to SM90/SM100/SM120; SM89 has no sparse-MLA
  backend.
- Community fork [`yhfgyyf/vllm-deepseek-v4-sm89`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)
  adds GLM-5.3-Flash on SM89/SM120 with a FlashInfer 0.6.18 SM89 JIT wheel and
  `FLASHINFER_MLA_SPARSE_SM120`; it is demonstrated on 8× RTX 4090 48 GB.
- On 4× L40 48 GB, the FP8 REAP50 checkpoint (160.6 GiB) fits with ~31 GiB
  for KV/activations. Use TP4, FP8 KV, chunked prefill 4K, `--max-num-seqs 4`,
  and prefix caching.
- The related [`fire3/vllm-deepseek-v4-flash-sm89`](https://github.com/fire3/vllm-deepseek-v4-flash-sm89)
  is a Triton sparse-MLA SM89 port validated on 8× L40S for DeepSeek V4; it is
  adaptable but not GLM-specific.
- **Caveat:** no published 4×L40 GLM-5.3 prefill number; 3–5K tok/s at 8K is
  an inference from the 8×H100/8×4090 anchors. Validate the 144-expert config
  and the unmerged SM89 indexer path before relying on it.

## 3. SGLang / vLLM / TensorRT-LLM on H100

- **SGLang v0.5.22** is the best-supported H100/H200 path: `--attention-backend
  dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --linear-attn-backend triton --kv-cache-dtype bfloat16
  --chunked-prefill-size 8192`. On H200, `flashmla_sparse_q8` is SM90-only and
  measured +5.8–12.1% on prefill-only requests.
- **vLLM v0.32** with `--attention-backend flashmla_sparse
  --kv-cache-dtype fp8` is the alternative; `fp8_ds_mla` adds ~30% decode on
  SM90. The default FlashInfer SM90 backend was 36–70% slower than FA3 for
  GLM-5.3 in vLLM #56564.
- **FP8 does not fit 2×80 GB** (160.6 GiB weights + KV); use NVFP4-W4A16 or a
  4-bit GGUF. **TensorRT-LLM** `glm5_next` is B200/SM100-only in its current
  guide; it is not an H100/L40 option.

## 4. What not to use

- **FlashMLA**: current main is SM100/103 only; the pre-2026-09-30 commit
  `ba89a3` supports SM90, never SM89.
- **FlashAttention-3 / DeepGEMM**: SM90+; no Ada path.
- **TensorRT-LLM**: B200 validated; no SM89/SM90 GLM-5.3 path.
- **SGLang on SM89**: the SM89 DSA indexer PR #31190 is closed/unmerged and not
  selected by serving code; the TileLang DSA kernel requests ~166 KiB dynamic
  shared memory versus Ada's ~99 KiB.
- **KTransformers**: SM89 FP8 and 1M context are real, but Layerwise Prefill
  moves experts to the CPU; measured full-model prefill was ~3.6 tok/s. It is a
  capacity tool, not a faster-prefill tool.

## Recommendation

1. **First, on the existing 2× L40S:** build upstream llama.cpp v0.7.0 with
   `-DCMAKE_CUDA_ARCHITECTURES=89`, apply the loader alias (or reconvert),
   use **IQ3_M/Q3_K_M** (fits 96 GB), `-ngl 99 -fa on -ub 1024 -c 131072`,
   and measure 32K/128K prefill. This is the lowest-cost way to test whether
   DSA gives the expected 2–3× at 128K.
2. **If 4× L40 is available:** try the vLLM SM89 fork + FP8 REAP50 for the
   fastest prefill on this hardware class; keep llama.cpp Q4 as the reliable
   fallback.
3. **If 2× H100 is acceptable:** SGLang v0.5.22 + TileLang DSA (NVFP4/W4A16)
   is the best-validated fast-prefill path, at roughly 3–5× the SM89 prefill.
4. **For 1M context:** use 4×48 GB or 2×H100; do not expect a 1-hour 1M run on
   2×48 GB with llama.cpp.

All prefills above assume the model stays fully GPU-resident. CPU expert
offload changes the bottleneck and is not a faster-prefill option.
