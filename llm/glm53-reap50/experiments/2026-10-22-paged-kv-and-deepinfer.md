# Paged-KV engines and the latest DeepSeek inference stack (2026-10-22)

Research note for moving GLM-5.3-Flash REAP50 from llama.cpp's single active
slot to a real paged-KV serving engine.

## Short answer

- **Stock vLLM v0.32.0 and SGLang v0.5.22** are the production paged-KV
  engines: PagedAttention / RadixAttention, continuous batching, chunked
  prefill, prefix caching, OpenAI server, metrics.
- On GLM5-Next they are officially **SM90+** (H100/H200, B200/B300, GB200/GB300,
  SM120). They do **not** have an SM80/SM89 sparse-DSA path.
- **2× A100 160 GB:** stock engines fail at attention-backend selection. The
  practical paged-KV route is a community SM80 fork + REAP50 NVFP4 W4A16.
- **2× H100 160 GB:** stock vLLM/SGLang work with REAP50 NVFP4 W4A16 (Marlin on
  Hopper), BF16 KV, TP2. FP8 does not fit.
- **3–4× L40 192 GB:** no validated public SM89 GLM-5.3 paged-KV path.
  KTransformers is the only SM89-blessed engine, but it streams experts from
  CPU RAM.
- **2× L40 96 GB:** no paged-KV engine with a fitting GLM checkpoint; llama.cpp
  remains the only option, and it has no GPU multi-prefix tree.

## Where to get the engines

| Engine | Get it | Version (Oct 2026) | GLM5-Next on SM80/89/90 |
|---|---|---|---|
| **vLLM** | `pip install vllm`; `docker pull vllm/vllm-openai:glm53-flash` | v0.32.0 | SM90/SM100/SM120 only; SM80/89 fail |
| **SGLang** | `pip install "sglang[all]"`; `lmsysorg/sglang:latest` / `:glm-5.3-flash` | v0.5.22 | SM90+/SM100/SM120; no SM80/89 DSA |
| **TensorRT-LLM** | NGC container; `pip install tensorrt_llm` | 1.3.0rc (GLM), 1.2.1 stable | Blackwell only for GLM-5.3 |
| **LMDeploy** | `pip install lmdeploy` | v0.18.0 | GLM5-Next not in matrix |
| **KTransformers** | `pip install "ktransformers[sglang]"` | v0.7.2 | SM89/SM120, FP8 + CPU experts |
| **LMCache** | `pip install lmcache` | v0.5.6 | KV tiering layer over vLLM/SGLang |
| **NVIDIA Dynamo** | `pip install ai-dynamo` / Helm | v1.5.1 | orchestrator, not an engine |
| **TokenSpeed** | `lightseekorg/tokenspeed-runner` | v0.1.1 | H100/H200/B200, not A100/L40 |
| **llama.cpp** | source/patch | v0.7.0 / fork | any CUDA, but DSA dense |

## Community SM80 paged-KV forks (A100 / A800)

These are the only demonstrated paged-KV GLM-5.3 paths on Ampere:

| Repo | Hardware / result | Notes |
|---|---|---|
| [`Malav-P/vllm` `glm53flash-a100`](https://github.com/Malav-P/vllm/tree/glm53flash-a100) | 8× A100 80 GB, **93 tok/s single-stream decode**, 262K | image `malavp/vllm-glm53flash:sm80`; TRITON_MLA_SPARSE, BF16 KV |
| [`Mrzhiyao/glm53-a800-vllm`](https://github.com/Mrzhiyao/glm53-a800-vllm) | 8× A800, FP8, TP4+PP2, **46.8 tok/s single, 304 tok/s @16×64**, 1M, **2.618M KV tokens** | build from `wtdcode/vllm-backport` + `ZJY0516/vllm` |
| [`promisezackr/glm53-flash-170hx-pp8`](https://github.com/promisezackr/glm53-flash-170hx-pp8) | 8× CMP 170HX SM80, NVFP4+Marlin, **123 tok/s code single**, 960K prompt in 208 s | PP8, DFlash2, 6.62M KV tokens |

No public **SM89** GLM-5.3 fork was found. `mochgolf/qsa-hisparse` is an SM89
sparse-attention reference for Qwen3.8, not GLM.

## Memory fit for REAP50

| Checkpoint | Size | Fits 2× A100 160 GB? | Fits 2× H100 160 GB? | Fits 4× L40 192 GB? |
|---|---:|---:|---:|---:|
| REAP50 FP8 v2 | 160.6 GiB | **No** | **No** | yes, only ~19 GiB spare |
| REAP50 NVFP4 v2 | ~98 GiB | **Yes**, ~55 GiB KV | **Yes**, ~55 GiB KV | yes, ~87 GiB KV |
| REAP50 GGUF Q4 | 87–99 GB | yes (llama.cpp only) | yes (llama.cpp only) | yes (llama.cpp only) |

On 2× A100 TP2 with NVFP4, expect ~1.7–2.1M BF16 KV tokens (about 13–16
concurrent 128K requests, or 2 at 1M). On 4× L40, ~1.4–1.5M tokens.

## Latest DeepSeek engine: DeepInfer

- [`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer), v0.1.0,
  repo created **2026-10-16**, Apache-2.0.
- It is the production engine behind V4.1-Flash, but the open release is
  **offline-only**: no HTTP/OpenAI server, no KV-cache store, no production
  E/P/D disaggregation, no request load balancer.
- It supports **exactly one architecture** (`DeepseekV41ForCausalLM`) and
  requires **SM100 or Ascend 950**. It cannot serve GLM-5.3-Flash.
- It has real paged KV (`BlockManager`/`block_table`/`ref_count`), continuous
  batching, chunked prefill, a separate state cache, partial prefix caching,
  FP4 KV, DSpark, and Engram.
- The HF `inference/` directory is a **static-batch reference**: right-padded
  prompt matrix, contiguous `[bsz, max_seq_len]` KV, no paging, no admission,
  no prefix cache.

### What transfers to GLM-5.3-Flash

| DeepInfer / DeepSeek idea | Transferable? | Notes |
|---|---|---|
| Paged KV + separate state cache | **Yes, already upstream** | GLM needs paged KV for 11 DSA layers plus a KDA recurrent-state pool; vLLM/SGLang implement the hybrid manager. Tune SGLang `--mamba-full-memory-ratio` / `--max-mamba-cache-size`. |
| DSA sparse attention / indexer | **Yes, already upstream** | GLM has `index_topk=2048`, `kpool=4`, NoPE MLA; SM90+ only. |
| k-pooling | **Yes** | Already a model property; no kernel port. |
| FP4 experts / FP8 dense | **Memory only on A100/L40/H100** | No native FP4; NVFP4 W4A16 via Marlin. On B200 native. |
| FP4 KV | **No** | GLM uses BF16 KV on H100; FP8 only Blackwell/MI355X. |
| DSpark / MTP | **Partly** | GLM has MTP; REAP50 removed it. Use a separate draft model or leave speculation off. |
| Engram / n-gram host streaming | **No Engram in GLM** | The analogous idea is SGLang Paged Experts for MoE weights. |
| DeepEP / EPLB | **Not at 2–4 GPUs** | Wide-EP wins start at EP64+; TP is simpler. |
| DeepGEMM grouped MoE | **H100 only** | Already default in SGLang on Hopper; unavailable on SM89. |
| 3FS / KV tiering | **Design only** | Use LMCache or SGLang HiCache; 3FS needs RDMA/SSD cluster. |
| P/D disaggregation | **No** | Needs multiple model copies; 2×80 GB cannot hold two 98 GiB copies. |
| Chunked prefill / async scheduling | **Yes, already upstream** | vLLM/SGLang have it. |

## Recommended next paged-KV run

**2× A100 160 GB (free now)** is the best available paged-KV experiment:

1. Pull `malavp/vllm-glm53flash:sm80` or build
   `Malav-P/vllm@glm53flash-a100`.
2. Download `GLM-5.3-Flash-REAP50-NVFP4-v2` (~98 GiB; root v2, not the
   unevaluated v3).
3. Launch TP2:
   `--attention-backend TRITON_MLA_SPARSE --kv-cache-dtype bfloat16 --block-size 128 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.94 --enable-prefix-caching --moe-backend marlin --load-format instanttensor --safetensors-load-strategy lazy`
4. Measure the same context/cache/session matrix. Expected 20–35 tok/s
   single-stream, 60–120 aggregate, prefill 1.5–4K tok/s — **inferred, not
   validated on 2 GPUs**.

**2× H100 160 GB (when free)** is the stock, validated path: vLLM v0.32 or
SGLang v0.5.22, REAP50 NVFP4 W4A16, BF16 KV, TP2, prefix caching. FP8 does not
fit.

**3–4× L40** is not a validated paged-KV path; KTransformers is the only
SM89-blessed engine and it streams experts from CPU RAM. Use it only for
capacity, not prefill speed.

## Sources

- DeepInfer: <https://github.com/deepseek-ai/DeepInfer>
- HF V4.1 reference: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>
- vLLM GLM-5.3 recipe: <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>
- vLLM v0.32: <https://github.com/vllm-project/vllm/releases/tag/v0.32.0>
- SGLang cookbook: <https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash>
- SGLang v0.5.22: <https://github.com/sgl-project/sglang/releases/tag/v0.5.22>
- SM80 A100 fork: <https://github.com/Malav-P/vllm/tree/glm53flash-a100>
- SM80 A800 fork: <https://github.com/Mrzhiyao/glm53-a800-vllm>
- SM80 CMP fork: <https://github.com/promisezackr/glm53-flash-170hx-pp8>
- NVFP4 checkpoint: <https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2>
- LMCache: <https://github.com/LMCache/LMCache>
