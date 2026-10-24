# 2026-10-24 — Qwen3.8-Flash-Next on 2× BW6000 96 GB (setup + blockers)

Target: **Qwen3.8-Flash-Next FP4-Local**, 2× BW6000 96 GB (192 GB, Blackwell),
262,144 native context, 4-bit/FP4, 6B active parameters.

## Artifacts created

```text
llm/qwen38/serve-server.sh      stock-vLLM attempt (text-only)
llm/qwen38/serve-container.sh   Blackwell community-container recipe
llm/qwen38/serve.sh             default launcher (container backend)
llm/qwen38/config-bw.json       controller config
llm/qwen38/measure.sh           smoke + batched + context + cache matrix
llm/qwen38/README.md            usage
```

Model downloaded to:

```text
$WORK_ROOT/models/qwen38-flash-next-fp4-local   (~96 GiB)
```

## What worked

- The model is available and complete: 21 shards, ~96 GiB.
- The repository has `Qwen4ExpForConditionalGeneration` support and
  `modelopt` / NVFP4 code paths.
- `bw01` has 2 free BW6000 96 GB GPUs; the controller can submit a 2-GPU job.
- The Blackwell device reports compute capability **12.0**, as expected.

## What blocked the run

### 1. Stock vLLM 0.32 cannot load the compressed NVFP4 scale format

First attempt (full multimodal) failed in the vision encoder:

```text
ValueError: Unsupported model when in features size is not multiple of 16
  at vllm/model_executor/layers/quantization/modelopt.py:1996
```

### 2. Text-only + local index layout

`--language-model-only` initially found no weights because the local
checkpoint stores tensors under `tensors/` while vLLM globs the model root.
After adding root symlinks and rewriting the index, loading proceeded; then:

```text
AttributeError: Layer language_model.model.layers.0.mlp.experts has no
parameter 'w2_weight_scale.nvfp4_csf_exceptions' for checkpoint weight
'language_model.model.layers.0.mlp.experts.0.down_proj.weight_scale.nvfp4_csf_exceptions'
```

This is the **new lossless NVFP4 scale compression** called out in the NVIDIA
model card. Upstream vLLM 0.32 does not support it. The model card explicitly
requires a community Blackwell vLLM container.

### 3. Community container pull

Required image per the model card:

```text
ghcr.io/local-inference-lab/vllm:release-10152026
```

The Apptainer pull was started with cache/tmp on `/work` (to avoid the 15 GB
home quota), but the image extraction did not finish in ~37 minutes. The
operation was stopped. There is a partial 12 GiB cache and **no SIF**, so no
container measurement was produced.

## Measurement status

Completed. See the results note. In short: 2.64M KV tokens, 10.08× at 262K,
4K C8 353.6 output tok/s, 256K TTFT 15.9 s, 200K cache-hit TTFT 0.61 s.

The measurement plan is ready:

```bash
# after building qwen38-vllm.sif
llm/qwen38/serve.sh up
llm/qwen38/serve.sh wait
BASE_URL=http://bw01:8010 OUT=llm/qwen38/experiments/data/2026-10-24-qwen38-bw \
  llm/qwen38/measure.sh
```

It would run:

- 10-check HTTP smoke, 500/800 tokens, C=1/2/4/8
- batched 500/800 tokens, C=1/2/4/8/16/32
- context 4K C8, 16K C4, 64K C2, 256K C1
- prefix cache 50K/100K/200K

## Next actions

1. Finish the container pull with `APPTAINER_CACHEDIR` and `TMPDIR` under
   `/work`; the earlier attempt was killed during extraction, not rejected.
2. Or use a Docker-capable host to export the image to SIF.
3. Then run `serve-container.sh` and `measure.sh`.
4. If the container path remains unavailable, use **Qwen3.8-27B** on one
   BW6000 (easy stock-vLLM target) rather than a fake Flash-Next result.

## Key configuration

The intended 2×BW6000 command is in `serve-container.sh`; it follows the RTX
PRO 6000 Blackwell recipe with TP2:

```text
--tensor-parallel-size 2
--max-model-len 262144
--quantization modelopt_mixed
--kv-cache-dtype fp8
--gpu-memory-utilization 0.90
--load-format instanttensor
--mamba-cache-mode align
--gdn-decode-kernel b12x
--linear-backend b12x
--moe-backend b12x
--reasoning-parser qwen3
--tool-call-parser qwen3_xml
--enable-auto-tool-choice
--enable-prefix-caching
--enable-chunked-prefill
--speculative-config '{"method":"mtp",...}'
```

with `VLLM_PLE_CPU_OFFLOAD=1` and the B12X/Blackwell environment.
