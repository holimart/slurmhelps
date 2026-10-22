# 2026-10-22/23 — paged-KV attempt on 2× A100 160 GB (blocked)

Attempted the SM80 vLLM fork + REAP50 NVFP4 v2 path on `ampere02`
(2× A100 80 GB, 160 GB).

## What was prepared

- Apptainer image `vllm-glm53-sm80.sif` from
  `docker://malavp/vllm-glm53flash:sm80` (vLLM 0.4 fork, SM80 sparse MLA).
- Checkpoint `GLM-5.3-Flash-REAP50-NVFP4-v2` (~99 GiB) downloaded to
  `$WORK_ROOT/models/glm53-reap50/nvfp4`.
- `paged-kv-run.sh` + `paged_probe.py`: vLLM TP2, BF16 KV, prefix caching,
  `TRITON_MLA_SPARSE`, `--block-size 128`, Marlin MoE.

## Where it stops

The engine recognizes the architecture and configures the hybrid
KDA/DSA + prefix-caching path, but weight loading fails:

```text
KeyError: 'layers.0.mlp.down_proj.weight'
  at vllm/models/glm5next/nvidia/model.py:857 load_weights
```

The checkpoint uses names like:

```text
model.language_model.layers.0.mlp.down_proj.weight
```

while this fork's wrapper expects the `language_model.model.*` convention
(per its own comment in `glm5next/nvidia/model.py`). The mismatch is in the
checkpoint serialization, not the GPU or the model math.

Tried:

- `--load-format safetensors` instead of `instanttensor`;
- `--language-model-only`;
- `--disable-custom-all-reduce`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`,
  `--enforce-eager`;
- a local `glm4v.py` mapping patch
  (`model.language_model.` → `language_model.model.`).

The loader still reaches the same key mismatch. The correct fix is either:

1. a loader alias/remap for this REAP50 NVFP4 checkpoint (the model loader must
   map its `model.language_model.*` names to the wrapper's
   `language_model.model.*` parameter names, or the checkpoint must be
   reconverted with the fork's expected serialization), or
2. a fork/checkpoint pair known to work together — the measured SM80 recipes
   use native FP8 (`8× A100`) or different conversions, not this NVFP4 v2
   artifact.

## Hardware / queue notes

- `ampere02` had both A100 GPUs free but two `beckd` CPU jobs on the node; the
  GPU job preempted/requeued them (ffa-preempt semantics). No A100 result.
- `hopper01` (2× H100) remained fully allocated, so the stock vLLM/SGLang
  path could not be run either.

## Conclusion

This is an **integration blocker**, not a performance result. The 2× A100
paged-KV run needs a working key mapping/reconversion before it can measure
TTFT, prefill, decode, cache hits, or cache capacity. The 2× L40 experiments
and Vulkan smoke are unaffected.
