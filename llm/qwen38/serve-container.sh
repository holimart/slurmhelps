#!/usr/bin/env bash
# Qwen3.8-Flash-Next FP4-Local in the Blackwell community vLLM container.
#
# The FP4-local checkpoint uses a compressed NVFP4 scale format that stock
# vLLM 0.32 cannot load; the model card requires the Local Inference Lab
# Blackwell container. This wrapper keeps the same worker/state contract.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

SIF=${QWEN38_SIF:-$WORK_ROOT/images/qwen38-vllm.sif}
MODEL=${QWEN38_MODEL:-$WORK_ROOT/models/qwen38-flash-next-fp4-local}
PORT=${PORT:-8010}
SERVED=${SERVED_MODEL_NAME:-qwen38-flash-next}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_DTYPE=${KV_DTYPE:-fp8}
MTP=${MTP:-1}

[ -f "$SIF" ] || { echo "qwen38 container: missing $SIF" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "qwen38 container: missing model $MODEL" >&2; exit 2; }

# Environment from the model card's RTX PRO 6000 recipe (non-DGX path).
export B12X_DENSE_SPLITK_TURBO=1
export B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0
export B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1
export B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1
export B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224
export B12X_DYNAMIC_SPLIT_FAST_PREPARE=1
export B12X_DYNAMIC_SPLIT_LOW_SMEM=1
export B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1
export B12X_DYNAMIC_WORK_SOURCE=persistent_grid
export B12X_W4A16_A4_PREFILL=1
export B12X_DISK_BACKEND=io_uring
export B12X_MHC_PDL=1
export B12X_PCIE_ONESHOT_BLOCK_LIMIT=4
export B12X_PCIE_ONESHOT_PDL=1
export B12X_PCIE_ONESHOT_THREADS=512
export CUDA_BINARY_LOADER_THREAD_COUNT=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY
export CUTE_DSL_ARCH=sm_120a
export INSTANTTENSOR_BACKEND=BUFFERED
export NCCL_BUFFSIZE=2097152
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=spcx
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAFETENSORS_FAST_GPU=1
export TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0
export TORCH_CUDA_ARCH_LIST=12.0a
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_NCCL_USE_COMM_NONBLOCKING=0
export VLLM_B12X_DENSE_ACTIVATION_MODE=auto
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
export VLLM_COMPUTE_NANS_IN_LOGITS=0
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1
export VLLM_LM_HEAD_A16=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_MTP_NVFP4_LM_HEAD=1
export VLLM_MXFP8_LM_HEAD=0
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0
export VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1
export VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_USE_FASTOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD=(apptainer exec --nv
  --bind "$WORK_ROOT:$WORK_ROOT"
  --env PYTHONUNBUFFERED=1
  --env HF_HUB_OFFLINE=1
  --env VLLM_PLE_CPU_OFFLOAD=1
  "$SIF" vllm serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --async-scheduling
  --block-size 64
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --dtype bfloat16
  --enable-auto-tool-choice
  --enable-chunked-prefill
  --no-enable-flashinfer-autotune
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --gdn-decode-kernel b12x
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --no-language-model-only
  --linear-backend b12x
  --load-format instanttensor
  --mamba-cache-mode align
  --mamba-ssm-cache-dtype auto
  --max-cudagraph-capture-size 64
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --mm-encoder-tp-mode data
  --mm-processor-cache-gb 0.0
  --moe-backend b12x
  --quantization modelopt_mixed
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --trust-remote-code)

if [ "$MTP" = 1 ]; then
  CMD+=(--speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","num_speculative_tokens":3}')
fi

printf '[qwen38-container] sif=%s model=%s tp=%s port=%s gpu_mem=%s\n' \
  "$SIF" "$MODEL" "$TP" "$PORT" "$GPU_MEM_UTIL"
printf '[qwen38-container] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
