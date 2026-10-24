#!/usr/bin/env bash
# Qwen3.8-Flash-Next 4-bit on 2x BW6000 96 GB (Blackwell).
#
# Model: nvidia/Qwen3.8-Flash-Next-FP4-Local (~96 GiB)
# This is an FP4/PLE-offload checkpoint; the controller supplies the same
# worker state/heartbeat/SIGTERM contract as the Qwen3.5 backends.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

MODEL=${QWEN38_MODEL:-$WORK_ROOT/models/qwen38-flash-next-fp4-local}
VLLM_BIN=${VLLM_BIN:-$WORK_ROOT/venvs/vllm032py/bin/vllm}
TP=${TP:-2}
PORT=${PORT:-8010}
SERVED=${SERVED_MODEL_NAME:-qwen38-flash-next}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_DTYPE=${KV_DTYPE:-bfloat16}
QUANT=${QUANT:-modelopt}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
PLE_CPU_OFFLOAD=${PLE_CPU_OFFLOAD:-1}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export JIT_NVCC_COMPILER=${JIT_NVCC_COMPILER:-$CUDA_HOME/bin/nvcc}
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-qwen38}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_PLE_CPU_OFFLOAD="$PLE_CPU_OFFLOAD"
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=${VLLM_PLE_OFFLOAD_READY_TIMEOUT:-3600}

[ -x "$VLLM_BIN" ] || { echo "qwen38 serve: missing vllm: $VLLM_BIN" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "qwen38 serve: missing model: $MODEL" >&2; exit 2; }

CMD=("$VLLM_BIN" serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --disable-custom-all-reduce
  --quantization "$QUANT"
  --distributed-executor-backend mp
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --load-format safetensors
  --no-enable-flashinfer-autotune
  --language-model-only
  --trust-remote-code)

if [ "$ENFORCE_EAGER" = 1 ]; then
  CMD+=(--enforce-eager)
fi

printf '[qwen38-serve] model=%s tp=%s port=%s max_len=%s quant=%s ple_cpu=%s\n' \
  "$MODEL" "$TP" "$PORT" "$MAX_MODEL_LEN" "$QUANT" "$PLE_CPU_OFFLOAD"
printf '[qwen38-serve] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
