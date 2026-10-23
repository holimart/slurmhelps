#!/usr/bin/env bash
# Qwen3.5 stock-vLLM backend for 2x A100 (A100 only).
#
# Variants:
#   QWEN_VARIANT=35b  -> one GPU, 35B-A3B, port 8001
#   QWEN_VARIANT=122b -> two GPUs, 122B-A10B, port 8002
#
# The slurmhelps controller/worker can launch this with:
#   slurmhelps-serve up --config llm/qwen35/config-122b.json \
#       --backend-command bash llm/qwen35/serve-122b.sh
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

VARIANT=${QWEN_VARIANT:-122b}
case "$VARIANT" in
  35b)
    MODEL=${QWEN_MODEL:-$WORK_ROOT/models/qwen35-35b-a3b-awq}
    TP=${TP:-1}; PORT=${PORT:-8001}; SERVED=${SERVED_MODEL_NAME:-qwen35-35b-a3b}
    ;;
  122b)
    MODEL=${QWEN_MODEL:-$WORK_ROOT/models/qwen35-122b-a10b-awq}
    TP=${TP:-2}; PORT=${PORT:-8002}; SERVED=${SERVED_MODEL_NAME:-qwen35-122b-a10b}
    ;;
  *) echo "unknown QWEN_VARIANT=$VARIANT (35b|122b)" >&2; exit 2 ;;
esac

VLLM_BIN=${VLLM_BIN:-$WORK_ROOT/venvs/vllm032py/bin/vllm}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_DTYPE=${KV_DTYPE:-bfloat16}

# Triton's CudaUtils compiler needs an explicit C compiler on the compute node.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export JIT_NVCC_COMPILER=${JIT_NVCC_COMPILER:-$CUDA_HOME/bin/nvcc}
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export PYTHONUNBUFFERED=1
# ampere02's two A100s are linked "NODE" (PCIe through the host bridge), not NVLink. TP2 NCCL
# init hung there twice (workers at 100% CPU, ~715 MiB of GPU memory for an hour, no log
# after the pynccl line), so peer-to-peer is off by default; set NCCL_P2P_DISABLE=0 to retry it.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-$VARIANT}

[ -x "$VLLM_BIN" ] || { echo "qwen serve: missing vllm: $VLLM_BIN" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "qwen serve: missing model: $MODEL" >&2; exit 2; }

CMD=("$VLLM_BIN" serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --disable-custom-all-reduce
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enforce-eager
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code)

printf '[qwen-serve] variant=%s model=%s tp=%s port=%s\n' "$VARIANT" "$MODEL" "$TP" "$PORT"
printf '[qwen-serve] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
