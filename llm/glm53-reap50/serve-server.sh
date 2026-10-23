#!/usr/bin/env bash
# Long-lived GLM-5.3-Flash REAP50 server on 2x A100 (SM80 fork).
#
# This is the backend process only: it starts vLLM and stays in the foreground.
# The slurmhelps serving worker/controller owns state, health polling, tunnels
# and preemption. No probe or benchmark runs here.
#
# Environment (all overrideable; defaults match the researched A100 setup):
#   WORK_ROOT                    /work/<site>/<user>
#   VLLM_SIF                     $WORK_ROOT/images/vllm-glm53-sm80.sif
#   NVFP4_MODEL                  $WORK_ROOT/models/glm53-reap50/nvfp4
#   SERVED_MODEL_NAME            glm53-reap50
#   PORT                         8000
#   MAX_MODEL_LEN                786432   (768K; 700K requests fit)
#   MAX_NUM_SEQS                 32
#   MAX_NUM_BATCHED_TOKENS       8192     (per scheduler step, not context)
#   GPU_MEM_UTIL                 0.94
#   KV_DTYPE                     bfloat16
#   BLOCK_SIZE                   128
#   ENABLE_CHUNKED_PREFILL       1
#   VLLM_PATCH_GLM4V             $WORK_ROOT/vllm-patch/glm4v.py
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT
# shellcheck source=config.env
source "$HELPER/config.env"

# This A100 backend needs the SM80 fork image.  config.env's VLLM_SIF points
# at the stock image, so it is intentionally not the default here.
SIF=${GLM53_SIF:-$WORK_ROOT/images/vllm-glm53-sm80.sif}
MODEL=${NVFP4_MODEL:-$WORK_ROOT/models/glm53-reap50/nvfp4}
PATCH=${VLLM_PATCH_GLM4V:-$WORK_ROOT/vllm-patch/glm4v.py}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-reap50}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-786432}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}
KV_DTYPE=${KV_DTYPE:-bfloat16}
BLOCK_SIZE=${BLOCK_SIZE:-128}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
ENABLE_TOOL_CALLS=${ENABLE_TOOL_CALLS:-1}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-glm47}

[ -f "$SIF" ] || { echo "serve-server: missing image: $SIF" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "serve-server: missing model: $MODEL" >&2; exit 2; }

BINDS=(--bind "$WORK_ROOT:$WORK_ROOT")
if [ -f "$PATCH" ]; then
  BINDS+=(--bind "$PATCH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4v.py")
fi

CMD=(apptainer exec --nv "${BINDS[@]}"
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600
  --env CC=gcc --env CXX=g++
  --env TRITON_CACHE_DIR=/tmp/triton-cache-glm53-serve
  --env NCCL_P2P_DISABLE=1 --env NCCL_IB_DISABLE=1
  "$SIF" vllm serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size 2
  --disable-custom-all-reduce
  --enforce-eager
  --attention-backend TRITON_MLA_SPARSE
  --kv-cache-dtype "$KV_DTYPE"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --moe-backend marlin
  --no-enable-flashinfer-autotune
  --load-format safetensors
  --language-model-only
  --trust-remote-code)

if [ "$ENABLE_CHUNKED_PREFILL" = 1 ]; then
  CMD+=(--enable-chunked-prefill)
fi
if [ "$ENABLE_TOOL_CALLS" = 1 ]; then
  CMD+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

printf '[serve-server] image=%s\n' "$SIF"
printf '[serve-server] model=%s\n' "$MODEL"
printf '[serve-server] port=%s max_len=%s max_seqs=%s batched_tokens=%s\n' \
  "$PORT" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"
printf '[serve-server] command:'; printf ' %q' "${CMD[@]}"; printf '\n'

# exec so the worker's SIGTERM goes to vLLM, not to a shell wrapper.
exec "${CMD[@]}"
