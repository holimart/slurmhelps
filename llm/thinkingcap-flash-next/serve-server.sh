#!/usr/bin/env bash
# ThinkingCap Qwen3.8-Flash-Next NVFP4 backend for 2x A100 80 GB (SM80).
#
# This is the A100 counterpart of llm/qwen38/serve-container.sh.  It serves
# bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 with stock vLLM:
#
#   * routed experts: NVFP4 W4A16 (compressed-tensors)
#   * n-gram table:   FP8 E4M3
#   * everything else: BF16
#
# A100 has no native FP4 tensor cores, so vLLM runs the weight-only FP4
# experts through its Marlin kernel.  This is slower than the BW6000 path, but
# it keeps the same checkpoint and the same OpenAI-compatible surface.
#
# Environment (all overrideable; defaults match config-a100.json):
#   WORK_ROOT                 /work/<site>/<user>
#   THINKINGCAP_MODEL         $WORK_ROOT/models/thinkingcap-flash-next-nvfp4
#   VLLM_BIN                  $WORK_ROOT/venvs/vllm032py/bin/vllm
#   SERVED_MODEL_NAME         thinkingcap-flash-next
#   PORT                      8011
#   MAX_MODEL_LEN             262144
#   MAX_NUM_SEQS              2
#   MAX_NUM_BATCHED_TOKENS    4096
#   GPU_MEM_UTIL              0.95
#   KV_DTYPE                  bfloat16
#   BLOCK_SIZE                64
#   MTP                       0  (set 1 for the checkpoint's MTP head)
#   LANGUAGE_MODEL_ONLY       0  (set 1 to drop the vision tower)
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

MODEL=${THINKINGCAP_MODEL:-$WORK_ROOT/models/thinkingcap-flash-next-nvfp4}
VLLM_BIN=${VLLM_BIN:-$WORK_ROOT/venvs/vllm032py/bin/vllm}
SERVED=${SERVED_MODEL_NAME:-thinkingcap-flash-next}
PORT=${PORT:-8011}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
KV_DTYPE=${KV_DTYPE:-bfloat16}
BLOCK_SIZE=${BLOCK_SIZE:-64}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
MTP=${MTP:-0}
MTP_DRAFT_TOKENS=${MTP_DRAFT_TOKENS:-3}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export JIT_NVCC_COMPILER=${JIT_NVCC_COMPILER:-$CUDA_HOME/bin/nvcc}
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-14400}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$WORK_ROOT/cache/thinkingcap/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WORK_ROOT/cache/thinkingcap/torchinductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$WORK_ROOT/cache/thinkingcap/vllm}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

[ -x "$VLLM_BIN" ] || { echo "thinkingcap serve: missing vllm: $VLLM_BIN" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "thinkingcap serve: missing model: $MODEL" >&2; exit 2; }
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "thinkingcap serve: missing index: $MODEL" >&2; exit 2; }

CMD=("$VLLM_BIN" serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --disable-custom-all-reduce
  --enforce-eager
  --dtype bfloat16
  --quantization compressed-tensors
  --moe-backend marlin
  --kv-cache-dtype "$KV_DTYPE"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
  --trust-remote-code)

if [ "$ENABLE_CHUNKED_PREFILL" = 1 ]; then
  CMD+=(--enable-chunked-prefill)
fi
if [ "$LANGUAGE_MODEL_ONLY" = 1 ]; then
  CMD+=(--language-model-only)
fi
if [ "$MTP" = 1 ]; then
  CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DRAFT_TOKENS,\"draft_sample_method\":\"probabilistic\"}")
fi

printf '[thinkingcap-serve] model=%s tp=%s port=%s max_len=%s max_seqs=%s mtp=%s\n' \
  "$MODEL" "$TP" "$PORT" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MTP"
printf '[thinkingcap-serve] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
