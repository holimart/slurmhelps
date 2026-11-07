#!/usr/bin/env bash
# Qwen3.8-Flash-Next FP4-Local on 2x H100 (hopper01, SM90) in the upstream
# vLLM nightly container.
#
# Upstream vLLM decodes the CSF NVFP4 expert scales at load and runs the
# routed experts through Marlin W4A16 (vllm-project/vllm#64312), and serves
# the NVFP4 PLE table from pinned host RAM (#56273, --engram-config
# cpu_offload). No b12x kernels or B12X_* environment: those are SM120-only.
#
# The checkpoint keeps its shards under tensors/, which upstream does not
# find yet (open #64739), so this serves a symlinked view whose shards and
# index sit at the root: $WORK_ROOT/models/qwen38-flash-next-fp4-local-view.
#
# Same worker/state contract as serve-container.sh: vLLM listens on
# 127.0.0.1:$PORT+1 behind the node gateway on $PORT (QWEN38_GATEWAY=1).
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

SIF=${QWEN38_SIF:-$WORK_ROOT/images/qwen38-vllm-nightly-20261106.sif}
MODEL=${QWEN38_MODEL:-$WORK_ROOT/models/qwen38-flash-next-fp4-local-view}
PORT=${PORT:-8020}
SERVED=${SERVED_MODEL_NAME:-qwen38-flash-next}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_DTYPE=${KV_DTYPE:-bfloat16}
# MTP stays off until the base path is measured on SM90.
MTP=${MTP:-0}
GATEWAY=${QWEN38_GATEWAY:-1}
BAD_WORDS_FILE=${QWEN38_BAD_WORDS:-$HELPER/bad_words.json}
REWRITE_RULES=${QWEN38_REWRITE_RULES-$HELPER/rewrite_rules.json}
REWRITE_LOG=${QWEN38_REWRITE_LOG:-${SLURMHELPS_SERVE_DIR:-$ROOT/.slurmhelps-serve/qwen38-flash-next-hopper}/rewrites.jsonl}
if [ "$GATEWAY" = 1 ]; then
  VLLM_HOST=127.0.0.1
  VLLM_PORT=${QWEN38_VLLM_PORT:-$((PORT + 1))}
else
  VLLM_HOST=0.0.0.0
  VLLM_PORT=$PORT
fi
CACHE_KEY=${QWEN38_CACHE_KEY:-qwen38-vllm-nightly-20261106-4fab5a0-sm90}
CACHE_ROOT=${QWEN38_CACHE_ROOT:-$WORK_ROOT/cache/qwen38/$CACHE_KEY}

[ -f "$SIF" ] || { echo "qwen38 hopper: missing $SIF" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "qwen38 hopper: missing model $MODEL" >&2; exit 2; }
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "qwen38 hopper: $MODEL has no root index (see header)" >&2; exit 2; }
for d in triton torchinductor cuda vllm xdg hf; do
  mkdir -p "$CACHE_ROOT/$d"
done

CMD=(apptainer exec --nv
  --bind "$WORK_ROOT:$WORK_ROOT"
  --bind "$CACHE_ROOT:/cache"
  --env PYTHONUNBUFFERED=1
  --env HF_HUB_OFFLINE=1
  --env TRITON_CACHE_DIR=/cache/triton
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
  --env CUDA_CACHE_PATH=/cache/cuda
  --env VLLM_CACHE_ROOT=/cache/vllm
  --env XDG_CACHE_HOME=/cache/xdg
  --env HF_HOME=/cache/hf
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID
  --env NCCL_IB_DISABLE=1
  --env NCCL_CUMEM_ENABLE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn
  --env CC=gcc
  --env CXX=g++
  "$SIF" vllm serve "$MODEL"
  --host "$VLLM_HOST" --port "$VLLM_PORT"
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --mm-encoder-tp-mode data
  --dtype bfloat16
  --engram-config '{"cpu_offload":true}'
  --enable-auto-tool-choice
  --enable-chunked-prefill
  --enable-prefix-caching
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --reasoning-parser qwen3
  --tool-call-parser "${QWEN38_TOOL_PARSER:-qwen3_xml}"
  --trust-remote-code)

if [ "$MTP" = 1 ]; then
  CMD+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

printf '[qwen38-hopper] sif=%s model=%s tp=%s port=%s gpu_mem=%s mtp=%s\n' \
  "$SIF" "$MODEL" "$TP" "$PORT" "$GPU_MEM_UTIL" "$MTP"
printf '[qwen38-hopper] persistent_cache=%s\n' "$CACHE_ROOT"
printf '[qwen38-hopper] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [ "$GATEWAY" = 1 ]; then
  [ -f "$BAD_WORDS_FILE" ] || { echo "qwen38 hopper: missing $BAD_WORDS_FILE" >&2; exit 2; }
  GW=(--config "$HELPER/config-hopper.json" --port "$PORT" --upstream-port "$VLLM_PORT"
      --bad-words-file "$BAD_WORDS_FILE")
  if [ -n "$REWRITE_RULES" ]; then
    [ -f "$REWRITE_RULES" ] || { echo "qwen38 hopper: missing $REWRITE_RULES" >&2; exit 2; }
    mkdir -p "$(dirname "$REWRITE_LOG")"
    GW+=(--rewrite-rules "$REWRITE_RULES" --rewrite-log "$REWRITE_LOG")
  fi
  exec "$ROOT/bin/serve_with_gateway.sh" "${GW[@]}" -- "${CMD[@]}"
fi
exec "${CMD[@]}"
