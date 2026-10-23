#!/usr/bin/env bash
# Paged-KV run for GLM-5.3-Flash REAP50 on 2x H100 80 GB (SM90), stock vLLM.
# Same measurement shape as paged-kv-run.sh but a host venv instead of the
# SM80 Apptainer image.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
source "$HELPER/config.env"

# Stock vLLM 0.32's mHC TileLang path uses DeepGEMM JIT, which needs a
# discoverable nvcc >= 12.9.  The compute nodes do not put CUDA on PATH.
export CUDA_HOME=/usr/local/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
log_nvcc() { "$JIT_NVCC_COMPILER" --version | tail -2; }
log_nvcc

VLLM_BIN=${VLLM_BIN:-$WORK_ROOT/venvs/vllm032py/bin/vllm}
MODEL=${NVFP4_MODEL:-$WORK_ROOT/models/glm53-reap50/nvfp4}
PORT=${PORT:-8000}
RESULT_DIR=${PAGED_RESULT_DIR:-$RUN_DIR/paged-kv-h100-$(date -u +%Y%m%dT%H%M%SZ)}
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
log() { printf '[paged-h100] %s\n' "$*"; }
stop_server() { if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap stop_server EXIT

[ -x "$VLLM_BIN" ] || { log "missing vllm: $VLLM_BIN"; exit 2; }
[ -f "$MODEL/config.json" ] || { log "missing model: $MODEL"; exit 2; }
log "vllm=$VLLM_BIN"
log "model=$MODEL"
log "result dir: $RESULT_DIR"

CMD=("$VLLM_BIN" serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name glm53-reap50
  --tensor-parallel-size 2
  --attention-backend FLASHMLA_SPARSE
  --kv-cache-dtype bfloat16
  --block-size 128
  --max-model-len 327680
  --max-num-seqs 8
  --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.94
  --enable-prefix-caching
  --moe-backend marlin
  --linear-backend marlin
  --load-format safetensors
  --no-enable-flashinfer-autotune
  --enforce-eager
  --trust-remote-code)

printf '[paged-h100] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
CC=gcc CXX=g++ TRITON_CACHE_DIR=/tmp/triton-cache-glm53-h100 "${CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log "server pid $SERVER_PID; waiting for health"
for _ in $(seq 1 3600); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then log "server ready"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "server exited during startup"
    tail -100 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { log "health timeout"; tail -100 "$SERVER_LOG" >&2; exit 1; }

{
  echo "== vLLM capacity =="
  grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache memory|model loading took' "$SERVER_LOG" || true
} > "$RESULT_DIR/capacity.txt"
cat "$RESULT_DIR/capacity.txt"

log "cache ladder 100/200/300K"
python3 "$HELPER/paged_probe.py" \
  --base-url "http://127.0.0.1:$PORT" --model glm53-reap50 \
  --prefixes "100000 200000 300000" --suffix 128 --osl 128 \
  --runs 1 --timeout 1800 --output-dir "$RESULT_DIR/cache"

for spec in "4096 8" "16384 4" "65536 2" "262144 1"; do
  set -- $spec
  ctx=$1; conc=$2
  prompt=$((ctx * 90 / 100))
  point="$RESULT_DIR/ctx-${ctx}-c${conc}"
  log "context=$ctx concurrency=$conc prompt=$prompt"
  timeout 1800 python3 "$HELPER/bench.py" \
    --base-url "http://127.0.0.1:$PORT" --model glm53-reap50 \
    --isl "$prompt" --osl 128 --concurrency "$conc" \
    --min-prompts "$conc" --max-prompts "$conc" --runs 1 --warmup 1 \
    --timeout 1500 --cache-salt-prefix "ctx${ctx}" \
    --output-dir "$point" || log "context point $ctx failed"
done

stop_server
log "done: $RESULT_DIR"
