#!/usr/bin/env bash
# Paged-KV run for GLM-5.3-Flash REAP50 on 2x A100 80 GB (SM80).
#
# Uses the community SM80 vLLM fork image and the REAP50 NVFP4 v2 checkpoint.
# Measures:
#   - cache ladder 100/200/300K: fill, P+128 hit, immediate repeat
#   - independent context/concurrency points: 4K C8, 16K C4, 64K C2, 256K C1
#
# No cold-miss requests. Miss is inferred from cumulative fill.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
source "$HELPER/config.env"

SIF=${VLLM_SIF:-$WORK_ROOT/images/vllm-glm53-sm80.sif}
MODEL=${NVFP4_MODEL:-$WORK_ROOT/models/glm53-reap50/nvfp4}
PORT=${PORT:-8000}
RESULT_DIR=${PAGED_RESULT_DIR:-$RUN_DIR/paged-kv-$(date -u +%Y%m%dT%H%M%SZ)}
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
log() { printf '[paged] %s\n' "$*"; }
stop_server() { if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap stop_server EXIT

[ -f "$SIF" ] || { log "missing image: $SIF"; exit 2; }
[ -f "$MODEL/config.json" ] || { log "missing model: $MODEL"; exit 2; }
log "image=$SIF"
log "model=$MODEL"
log "result dir: $RESULT_DIR"

CMD=(apptainer exec --nv --bind "$WORK_ROOT:$WORK_ROOT"
  --bind "$WORK_ROOT/vllm-patch/glm4v.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4v.py"
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600
  --env CC=gcc --env CXX=g++
  --env TRITON_CACHE_DIR=/tmp/triton-cache-glm53
  --env NCCL_P2P_DISABLE=1 --env NCCL_IB_DISABLE=1
  "$SIF" vllm serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name glm53-reap50
  --tensor-parallel-size 2
  --disable-custom-all-reduce
  --enforce-eager
  --attention-backend TRITON_MLA_SPARSE
  --kv-cache-dtype bfloat16
  --block-size 128
  --max-model-len 327680
  --max-num-seqs 8
  --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.94
  --enable-prefix-caching
  --moe-backend marlin
  --no-enable-flashinfer-autotune --load-format safetensors
  --language-model-only
  --trust-remote-code)

printf '[paged] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log "server pid $SERVER_PID; waiting for health"
for _ in $(seq 1 3600); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then log "server ready"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "server exited during startup"
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { log "health timeout"; tail -80 "$SERVER_LOG" >&2; exit 1; }

# Capacity lines from the vLLM startup log.
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
