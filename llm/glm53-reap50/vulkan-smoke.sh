#!/usr/bin/env bash
# Short Vulkan smoke test for GLM5-Next/REAP50 on 2x L40S.
#
# The fork implements ggml_mhc_sinkhorn only for CPU and CUDA. The Vulkan
# backend therefore has to fall back to CPU for that operator; this test only
# establishes whether the model loads and produces tokens, not performance.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
source "$HELPER/config.env"

RESULT_DIR=${VULKAN_RESULT_DIR:-$RUN_DIR/vulkan-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RESULT_DIR"
log() { printf '[vulkan] %s\n' "$*"; }
stop_server() { "$HELPER/serve.sh" --stop >/dev/null 2>&1 || true; }
trap stop_server EXIT

export LLAMA_SERVER=${LLAMA_SERVER:-${WORK_ROOT}/build/llama-vulkan/bin/llama-server}
export LLAMA_FA=${LLAMA_FA:-off}
export LLAMA_KV_TYPE=${LLAMA_KV_TYPE:-f16}
export LLAMA_BATCH=${LLAMA_BATCH:-512}
export LLAMA_UBATCH=${LLAMA_UBATCH:-128}
export LLAMA_MMAP=${LLAMA_MMAP:-0}
export LLAMA_PROMPT_CACHE=${LLAMA_PROMPT_CACHE:-1}
export CTX_TOTAL=${CTX_TOTAL:-8192}
export MAX_RUNNING=${MAX_RUNNING:-1}
export PORT=${PORT:-30000}

log "server=$LLAMA_SERVER"
log "vulkan icd: $(ls /usr/share/vulkan/icd.d/*nvidia* 2>/dev/null | head -1)"
{
  echo "== llama-server =="
  "$LLAMA_SERVER" --version 2>&1 || true
  echo
  echo "== vulkaninfo =="
  if command -v vulkaninfo >/dev/null 2>&1; then vulkaninfo --summary 2>&1 | head -80; fi
} > "$RESULT_DIR/backend.txt"

if ! "$HELPER/serve.sh" --engine llamacpp --variant "${VULKAN_VARIANT:-$VARIANT}" --background --wait; then
  log "server did not start; see $RUN_DIR/server.log"
  cp -f "$RUN_DIR/server.log" "$RESULT_DIR/server.log" 2>/dev/null || true
  exit 1
fi
cp -f "$RUN_DIR/server.log" "$RESULT_DIR/server.log" 2>/dev/null || true

log "server up; sending one 4K prompt"
rc=0
timeout 600 "$HELPER/bench.sh" \
  --base-url "http://127.0.0.1:${PORT}" --model "$SERVED_MODEL_NAME" \
  --isl 4096 --osl 8 --concurrency 1 --min-prompts 1 --max-prompts 1 \
  --runs 1 --warmup 1 --timeout 600 --output-dir "$RESULT_DIR" || rc=$?
log "bench rc=$rc"
stop_server
cp -f "$RUN_DIR/server.log" "$RESULT_DIR/server.log" 2>/dev/null || true
exit "$rc"
