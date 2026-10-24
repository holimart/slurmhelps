#!/usr/bin/env bash
# Measurement matrix for Qwen3.8-Flash-Next on 2x BW6000 (Blackwell).
#
# Assumes the server is already up (serve.sh up / wait) and reachable at
# $BASE_URL. Runs the same matrix shape as the GLM/Qwen A100 experiments:
#   - 10-check smoke
#   - batched 500/800-token concurrency sweep
#   - context sweep 4K/16K/64K/262K
#   - prefix-cache ladder 50K/100K/200K
set -euo pipefail
HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
BASE_URL=${BASE_URL:-http://bw01:8010}
MODEL=${SERVED_MODEL_NAME:-qwen38-flash-next}
OUT=${OUT:-$HELPER/experiments/data/2026-10-24-qwen38-bw}
TIMEOUT=${TIMEOUT:-1800}
mkdir -p "$OUT"

log() { printf '[qwen38-measure] %s\n' "$*"; }

log "smoke @ $BASE_URL"
python3 "$ROOT/tests/serve_smoke.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --contexts "500 800" --concurrency "1 2 4 8" \
  --model-max-len 262144 --timeout "$TIMEOUT" \
  --output "$OUT/smoke.json" || log "smoke had failures"

log "batched 500/800 concurrency"
python3 "$ROOT/llm/glm53-reap50/bench.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --isl 500 --osl 32 --concurrency "1 2 4 8 16 32" \
  --min-prompts 32 --max-prompts 32 --runs 1 --warmup 1 \
  --timeout "$TIMEOUT" --cache-salt-prefix "qwen38-500" \
  --output-dir "$OUT/batch-500" || log "batch-500 failed"

python3 "$ROOT/llm/glm53-reap50/bench.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --isl 800 --osl 32 --concurrency "1 2 4 8 16 32" \
  --min-prompts 32 --max-prompts 32 --runs 1 --warmup 1 \
  --timeout "$TIMEOUT" --cache-salt-prefix "qwen38-800" \
  --output-dir "$OUT/batch-800" || log "batch-800 failed"

for spec in "4096 8" "16384 4" "65536 2" "262144 1"; do
  set -- $spec
  ctx=$1; conc=$2
  prompt=$((ctx * 90 / 100))
  log "context=$ctx C=$conc prompt=$prompt"
  python3 "$ROOT/llm/glm53-reap50/bench.py" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --isl "$prompt" --osl 128 --concurrency "$conc" \
    --min-prompts "$conc" --max-prompts "$conc" --runs 1 --warmup 1 \
    --timeout "$TIMEOUT" --cache-salt-prefix "qwen38-ctx$ctx" \
    --output-dir "$OUT/ctx-$ctx-c$conc" || log "ctx $ctx failed"
done

log "prefix-cache ladder"
python3 "$ROOT/llm/glm53-reap50/paged_probe.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --prefixes "50000 100000 200000" --suffix 128 --osl 128 \
  --runs 1 --timeout "$TIMEOUT" --output-dir "$OUT/cache" || log "cache ladder failed"

log "done: $OUT"
