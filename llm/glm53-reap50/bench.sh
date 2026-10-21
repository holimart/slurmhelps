#!/usr/bin/env bash
# Run the batched-throughput benchmark against an already-running server.
#
# usage: bench.sh [--base-url URL] [--model NAME] [--chat]
#                 [--concurrency "1 2 4 8 16 32 64"] [--isl 2048] [--osl 256]
#                 [--runs 3] [--workers 1] [--shared-prefix 0]
#                 [--cache-regime cold|warm-shared] [--no-flush-cache]
#                 [--output-dir DIR] [--dry-run]
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
# shellcheck source=config.env
source "$HELPER/config.env"

BASE_URL="http://127.0.0.1:${PORT}"
MODEL=$SERVED_MODEL_NAME
CONCURRENCY=$BENCH_CONCURRENCY
ISL=$BENCH_ISL
OSL=$BENCH_OSL
RUNS=$BENCH_RUNS
WORKERS=$BENCH_WORKERS
MIN_PROMPTS=$BENCH_MIN_PROMPTS
MAX_PROMPTS=$BENCH_MAX_PROMPTS
SHARED_PREFIX=$BENCH_SHARED_PREFIX
CACHE_REGIME=$BENCH_CACHE_REGIME
FLUSH=1
OUTPUT_DIR=
CHAT=0
DRY_RUN=0
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url)      BASE_URL=$2; shift 2 ;;
    --model)         MODEL=$2; shift 2 ;;
    --chat)          CHAT=1; shift ;;
    --concurrency)   CONCURRENCY=$2; shift 2 ;;
    --isl)           ISL=$2; shift 2 ;;
    --osl)           OSL=$2; shift 2 ;;
    --runs)          RUNS=$2; shift 2 ;;
    --workers)       WORKERS=$2; shift 2 ;;
    --min-prompts)   MIN_PROMPTS=$2; shift 2 ;;
    --max-prompts)   MAX_PROMPTS=$2; shift 2 ;;
    --shared-prefix) SHARED_PREFIX=$2; shift 2 ;;
    --cache-regime)  CACHE_REGIME=$2; shift 2 ;;
    --no-flush-cache) FLUSH=0; shift ;;
    --output-dir)    OUTPUT_DIR=$2; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --)              shift; PASSTHROUGH=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RUN_DIR/bench-$(date -u +%Y%m%dT%H%M%SZ)"
fi

cmd=(python3 "$HELPER/bench.py"
  --base-url "$BASE_URL" --model "$MODEL"
  --concurrency "$CONCURRENCY" --isl "$ISL" --osl "$OSL"
  --runs "$RUNS" --workers "$WORKERS"
  --min-prompts "$MIN_PROMPTS" --max-prompts "$MAX_PROMPTS"
  --shared-prefix "$SHARED_PREFIX" --cache-regime "$CACHE_REGIME"
  --output-dir "$OUTPUT_DIR")
if [ "$CHAT" = 1 ]; then cmd+=(--chat); fi
if [ "$FLUSH" = 0 ]; then cmd+=(--no-flush-cache); fi
cmd+=("${PASSTHROUGH[@]}")

printf '[bench]'; printf ' %q' "${cmd[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ]; then exit 0; fi
"${cmd[@]}"
