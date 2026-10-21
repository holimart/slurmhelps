#!/usr/bin/env bash
# Node-side orchestrator: install (optional) -> serve -> wait -> benchmark -> stop.
#
# Called by glm53-reap50.sbatch, or directly inside an allocation:
#   srun -p ... --gres=gpu:2 ... bash run-job.sh
#
# Environment (all optional; config.env supplies defaults):
#   ENGINE, VARIANT, INSTALL, BENCH, SERVE, BENCH_*, PORT, TP, ...
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
# submit.sh encodes the list with '+' because Slurm --export is comma-separated.
BENCH_CONCURRENCY=${BENCH_CONCURRENCY//+/ }

INSTALL=${INSTALL:-1}
SERVE=${SERVE:-1}
BENCH=${BENCH:-1}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=${RESULT_DIR:-$RUN_DIR/$STAMP}
# Per-job server.pid/server.log: two jobs on two nodes must not share the
# same files under the common runs directory.
export RUN_DIR="$RESULT_DIR/runtime"
mkdir -p "$RESULT_DIR" "$RUN_DIR"

log() { printf '[job] %s\n' "$*"; }
cleanup() {
  local rc=$?
  if [ -f "$RUN_DIR/server.pid" ]; then
    "$HELPER/serve.sh" --stop >/dev/null 2>&1 || true
  fi
  if [ -f "$RUN_DIR/server.log" ]; then
    cp -f "$RUN_DIR/server.log" "$RESULT_DIR/server.log" 2>/dev/null || true
  fi
  log "result dir: $RESULT_DIR"
  trap - EXIT
  exit "$rc"
}
trap cleanup EXIT

log "host=$(hostname) job=${SLURM_JOB_ID:-manual} engine=$ENGINE variant=$VARIANT"
{
  echo "== nvidia-smi =="
  nvidia-smi 2>&1 || true
  echo
  echo "== cpu/mem =="
  lscpu 2>&1 | egrep 'Model name|Socket|Core|Thread|CPU\(s\)' || true
  free -g || true
  echo
  echo "== disk =="
  df -h "$WORK_ROOT" "$MODEL_ROOT" 2>&1 || true
} > "$RESULT_DIR/host.txt"

if [ "$INSTALL" = 1 ]; then
  log "install"
  "$HELPER/install.sh" --engine "$ENGINE" --variant "$VARIANT"
fi

if [ "${CACHE_SWEEP:-0}" = 1 ]; then
  log "cache-reuse sweep"
  exec "$HELPER/cache-sweep.sh" --engine "$ENGINE" --variant "$VARIANT" \
    --result-dir "$RESULT_DIR"
fi

if [ "${CONTEXT_SWEEP:-0}" = 1 ]; then
  log "long-context sweep"
  exec "$HELPER/context-sweep.sh" --engine "$ENGINE" --variant "$VARIANT" \
    --result-dir "$RESULT_DIR"
fi

if [ "$SERVE" = 1 ]; then
  log "serve"
  "$HELPER/serve.sh" --engine "$ENGINE" --variant "$VARIANT" --background --wait
fi

if [ "$BENCH" = 1 ]; then
  log "benchmark"
  "$HELPER/bench.sh" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "$SERVED_MODEL_NAME" \
    --concurrency "$BENCH_CONCURRENCY" \
    --isl "$BENCH_ISL" --osl "$BENCH_OSL" \
    --runs "$BENCH_RUNS" --workers "$BENCH_WORKERS" \
    --min-prompts "$BENCH_MIN_PROMPTS" --max-prompts "$BENCH_MAX_PROMPTS" \
    --shared-prefix "$BENCH_SHARED_PREFIX" \
    --cache-regime "$BENCH_CACHE_REGIME" \
    --output-dir "$RESULT_DIR"
fi

if [ "${KEEP_ALIVE:-0}" = 1 ]; then
  log "server stays up until the Slurm time limit; pid $(cat "$RUN_DIR/server.pid" 2>/dev/null || echo '?')"
  trap - EXIT
  while true; do sleep 3600; done
fi

cat > "$RESULT_DIR/job.json" <<EOF
{
  "host": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "engine": "$ENGINE",
  "variant": "$VARIANT",
  "tp": $TP,
  "dp": $DP,
  "max_len": $MAX_LEN,
  "max_running": $MAX_RUNNING,
  "max_prefill": $MAX_PREFILL,
  "chunked_prefill": $CHUNKED_PREFILL,
  "kv_dtype": "$KV_DTYPE",
  "mem_fraction": $MEM_FRACTION,
  "schedule_policy": "$SCHEDULE_POLICY",
  "bench_concurrency": "$BENCH_CONCURRENCY",
  "bench_isl": $BENCH_ISL,
  "bench_osl": $BENCH_OSL,
  "bench_runs": $BENCH_RUNS,
  "result_dir": "$RESULT_DIR"
}
EOF

log "done"
