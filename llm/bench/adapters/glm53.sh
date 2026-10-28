#!/usr/bin/env bash
# glm53.sh — thin adapter over llm/glm53-reap50 (SGLang/vLLM/llama.cpp).
#
# Wraps: bench.sh, context-sweep.sh, cache-sweep.sh, submit.sh. The scripts
# keep their own defaults from config.env; this adapter only forwards flags and
# sets WORK_ROOT / RESULT_DIR.
#
# usage: glm53.sh [--dry-run] --runner submit|bench|context|cache
#                 [--engine sglang|vllm|llamacpp] [--variant nvfp4|fp8|q4_k_s|...]
#                 [--concurrency "1 2 4 8"] [--isl N] [--osl N] [--runs N]
#                 [--contexts "4096:8 16384:4"] [--prefixes "100000 200000"]
#                 [--suffix N] [--ctx N] [--base-url URL] [--served-model NAME]
#                 [--outdir DIR] [--budget-min N] [--context-sweep] [--cache-sweep]
#                 [--node NODE] [--partition P] [--gres gpu:2] [--cpus N]
#                 [--mem 256G] [--time 12:00:00] [--account ACCT]
#
# `submit` creates a Slurm job (submit.sh owns the --export encoding; never put
# a comma list in --export by hand). `context`/`cache` start their own server,
# so they belong inside an allocation, not on the login node.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
HELPER="$ROOT/llm/glm53-reap50"

RUNNER=""
ENGINE=""
VARIANT=""
CONCURRENCY=""
ISL=""
OSL=""
RUNS=""
CONTEXTS=""
PREFIXES=""
SUFFIX=""
CTX=""
BASE_URL=""
SERVED=""
OUTDIR=""
BUDGET_MIN=""
CONTEXT_SWEEP=0
CACHE_SWEEP=0
NODE=""
PARTITION=""
GRES=""
CPUS=""
MEM=""
TIME=""
ACCOUNT=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --runner) RUNNER=${2:?--runner needs submit|bench|context|cache}; shift 2 ;;
    --engine) ENGINE=${2:?}; shift 2 ;;
    --variant) VARIANT=${2:?}; shift 2 ;;
    --concurrency) CONCURRENCY=${2:?}; shift 2 ;;
    --isl) ISL=${2:?}; shift 2 ;;
    --osl) OSL=${2:?}; shift 2 ;;
    --runs) RUNS=${2:?}; shift 2 ;;
    --contexts) CONTEXTS=${2:?}; shift 2 ;;
    --prefixes) PREFIXES=${2:?}; shift 2 ;;
    --suffix) SUFFIX=${2:?}; shift 2 ;;
    --ctx) CTX=${2:?}; shift 2 ;;
    --base-url) BASE_URL=${2:?}; shift 2 ;;
    --served-model) SERVED=${2:?}; shift 2 ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --context-sweep) CONTEXT_SWEEP=1; shift ;;
    --cache-sweep) CACHE_SWEEP=1; shift ;;
    --node) NODE=${2:?}; shift 2 ;;
    --partition) PARTITION=${2:?}; shift 2 ;;
    --gres) GRES=${2:?}; shift 2 ;;
    --cpus) CPUS=${2:?}; shift 2 ;;
    --mem) MEM=${2:?}; shift 2 ;;
    --time) TIME=${2:?}; shift 2 ;;
    --account) ACCOUNT=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "glm53.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$RUNNER" ] || { echo "glm53.sh: --runner is required" >&2; exit 2; }
[ -n "${WORK_ROOT:-}" ] || WORK_ROOT="$(cd "$ROOT/.." && pwd)"

env_args=()
[ -n "$OUTDIR" ] && env_args+=(RESULT_DIR="$OUTDIR")
env_args+=(WORK_ROOT="$WORK_ROOT")

case "$RUNNER" in
  submit)
    cmd=(bash "$HELPER/submit.sh")
    [ -n "$NODE" ] && cmd+=(--node "$NODE")
    [ -n "$PARTITION" ] && cmd+=(--partition "$PARTITION")
    [ -n "$GRES" ] && cmd+=(--gres "$GRES")
    [ -n "$CPUS" ] && cmd+=(--cpus "$CPUS")
    [ -n "$MEM" ] && cmd+=(--mem "$MEM")
    [ -n "$TIME" ] && cmd+=(--time "$TIME")
    [ -n "$ACCOUNT" ] && cmd+=(--account "$ACCOUNT")
    [ -n "$ENGINE" ] && cmd+=(--engine "$ENGINE")
    [ -n "$VARIANT" ] && cmd+=(--variant "$VARIANT")
    [ -n "$CONCURRENCY" ] && cmd+=(--concurrency "$CONCURRENCY")
    [ -n "$ISL" ] && cmd+=(--isl "$ISL")
    [ -n "$OSL" ] && cmd+=(--osl "$OSL")
    [ -n "$RUNS" ] && cmd+=(--runs "$RUNS")
    [ -n "$CONTEXTS" ] && cmd+=(--contexts "$CONTEXTS")
    [ -n "$PREFIXES" ] && cmd+=(--cache-prefixes "$PREFIXES")
    [ -n "$SUFFIX" ] && cmd+=(--cache-suffix "$SUFFIX")
    [ -n "$CTX" ] && cmd+=(--cache-ctx "$CTX")
    [ -n "$BUDGET_MIN" ] && cmd+=(--budget-min "$BUDGET_MIN")
    [ "$CONTEXT_SWEEP" = 1 ] && cmd+=(--context-sweep)
    [ "$CACHE_SWEEP" = 1 ] && cmd+=(--cache-sweep)
    # submit.sh has its own --dry-run (prints the sbatch line, creates the result dir).
    [ "$DRY" = 1 ] && cmd+=(--dry-run)
    ;;
  context)
    cmd=(bash "$HELPER/context-sweep.sh")
    [ -n "$ENGINE" ] && cmd+=(--engine "$ENGINE")
    [ -n "$VARIANT" ] && cmd+=(--variant "$VARIANT")
    [ -n "$CONTEXTS" ] && cmd+=(--contexts "$CONTEXTS")
    [ -n "$OSL" ] && cmd+=(--osl "$OSL")
    [ -n "$BUDGET_MIN" ] && cmd+=(--budget-min "$BUDGET_MIN")
    [ -n "$OUTDIR" ] && cmd+=(--result-dir "$OUTDIR")
    ;;
  cache)
    cmd=(bash "$HELPER/cache-sweep.sh")
    [ -n "$ENGINE" ] && cmd+=(--engine "$ENGINE")
    [ -n "$VARIANT" ] && cmd+=(--variant "$VARIANT")
    [ -n "$PREFIXES" ] && cmd+=(--prefixes "$PREFIXES")
    [ -n "$SUFFIX" ] && cmd+=(--suffix "$SUFFIX")
    [ -n "$OSL" ] && cmd+=(--osl "$OSL")
    [ -n "$CTX" ] && cmd+=(--ctx "$CTX")
    [ -n "$OUTDIR" ] && cmd+=(--result-dir "$OUTDIR")
    ;;
  bench)
    cmd=(bash "$HELPER/bench.sh")
    [ -n "$BASE_URL" ] && cmd+=(--base-url "$BASE_URL")
    [ -n "$SERVED" ] && cmd+=(--model "$SERVED")
    [ -n "$CONCURRENCY" ] && cmd+=(--concurrency "$CONCURRENCY")
    [ -n "$ISL" ] && cmd+=(--isl "$ISL")
    [ -n "$OSL" ] && cmd+=(--osl "$OSL")
    [ -n "$RUNS" ] && cmd+=(--runs "$RUNS")
    [ -n "$OUTDIR" ] && cmd+=(--output-dir "$OUTDIR")
    ;;
  *) echo "glm53.sh: --runner must be submit|bench|context|cache" >&2; exit 2 ;;
esac

full=(env "${env_args[@]}" "${cmd[@]}")
if [ "$DRY" = 1 ] && [ "$RUNNER" != "submit" ]; then
  printf '%q ' "${full[@]}"; printf '\n'
  exit 0
fi
if [ "$DRY" = 1 ] && [ "$RUNNER" = "submit" ]; then
  # submit.sh --dry-run prints the exact sbatch invocation without submitting.
  printf '%q ' "${full[@]}"; printf '\n'
  echo "glm53.sh: run the line above to see submit.sh's exact sbatch command" >&2
  exit 0
fi
exec "${full[@]}"
