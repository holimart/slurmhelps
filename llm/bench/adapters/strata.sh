#!/usr/bin/env bash
# strata.sh — thin adapter: recipe flags -> llm/strata/{strata-bench,knob-sweep,warm-cache}.sbatch
#
# The runners are NOT modified here. This file only chooses the right entry
# point, pins --nice=100, passes resources on the sbatch command line (never
# through --export, which splits comma lists) and prints the exact command for
# --dry-run.
#
# usage: strata.sh [--dry-run] --runner matrix|knobs|warm --model iq3s|coder|q2_0
#                  [--cells ctx:C,...] [--warm-cells ...] [--warm-repeats N]
#                  [--contexts LIST] [--batches LIST] [--variants LIST]
#                  [--max-context N] [--max-tokens N] [--requests-per-slot N]
#                  [--gpus 0,1] [--node NAME] [--outdir DIR] [--budget-min N]
#                  [--partition P] [--nodelist NODE] [--gres gpu:N]
#                  [--cpus N] [--mem 256G] [--time 11:30:00] [--account ACCT]
#                  [--self-test]
#
# --self-test runs the runner's stub-server test directly (no Slurm, no GPU)
# and is the only mode allowed on the login node.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
HELPER=${STRATAHELPER:-$ROOT/llm/strata}

RUNNER=""
MODEL=""
CELLS=""
WARM_CELLS=""
WARM_REPEATS=""
CONTEXTS=""
BATCHES=""
VARIANTS=""
MAX_CONTEXT=""
MAX_TOKENS=""
RPS=""
GPUS=""
NODE=""
OUTDIR=""
BUDGET_MIN=""
PARTITION=""
NODELIST=""
GRES=""
CPUS=""
MEM=""
TIME=""
ACCOUNT=""
SELF_TEST=0
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --runner) RUNNER=${2:?--runner needs matrix|knobs|warm}; shift 2 ;;
    --model) MODEL=${2:?--model needs a pack}; shift 2 ;;
    --cells) CELLS=${2:?}; shift 2 ;;
    --warm-cells) WARM_CELLS=${2:?}; shift 2 ;;
    --warm-repeats) WARM_REPEATS=${2:?}; shift 2 ;;
    --contexts) CONTEXTS=${2:?}; shift 2 ;;
    --batches) BATCHES=${2:?}; shift 2 ;;
    --variants) VARIANTS=${2:?}; shift 2 ;;
    --max-context) MAX_CONTEXT=${2:?}; shift 2 ;;
    --max-tokens) MAX_TOKENS=${2:?}; shift 2 ;;
    --requests-per-slot) RPS=${2:?}; shift 2 ;;
    --gpus) GPUS=${2:?}; shift 2 ;;
    --node) NODE=${2:?}; shift 2 ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --partition) PARTITION=${2:?}; shift 2 ;;
    --nodelist) NODELIST=${2:?}; shift 2 ;;
    --gres) GRES=${2:?}; shift 2 ;;
    --cpus) CPUS=${2:?}; shift 2 ;;
    --mem) MEM=${2:?}; shift 2 ;;
    --time) TIME=${2:?}; shift 2 ;;
    --account) ACCOUNT=${2:?}; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "strata.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$RUNNER" ] || { echo "strata.sh: --runner is required" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "strata.sh: --model is required" >&2; exit 2; }

runner_args=()
case "$RUNNER" in
  matrix)
    SBATCH=$HELPER/strata-bench.sbatch
    RUNNER_SH=$HELPER/bench-matrix.sh
    runner_args+=(--model "$MODEL")
    [ -n "$CELLS" ] && runner_args+=(--cells "$CELLS")
    [ -n "$WARM_CELLS" ] && runner_args+=(--warm-cells "$WARM_CELLS")
    [ -n "$WARM_REPEATS" ] && runner_args+=(--warm-repeats "$WARM_REPEATS")
    [ -n "$MAX_TOKENS" ] && runner_args+=(--max-tokens "$MAX_TOKENS")
    [ -n "$RPS" ] && runner_args+=(--requests-per-slot "$RPS")
    ;;
  knobs)
    SBATCH=$HELPER/knob-sweep.sbatch
    RUNNER_SH=$HELPER/knob-sweep.sh
    runner_args+=(--model "$MODEL")
    [ -n "$NODE" ] && runner_args+=(--node "$NODE")
    [ -n "$CONTEXTS" ] && runner_args+=(--contexts "$CONTEXTS")
    [ -n "$BATCHES" ] && runner_args+=(--batches "$BATCHES")
    [ -n "$VARIANTS" ] && runner_args+=(--variants "$VARIANTS")
    [ -n "$MAX_CONTEXT" ] && runner_args+=(--max-context "$MAX_CONTEXT")
    ;;
  warm)
    SBATCH=$HELPER/warm-cache.sbatch
    RUNNER_SH=$HELPER/warm-cache.sh
    runner_args+=(--model "$MODEL")
    [ -n "$CELLS" ] && runner_args+=(--cells "$CELLS")
    [ -n "$WARM_REPEATS" ] && runner_args+=(--warm-repeats "$WARM_REPEATS")
    [ -n "$MAX_CONTEXT" ] && runner_args+=(--max-context "$MAX_CONTEXT")
    [ -n "$MAX_TOKENS" ] && runner_args+=(--max-tokens "$MAX_TOKENS")
    [ -n "$NODE" ] && runner_args+=(--node "$NODE")
    ;;
  *) echo "strata.sh: --runner must be matrix|knobs|warm" >&2; exit 2 ;;
esac

[ -n "$GPUS" ] && runner_args+=(--gpus "$GPUS")
[ -n "$BUDGET_MIN" ] && runner_args+=(--budget-min "$BUDGET_MIN")
[ -n "$OUTDIR" ] && runner_args+=(--outdir "$OUTDIR")

if [ "$SELF_TEST" = 1 ]; then
  cmd=(bash "$RUNNER_SH" --self-test)
  if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
  exec "${cmd[@]}"
fi

sbatch_args=(sbatch --nice=100)
[ -n "$PARTITION" ] && sbatch_args+=(--partition "$PARTITION")
[ -n "$NODELIST" ] && sbatch_args+=(--nodelist "$NODELIST")
[ -n "$GRES" ] && sbatch_args+=(--gres "$GRES")
[ -n "$CPUS" ] && sbatch_args+=(--cpus-per-task "$CPUS")
[ -n "$MEM" ] && sbatch_args+=(--mem "$MEM")
[ -n "$TIME" ] && sbatch_args+=(--time "$TIME")
[ -n "$ACCOUNT" ] && sbatch_args+=(--account "$ACCOUNT")

cmd=("${sbatch_args[@]}" "$SBATCH" "${runner_args[@]}")
if [ "$DRY" = 1 ]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || { echo "strata.sh: sbatch not found" >&2; exit 127; }
exec "${cmd[@]}"
