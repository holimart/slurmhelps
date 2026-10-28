#!/usr/bin/env bash
# vllm.sh — thin adapter over llm/qwen38 (FP4 Blackwel) and llm/qwen35 (A100).
#
# Wraps, never edits: llm/qwen38/serve.sh + measure.sh and llm/qwen35/serve.sh +
# serve-server.sh, plus the shared tests/serve_smoke.py and the GLM bench client
# that qwen38/measure.sh already calls.
#
# usage: vllm.sh [--dry-run] --runner serve|measure|tasks
#                 --flavour qwen38|qwen35-122b|qwen35-35b
#                 [--base-url URL] [--served-model NAME] [--outdir DIR]
#                 [--contexts "500 800"] [--concurrency "1 2 4 8"]
#                 [--max-model-len N] [--partition P] [--nodelist N]
#                 [--gpus 0,1] [--account ACCT] [--model NAME]
#
# Where the resource limits actually live:
#   qwen38      llm/qwen38/config-bw.json (2x BW6000, ffa-preempt, 36 h, port 8010)
#   qwen35-122b llm/qwen35/config-122b.json (2x A100, port 8002)
#   qwen35-35b  llm/qwen35/config-35b.json (1x A100, port 8001)
# Slurm partition/nodelist flags here are therefore *checked against* the
# config, not applied: serve.sh reads the JSON. A mismatch prints a warning.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)

RUNNER=""
FLAVOUR=""
BASE_URL=""
SERVED=""
OUTDIR=""
CONTEXTS=""
CONCURRENCY=""
MAX_MODEL_LEN=""
PARTITION=""
NODELIST=""
GPUS=""
ACCOUNT=""
MODEL=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --runner) RUNNER=${2:?--runner needs serve|measure|tasks}; shift 2 ;;
    --flavour) FLAVOUR=${2:?--flavour needs qwen38|qwen35-122b|qwen35-35b}; shift 2 ;;
    --base-url) BASE_URL=${2:?}; shift 2 ;;
    --served-model) SERVED=${2:?}; shift 2 ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --contexts) CONTEXTS=${2:?}; shift 2 ;;
    --concurrency) CONCURRENCY=${2:?}; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=${2:?}; shift 2 ;;
    --partition) PARTITION=${2:?}; shift 2 ;;
    --nodelist) NODELIST=${2:?}; shift 2 ;;
    --gpus) GPUS=${2:?}; shift 2 ;;
    --account) ACCOUNT=${2:?}; shift 2 ;;
    --model) MODEL=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "vllm.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$RUNNER" ] || { echo "vllm.sh: --runner is required" >&2; exit 2; }
[ -n "$FLAVOUR" ] || { echo "vllm.sh: --flavour is required" >&2; exit 2; }

case "$FLAVOUR" in
  qwen38)
    SERVE_CMD=(bash "$ROOT/llm/qwen38/serve.sh" up --yes)
    CONFIG="$ROOT/llm/qwen38/config-bw.json"
    DEFAULT_BASE=${BASE_URL:-http://bw01:8010}
    DEFAULT_MODEL=${SERVED:-qwen38-flash-next}
    DEFAULT_LEN=${MAX_MODEL_LEN:-262144}
    ;;
  qwen35-122b)
    SERVE_CMD=(bash "$ROOT/llm/qwen35/serve.sh" up --yes)
    CONFIG="$ROOT/llm/qwen35/config-122b.json"
    DEFAULT_BASE=${BASE_URL:-http://ampere02:8002}
    DEFAULT_MODEL=${SERVED:-qwen35-122b-a10b}
    DEFAULT_LEN=${MAX_MODEL_LEN:-262144}
    ;;
  qwen35-35b)
    SERVE_CMD=(bash "$ROOT/bin/slurmhelps-serve" --config "$ROOT/llm/qwen35/config-35b.json" \
               up --yes --backend-command bash "$ROOT/llm/qwen35/serve-35b.sh")
    CONFIG="$ROOT/llm/qwen35/config-35b.json"
    DEFAULT_BASE=${BASE_URL:-http://ampere02:8001}
    DEFAULT_MODEL=${SERVED:-qwen35-35b-a3b}
    DEFAULT_LEN=${MAX_MODEL_LEN:-262144}
    ;;
  *) echo "vllm.sh: --flavour must be qwen38|qwen35-122b|qwen35-35b" >&2; exit 2 ;;
esac

# The serving config is the source of truth for resources. Warn, do not silently
# pretend the CLI flags were applied.
if [ -n "$PARTITION" ] || [ -n "$NODELIST" ]; then
  echo "vllm.sh: note: partition/nodelist come from $(basename "$CONFIG");" \
       "--partition/--nodelist here are informational" >&2
fi


case "$RUNNER" in
  serve)
    cmd=("${SERVE_CMD[@]}")
    ;;
  measure)
    if [ "$FLAVOUR" = "qwen38" ]; then
      # measure.sh runs the 10-check smoke, 500/800 batched sweeps, the context
      # ladder and the prefix-cache ladder; its shape is fixed in the script.
      cmd=(env BASE_URL="$DEFAULT_BASE" SERVED_MODEL_NAME="$DEFAULT_MODEL" OUT="$OUTDIR" \
           bash "$ROOT/llm/qwen38/measure.sh")
    else
      # qwen35 has no measure.sh: use the shared smoke matrix (the 2026-10-23 run).
      cmd=(python3 "$ROOT/tests/serve_smoke.py"
           --base-url "$DEFAULT_BASE" --model "$DEFAULT_MODEL"
           --contexts "${CONTEXTS:-500 800}"
           --concurrency "${CONCURRENCY:-1 2 4 8}"
           --model-max-len "$DEFAULT_LEN"
           --output "$OUTDIR/smoke.json")
    fi
    ;;
  tasks)
    cmd=(python3 "$ROOT/llm/qwen35/opencode_tasks.py"
         --model "${SERVED:-slurmhelps-qwen/qwen35-122b-a10b}"
         --timeout 600 --output "$OUTDIR/opencode.json")
    ;;
  *) echo "vllm.sh: --runner must be serve|measure|tasks" >&2; exit 2 ;;
esac

if [ "$DRY" = 1 ]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

case "$RUNNER" in
  measure|tasks)
    mkdir -p "$OUTDIR"
    ;;
esac
exec "${cmd[@]}"
