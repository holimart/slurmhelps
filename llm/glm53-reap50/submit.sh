#!/usr/bin/env bash
# Submit the GLM-5.3-Flash REAP50 install + serve + benchmark job.
#
# usage: submit.sh [--node NODE] [--partition PART] [--gres gpu:2]
#                  [--gpus N] [--cpus N] [--mem 256G] [--time 12:00:00]
#                  [--account ACCT] [--engine sglang|vllm|llamacpp]
#                  [--variant nvfp4|fp8|q4_k_s|q4_k_m|iq4_xs|iq3_m]
#                  [--tp N] [--port N] [--concurrency "1 2 4 8"]
#                  [--isl N] [--osl N] [--runs N] [--workers N]
#                  [--no-install] [--no-bench] [--serve-only]
#                  [--wait] [--dry-run]
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

get_cfg() { python3 "$ROOT/lib/slurmhelps/config.py" get "$1" 2>/dev/null || true; }

PARTITION=${PARTITION:-$(get_cfg site.partition_long)}
ACCOUNT=${ACCOUNT:-$(get_cfg cluster.account)}
NODE=${NODE:-}
GPUS=${GPUS:-2}
GRES=${GRES:-gpu:${GPUS}}
CPUS=${CPUS:-32}
MEM=${MEM:-256G}
TIME=${TIME:-12:00:00}
INSTALL=${INSTALL:-1}
BENCH=${BENCH:-1}
KEEP_ALIVE=${KEEP_ALIVE:-0}
WAIT=0
DRY_RUN=0
CONCURRENCY=$BENCH_CONCURRENCY
ISL=$BENCH_ISL
OSL=$BENCH_OSL
RUNS=$BENCH_RUNS
WORKERS=$BENCH_WORKERS
MIN_PROMPTS=$BENCH_MIN_PROMPTS
MAX_PROMPTS=$BENCH_MAX_PROMPTS
SHARED_PREFIX=$BENCH_SHARED_PREFIX
CACHE_REGIME=$BENCH_CACHE_REGIME
CONTEXT_SWEEP=${CONTEXT_SWEEP:-0}
CONTEXTS=${CONTEXTS:-4096:8+16384:4+65536:2+262144:1+524288:1+1048576:1}
SWEEP_OSL=${SWEEP_OSL:-128}
SWEEP_BUDGET_MIN=${SWEEP_BUDGET_MIN:-55}
SWEEP_TIMEOUT_MULT=${SWEEP_TIMEOUT_MULT:-1}

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --node)         NODE=$2; shift 2 ;;
    --partition)    PARTITION=$2; shift 2 ;;
    --gres)         GRES=$2; shift 2 ;;
    --gpus)         GPUS=$2; GRES=gpu:${GPUS}; shift 2 ;;
    --cpus)         CPUS=$2; shift 2 ;;
    --mem)          MEM=$2; shift 2 ;;
    --time)         TIME=$2; shift 2 ;;
    --account)      ACCOUNT=$2; shift 2 ;;
    --model-root)   MODEL_ROOT=$2; shift 2 ;;
    --engine)       ENGINE=$2; shift 2 ;;
    --variant)      VARIANT=$2; shift 2 ;;
    --tp)           TP=$2; shift 2 ;;
    --port)         PORT=$2; shift 2 ;;
    --concurrency)  CONCURRENCY=$2; shift 2 ;;
    --isl)          ISL=$2; shift 2 ;;
    --osl)          OSL=$2; shift 2 ;;
    --runs)         RUNS=$2; shift 2 ;;
    --workers)      WORKERS=$2; shift 2 ;;
    --min-prompts)  MIN_PROMPTS=$2; shift 2 ;;
    --max-prompts)  MAX_PROMPTS=$2; shift 2 ;;
    --quick)        CONCURRENCY="1 2 4 8"; RUNS=1; ISL=1024; OSL=128
                    MIN_PROMPTS=32; MAX_PROMPTS=128; shift ;;
    --context-sweep) CONTEXT_SWEEP=1; shift ;;
    --contexts)     CONTEXTS=$2; CONTEXTS=${CONTEXTS// /+}; shift 2 ;;
    --sweep-osl)    SWEEP_OSL=$2; shift 2 ;;
    --budget-min)   SWEEP_BUDGET_MIN=$2; shift 2 ;;
    --timeout-mult) SWEEP_TIMEOUT_MULT=$2; shift 2 ;;
    --shared-prefix) SHARED_PREFIX=$2; shift 2 ;;
    --cache-regime) CACHE_REGIME=$2; shift 2 ;;
    --no-install)   INSTALL=0; shift ;;
    --no-bench)     BENCH=0; shift ;;
    --serve-only)   BENCH=0; KEEP_ALIVE=1; shift ;;
    --keep-alive)   KEEP_ALIVE=1; shift ;;
    --wait)         WAIT=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

CONTEXTS=${CONTEXTS// /+}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=${RESULT_DIR:-$RUN_DIR/$STAMP}
mkdir -p "$RESULT_DIR"

# Slurm --export is comma-separated, so the concurrency list must not contain
# commas. Encode it with '+' and let run-job.sh / bench.py expand it.
CONC_EXPORT=$(printf '%s' "$CONCURRENCY" | tr ' ,' '++')
EXPORTS="ALL,ENGINE=$ENGINE,VARIANT=$VARIANT,INSTALL=$INSTALL,BENCH=$BENCH,KEEP_ALIVE=$KEEP_ALIVE"
EXPORTS+=",WORK_ROOT=$WORK_ROOT,MODEL_ROOT=$MODEL_ROOT,RESULT_DIR=$RESULT_DIR"
EXPORTS+=",PORT=$PORT,TP=$TP,DP=$DP,SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
EXPORTS+=",MAX_LEN=$MAX_LEN,MAX_RUNNING=$MAX_RUNNING,MAX_PREFILL=$MAX_PREFILL"
EXPORTS+=",CHUNKED_PREFILL=$CHUNKED_PREFILL,KV_DTYPE=$KV_DTYPE"
EXPORTS+=",MEM_FRACTION=$MEM_FRACTION,GPU_MEM_UTIL=$GPU_MEM_UTIL"
EXPORTS+=",SCHEDULE_POLICY=$SCHEDULE_POLICY,CUDA_GRAPH_BS=$CUDA_GRAPH_BS"
EXPORTS+=",BENCH_CONCURRENCY=$CONC_EXPORT,BENCH_ISL=$ISL,BENCH_OSL=$OSL"
EXPORTS+=",BENCH_RUNS=$RUNS,BENCH_WORKERS=$WORKERS"
EXPORTS+=",BENCH_MIN_PROMPTS=$MIN_PROMPTS,BENCH_MAX_PROMPTS=$MAX_PROMPTS"
EXPORTS+=",BENCH_SHARED_PREFIX=$SHARED_PREFIX,BENCH_CACHE_REGIME=$CACHE_REGIME"
EXPORTS+=",CONTEXT_SWEEP=$CONTEXT_SWEEP,CONTEXTS=$CONTEXTS"
EXPORTS+=",SWEEP_OSL=$SWEEP_OSL,SWEEP_BUDGET_MIN=$SWEEP_BUDGET_MIN"
EXPORTS+=",SWEEP_TIMEOUT_MULT=$SWEEP_TIMEOUT_MULT"
EXPORTS+=",LLAMA_CUDA_HOME=${LLAMA_CUDA_HOME:-}"
EXPORTS+=",GLM53_HELPER=$HELPER,SLURMHELPS_ROOT=$ROOT"

args=(--parsable
  --job-name="glm53-${ENGINE}-${VARIANT}"
  --nodes=1 --ntasks=1
  --gres="$GRES"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
  --output="$RESULT_DIR/slurm-%j.out"
  --export="$EXPORTS")
if [ -n "$ACCOUNT" ]; then args+=(--account "$ACCOUNT"); fi
if [ -n "$PARTITION" ]; then args+=(--partition "$PARTITION"); fi
if [ -n "$NODE" ]; then args+=(--nodelist "$NODE"); fi
if [ "$WAIT" = 1 ]; then args+=(--wait); fi

printf '[submit] result dir: %s\n' "$RESULT_DIR"
printf '[submit] sbatch'; printf ' %q' "${args[@]}"; printf ' %q\n' "$HELPER/glm53-reap50.sbatch"
if [ "$DRY_RUN" = 1 ]; then exit 0; fi

job=$(sbatch "${args[@]}" "$HELPER/glm53-reap50.sbatch")
printf '[submit] job %s\n' "$job"
printf '[submit] follow: squeue -j %s ; tail -f %s/slurm-%s.out\n' "$job" "$RESULT_DIR" "$job"
