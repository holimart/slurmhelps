#!/usr/bin/env bash
# Start (or stop) the GLM-5.3-Flash REAP50 server.
#
# usage: serve.sh [--engine sglang|vllm|llamacpp] [--variant nvfp4|fp8|gguf]
#                 [--model PATH] [--port N] [--tp N] [--dp N]
#                 [--max-len N] [--max-running N] [--max-prefill N]
#                 [--background] [--wait] [--stop] [--dry-run]
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

log()  { printf '[serve] %s\n' "$*"; }
die()  { printf '[serve] ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '+ %s\n' "$*"; else "$@"; fi; }

DRY_RUN=0
BACKGROUND=0
WAIT=0
STOP=0
MODEL_OVERRIDE=
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)      ENGINE=$2; shift 2 ;;
    --variant)     VARIANT=$2; shift 2 ;;
    --model)       MODEL_OVERRIDE=$2; shift 2 ;;
    --port)        PORT=$2; shift 2 ;;
    --tp)          TP=$2; shift 2 ;;
    --dp)          DP=$2; shift 2 ;;
    --max-len)     MAX_LEN=$2; shift 2 ;;
    --max-running) MAX_RUNNING=$2; shift 2 ;;
    --max-prefill) MAX_PREFILL=$2; shift 2 ;;
    --kv-dtype)    KV_DTYPE=$2; shift 2 ;;
    --background)  BACKGROUND=1; shift ;;
    --wait)        WAIT=1; shift ;;
    --stop)        STOP=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

mkdir -p "$RUN_DIR"

if [ "$STOP" = 1 ]; then
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      log "stopping pid $pid"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  else
    log "no pid file at $PID_FILE"
  fi
  exit 0
fi

case "$VARIANT" in
  nvfp4) MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/nvfp4} ;;
  fp8)   MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/fp8} ;;
  q4_k_s) MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/gguf/GLM-5.3-Flash-REAP50-Q4_K_S.gguf} ;;
  q4_k_m) MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/gguf/GLM-5.3-Flash-REAP50-Q4_K_M.gguf} ;;
  iq4_xs) MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/gguf/GLM-5.3-Flash-REAP50-IQ4_XS.gguf} ;;
  iq3_m)  MODEL=${MODEL_OVERRIDE:-$MODEL_ROOT/gguf/GLM-5.3-Flash-REAP50-IQ3_M.gguf} ;;
  *) die "unknown variant: $VARIANT" ;;
esac

case "$ENGINE" in
  sglang)   : "${PORT:=30000}" ;;
  vllm)     : "${PORT:=8000}" ;;
  llamacpp) : "${PORT:=8080}" ;;
  *) die "engine must be sglang, vllm or llamacpp" ;;
esac

if [ ! -e "$MODEL" ] && [ "$DRY_RUN" = 0 ]; then
  die "model path not found: $MODEL (run install.sh first)"
fi

bind_args=()
if [ "$MODEL_ROOT" != "$WORK_ROOT" ] && [[ "$MODEL_ROOT" != "$WORK_ROOT/"* ]]; then
  bind_args+=(--bind "$MODEL_ROOT:$MODEL_ROOT")
fi

cmd=()
case "$ENGINE" in
  sglang)
    if [ -x "$VENV_DIR/sglang/bin/python" ]; then
      cmd=("$VENV_DIR/sglang/bin/python" -m sglang.launch_server)
    else
      if [ ! -f "$SGLANG_SIF" ] && [ "$DRY_RUN" = 0 ]; then die "container missing: $SGLANG_SIF (run install.sh)"; fi
      cmd=(python3 -m sglang.launch_server)
    fi
    cmd+=(--model-path "$MODEL" --host "$HOST" --port "$PORT"
      --served-model-name "$SERVED_MODEL_NAME"
      --tp "$TP" --mem-fraction-static "$MEM_FRACTION"
      --max-running-requests "$MAX_RUNNING"
      --max-prefill-tokens "$MAX_PREFILL"
      --chunked-prefill-size "$CHUNKED_PREFILL"
      --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
      --schedule-policy "$SCHEDULE_POLICY"
      --kv-cache-dtype "$KV_DTYPE")
    if [ "$DP" -gt 1 ]; then cmd+=(--dp "$DP"); fi
    if [ "$ENABLE_METRICS" = 1 ]; then cmd+=(--enable-metrics); fi
    if [ "$ENABLE_CACHE_REPORT" = 1 ]; then cmd+=(--enable-cache-report); fi
    if [ "$ENABLE_PREFIX_CACHE" != 1 ]; then cmd+=(--disable-radix-cache); fi
    if [ "${HICACHE:-0}" = 1 ]; then cmd+=(--enable-hierarchical-cache --hicache-size "${HICACHE_SIZE:-180}"); fi
    if [ ! -x "$VENV_DIR/sglang/bin/python" ]; then
      cmd=(apptainer exec --nv --bind "$WORK_ROOT:$WORK_ROOT" "${bind_args[@]}" "$SGLANG_SIF" "${cmd[@]}")
    fi
    ;;
  vllm)
    if [ -x "$VENV_DIR/vllm/bin/vllm" ]; then
      cmd=("$VENV_DIR/vllm/bin/vllm" serve)
    else
      if [ ! -f "$VLLM_SIF" ] && [ "$DRY_RUN" = 0 ]; then die "container missing: $VLLM_SIF (run install.sh)"; fi
      cmd=(vllm serve)
    fi
    cmd+=("$MODEL" --host "$HOST" --port "$PORT"
      --served-model-name "$SERVED_MODEL_NAME"
      --tensor-parallel-size "$TP"
      --max-model-len "$MAX_LEN"
      --max-num-seqs "$MAX_RUNNING"
      --max-num-batched-tokens "$MAX_PREFILL"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --enable-chunked-prefill --enable-prefix-caching
      --enable-prompt-tokens-details
      --kv-cache-dtype "$KV_DTYPE")
    if [ "$DP" -gt 1 ]; then cmd+=(--data-parallel-size "$DP"); fi
    if [ "${EP:-0}" = 1 ]; then cmd+=(--enable-expert-parallel); fi
    if [ -n "${VLLM_ATTENTION_BACKEND:-}" ]; then cmd+=(--attention-backend "$VLLM_ATTENTION_BACKEND"); fi
    if [ ! -x "$VENV_DIR/vllm/bin/vllm" ]; then
      cmd=(apptainer exec --nv --bind "$WORK_ROOT:$WORK_ROOT" "${bind_args[@]}" "$VLLM_SIF" "${cmd[@]}")
    fi
    ;;
  llamacpp)
    if [ ! -f "$LLAMA_SERVER" ] && [ "$DRY_RUN" = 0 ]; then die "llama-server missing: $LLAMA_SERVER (run install.sh)"; fi
    case "$MODEL" in *.gguf) ;; *) die "llamacpp needs a .gguf model, got $MODEL" ;; esac
    # The binary may be linked against a toolkit different from /usr/local/cuda
    # (V100 needs 12.5 for sm_70; L40 was built with 12.5 for CCCL reasons).
    if [ -n "${LLAMA_CUDA_HOME:-}" ] && [ -d "$LLAMA_CUDA_HOME/lib64" ]; then
      export LD_LIBRARY_PATH="$LLAMA_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
    cmd=("$LLAMA_SERVER" -m "$MODEL" --host "$HOST" --port "$PORT"
      -ngl 99 --split-mode layer
      -np "$MAX_RUNNING" -c "$CTX_TOTAL"
      -b "$LLAMA_BATCH" -ub "$LLAMA_UBATCH"
      -fa on -ctk "$LLAMA_KV_TYPE" -ctv "$LLAMA_KV_TYPE"
      --cache-prompt --cache-reuse "$LLAMA_CACHE_REUSE")
    # NFS mmap page-faults ~10 MB/s here; sequential --no-mmap is much faster
    # and the 82 GiB fits in the job's 256 GiB RAM.
    if [ "${LLAMA_MMAP:-0}" = 0 ]; then cmd+=(--no-mmap); fi
    if [ "${LLAMA_PROMPT_CACHE:-1}" = 0 ]; then cmd+=(--no-cache-prompt); fi
    cmd+=(
      --metrics
      --override-kv tokenizer.ggml.eot_token_id=int:154827
      --override-kv tokenizer.ggml.eom_token_id=int:154829)
    if [ "${LLAMA_JINJA:-0}" = 1 ]; then cmd+=(--jinja); fi
    ;;
esac

log "engine=$ENGINE variant=$VARIANT"
log "model=$MODEL"
log "port=$PORT tp=$TP dp=$DP max_running=$MAX_RUNNING max_len=$MAX_LEN"
printf '[serve] command:'; printf ' %q' "${cmd[@]}"; printf '\n'

if [ "$DRY_RUN" = 1 ]; then exit 0; fi

if [ "$BACKGROUND" = 1 ]; then
  : > "$LOG_FILE"
  nohup "${cmd[@]}" >> "$LOG_FILE" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  log "started pid $pid, log $LOG_FILE"
  if [ "$WAIT" = 1 ]; then
    log "waiting for http://127.0.0.1:$PORT/health (timeout ${HEALTH_TIMEOUT_S}s)"
    for _ in $(seq 1 "$HEALTH_TIMEOUT_S"); do
      if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        log "server ready"
        exit 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        log "server exited during startup; last log lines:"
        tail -60 "$LOG_FILE" >&2 || true
        exit 1
      fi
      sleep 1
    done
    log "health check timed out; last log lines:"
    tail -60 "$LOG_FILE" >&2 || true
    exit 1
  fi
  exit 0
fi

exec "${cmd[@]}"
