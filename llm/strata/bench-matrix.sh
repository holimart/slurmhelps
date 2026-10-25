#!/usr/bin/env bash
# bench-matrix.sh — run the Strata benchmark matrix for the Qwen3.8-Flash-Next GGUFs.
#
# For each selected model pack it
#   1. copies the pinned GGUF shards to node-local scratch (fast engine load),
#   2. starts ONE Strata server for the whole matrix:
#        strata launch --model-path DIR --context <max CTXS> --parallel <max BATCHES>
#          --kv int8 --vision off --access local --port <free> --no-api-key
#          --data-dir <persistent> --yes --no-tui --restart
#   3. waits for GET /v1/models readiness (default cap 45 min),
#   4. runs strata_probe.py once per (context, concurrency) cell under timeout,
#   5. appends the cell's JSON line to OUTDIR/<model>/cells.jsonl and one CSV
#      row to OUTDIR/<model>/summary.csv:
#        ctx,concurrency,decode_tok_s_p50,aggregate_tok_s,ttft_s_p50,ttft_s_p95,status
#      status is ok | fail | server-dead | skipped-budget.
# A cell where every request failed costs at most one server restart (max 3
# per matrix) and one retry. A wall-clock budget stops opening new cells so
# the job always ends with clean output. The EXIT trap stops the server
# (`strata launch --stop`, pkill fallback) and removes the scratch copy.
#
# Intended to run INSIDE a Slurm job on a GPU node (see strata-bench.sbatch);
# never run the real matrix on the login node.
#
# Usage:
#   bench-matrix.sh [--model iq3s|coder|both] [--contexts 1024,4096]
#                   [--batches 1,4] [--gpus 0,1] [--max-tokens N]
#                   [--requests-per-slot N] [--outdir DIR] [--budget-min MIN]
#                   [--self-test]
#
#   --model X            pack(s) to benchmark (default: both; MODEL_NAME works)
#   --contexts LIST      prompt size in tokens, comma-separated
#                        (default "1024,4096"; V100 nodes: use only
#                         1024,4096,16384,65536 — docs/NVIDIA_V100.md)
#   --batches LIST       concurrent clients per cell, comma-separated
#                        (default "1,4"; strata launch caps --parallel at 8)
#   --gpus LIST          GPU numbers for strata launch --gpu, e.g. 0,1
#                        (default: auto / one card; GPUS_PER_NODE=2 -> 0,1)
#   --max-tokens N       completion length per request (default 128)
#   --requests-per-slot N  sequential requests per concurrent client (default 3)
#   --outdir DIR         results root (default $RESULTS_ROOT/strata-bench-<job>-<ts>)
#   --budget-min MIN     stop opening new cells after this much wall time
#                        (default 420; job time must leave slack to clean up)
#   --self-test          plumbing test with tests/stub_server.py + a 1 MB dummy
#                        gguf: no Slurm, no GPU, no Strata, no model download.
#                        Binds 127.0.0.1 only and kills the stub on exit.
#
# Environment overrides (the CLI flags win):
#   STRATA_ROOT   Strata install from install-engine.sh (default /work/ffa/holecem/strata-engine)
#   MODELS_ROOT   GGUF library from fetch-models.sh  (default /work/ffa/holecem/models/strata)
#   DATA_DIR      engine data *staging root*; each job gets a private
#                 $DATA_DIR/jobs/<job-id> seeded from its mtp/ and packs/ trees
#                 (default $MODELS_ROOT/_strata-data). The private copy is
#                 removed on exit unless KEEP_DATA_DIR=1.
#   MODEL_NAME    iq3s|coder|both (same as --model; unknown values are fatal)
#   GPUS_PER_NODE number of cards to use (default 1; >1 passes --gpu 0,1,...)
#   RESULTS_ROOT  where --outdir defaults live (default /work/ffa/holecem/results)
#   SCRATCH_ROOT  node-local root for the model copy (default /tmp)
#   COPY_MODELS   1 (default) copy shards to SCRATCH_ROOT; 0 read straight from
#                 Lustre (slow first-token on big contexts, but /tmp is small)
#   READY_TIMEOUT_S per-server readiness cap (default 2700)
#   CXX_LIB_DIR   dir with a newer libstdc++.so.6 for the cuda13 engine
#                 (default the reusable /work/ffa/holecem/strata-runs/
#                 2026-10-25-l40x1/deps/cxx/lib)
#   SELFTEST_DIR  scratch for --self-test (default /work/ffa/holecem/tmp/strata-agent-selftest)
#
# Slurm note: NEVER put comma lists in --export. Slurm splits the value at
# commas (CTXS=1024,4096 becomes CTXS=1024 plus a bogus 4096 key). Pass
# --contexts/--batches as trailing CLI args to strata-bench.sbatch instead.
#
# Gotchas (verified against Strata docs/, see README.md):
#   - `strata launch` downloads the ~5-6 GB MTP draft layer from the original
#     Qwen checkpoint on first run unless it is already in --data-dir; that is
#     why DATA_DIR is persistent shared storage. Without network the launch
#     still succeeds and runs without the drafter (lower decode numbers).
#   - One server per matrix keeps engine load (~2 min IQ3_S + table mmap) to
#     once per model; per-cell restarts would double the job.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STRATA_ROOT=${STRATA_ROOT:-/work/ffa/holecem/strata-engine}
MODELS_ROOT=${MODELS_ROOT:-/work/ffa/holecem/models/strata}
DATA_DIR=${DATA_DIR:-$MODELS_ROOT/_strata-data}
RESULTS_ROOT=${RESULTS_ROOT:-/work/ffa/holecem/results}
SCRATCH_ROOT=${SCRATCH_ROOT:-/tmp}
COPY_MODELS=${COPY_MODELS:-1}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2700}
SELFTEST_DIR=${SELFTEST_DIR:-/work/ffa/holecem/tmp/strata-agent-selftest}
CXX_LIB_DIR=${CXX_LIB_DIR:-/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib}

# MODEL_NAME is the documented interface; MODEL is accepted as a legacy alias.
MODEL=${MODEL_NAME:-${MODEL:-both}}
CTXS=${CTXS:-1024,4096}
BATCHES=${BATCHES:-1,4}
MAX_TOKENS=${MAX_TOKENS:-128}
RPS=${RPS:-3}
BUDGET_MIN=${BUDGET_MIN:-420}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
GPU_LIST=${GPU_LIST:-}
OUTDIR=${OUTDIR:-}
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:?--model needs iq3s|coder|both}; shift 2 ;;
    --model=*) MODEL=${1#*=}; shift ;;
    --contexts) CTXS=${2:?--contexts needs e.g. 1024,4096}; shift 2 ;;
    --contexts=*) CTXS=${1#*=}; shift ;;
    --batches) BATCHES=${2:?--batches needs e.g. 1,4}; shift 2 ;;
    --batches=*) BATCHES=${1#*=}; shift ;;
    --gpus) GPU_LIST=${2:?--gpus needs e.g. 0,1}; shift 2 ;;
    --gpus=*) GPU_LIST=${1#*=}; shift ;;
    --max-tokens) MAX_TOKENS=${2:?}; shift 2 ;;
    --max-tokens=*) MAX_TOKENS=${1#*=}; shift ;;
    --requests-per-slot) RPS=${2:?}; shift 2 ;;
    --requests-per-slot=*) RPS=${1#*=}; shift ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --outdir=*) OUTDIR=${1#*=}; shift ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --budget-min=*) BUDGET_MIN=${1#*=}; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,76p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "bench-matrix: unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${MODEL,,}" in
  iq3s|iq3_s) MODEL=iq3s ;;
  coder|iq1_m) MODEL=coder ;;
  both|all) MODEL=both ;;
  *) echo "bench-matrix: MODEL_NAME/MODEL/--model must be iq3s|coder|both (got '$MODEL')" >&2; exit 2 ;;
esac
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || [ "$GPUS_PER_NODE" -lt 1 ]; then
  echo "bench-matrix: GPUS_PER_NODE must be a positive integer (got '$GPUS_PER_NODE')" >&2
  exit 2
fi
if [ -n "$GPU_LIST" ]; then
  if ! [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "bench-matrix: --gpus/GPU_LIST must look like 0 or 0,1 (got '$GPU_LIST')" >&2
    exit 2
  fi
elif [ "$GPUS_PER_NODE" -gt 1 ]; then
  GPU_LIST=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
fi
list_max() { printf '%s\n' "${1//,/ }" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1; }
MAXCTX=$(list_max "$CTXS"); MAXPAR=$(list_max "$BATCHES")
[ -n "$MAXCTX" ] && [ -n "$MAXPAR" ] || { echo "bench-matrix: --contexts/--batches need numbers" >&2; exit 2; }
if [ "$MAXPAR" -gt 8 ]; then
  echo "bench-matrix: --batches capped at 8 (strata launch --parallel range)" >&2; MAXPAR=8
fi

JOB=${SLURM_JOB_ID:-selftest}
TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ -z "$OUTDIR" ]; then OUTDIR=$RESULTS_ROOT/strata-bench-$JOB-$TS; fi
PROBE=$here/strata_probe.py
STUB=$here/tests/stub_server.py

log() { printf '[strata-bench] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

# ---- per-job isolation (strata_control/paths.py honours these) ---------------
JOB_LOCAL=$SCRATCH_ROOT/strata-$JOB
JOB_SCRATCH=$JOB_LOCAL/model
STATE_HOME=$JOB_LOCAL/state
mkdir -p "$STATE_HOME"
export STRATA_PROFILE_DIR="$JOB_LOCAL/profile"
export STRATA_RUNTIME_DIR="$JOB_LOCAL/runtime"
export XDG_STATE_HOME="$STATE_HOME"
export XDG_CONFIG_HOME="$JOB_LOCAL/config"
# Control scans $XDG_DATA_HOME/strata/components by default; point it at the
# shared install (strata_control/engine_component.py: ENV_DIR wins). Without
# this the components look absent and serve refuses to download one itself.
export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
mkdir -p "$STRATA_PROFILE_DIR" "$STRATA_RUNTIME_DIR" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME"

SERVER=none          # none | strata | stub
SERVER_PID=""        # stub pid
SERVER_MODEL=""      # id reported by /v1/models
CUR_MODEL=""
CUR_DIR=""
SHARED_DATA_DIR=${DATA_DIR:-}
JOB_DATA_DIR=""
DATA_ISOLATED=0
PREPARE_LOCKED=0

# Engine launch logs live under the per-job state dir inside /tmp; the EXIT
# trap used to delete them before anyone could read them. Archive them first.
archive_engine_logs() {
  local dest=$OUTDIR/${CUR_MODEL:-unknown}/engine-logs
  [ -d "$XDG_STATE_HOME/strata" ] || return 0
  mkdir -p "$dest" 2>/dev/null || return 0
  cp -a "$XDG_STATE_HOME/strata/." "$dest/" 2>/dev/null || true
  log "archived engine logs to $dest"
}

stop_everything() {
  local rc=$?
  if [ "$SERVER" = strata ]; then
    "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
    pkill -f "$STRATA_ROOT/components/engine" 2>/dev/null || true
  elif [ "$SERVER" = stub ] && [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER=none
  archive_engine_logs || true
  release_prepare_lock
  [ "${KEEP_SCRATCH:-0}" = 1 ] || rm -rf "$JOB_LOCAL" 2>/dev/null || true
  if [ "$DATA_ISOLATED" = 1 ] && [ "${KEEP_DATA_DIR:-0}" != 1 ]; then
    rm -rf "$JOB_DATA_DIR" 2>/dev/null || true
  fi
  return $rc
}
trap stop_everything EXIT

# ---- libstdc++ floor (cuda13 engine on EL9) -----------------------------------
# The cuda13 engine links GLIBCXX_3.4.30+, but the gpu-ffa EL9 image tops out at
# GLIBCXX_3.4.29.  A newer libstdc++ is reusable from the l40x1 run; prepend it
# when needed instead of burning the 45-min readiness timeout on exec failures.
precheck_engine_libs() {
  local cap variant bin need lib
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ')
  if [ -z "$cap" ]; then
    if nvidia-smi -L 2>/dev/null | grep -qi 'V100'; then cap=7.0; else cap=8.9; fi
  fi
  case "$cap" in 7.*) variant=cuda12 ;; *) variant=cuda13 ;; esac
  bin=$(ls -1d "$STRATA_ROOT"/components/engine/*cuda-${variant}*/strata 2>/dev/null | head -1)
  [ -n "$bin" ] || return 0
  need=$(strings "$bin" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1)
  [ -n "$need" ] || return 0
  lib=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}')
  if [ -n "$lib" ] && strings "$lib" 2>/dev/null | grep -qx "$need"; then
    return 0
  fi
  if [ -f "$CXX_LIB_DIR/libstdc++.so.6" ] \
     && strings "$CXX_LIB_DIR/libstdc++.so.6" 2>/dev/null | grep -qx "$need"; then
    export LD_LIBRARY_PATH="$CXX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "precheck: engine $variant needs $need; prepended $CXX_LIB_DIR to LD_LIBRARY_PATH"
    return 0
  fi
  log "FATAL: engine $variant requires $need but the node libstdc++ (${lib:-not found}) lacks it."
  log "       Stage a newer libstdc++ (conda-forge libstdcxx-ng) and either export"
  log "       LD_LIBRARY_PATH or set CXX_LIB_DIR to the directory holding libstdc++.so.6."
  return 5
}

# ---- private per-job engine data dir -----------------------------------------
# The old runner shared --data-dir across concurrent jobs; both engines wrote
# mtp/rt/experts.bin at once and one read a truncated file.  Seed a private
# tree (hardlinks for the read-only tensors/packs, a real copy of the small
# mutable rt/ tree) and never let two jobs build the shared tree concurrently.
seed_private_data_dir() { # shared private
  local shared=$1 private=$2
  rm -rf "$private"
  mkdir -p "$private" || return 1
  [ -d "$shared/mtp" ] || return 1
  cp -al "$shared/mtp" "$private/mtp" 2>/dev/null \
    || cp -a "$shared/mtp" "$private/mtp" 2>/dev/null || return 1
  # rt/ is small (~0.8 GiB) and is the only tree the engine may rewrite.
  if [ -d "$shared/mtp/rt" ]; then
    rm -rf "$private/mtp/rt"
    cp -a "$shared/mtp/rt" "$private/mtp/rt" 2>/dev/null || return 1
  fi
  if [ -d "$shared/packs" ]; then
    cp -al "$shared/packs" "$private/packs" 2>/dev/null \
      || cp -a "$shared/packs" "$private/packs" 2>/dev/null || return 1
  fi
  [ -f "$private/mtp/rt/dense.bin" ] \
    && [ -f "$private/mtp/rt/experts.bin" ] \
    && [ -f "$private/mtp/rt/draft_vocab.bin" ] || return 1
  return 0
}

acquire_prepare_lock() {
  [ "$DATA_ISOLATED" = 1 ] && return 0
  mkdir -p "$SHARED_DATA_DIR" || return 1
  exec 9>"$SHARED_DATA_DIR/.prepare.lock"
  if ! flock 9; then exec 9>&-; return 1; fi
  PREPARE_LOCKED=1
  log "holding $SHARED_DATA_DIR/.prepare.lock while the shared drafter/pack tree is prepared"
}

release_prepare_lock() {
  [ "$PREPARE_LOCKED" = 1 ] || return 0
  flock -u 9 2>/dev/null || true
  exec 9>&-
  PREPARE_LOCKED=0
}

free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
}

http_json() { # url timeout_s -> exit 0 + body on 2xx
  python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as r:
        sys.stdout.write(r.read(1 << 20).decode("utf-8", "replace"))
except Exception as exc:
    print("http-error: %s" % exc, file=sys.stderr); raise SystemExit(1)
PY
}

server_alive() { # port -> 0 if /v1/models answers JSON
  http_json "http://127.0.0.1:$1/v1/models" 10 >/dev/null 2>&1
}

wait_ready() { # port -> 0 once /v1/models answers, 1 after READY_TIMEOUT_S
  local port=$1 t0 now
  t0=$(date +%s)
  while :; do
    if http_json "http://127.0.0.1:$port/v1/models" 10 | grep -q '"id"'; then
      return 0
    fi
    now=$(date +%s)
    if [ $((now - t0)) -ge "$READY_TIMEOUT_S" ]; then return 1; fi
    sleep 10
  done
}

jsonl_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# last JSONL line -> the four CSV numeric fields, comma-joined
# ("decode,aggregate,ttft50,ttft95"; ",,," when absent/unparsable)
csv_fields() { # cells.jsonl lineno_before
  local f=$1 before=$2
  if [ "$(jsonl_lines "$f")" -le "$before" ]; then echo ",,,"; return; fi
  python3 - "$f" <<'PY'
import json, sys
lines = [l for l in open(sys.argv[1]) if l.strip()]
try:
    d = json.loads(lines[-1])
except Exception:
    d = {}
def f(k):
    v = d.get(k)
    return "%.2f" % v if isinstance(v, (int, float)) else ""
print(",".join([f("decode_tok_s_p50"), f("aggregate_tok_s"),
                f("ttft_s_p50"), f("ttft_s_p95")]))
PY
}

# ---- server control -----------------------------------------------------------
launch_strata_server() { # model_dir port
  local dir=$1 port=$2 logf=$OUTDIR/$CUR_MODEL/launch-$(date -u +%H%M%S).log
  local gpu_args=()
  [ -n "$GPU_LIST" ] && gpu_args=(--gpu "$GPU_LIST")
  mkdir -p "$OUTDIR/$CUR_MODEL"
  log "launch: strata launch --model-path $dir --context $MAXCTX --parallel $MAXPAR${GPU_LIST:+ --gpu $GPU_LIST} --kv int8 --vision off --access local --port $port --data-dir $DATA_DIR (log $logf)"
  mkdir -p "$DATA_DIR"
  "$STRATA_ROOT/bin/strata" launch \
    --model-path "$dir" \
    --context "$MAXCTX" \
    --parallel "$MAXPAR" \
    ${gpu_args[@]+"${gpu_args[@]}"} \
    --kv int8 \
    --vision off \
    --access local \
    --port "$port" \
    --no-api-key \
    --data-dir "$DATA_DIR" \
    --yes --no-tui --restart >"$logf" 2>&1
  SERVER=strata
  if ! wait_ready "$port"; then
    log "server not ready after ${READY_TIMEOUT_S}s; tail of $logf:"
    tail -n 30 "$logf" || true
    return 1
  fi
  SERVER_MODEL=$(http_json "http://127.0.0.1:$port/v1/models" 10 \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)
  [ -n "$SERVER_MODEL" ] || SERVER_MODEL=strata
  log "server ready on 127.0.0.1:$port, model id: $SERVER_MODEL"
}

stop_strata_server() {
  "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
  pkill -f "$STRATA_ROOT/components/engine" 2>/dev/null || true
  SERVER=none
}

start_stub() { # port -> starts stub, waits ready
  mkdir -p "$OUTDIR/selftest"
  python3 "$STUB" --port "$1" >"$OUTDIR/selftest/stub.log" 2>&1 &
  SERVER_PID=$!
  SERVER=stub
  sleep 0.5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stub died:"; cat "$OUTDIR/selftest/stub.log" || true; return 1
  fi
  if ! wait_ready "$1"; then log "stub never became ready"; return 1; fi
  SERVER_MODEL=strata-selftest
}

# ---- model staging ------------------------------------------------------------
# pinned expected sizes (fetch-models.sh / setup.py); complete == exact bytes,
# which also guards against staging a shard a download job has not finished.
size_of() { case "$1" in
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) echo 54817524224 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) echo 28800138432 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf) echo 29608446496 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf) echo 28800138432 ;;
  *) echo 0 ;;
esac }

prepare_model_dir() { # model_name -> sets CUR_DIR
  CUR_MODEL=$1
  local sub cands=() dir cand f ok sz
  case "$1" in
    iq3s)  sub=iq3s
           want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf)
           cands=("$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S") ;;
    coder) sub=coder
           want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf)
           cands=("$MODELS_ROOT/coder" "$MODELS_ROOT/models/coder-IQ1_M") ;;
    *) echo "bench-matrix: unknown model $1" >&2; return 2 ;;
  esac
  dir=""
  for cand in "${cands[@]}"; do
    ok=1
    for f in "${want[@]}"; do
      sz=$(stat -c %s "$cand/$f" 2>/dev/null || echo 0)
      [ "$sz" = "$(size_of "$f")" ] || { ok=0; break; }
    done
    if [ "$ok" = 1 ]; then dir=$cand; log "pack $1 complete in $dir"; break; fi
    log "pack $1 incomplete or absent under $cand (download job still running?)"
  done
  if [ -z "$dir" ]; then
    log "FATAL: no complete pack for $1; checked: ${cands[*]}"
    log "       finish llm/strata/fetch-models.sh --only $1 (or the download job) first"
    return 3
  fi
  if [ "$COPY_MODELS" = 1 ]; then
    local need avail
    need=0
    for f in "${want[@]}"; do need=$((need + $(stat -c %s "$dir/$f"))); done
    avail=$(df -B1 --output=avail "$SCRATCH_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
    if [ "${avail:-0}" -lt $((need + (need / 20) + 2147483648)) ]; then
      log "FATAL: $SCRATCH_ROOT has ${avail:-?} bytes free, need $((need + need / 20)) (+5% margin)."
      log "       set SCRATCH_ROOT to a bigger node-local disk, or COPY_MODELS=0 to read from Lustre."
      return 4
    fi
    rm -rf "$JOB_SCRATCH"; mkdir -p "$JOB_SCRATCH"
    log "copying $((need / 1073741824)) GiB of shards to $JOB_SCRATCH"
    for f in "${want[@]}"; do
      log "  cp $f"
      cp "$dir/$f" "$JOB_SCRATCH/$f"
    done
    CUR_DIR=$JOB_SCRATCH
  else
    CUR_DIR=$dir
    log "COPY_MODELS=0: reading shards straight from $dir (expect slower startup)"
  fi
}

# ---- the matrix -----------------------------------------------------------------
run_matrix() { # port ; uses CUR_MODEL/CUR_DIR/SERVER_MODEL, OUTDIR
  local port=$1 ctx c f status rc restarts=0 t0 now cells_dir cells csv before
  cells_dir=$OUTDIR/$CUR_MODEL
  mkdir -p "$cells_dir"
  cells=$cells_dir/cells.jsonl
  csv=$cells_dir/summary.csv
  [ -f "$csv" ] || echo "ctx,concurrency,decode_tok_s_p50,aggregate_tok_s,ttft_s_p50,ttft_s_p95,status" > "$csv"
  t0=$(date +%s)
  for ctx in ${CTXS//,/ }; do
    for c in ${BATCHES//,/ }; do
      now=$(date +%s)
      if [ $(((now - t0) / 60)) -ge "$BUDGET_MIN" ]; then
        log "budget ${BUDGET_MIN}min reached — remaining cells skipped (ctx=$ctx C=$c ...)"
        echo "$ctx,$c,,,,,skipped-budget" >> "$csv"
        continue
      fi
      if [ "$SERVER" != stub ] && ! server_alive "$port"; then
        log "server is down before ctx=$ctx C=$c"
        echo "$ctx,$c,,,,,server-dead" >> "$csv"
        continue
      fi
      before=$(jsonl_lines "$cells")
      log "cell ctx=$ctx C=$c (rps=$RPS max_tokens=$MAX_TOKENS)"
      rc=0
      timeout $((90 + RPS * 1800)) python3 "$PROBE" \
        --base-url "http://127.0.0.1:$port" --model "$SERVER_MODEL" \
        --context "$ctx" --concurrency "$c" --max-context "$MAXCTX" \
        --max-tokens "$MAX_TOKENS" \
        --requests-per-slot "$RPS" --json-out "$cells" || rc=$?
      if [ "$rc" = 1 ] && [ "$restarts" -lt 3 ] && [ "$SERVER" != stub ] \
         && ! server_alive "$port"; then
        restarts=$((restarts + 1))
        log "cell failed with dead server — restart $restarts/3"
        if launch_strata_server "$CUR_DIR" "$port"; then
          before=$(jsonl_lines "$cells")
          rc=0
          timeout $((90 + RPS * 1800)) python3 "$PROBE" \
            --base-url "http://127.0.0.1:$port" --model "$SERVER_MODEL" \
            --context "$ctx" --concurrency "$c" --max-context "$MAXCTX" \
            --max-tokens "$MAX_TOKENS" \
            --requests-per-slot "$RPS" --json-out "$cells" || rc=$?
        fi
      fi
      case $rc in
        0) status=ok ;;
        1) status=fail ;;
        *) status="rc-$rc" ;;
      esac
      f=$(csv_fields "$cells" "$before")
      echo "$ctx,$c,$f,$status" >> "$csv"
      log "cell ctx=$ctx C=$c -> $status"
    done
  done
  cp "$cells" "$OUTDIR/cells-$CUR_MODEL.jsonl" 2>/dev/null || true
  cp "$csv" "$OUTDIR/summary-$CUR_MODEL.csv" 2>/dev/null || true
}

# ---- self test --------------------------------------------------------------------
self_test() {
  log "SELF-TEST: stub server + 1 MB dummy gguf, no Slurm/GPU/Strata needed"
  [ -f "$STUB" ] || { log "FATAL: $STUB missing"; exit 2; }
  rm -rf "$SELFTEST_DIR"; mkdir -p "$SELFTEST_DIR/model"
  head -c 1048576 /dev/urandom > "$SELFTEST_DIR/model/dummy-00001-of-00001.gguf"
  OUTDIR=$SELFTEST_DIR/out
  mkdir -p "$OUTDIR"
  CTXS=512,1024
  BATCHES=1,2
  MAXCTX=1024
  MAXPAR=2
  MAX_TOKENS=32
  RPS=2
  local port rc=0 cells csv n
  port=$(free_port)
  start_stub "$port" || exit 2
  CUR_MODEL=selftest
  CUR_DIR=$SELFTEST_DIR/model
  cells=$OUTDIR/selftest/cells.jsonl
  csv=$OUTDIR/selftest/summary.csv
  run_matrix "$port"
  n=$(jsonl_lines "$cells")
  log "cells.jsonl lines: $n (expected 4)"
  [ "$n" = 4 ] || rc=1
  head -1 "$csv" | grep -q '^ctx,concurrency,decode_tok_s_p50,aggregate_tok_s,ttft_s_p50,ttft_s_p95,status$' \
    || { log "FAIL: summary.csv header wrong"; rc=1; }
  n=$(grep -vc '^ctx,' "$csv" 2>/dev/null || true)
  log "summary.csv data rows: $n (expected 4)"
  [ "$n" = 4 ] || rc=1
  # stub reports engine timings: predicted_ms gives exactly 100 tok/s decode
  python3 - "$cells" <<'PY' || rc=1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [r for r in rows if r.get("error") or not r.get("decode_tok_s_p50")]
for r in rows:
    print("  cell ctx=%s C=%s target=%s clamped=%s decode=%s agg=%s ttft=%s ok=%s/%s"
          % (r["context"], r["concurrency"], r.get("prompt_target_tokens"),
             r.get("prompt_clamped"), r["decode_tok_s_p50"], r["aggregate_tok_s"],
             r["ttft_s_p50"], r["requests_ok"],
             r["requests_ok"] + r["requests_failed"]))
assert not bad, "cells with error or missing decode: %s" % bad
for r in rows:
    assert 90 <= r["decode_tok_s_p50"] <= 110, "decode p50 %s outside stub band" % r["decode_tok_s_p50"]
    assert r["requests_failed"] == 0, "failed requests in %s" % (r,)
    cal = r.get("calibration") or {}
    assert cal.get("prompt_tokens") and 3.0 <= cal.get("chars_per_token", 0) <= 5.0, \
        "calibration not recorded/plausible: %s" % cal
    assert r.get("prompt_target_tokens"), "prompt target not recorded"
    if r["context"] == 1024:
        assert r["prompt_clamped"] is True, "1024 cell should clamp to the 1024 window"
        assert r["prompt_target_tokens"] <= 1024 - 32 - 128, "clamp left no answer headroom"
print("  JSONL contract + stub decode band (90..110 tok/s): ok")
print("  prompt calibration + window clamp: ok")
PY
  grep -q ',ok$' "$csv" || { log "FAIL: no status=ok rows"; rc=1; }
  # Engine logs must be archived before the EXIT trap deletes the scratch dir.
  mkdir -p "$XDG_STATE_HOME/strata/launch"
  echo "selftest engine log" > "$XDG_STATE_HOME/strata/launch/selftest.engine.log"
  CUR_MODEL=selftest
  archive_engine_logs
  [ -f "$OUTDIR/selftest/engine-logs/launch/selftest.engine.log" ] \
    || { log "FAIL: engine log was not archived"; rc=1; }

  # Private data dir: read-only trees may be hardlinked, but the mutable rt/
  # tree must be isolated from the shared staging copy.
  local shared=$SELFTEST_DIR/shared-data private=$SELFTEST_DIR/private-data
  mkdir -p "$shared/mtp/rt" "$shared/mtp/tensors" "$shared/packs/iq3_s"
  echo tensors > "$shared/mtp/tensors/a.bin"
  echo dense > "$shared/mtp/rt/dense.bin"
  echo experts > "$shared/mtp/rt/experts.bin"
  echo vocab > "$shared/mtp/rt/draft_vocab.bin"
  echo pack > "$shared/packs/iq3_s/pack.bin"
  if seed_private_data_dir "$shared" "$private"; then
    [ -f "$private/mtp/tensors/a.bin" ] && [ -f "$private/packs/iq3_s/pack.bin" ] \
      || { log "FAIL: private data dir missing hardlinked trees"; rc=1; }
    echo changed > "$private/mtp/rt/experts.bin"
    [ "$(cat "$shared/mtp/rt/experts.bin")" = experts ] \
      || { log "FAIL: private rt write leaked into the shared data dir"; rc=1; }
  else
    log "FAIL: seed_private_data_dir rejected a complete fake tree"; rc=1
  fi
  kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER=none
  if [ "$rc" = 0 ]; then
    log "SELF-TEST PASS — results under $OUTDIR (self-test scratch kept)"
  else
    log "SELF-TEST FAIL — see $OUTDIR"
  fi
  exit "$rc"
}

# ---- main ---------------------------------------------------------------------------
if [ "$SELF_TEST" = 1 ]; then self_test; fi

mkdir -p "$OUTDIR"
log "run: job=$JOB model=$MODEL contexts=$CTXS batches=$BATCHES max_tokens=$MAX_TOKENS rps=$RPS budget=${BUDGET_MIN}min outdir=$OUTDIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  log "FATAL: no SLURM_JOB_ID — the real matrix belongs on a GPU node (sbatch strata-bench.sbatch)."
  log "       Use --self-test to check the plumbing on the login node."
  exit 2
fi
[ -x "$STRATA_ROOT/bin/strata" ] || { log "FATAL: $STRATA_ROOT/bin/strata missing — run install-engine.sh first"; exit 3; }
precheck_engine_libs || exit $?

if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  if [ "${visible_gpus:-0}" -lt "$GPUS_PER_NODE" ]; then
    log "FATAL: GPUS_PER_NODE=$GPUS_PER_NODE but only ${visible_gpus:-0} GPU(s) visible on this node"
    exit 4
  fi
fi

SHARED_DATA_DIR=$DATA_DIR
JOB_DATA_DIR=$SHARED_DATA_DIR/jobs/$JOB
if seed_private_data_dir "$SHARED_DATA_DIR" "$JOB_DATA_DIR"; then
  DATA_DIR=$JOB_DATA_DIR
  DATA_ISOLATED=1
  log "private engine data dir: $DATA_DIR (seeded from $SHARED_DATA_DIR)"
else
  rm -rf "$JOB_DATA_DIR" 2>/dev/null || true
  log "shared engine data dir has no complete drafter tree; using $SHARED_DATA_DIR with a prepare lock"
fi

models=both
[ "$MODEL" = both ] && models="iq3s coder" || models=$MODEL
for m in $models; do
  log "=== model pack: $m ==="
  prepare_model_dir "$m" || { log "skipping $m (staging failed)"; continue; }
  port=$(free_port)
  acquire_prepare_lock || { log "FATAL: cannot lock $SHARED_DATA_DIR"; exit 6; }
  if ! launch_strata_server "$CUR_DIR" "$port"; then
    release_prepare_lock
    log "server for $m never became ready; recording failed matrix"
    mkdir -p "$OUTDIR/$m"
    for ctx in ${CTXS//,/ }; do for c in ${BATCHES//,/ }; do
      echo "$ctx,$c,,,,,server-dead" >> "$OUTDIR/$m/summary.csv"
    done; done
    stop_strata_server
    continue
  fi
  release_prepare_lock
  run_matrix "$port"
  stop_strata_server
  archive_engine_logs
done

log "done. results in $OUTDIR"
