#!/usr/bin/env bash
# warm-cache.sh — measure warm (cache-reuse) TTFT against the stored cold cells.
#
# The cold matrix (strata-runs/2026-10-25-l40x1 and -v100x1) salted every
# prompt, so every request was a full prefill (`reused=0`).  This runner keeps
# the exact cold launch recipe and asks the probe for the missing half:
#
#   --mode warm: one saltless request per slot, then --warm-repeats identical
#   requests.  Repeat 1 is the cold reference (it can itself hit a checkpoint
#   left by an earlier cell with the same context), repeats 2..N are warm.
#
# It stages the pack ONCE (node-local scratch), seeds a private engine data dir
# once (the same helper logic as bench-matrix/knob-sweep), starts ONE server
# with the baseline recipe, and runs every requested (context, concurrency)
# cell in warm mode.  Engine `reused` tokens and the conversation_cache block
# (budget/slots/parks/restores) are captured per request and per cell.
#
# The baseline recipe is the one the cold runs used (engine argv captured in
# their logs):
#   --kv int8 --kv-resident 32768 --expert-cache auto --prefill auto
#   --spec 4 --spec-min-p 0.5 --batch 4
# `--parallel 4` is the cold matrix's server shape even when a cell measures
# C=1; the server turns `--parallel 4` into the engine's `--batch 4` (passing
# both `--parallel` and `--set batch=4` is refused as a conflict).
#
# The model's installed run config (STRATA_ROOT/install/strata-<model>.json) is
# shared mutable state: a concurrent `strata launch` of another variant can
# leave `"mtp": false` there and strip `--mtp` from the drafter.  To stay on the
# baseline recipe this runner passes `--mtp on` and re-adds the drafter path
# explicitly with `--set mtp=<private>/mtp/rt`, then verifies the engine argv.
#
# Usage:
#   warm-cache.sh [--model iq3s|coder|both] [--contexts 1024,4096,16384,65536]
#                 [--batches 1] [--cells "1024:1,4096:1,4096:4"]
#                 [--warm-repeats 3] [--max-context 131072]
#                 [--max-tokens 128] [--budget-min 120]
#                 [--node uk-g6-06] [--gpus 0] [--outdir DIR]
#                 [--plan] [--self-test]
#
#   --cells SPEC      explicit ctx:concurrency list; overrides --contexts and
#                     --batches (use it for "C4 only at 1024/4096").
#   --max-context N   server window passed to strata launch (default 131072,
#                     the L40 baseline; use 65536 on a V100).
#   --budget-min MIN  stop opening new cells after this wall time (default 120).
#   --node NAME       abort unless the Slurm job landed on that node.
#   --plan            print the exact launch/cell plan; launches nothing.
#   --self-test       stub server (tests/stub_server.py --cache-reuse) + a
#                     1 MB dummy gguf; proves the summary/JSONL plumbing without
#                     Slurm, GPU, Strata or a model download.
#
# Environment overrides (CLI wins): STRATA_ROOT, MODELS_ROOT, DATA_DIR,
# RESULTS_ROOT, SCRATCH_ROOT, COPY_MODELS, READY_TIMEOUT_S, CELL_TIMEOUT_S,
# CXX_LIB_DIR, KEEP_DATA_DIR=1, KEEP_SCRATCH=1.
#
# Always submit through warm-cache.sbatch with --nice=100 (and never run the
# real cells on the login node).  Comma lists belong in trailing CLI args,
# never in --export (Slurm splits them).
#
# Status vocabulary: ok | fail | server-dead | launch-fail | skipped-budget.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STRATA_ROOT=${STRATA_ROOT:-/work/ffa/holecem/strata-engine}
MODELS_ROOT=${MODELS_ROOT:-/work/ffa/holecem/models/strata}
DATA_DIR=${DATA_DIR:-$MODELS_ROOT/_strata-data}
RESULTS_ROOT=${RESULTS_ROOT:-/work/ffa/holecem/results}
SCRATCH_ROOT=${SCRATCH_ROOT:-/tmp}
COPY_MODELS=${COPY_MODELS:-1}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2700}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-3600}
SELFTEST_DIR=${SELFTEST_DIR:-/work/ffa/holecem/tmp/strata-warm-selftest}
CXX_LIB_DIR=${CXX_LIB_DIR:-/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib}

MODEL=iq3s
CTXS=1024,4096,16384,65536
BATCHES=1
CELLS=""
WARM_REPEATS=3
MAX_CONTEXT=131072
MAX_TOKENS=128
BUDGET_MIN=120
OUTDIR=""
NODE=""
GPU_LIST=${GPU_LIST:-0}
PLAN=0
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:?--model needs iq3s|coder|both}; shift 2 ;;
    --model=*) MODEL=${1#*=}; shift ;;
    --contexts) CTXS=${2:?--contexts needs e.g. 1024,4096}; shift 2 ;;
    --contexts=*) CTXS=${1#*=}; shift ;;
    --batches) BATCHES=${2:?--batches needs e.g. 1,4}; shift 2 ;;
    --batches=*) BATCHES=${1#*=}; shift ;;
    --cells) CELLS=${2:?--cells needs e.g. 1024:1,4096:4}; shift 2 ;;
    --cells=*) CELLS=${1#*=}; shift ;;
    --warm-repeats) WARM_REPEATS=${2:?}; shift 2 ;;
    --warm-repeats=*) WARM_REPEATS=${1#*=}; shift ;;
    --max-context) MAX_CONTEXT=${2:?}; shift 2 ;;
    --max-context=*) MAX_CONTEXT=${1#*=}; shift ;;
    --max-tokens) MAX_TOKENS=${2:?}; shift 2 ;;
    --max-tokens=*) MAX_TOKENS=${1#*=}; shift ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --budget-min=*) BUDGET_MIN=${1#*=}; shift ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --outdir=*) OUTDIR=${1#*=}; shift ;;
    --node) NODE=${2:?--node needs e.g. uk-g6-06}; shift 2 ;;
    --node=*) NODE=${1#*=}; shift ;;
    --gpus) GPU_LIST=${2:?--gpus needs e.g. 0}; shift 2 ;;
    --gpus=*) GPU_LIST=${1#*=}; shift ;;
    --plan) PLAN=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,67p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "warm-cache: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODEL,,}" in
  iq3s|iq3_s) MODEL=iq3s ;;
  coder|iq1_m) MODEL=coder ;;
  both|all) MODEL=both ;;
  *) echo "warm-cache: --model must be iq3s|coder|both (got '$MODEL')" >&2; exit 2 ;;
esac
if ! [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "warm-cache: --gpus must look like 0 or 0,1 (got '$GPU_LIST')" >&2; exit 2
fi
if ! [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] || [ "$MAX_CONTEXT" -lt 1024 ]; then
  echo "warm-cache: --max-context must be an integer >= 1024" >&2; exit 2
fi
if ! [[ "$BUDGET_MIN" =~ ^[0-9]+$ ]]; then
  echo "warm-cache: --budget-min must be an integer" >&2; exit 2
fi
if ! [[ "$WARM_REPEATS" =~ ^[0-9]+$ ]] || [ "$WARM_REPEATS" -lt 1 ]; then
  echo "warm-cache: --warm-repeats must be an integer >= 1" >&2; exit 2
fi

# Cell plan: "ctx C" lines.  --cells wins over --contexts/--batches.
cell_plan() {
  local spec=${CELLS:-} ctx c
  if [ -n "$spec" ]; then
    printf '%s\n' "${spec//,/$'\n'}" | tr ':' ' '
    return 0
  fi
  for ctx in ${CTXS//,/ }; do
    for c in ${BATCHES//,/ }; do
      printf '%s %s\n' "$ctx" "$c"
    done
  done
}
while read -r ctx c; do
  [ -n "$ctx" ] || continue
  [[ "$ctx" =~ ^[0-9]+$ ]] && [[ "$c" =~ ^[0-9]+$ ]] \
    || { echo "warm-cache: bad cell '$ctx:$c'" >&2; exit 2; }
  if [ "$ctx" -gt "$MAX_CONTEXT" ]; then
    echo "warm-cache: cell ctx=$ctx exceeds server window --max-context $MAX_CONTEXT" >&2
    exit 2
  fi
done < <(cell_plan)
MAXPAR=4          # the cold matrix served up to 4 slots for every cell
MAXCTX=$MAX_CONTEXT

LOGNAME_WARM=warm-cache
log() { printf '[strata-warm] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

JOB=${SLURM_JOB_ID:-selftest}
TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ -z "$OUTDIR" ]; then OUTDIR=$RESULTS_ROOT/strata-warm-$JOB-$TS; fi
PROBE=$here/strata_probe.py
STUB=$here/tests/stub_server.py

# ---- per-job isolation (strata_control/paths.py honours these) ---------------
JOB_LOCAL=$SCRATCH_ROOT/strata-warm-$JOB
JOB_SCRATCH=$JOB_LOCAL/model
export STRATA_PROFILE_DIR="$JOB_LOCAL/profile"
export STRATA_RUNTIME_DIR="$JOB_LOCAL/runtime"
export XDG_STATE_HOME="$JOB_LOCAL/state"
export XDG_CONFIG_HOME="$JOB_LOCAL/config"
export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
mkdir -p "$STRATA_PROFILE_DIR" "$STRATA_RUNTIME_DIR" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME"

SERVER=none
SERVER_PID=""
SERVER_MODEL=""
CUR_MODEL=""
CUR_DIR=""
JOB_DATA_DIR=""
DATA_ISOLATED=0
PREPARE_LOCKED=0
STARTED_AT=$(date +%s)
SERVER_RESTARTS=0
PORT=""

stop_everything() {
  local rc=$?
  if [ "$SERVER" = strata ]; then
    "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
    # Precise fallback: only an engine reading THIS job's private data dir.
    [ -n "$JOB_DATA_DIR" ] && pkill -f "$JOB_DATA_DIR/mtp" 2>/dev/null || true
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

free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
}
http_json() { # url timeout_s
  python3 - "$1" "$2" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as r:
        sys.stdout.write(r.read(1 << 20).decode("utf-8", "replace"))
except Exception as exc:
    print("http-error: %s" % exc, file=sys.stderr); raise SystemExit(1)
PY
}
server_alive() { http_json "http://127.0.0.1:$1/v1/models" 10 >/dev/null 2>&1; }
wait_ready() {
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

# ---- libstdc++ floor (cuda13 engine on EL9) ----------------------------------
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
  # No `strings | grep -q`: grep -q exits early, strings takes SIGPIPE, and
  # `set -o pipefail` turns a successful match into status 141.
  has_symbol() { strings "$1" 2>/dev/null | awk -v n="$need" '$0==n{f=1} END{exit !f}'; }
  if [ -n "$lib" ] && has_symbol "$lib"; then return 0; fi
  if [ -f "$CXX_LIB_DIR/libstdc++.so.6" ] && has_symbol "$CXX_LIB_DIR/libstdc++.so.6"; then
    export LD_LIBRARY_PATH="$CXX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "precheck: engine $variant needs $need; prepended $CXX_LIB_DIR to LD_LIBRARY_PATH"
    return 0
  fi
  log "FATAL: engine $variant requires $need but the node libstdc++ (${lib:-not found}) lacks it."
  return 5
}

# ---- private per-job engine data dir (same logic as bench-matrix) ------------
seed_private_data_dir() { # shared private
  local shared=$1 private=$2
  rm -rf "$private"
  mkdir -p "$private" || return 1
  [ -d "$shared/mtp" ] || return 1
  cp -al "$shared/mtp" "$private/mtp" 2>/dev/null \
    || cp -a "$shared/mtp" "$private/mtp" 2>/dev/null || return 1
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
  mkdir -p "$DATA_DIR" || return 1
  exec 9>"$DATA_DIR/.prepare.lock"
  flock -w 300 9 || { exec 9>&-; return 1; }
  PREPARE_LOCKED=1
}
release_prepare_lock() {
  [ "$PREPARE_LOCKED" = 1 ] || return 0
  flock -u 9 2>/dev/null || true
  exec 9>&-
  PREPARE_LOCKED=0
}

# ---- model staging (once per job) --------------------------------------------
size_of() { case "$1" in
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) echo 54817524224 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) echo 28800138432 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf) echo 29608446496 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf) echo 28800138432 ;;
  *) echo 0 ;;
esac }

prepare_model_dir() { # model -> CUR_DIR
  CUR_MODEL=$1
  local cands=() want=() dir f ok sz need avail
  case "$1" in
    iq3s) want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf)
          cands=("$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S") ;;
    coder) want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
                  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf)
          cands=("$MODELS_ROOT/coder" "$MODELS_ROOT/models/coder-IQ1_M") ;;
    *) return 2 ;;
  esac
  dir=""
  for cand in "${cands[@]}"; do
    ok=1
    for f in "${want[@]}"; do
      sz=$(stat -c %s "$cand/$f" 2>/dev/null || echo 0)
      [ "$sz" = "$(size_of "$f")" ] || { ok=0; break; }
    done
    if [ "$ok" = 1 ]; then dir=$cand; log "pack $1 complete in $dir"; break; fi
    log "pack $1 incomplete or absent under $cand"
  done
  if [ -z "$dir" ]; then
    log "FATAL: no complete pack for $1; finish fetch-models.sh --only $1 first"
    return 3
  fi
  if [ "$COPY_MODELS" = 1 ]; then
    need=0
    for f in "${want[@]}"; do need=$((need + $(stat -c %s "$dir/$f"))); done
    avail=$(df -B1 --output=avail "$SCRATCH_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
    if [ "${avail:-0}" -lt $((need + (need / 20) + 2147483648)) ]; then
      log "FATAL: $SCRATCH_ROOT has ${avail:-?} bytes free, need $((need + need / 20)) (+5%)."
      log "       set SCRATCH_ROOT to a bigger node-local disk, or COPY_MODELS=0."
      return 4
    fi
    rm -rf "$JOB_SCRATCH"; mkdir -p "$JOB_SCRATCH"
    log "staging $((need / 1073741824)) GiB of shards ONCE to $JOB_SCRATCH"
    for f in "${want[@]}"; do cp "$dir/$f" "$JOB_SCRATCH/$f"; done
    CUR_DIR=$JOB_SCRATCH
  else
    CUR_DIR=$dir
    log "COPY_MODELS=0: reading shards straight from $dir"
  fi
}

# ---- server control ----------------------------------------------------------
launch_server() { # model_dir port datadir
  local dir=$1 port=$2 datadir=$3 logf
  logf=$OUTDIR/$CUR_MODEL/launch-$(date -u +%H%M%S).log
  local -a gpu_args=()
  [ -n "$GPU_LIST" ] && gpu_args=(--gpu "$GPU_LIST")
  mkdir -p "$OUTDIR/$CUR_MODEL"
  # The cold matrix's recipe.  --set is the launch CLI's advanced-setting
  # channel; --parallel 4 becomes the engine's --batch 4 (passing --set batch
  # too is refused as a conflict).  --mtp on + --set mtp= re-add the drafter
  # even if a concurrent launch left "mtp": false in the shared install config.
  log "launch: strata launch --model-path $dir --context $MAX_CONTEXT --parallel $MAXPAR --mtp on${GPU_LIST:+ --gpu $GPU_LIST} --kv int8 --data-dir $datadir --set kv-resident=32768 --set expert-cache=auto --set prefill=auto --set ple-io=ram --set spec=4 --set spec-min-p=0.5 --set mtp=$datadir/mtp/rt (log $logf)"
  if ! "$STRATA_ROOT/bin/strata" launch \
      --model-path "$dir" \
      --context "$MAX_CONTEXT" \
      --parallel "$MAXPAR" \
      --mtp on \
      ${gpu_args[@]+"${gpu_args[@]}"} \
      --kv int8 --vision off --access local --port "$port" --no-api-key \
      --data-dir "$datadir" --yes --no-tui --restart \
      --set kv-resident=32768 --set expert-cache=auto --set prefill=auto \
      --set ple-io=ram --set spec=4 --set spec-min-p=0.5 \
      --set "mtp=$datadir/mtp/rt" >"$logf" 2>&1; then
    log "launch failed; tail of $logf:"
    tail -n 20 "$logf" || true
    return 1
  fi
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
  verify_engine_recipe
}

# Guard against the shared install config drifting from the cold recipe: the
# engine's own start line is the evidence (it names the drafter and --batch).
verify_engine_recipe() {
  local elog line want
  elog=$(ls -1t "$XDG_STATE_HOME"/strata/launch/*.engine.log 2>/dev/null | head -1 || true)
  if [ -z "$elog" ]; then
    log "WARN: no engine log yet; cannot verify --mtp/--batch recipe"
    return 0
  fi
  line=$(grep -m1 "engine started:" "$elog" || true)
  if [ -z "$line" ]; then
    log "WARN: engine log has no 'engine started:' line; recipe unverified"
    return 0
  fi
  for want in "--mtp " "--batch 4" "--kv int8" "--kv-resident 32768" \
              "--expert-cache auto" "--spec 4" "--spec-min-p 0.5"; do
    case "$line" in
      *"$want"*) ;;
      *) log "FATAL: engine argv is missing '$want' (shared install config drift?)"; \
         log "       $line"; return 1 ;;
    esac
  done
  log "engine recipe verified: --mtp --batch 4 --kv int8 --kv-resident 32768 --expert-cache auto --spec 4 --spec-min-p 0.5"
}

stop_server() {
  "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
  [ -n "$JOB_DATA_DIR" ] && pkill -f "$JOB_DATA_DIR/mtp" 2>/dev/null || true
  SERVER=none
}

start_stub() { # port
  mkdir -p "$OUTDIR/selftest"
  python3 "$STUB" --port "$1" --cache-reuse >"$OUTDIR/selftest/stub.log" 2>&1 &
  SERVER_PID=$!
  SERVER=stub
  sleep 0.5
  kill -0 "$SERVER_PID" 2>/dev/null || { log "stub died"; cat "$OUTDIR/selftest/stub.log"; return 1; }
  wait_ready "$1" || { log "stub never became ready"; return 1; }
  SERVER_MODEL=strata-selftest
}

# ---- engine evidence ---------------------------------------------------------
archive_engine_logs() {
  local dest=$OUTDIR/${CUR_MODEL:-unknown}/engine-logs
  [ -d "$XDG_STATE_HOME/strata" ] || return 0
  mkdir -p "$dest" 2>/dev/null || return 0
  cp -a "$XDG_STATE_HOME/strata/." "$dest/" 2>/dev/null || true
  local argv; argv=$(ls -1t "$dest"/launch/*.engine.log 2>/dev/null | head -1 || true)
  if [ -n "$argv" ]; then
    grep -m1 "engine started:" "$argv" > "$OUTDIR/${CUR_MODEL:-unknown}/engine-argv.txt" 2>/dev/null || true
    grep -E "conversation cache sized|batch_slots=|expert cache [0-9]+ slots|KV streaming:" "$argv" \
      | head -20 > "$OUTDIR/${CUR_MODEL:-unknown}/engine-facts.txt" 2>/dev/null || true
  fi
  local cfg; cfg=$(ls -1t "$dest"/launch/*.json 2>/dev/null | head -1 || true)
  [ -n "$cfg" ] && cp "$cfg" "$OUTDIR/${CUR_MODEL:-unknown}/launch-config.json" 2>/dev/null || true
  log "archived engine logs -> $dest"
}

capture_cache_budget() { # port -> writes cache-budget.json
  local port=$1 dest=$OUTDIR/$CUR_MODEL/cache-budget.json
  http_json "http://127.0.0.1:$port/metrics" 15 > "$OUTDIR/$CUR_MODEL/metrics-start.json" 2>/dev/null || true
  python3 - "$OUTDIR/$CUR_MODEL/metrics-start.json" "$dest" <<'PY'
import json, os, sys
src, dest = sys.argv[1], sys.argv[2]
out = {"source": "GET /metrics at server start"}
try:
    raw = json.load(open(src))
except Exception as exc:
    raw = {}
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
cc = raw.get("conversation_cache")
out["conversation_cache"] = cc if isinstance(cc, dict) else None
out["budget_mib"] = (cc or {}).get("budget_mib")
out["slots"] = (cc or {}).get("slots")
json.dump(out, open(dest, "w"), indent=2)
print(json.dumps(out))
PY
}

# ---- cells -------------------------------------------------------------------
CSV_HEADER='model,ctx,concurrency,warm_repeats,ttft_cold_s,ttft_warm_p50,ttft_warm_p95,reused_tokens_p50,reused_tokens_total,reuse_engaged,ttft_speedup,decode_p50,status,error'
csv_escape() { local s=${1:-}; s=${s//$'\n'/ }; s=${s//$'\r'/ }; s=${s//\"/\"\"}; printf '"%s"' "$s"; }

cell_fields() { # cells.jsonl lineno_before
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, before = sys.argv[1], int(sys.argv[2])
lines = [l for l in open(path) if l.strip()] if os.path.exists(path) else []
d = {}
if len(lines) > before:
    try: d = json.loads(lines[-1])
    except Exception: d = {}
def f(k, fmt="%.4f"):
    v = d.get(k)
    return fmt % v if isinstance(v, (int, float)) else ""
reuse = d.get("cache_reuse_engaged")
print("\x1f".join([
    f("ttft_cold_s"), f("ttft_warm_s_p50"), f("ttft_warm_s_p95"),
    f("reused_tokens_p50", "%.0f"), f("reused_tokens_total", "%.0f"),
    "1" if reuse is True else ("0" if reuse is False else ""),
    f("warm_ttft_speedup"), f("decode_tok_s_p50", "%.2f"),
    str(d.get("error") or "").replace("\t", " ")]))
PY
}

run_cell() { # ctx c
  local ctx=$1 c=$2
  local dir=$OUTDIR/$CUR_MODEL
  local cells=$dir/cells.jsonl csv=$dir/summary.csv
  local before rc status fields t_cold t_w50 t_w95 r50 rtot engaged speed dec err
  mkdir -p "$dir"
  [ -f "$csv" ] || printf '%s\n' "$CSV_HEADER" > "$csv"
  if [ $((( $(date +%s) - STARTED_AT) / 60)) -ge "$BUDGET_MIN" ]; then
    log "budget ${BUDGET_MIN}min reached — ctx=$ctx C=$c skipped"
    printf '%s,%s,%s,%s,,,,,,,,,skipped-budget,%s\n' \
      "$CUR_MODEL" "$ctx" "$c" "$WARM_REPEATS" "$(csv_escape "wall budget ${BUDGET_MIN}min exhausted")" >> "$csv"
    return 0
  fi
  if [ "$SERVER" != stub ] && ! server_alive "$PORT"; then
    log "server is down before ctx=$ctx C=$c"
    printf '%s,%s,%s,%s,,,,,,,,,server-dead,%s\n' \
      "$CUR_MODEL" "$ctx" "$c" "$WARM_REPEATS" "$(csv_escape "server not answering /v1/models")" >> "$csv"
    return 0
  fi
  before=$( [ -f "$cells" ] && wc -l < "$cells" || echo 0 )
  log "cell warm ctx=$ctx C=$c (repeats=$WARM_REPEATS max_tokens=$MAX_TOKENS)"
  rc=0
  timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
    --base-url "http://127.0.0.1:$PORT" --model "$SERVER_MODEL" \
    --context "$ctx" --concurrency "$c" --max-context "$MAXCTX" \
    --max-tokens "$MAX_TOKENS" --mode warm --warm-repeats "$WARM_REPEATS" \
    --json-out "$cells" || rc=$?
  if [ "$rc" = 1 ] && [ "$SERVER_RESTARTS" -lt 2 ] && [ "$SERVER" != stub ] \
     && ! server_alive "$PORT"; then
    SERVER_RESTARTS=$((SERVER_RESTARTS + 1))
    log "cell lost the server — relaunch $SERVER_RESTARTS/2"
    stop_server
    if launch_server "$CUR_DIR" "$PORT" "$JOB_DATA_DIR"; then
      before=$( [ -f "$cells" ] && wc -l < "$cells" || echo 0 )
      rc=0
      timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVER_MODEL" \
        --context "$ctx" --concurrency "$c" --max-context "$MAXCTX" \
        --max-tokens "$MAX_TOKENS" --mode warm --warm-repeats "$WARM_REPEATS" \
        --json-out "$cells" || rc=$?
    else
      rc=7
    fi
  fi
  case $rc in
    0) status=ok ;;
    1) status=fail ;;
    7) status=server-dead ;;
    124) status=fail ;;
    *) status=fail ;;
  esac
  if [ "$SERVER" != stub ] && ! server_alive "$PORT"; then status=server-dead; fi
  IFS=$'\x1f' read -r t_cold t_w50 t_w95 r50 rtot engaged speed dec err \
    < <(cell_fields "$cells" "$before") || true
  if [ "$rc" = 124 ]; then err="probe timeout after ${CELL_TIMEOUT_S}s"; fi
  if [ -z "$err" ] && [ "$status" != ok ]; then err="probe exit $rc"; fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$CUR_MODEL" "$ctx" "$c" "$WARM_REPEATS" \
    "${t_cold:-}" "${t_w50:-}" "${t_w95:-}" "${r50:-}" "${rtot:-}" \
    "${engaged:-}" "${speed:-}" "${dec:-}" "$status" "$(csv_escape "$err")" >> "$csv"
  log "cell ctx=$ctx C=$c -> $status cold=${t_cold:-?}s warm50=${t_w50:-?}s reused50=${r50:-?} engaged=${engaged:-?}${err:+ ($err)}"
}

# ---- plan --------------------------------------------------------------------
print_plan() {
  echo "model=$MODEL node=${NODE:-any} server_window=$MAX_CONTEXT parallel=$MAXPAR gpus=$GPU_LIST"
  echo "outdir=$OUTDIR budget=${BUDGET_MIN}min warm_repeats=$WARM_REPEATS max_tokens=$MAX_TOKENS"
  echo "launch: strata launch --model-path <staged> --context $MAX_CONTEXT --parallel $MAXPAR --mtp on --gpu $GPU_LIST --kv int8 --data-dir <private> --set kv-resident=32768 --set expert-cache=auto --set prefill=auto --set ple-io=ram --set spec=4 --set spec-min-p=0.5 --set mtp=<private>/mtp/rt"
  echo "cells:"
  cell_plan | grep -v '^$' | while read -r ctx c; do
    printf '  ctx=%-7s C=%s  warm repeats=%s (1 cold + %s warm)\n' "$ctx" "$c" "$WARM_REPEATS" "$WARM_REPEATS"
  done
}

# ---- self test ---------------------------------------------------------------
self_test() {
  log "SELF-TEST: stub server (--cache-reuse) + 1 MB dummy gguf, no Slurm/GPU/Strata"
  [ -f "$STUB" ] || { log "FATAL: $STUB missing"; exit 2; }
  rm -rf "$SELFTEST_DIR"; mkdir -p "$SELFTEST_DIR/model" "$SELFTEST_DIR/out"
  head -c 1048576 /dev/urandom > "$SELFTEST_DIR/model/dummy-00001-of-00001.gguf"
  OUTDIR=$SELFTEST_DIR/out
  CUR_MODEL=selftest
  CUR_DIR=$SELFTEST_DIR/model
  CELL_TIMEOUT_S=120
  MAXTOK_SAVE=$MAX_TOKENS; MAX_TOKENS=8
  MAXCTX=8192
  WARM_REPEATS=3
  CTXS=1024; BATCHES=1; CELLS=""
  STARTED_AT=$(date +%s)
  BUDGET_MIN=10
  local rc=0 port n
  port=$(free_port)
  start_stub "$port" || exit 2
  PORT=$port
  run_cell 1024 1 || rc=1
  # second cell must also run (server persists across cells)
  run_cell 512 1 || rc=1
  local csv=$OUTDIR/$CUR_MODEL/summary.csv cells=$OUTDIR/$CUR_MODEL/cells.jsonl
  n=$(grep -vc '^model,' "$csv" 2>/dev/null || true)
  log "summary.csv data rows: $n (expected 2)"
  [ "$n" = 2 ] || rc=1
  head -1 "$csv" | grep -q "^$CSV_HEADER\$" || { log "FAIL: summary header wrong"; rc=1; }
  python3 - "$cells" <<'PY' || rc=1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(rows) == 2, "expected 2 JSONL cells, got %d" % len(rows)
for r in rows:
    assert r.get("mode") == "warm", "mode missing: %s" % r
    assert r.get("ttft_cold_s") and r.get("ttft_warm_s_p50"), "warm TTFT fields missing: %s" % r
    assert r.get("ttft_warm_s_p50") <= r["ttft_cold_s"] * 0.6, "stub TTFT did not drop"
    assert (r.get("reused_tokens_p50") or 0) > 0, "no engine reuse reported"
    assert r.get("cache_reuse_engaged") is True, "reuse not marked engaged"
    assert (r.get("warm_ttft_speedup") or 0) > 1.2, "no warm speedup"
    assert r.get("conversation_cache_budget_mib") == 8192, "cache budget missing"
    assert len(r.get("per_request") or []) == 4, "expected 1+3 requests"
print("  warm JSONL contract + deterministic stub speedup: ok")
PY
  kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER=none
  MAX_TOKENS=$MAXTOK_SAVE
  if [ "$rc" = 0 ]; then
    log "SELF-TEST PASS — results under $OUTDIR (self-test scratch kept)"
  else
    log "SELF-TEST FAIL — see $OUTDIR"
  fi
  exit "$rc"
}

if [ "$SELF_TEST" = 1 ]; then self_test; fi

if [ "$PLAN" = 1 ]; then print_plan; exit 0; fi

mkdir -p "$OUTDIR"
log "run: job=$JOB model=$MODEL cells=$(cell_plan | tr '\n' ' ') warm_repeats=$WARM_REPEATS server_window=$MAX_CONTEXT max_tokens=$MAX_TOKENS budget=${BUDGET_MIN}min outdir=$OUTDIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  log "FATAL: no SLURM_JOB_ID — warm cells belong on a GPU node (sbatch warm-cache.sbatch)."
  log "       Use --plan or --self-test for the login-node checks."
  exit 2
fi
[ -x "$STRATA_ROOT/bin/strata" ] || { log "FATAL: $STRATA_ROOT/bin/strata missing — run install-engine.sh first"; exit 3; }
precheck_engine_libs || exit $?

if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  if [ "${visible_gpus:-0}" -lt 1 ]; then
    log "FATAL: no GPU visible on this node"; exit 4
  fi
  log "GPUs visible: $visible_gpus; using --gpu ${GPU_LIST:-auto}"
fi

if [ -n "$NODE" ]; then
  host=$(hostname -s)
  nodestr="${SLURM_JOB_NODELIST:-}"
  ok=0
  case "$host" in "$NODE"|"$NODE".*) ok=1 ;; esac
  case "$nodestr" in *"$NODE"*) ok=1 ;; esac
  if [ "$ok" != 1 ]; then
    log "FATAL: expected node $NODE but this job is on host=$host nodelist=$nodestr"
    exit 2
  fi
  log "node check: $NODE ok (host=$host nodelist=$nodestr)"
fi

models=$MODEL
[ "$MODEL" = both ] && models="iq3s coder"
for m in $models; do
  log "=== model pack: $m ==="
  CUR_MODEL=$m
  if ! prepare_model_dir "$m"; then
    log "skipping $m (staging failed)"
    mkdir -p "$OUTDIR/$m"
    [ -f "$OUTDIR/$m/summary.csv" ] || printf '%s\n' "$CSV_HEADER" > "$OUTDIR/$m/summary.csv"
    while read -r ctx c; do
      [ -n "$ctx" ] || continue
      printf '%s,%s,%s,%s,,,,,,,,,%s,%s\n' \
        "$m" "$ctx" "$c" "$WARM_REPEATS" "launch-fail" \
        "$(csv_escape "model staging failed")" >> "$OUTDIR/$m/summary.csv"
    done < <(cell_plan)
    continue
  fi

  JOB_DATA_DIR=$DATA_DIR/jobs/$JOB-warm-$m
  if acquire_prepare_lock && seed_private_data_dir "$DATA_DIR" "$JOB_DATA_DIR"; then
    DATA_ISOLATED=1
    release_prepare_lock
    log "private engine data dir: $JOB_DATA_DIR"
  else
    release_prepare_lock
    rm -rf "$JOB_DATA_DIR" 2>/dev/null || true
    JOB_DATA_DIR=$DATA_DIR
    DATA_ISOLATED=0
    log "shared engine data dir incomplete; using $DATA_DIR unisolated"
  fi

  PORT=$(free_port)
  mkdir -p "$OUTDIR/$m"
  [ -f "$OUTDIR/$m/summary.csv" ] || printf '%s\n' "$CSV_HEADER" > "$OUTDIR/$m/summary.csv"
  if ! launch_server "$CUR_DIR" "$PORT" "$JOB_DATA_DIR"; then
    log "server for $m never became ready; recording failed cells"
    while read -r ctx c; do
      [ -n "$ctx" ] || continue
      printf '%s,%s,%s,%s,,,,,,,,,%s,%s\n' \
        "$m" "$ctx" "$c" "$WARM_REPEATS" "launch-fail" \
        "$(csv_escape "strata launch failed or readiness timeout")" >> "$OUTDIR/$m/summary.csv"
    done < <(cell_plan)
    stop_server
    archive_engine_logs
    [ "$DATA_ISOLATED" = 1 ] && { [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$JOB_DATA_DIR"; }
    JOB_DATA_DIR=""; DATA_ISOLATED=0
    continue
  fi
  capture_cache_budget "$PORT" || true
  while read -r ctx c; do
    [ -n "$ctx" ] || continue
    run_cell "$ctx" "$c"
  done < <(cell_plan)
  stop_server
  archive_engine_logs
  [ "$DATA_ISOLATED" = 1 ] && { [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$JOB_DATA_DIR"; }
  JOB_DATA_DIR=""; DATA_ISOLATED=0
done

elapsed=$(( ($(date +%s) - STARTED_AT) / 60 ))
log "done in ${elapsed}min. results: $OUTDIR"
