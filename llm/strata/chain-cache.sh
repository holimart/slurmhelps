#!/usr/bin/env bash
# chain-cache.sh — checkpoint-chain (prefix-growth) measurement.
#
# Adapted from warm-cache.sh (same staging / private DATA_DIR / readiness /
# engine-verification / stop pattern; copy kept separate so the V100 campaign's
# runner files are not edited while a job is running).  Differences:
#
#   * the probe runs in --mode chain: ONE conversation, one sequential request
#     per stop, each prompt an exact prefix extension of the previous one;
#   * a stop ladder replaces the ctx:C cell list (chain is C1 by definition);
#   * the summary is one row PER TURN (stop, prompt, reused, TTFT, delta
#     prefill, strict decode/TPOT, aggregate);
#   * the default server window is 262144 and each turn generates 512 tokens.
#
# What it proves: turn 1 at ~1k is a cold prefill; turns at ~100k/~200k/~262k
# should report engine `reused` tokens approximately equal to the previous
# prompt, and their TTFT should be driven by the delta prefill, not by the full
# context.  If reuse does not engage, the real numbers are still recorded --
# that is a finding (8 GiB / 4-slot conversation cache, eviction, checkpoint
# granularity), not a failed run.
#
# The launch recipe is the cold matrix / warm-cache baseline, unchanged:
#   --kv int8 --kv-resident 32768 --expert-cache auto --prefill auto
#   --spec 4 --spec-min-p 0.5 --batch 4
# (`--parallel 4` is the server shape even though the chain itself is C1.)
#
# Usage:
#   chain-cache.sh [--model iq3s|coder|both]
#                  [--stops 1024,100000,200000,262144]
#                  [--tokens-per-turn 512] [--chain-last-repeats 0]
#                  [--max-context 262144] [--budget-min 60]
#                  [--node NAME] [--gpus 0] [--outdir DIR]
#                  [--plan] [--self-test]
#
#   --stops LIST       prefix-growth ladder in tokens; strictly increasing.
#   --max-context N    server window (default 262144; each stop is clamped to
#                      window - tokens_per_turn - pad_safety).
#   --tokens-per-turn  completion tokens generated on every turn (default 512).
#   --chain-last-repeats N  extra repeats of the last stop (default 0; 1 is a
#                      cheap second sample at the top of the ladder).
#   --budget-min MIN   stop opening work after this wall time (default 60).
#   --node NAME        abort unless the Slurm job landed on that node.
#   --plan             print the launch/chain plan; launches nothing.
#   --self-test        stub server (tests/stub_server.py --cache-reuse) + a
#                      1 MB dummy gguf; proves the per-turn CSV/JSONL and
#                      prefix reuse without Slurm, GPU, Strata or a model.
#
# Environment overrides (CLI wins): STRATA_ROOT, MODELS_ROOT, DATA_DIR,
# RESULTS_ROOT, SCRATCH_ROOT, COPY_MODELS, READY_TIMEOUT_S, CELL_TIMEOUT_S,
# CXX_LIB_DIR, KEEP_DATA_DIR=1, KEEP_SCRATCH=1.
#
# Always submit through chain-cache.sbatch with --nice=100 (and never run the
# real chain on the login node).  Comma lists belong in trailing CLI args,
# never in --export (Slurm splits them).
#
# Status vocabulary per turn: ok | fail | server-dead | launch-fail |
# skipped-budget.
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
SELFTEST_DIR=${SELFTEST_DIR:-/work/ffa/holecem/tmp/strata-chain-selftest}
CXX_LIB_DIR=${CXX_LIB_DIR:-/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib}

MODEL=iq3s
STOPS=1024,100000,200000,262144
TOKENS_PER_TURN=512
CHAIN_LAST_REPEATS=0
MAX_CONTEXT=262144
PAD_SAFETY=128
BUDGET_MIN=60
OUTDIR=""
NODE=""
GPU_LIST=${GPU_LIST:-0}
PLAN=0
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:?--model needs iq3s|coder|both}; shift 2 ;;
    --model=*) MODEL=${1#*=}; shift ;;
    --stops) STOPS=${2:?--stops needs e.g. 1024,100000}; shift 2 ;;
    --stops=*) STOPS=${1#*=}; shift ;;
    --tokens-per-turn) TOKENS_PER_TURN=${2:?}; shift 2 ;;
    --tokens-per-turn=*) TOKENS_PER_TURN=${1#*=}; shift ;;
    --chain-last-repeats) CHAIN_LAST_REPEATS=${2:?}; shift 2 ;;
    --chain-last-repeats=*) CHAIN_LAST_REPEATS=${1#*=}; shift ;;
    --max-context) MAX_CONTEXT=${2:?}; shift 2 ;;
    --max-context=*) MAX_CONTEXT=${1#*=}; shift ;;
    --pad-safety) PAD_SAFETY=${2:?}; shift 2 ;;
    --pad-safety=*) PAD_SAFETY=${1#*=}; shift ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --budget-min=*) BUDGET_MIN=${1#*=}; shift ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --outdir=*) OUTDIR=${1#*=}; shift ;;
    --node) NODE=${2:?--node needs e.g. uk-g6-02}; shift 2 ;;
    --node=*) NODE=${1#*=}; shift ;;
    --gpus) GPU_LIST=${2:?--gpus needs e.g. 0}; shift 2 ;;
    --gpus=*) GPU_LIST=${1#*=}; shift ;;
    --plan) PLAN=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,57p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "chain-cache: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODEL,,}" in
  iq3s|iq3_s) MODEL=iq3s ;;
  coder|iq1_m) MODEL=coder ;;
  both|all) MODEL=both ;;
  *) echo "chain-cache: --model must be iq3s|coder|both (got '$MODEL')" >&2; exit 2 ;;
esac
if ! [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "chain-cache: --gpus must look like 0 or 0,1 (got '$GPU_LIST')" >&2; exit 2
fi
if ! [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] || [ "$MAX_CONTEXT" -lt 1024 ]; then
  echo "chain-cache: --max-context must be an integer >= 1024" >&2; exit 2
fi
if ! [[ "$BUDGET_MIN" =~ ^[0-9]+$ ]]; then
  echo "chain-cache: --budget-min must be an integer" >&2; exit 2
fi
if ! [[ "$TOKENS_PER_TURN" =~ ^[0-9]+$ ]] || [ "$TOKENS_PER_TURN" -lt 2 ]; then
  echo "chain-cache: --tokens-per-turn must be an integer >= 2" >&2; exit 2
fi
if ! [[ "$CHAIN_LAST_REPEATS" =~ ^[0-9]+$ ]]; then
  echo "chain-cache: --chain-last-repeats must be an integer >= 0" >&2; exit 2
fi
if ! [[ "$PAD_SAFETY" =~ ^[0-9]+$ ]]; then
  echo "chain-cache: --pad-safety must be an integer >= 0" >&2; exit 2
fi

# ---- stop ladder validation --------------------------------------------------
IFS=',' read -r -a STOP_ARR <<< "$STOPS"
if [ "${#STOP_ARR[@]}" -lt 1 ]; then
  echo "chain-cache: --stops needs at least one size" >&2; exit 2
fi
prev_stop=0
for stop in "${STOP_ARR[@]}"; do
  [[ "$stop" =~ ^[0-9]+$ ]] \
    || { echo "chain-cache: bad stop '$stop' in --stops $STOPS" >&2; exit 2; }
  if [ "$stop" -le "$prev_stop" ]; then
    echo "chain-cache: stops must be strictly increasing (got $STOPS)" >&2; exit 2
  fi
  prev_stop=$stop
done
if [ "${STOP_ARR[${#STOP_ARR[@]}-1]}" -gt "$MAX_CONTEXT" ]; then
  echo "chain-cache: last stop ${STOP_ARR[${#STOP_ARR[@]}-1]} exceeds --max-context $MAX_CONTEXT" >&2
  exit 2
fi
MAXPAR=4          # same server shape as the cold matrix / warm-cache
MAXCTX=$MAX_CONTEXT

log() { printf '[strata-chain] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

JOB=${SLURM_JOB_ID:-selftest}
TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ -z "$OUTDIR" ]; then OUTDIR=$RESULTS_ROOT/strata-chain-$JOB-$TS; fi
PROBE=$here/../bench/probe.py
[ -f "$PROBE" ] || PROBE=$here/strata_probe.py
STUB=$here/tests/stub_server.py

# ---- per-job isolation (strata_control/paths.py honours these) ---------------
JOB_LOCAL=$SCRATCH_ROOT/strata-chain-$JOB
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

capture_cache_budget() { # port -> cache-budget.json
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

# ---- per-turn summary --------------------------------------------------------
# One CSV row per chain turn, projected straight from the probe's JSONL lines.
append_chain_summary() { # cells_jsonl line_offset csv model
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import csv, json, os, sys
cells, offset, csv_path, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
header = ["model", "turn", "repeat", "stop", "prev_stop", "prompt_tokens", "reused_tokens",
          "reuse_fraction", "ttft_s", "prefill_s", "prefill_delta_tokens",
          "decode_s", "tpot_ms", "decode_tok_s_strict", "decode_tok_s",
          "aggregate_tok_s", "completion_tokens", "timing_source",
          "status", "error"]
rows = []
if os.path.exists(cells):
    lines = [l for l in open(cells) if l.strip()]
    for line in lines[offset:]:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("mode") != "chain":
            continue
        row = {"model": model}
        for key in header[1:]:
            row[key] = rec.get(key)
        rows.append(row)
exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
with open(csv_path, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
    if not exists:
        writer.writeheader()
    writer.writerows(rows)
for row in rows:
    def fmt(key, spec):
        value = row.get(key)
        return spec % value if isinstance(value, (int, float)) else ""
    print("  turn stop=%-7s prompt=%-7s reused=%-7s ttft=%s delta=%-7s "
          "strict=%s tpot=%s agg=%s %s" % (
              row.get("stop"), row.get("prompt_tokens"),
              row.get("reused_tokens"),
              fmt("ttft_s", "%.3f"), row.get("prefill_delta_tokens"),
              fmt("decode_tok_s_strict", "%.1f"), fmt("tpot_ms", "%.2f"),
              fmt("aggregate_tok_s", "%.2f"), row.get("status")))
PY
}

write_failed_rows() { # status error
  local status=$1 err=$2 dest=$OUTDIR/$CUR_MODEL/summary.csv
  mkdir -p "$OUTDIR/$CUR_MODEL"
  python3 - "$dest" "$CUR_MODEL" "$STOPS" "$status" "$err" <<'PY'
import csv, os, sys
path, model, stops, status, err = sys.argv[1:6]
header = ["model", "turn", "repeat", "stop", "prev_stop", "prompt_tokens", "reused_tokens",
          "reuse_fraction", "ttft_s", "prefill_s", "prefill_delta_tokens",
          "decode_s", "tpot_ms", "decode_tok_s_strict", "decode_tok_s",
          "aggregate_tok_s", "completion_tokens", "timing_source",
          "status", "error"]
exists = os.path.exists(path) and os.path.getsize(path) > 0
with open(path, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
    if not exists:
        writer.writeheader()
    for turn, stop in enumerate([s for s in stops.split(",") if s.strip()], 1):
        writer.writerow({"model": model, "turn": turn, "repeat": 1,
                         "stop": stop, "status": status, "error": err})
PY
}

run_chain() {
  local dir=$OUTDIR/$CUR_MODEL
  local cells=$dir/cells.jsonl
  local before rc status
  mkdir -p "$dir"
  if [ $((( $(date +%s) - STARTED_AT) / 60)) -ge "$BUDGET_MIN" ]; then
    log "budget ${BUDGET_MIN}min reached — chain skipped"
    write_failed_rows skipped-budget "wall budget ${BUDGET_MIN}min exhausted"
    return 0
  fi
  if [ "$SERVER" != stub ] && ! server_alive "$PORT"; then
    log "server is down before the chain"
    write_failed_rows server-dead "server not answering /v1/models"
    return 0
  fi
  before=$( [ -f "$cells" ] && wc -l < "$cells" || echo 0 )
  log "chain stops=$STOPS tokens/turn=$TOKENS_PER_TURN last_repeats=$CHAIN_LAST_REPEATS (max_context=$MAXCTX)"
  rc=0
  timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
    --base-url "http://127.0.0.1:$PORT" --model "$SERVER_MODEL" \
    --concurrency 1 --max-context "$MAXCTX" --pad-safety "$PAD_SAFETY" \
    --mode chain --stops "$STOPS" --tokens-per-turn "$TOKENS_PER_TURN" \
    --chain-last-repeats "$CHAIN_LAST_REPEATS" \
    --json-out "$cells" || rc=$?
  if [ "$rc" = 1 ] && [ "$SERVER_RESTARTS" -lt 2 ] && [ "$SERVER" != stub ] \
     && ! server_alive "$PORT"; then
    SERVER_RESTARTS=$((SERVER_RESTARTS + 1))
    log "chain lost the server — relaunch $SERVER_RESTARTS/2 (the chain restarts from turn 1)"
    stop_server
    # Drop the failed turn lines; a restarted server cannot continue this chain.
    if [ -f "$cells" ]; then
      head -n "$before" "$cells" > "$cells.tmp" && mv "$cells.tmp" "$cells"
    fi
    if launch_server "$CUR_DIR" "$PORT" "$JOB_DATA_DIR"; then
      rc=0
      timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVER_MODEL" \
        --concurrency 1 --max-context "$MAXCTX" --pad-safety "$PAD_SAFETY" \
        --mode chain --stops "$STOPS" --tokens-per-turn "$TOKENS_PER_TURN" \
        --chain-last-repeats "$CHAIN_LAST_REPEATS" \
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
  append_chain_summary "$cells" "$before" "$dir/summary.csv" "$CUR_MODEL" || true
  if [ "$rc" = 124 ]; then
    write_failed_rows fail "probe timeout after ${CELL_TIMEOUT_S}s"
  fi
  log "chain -> $status (probe exit $rc); summary: $dir/summary.csv"
}

# ---- plan --------------------------------------------------------------------
print_plan() {
  echo "model=$MODEL node=${NODE:-any} server_window=$MAX_CONTEXT parallel=$MAXPAR gpus=$GPU_LIST"
  echo "outdir=$OUTDIR budget=${BUDGET_MIN}min stops=$STOPS tokens/turn=$TOKENS_PER_TURN last_repeats=$CHAIN_LAST_REPEATS"
  echo "launch: strata launch --model-path <staged> --context $MAX_CONTEXT --parallel $MAXPAR --mtp on --gpu $GPU_LIST --kv int8 --data-dir <private> --set kv-resident=32768 --set expert-cache=auto --set prefill=auto --set ple-io=ram --set spec=4 --set spec-min-p=0.5 --set mtp=<private>/mtp/rt"
  echo "chain (sequential, one conversation slot):"
  local prev=0
  for stop in "${STOP_ARR[@]}"; do
    printf '  turn stop=%-7s prev=%-7s (prompt is a prefix extension; answer %s tokens)\n' \
      "$stop" "$prev" "$TOKENS_PER_TURN"
    prev=$stop
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
  MAXCTX=4096; MAX_CONTEXT=4096
  TOKENS_PER_TURN=8
  STOPS=512,1024,2048,4000
  IFS=',' read -r -a STOP_ARR <<< "$STOPS"
  CHAIN_LAST_REPEATS=0
  STARTED_AT=$(date +%s)
  BUDGET_MIN=10
  local rc=0 port csv cells n
  port=$(free_port)
  start_stub "$port" || exit 2
  PORT=$port
  run_chain || rc=1
  csv=$OUTDIR/$CUR_MODEL/summary.csv
  cells=$OUTDIR/$CUR_MODEL/cells.jsonl
  n=$(grep -vc '^model,' "$csv" 2>/dev/null || true)
  log "summary.csv data rows: $n (expected 4)"
  [ "$n" = 4 ] || rc=1
  python3 - "$cells" "$port" <<'PY' || rc=1
import json, sys, urllib.request
cells, port = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(cells) if l.strip()]
assert len(rows) == 4, "expected 4 chain turns, got %d" % len(rows)
assert [r["status"] for r in rows] == ["ok"] * 4, rows
assert rows[0]["reused_tokens"] == 0, "turn 1 must be cold"
for i in range(1, len(rows)):
    prev, cur = rows[i - 1], rows[i]
    assert cur["reused_tokens"] > prev["reused_tokens"], "reuse did not grow"
    assert abs(cur["reused_tokens"] - prev["prompt_tokens"]) <= 8, cur
    assert cur["prefill_delta_tokens"] < cur["prompt_tokens"], cur
    assert cur["decode_tok_s_strict"] > 0 and cur["tpot_ms"] > 0, cur
with urllib.request.urlopen("http://127.0.0.1:%d/debug/state" % port, timeout=5) as resp:
    state = json.loads(resp.read().decode("utf-8"))
texts = [r["text"] for r in state["requests"]][-4:]
assert len(texts) == 4, "debug state missing chain prompts"
for earlier, later in zip(texts, texts[1:]):
    assert later.startswith(earlier), "chain turn is not a prefix extension"
print("  chain JSONL contract + prefix growth + engine reuse: ok")
PY
  kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER=none
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
log "run: job=$JOB model=$MODEL stops=$STOPS tokens/turn=$TOKENS_PER_TURN repeats=$CHAIN_LAST_REPEATS server_window=$MAX_CONTEXT budget=${BUDGET_MIN}min outdir=$OUTDIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  log "FATAL: no SLURM_JOB_ID — chain cells belong on a GPU node (sbatch chain-cache.sbatch)."
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
    write_failed_rows launch-fail "model staging failed"
    continue
  fi

  JOB_DATA_DIR=$DATA_DIR/jobs/$JOB-chain-$m
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
  if ! launch_server "$CUR_DIR" "$PORT" "$JOB_DATA_DIR"; then
    log "server for $m never became ready; recording failed turns"
    write_failed_rows launch-fail "strata launch failed or readiness timeout"
    stop_server
    archive_engine_logs
    [ "$DATA_ISOLATED" = 1 ] && { [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$JOB_DATA_DIR"; }
    JOB_DATA_DIR=""; DATA_ISOLATED=0
    continue
  fi
  capture_cache_budget "$PORT" || true
  run_chain
  stop_server
  archive_engine_logs
  [ "$DATA_ISOLATED" = 1 ] && { [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$JOB_DATA_DIR"; }
  JOB_DATA_DIR=""; DATA_ISOLATED=0
done

elapsed=$(( ($(date +%s) - STARTED_AT) / 60 ))
log "done in ${elapsed}min. results: $OUTDIR"
