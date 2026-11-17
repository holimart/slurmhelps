#!/usr/bin/env bash
# Qwen3.8-Flash-Next IQ3_S GGUF on 4x V100-32G (volta05) with the Strata engine:
# the lowest-priority fallback replica behind the login-node router
# (deployment qwen38-flash-next-strata-volta, operator 2026-11-17).
#
# Same worker/state contract as serve-container*.sh: the worker runs this
# script with PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, KV_DTYPE in the environment
# and polls http://127.0.0.1:$PORT/health; SIGTERM (Slurm preemption) stops it.
#
#   1. stage the pinned IQ3_S shards (54.8 GB + 28.8 GB) to node-local
#      $SCRATCH_ROOT/strata-volta-<job> (exact pinned sizes, idempotent: a
#      shard with the right size is not copied again; STRATA_VERIFY_SHA=1 also
#      checks the pinned sha256 of the staged copy);
#   2. seed a PRIVATE engine data dir $DATA_DIR/jobs/strata-volta-<job> from
#      the shared _strata-data tree (hardlinked mtp/ and packs/, copied mtp/rt)
#      and isolate Strata's profile/runtime/state/config dirs per job, so a
#      concurrent launch cannot rewrite this job's drafter or knobs;
#   3. strata launch --model-path <staged> --context $MAX_MODEL_LEN
#        --parallel $MAX_NUM_SEQS --kv int8 --vision off --mtp on --gpu 0,1,2,3
#        --access local --port $PORT+1 --no-api-key  (+ the pinned V100 recipe
#        via --set, as llm/strata/bench-matrix.sh), wait until GET /v1/models
#      lists a model (Strata's /health answers 200 before the model is
#      loaded, so it is not a readiness signal), and check the engine start
#      line for --mtp and --kv int8 (shared install config drift);
#   4. only then start the node gateway on 0.0.0.0:$PORT -> 127.0.0.1:$PORT+1
#      (response rewrite rules, NO bad_words: Strata has no bad_words
#      sampling parameter), so the worker's /health on $PORT means "loaded";
#   5. watch both; exit non-zero with a message if either dies. The EXIT trap
#      stops this job's Strata (strata launch --stop with the per-job runtime
#      dir, then a kill scoped to this job's scratch paths) and removes the
#      staged copy and the private data dir.
#
# Never run on the login node (no SLURM_JOB_ID -> refuses). --plan prints the
# resolved paths and the launch command and exits without touching anything.
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
: "${WORK_ROOT:=$(cd "$ROOT/.." && pwd)}"
export WORK_ROOT

PLAN=0
[ "${1:-}" = "--plan" ] && PLAN=1

STRATA_ROOT=${STRATA_ROOT:-$WORK_ROOT/strata-engine}
MODELS_ROOT=${MODELS_ROOT:-$WORK_ROOT/models/strata}
SHARED_DATA_DIR=${DATA_DIR:-$MODELS_ROOT/_strata-data}
SCRATCH_ROOT=${SCRATCH_ROOT:-/tmp}
CXX_LIB_DIR=${CXX_LIB_DIR:-$WORK_ROOT/strata-runs/2026-10-25-l40x1/deps/cxx/lib}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2700}
PORT=${PORT:-8030}
UPORT=${STRATA_PORT:-$((PORT + 1))}
CONTEXT=${MAX_MODEL_LEN:-262144}
PARALLEL=${MAX_NUM_SEQS:-4}
KV=${KV_DTYPE:-int8}
# Parked-conversation cache (RAM) between requests. Strata's defaults (8 GiB, 4 slots) hold
# only 2-3 agent conversations of 100-145k tokens, so 7 lanes evicted each other and every
# tool call re-read ~100k prompt tokens (25-35 s); operator 2026-11-17: 192 GiB, 48 slots.
CONV_CACHE_MIB=${STRATA_CONV_CACHE_MIB:-196608}
CONV_CACHE_SLOTS=${STRATA_CONV_CACHE_SLOTS:-48}
GPU_LIST=${STRATA_GPUS:-0,1,2,3}
JOB=${SLURM_JOB_ID:-plan}
TAG=strata-volta-$JOB
JOB_LOCAL=$SCRATCH_ROOT/$TAG
PACK_DIR=$JOB_LOCAL/model
JOB_DATA_DIR=$SHARED_DATA_DIR/jobs/$TAG
REWRITE_RULES=${QWEN38_REWRITE_RULES-$HELPER/rewrite_rules.json}
STATE_DIR=${SLURMHELPS_SERVE_DIR:-$ROOT/.slurmhelps-serve/qwen38-flash-next-strata-volta}
REWRITE_LOG=${QWEN38_REWRITE_LOG:-$STATE_DIR/rewrites.jsonl}
LOCK=${STRATA_LAUNCH_LOCK:-$STRATA_ROOT/install/.qwen38-strata-volta.lock}
STRATA=$STRATA_ROOT/bin/strata

# Pinned IQ3_S shards: exact size + sha256 (llm/strata/fetch-models.sh, models/strata/manifest.txt).
SHARDS=(Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
        Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf)
# STRATA_TEST_SHARD_SIZE (tests/test_qwen38_strata_backend.py only) replaces both pinned sizes.
size_of() {
  if [ -n "${STRATA_TEST_SHARD_SIZE:-}" ]; then echo "$STRATA_TEST_SHARD_SIZE"; return; fi
  case "$1" in
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) echo 54817524224 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) echo 28800138432 ;;
  *) echo 0 ;;
esac; }
sha_of() { case "$1" in
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) echo 4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) echo 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113 ;;
  *) echo "" ;;
esac; }

log() { printf '[qwen38-strata] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { log "FATAL: $1"; exit "${2:-1}"; }

launch_args() { # data_dir -> the strata launch argv (one per line)
  printf '%s\n' launch --model-path "$PACK_DIR" --context "$CONTEXT" --parallel "$PARALLEL" \
    --mtp on --gpu "$GPU_LIST" --kv "$KV" --vision off --access local --port "$UPORT" \
    --no-api-key --data-dir "$1" --yes --no-tui --restart \
    --set kv-resident=32768 --set expert-cache=auto --set prefill=auto \
    --set ple-io=ram --set spec=4 --set spec-min-p=0.5 --set "mtp=$1/mtp/rt" \
    --set "conversation-cache-mib=$CONV_CACHE_MIB" --set "conversation-cache-slots=$CONV_CACHE_SLOTS"
}

if [ "$PLAN" = 1 ]; then
  echo "job_local=$JOB_LOCAL"
  echo "pack_dir=$PACK_DIR"
  echo "data_dir=$JOB_DATA_DIR (seeded from $SHARED_DATA_DIR)"
  echo "strata_port=127.0.0.1:$UPORT gateway_port=0.0.0.0:$PORT"
  echo "rewrite_rules=${REWRITE_RULES:-none} bad_words=none"
  mapfile -t ARGS < <(launch_args "$JOB_DATA_DIR")
  printf 'command: %q' "$STRATA"; printf ' %q' "${ARGS[@]}"; printf '\n'
  exit 0
fi

[ -n "${SLURM_JOB_ID:-}" ] || die "no SLURM_JOB_ID: this backend runs inside the serving job, never on the login node" 2
[ -x "$STRATA" ] || die "$STRATA missing: run llm/strata/install-engine.sh" 3
[[ "$PARALLEL" =~ ^[1-8]$ ]] || die "MAX_NUM_SEQS=$PARALLEL: strata launch --parallel accepts 1..8" 2
case "$KV" in int8|q4_0|k8v4|fp16) ;; *) die "KV_DTYPE=$KV: Strata supports int8|q4_0|k8v4|fp16" 2 ;; esac
if [ -n "$REWRITE_RULES" ] && [ ! -f "$REWRITE_RULES" ]; then die "missing $REWRITE_RULES" 2; fi

# ---- per-job isolation (strata_control/paths.py honours these) ---------------
export STRATA_PROFILE_DIR=$JOB_LOCAL/profile
export STRATA_RUNTIME_DIR=$JOB_LOCAL/runtime
export XDG_STATE_HOME=$JOB_LOCAL/state
export XDG_CONFIG_HOME=$JOB_LOCAL/config
export STRATA_COMPONENTS_DIR=$STRATA_ROOT/components
export STRATA_NO_PROMPT=1

STARTED=0
GATEWAY_PID=""
LOCKED=0
SLEEP_PID=""

stop_strata() {
  if [ "$STARTED" = 1 ]; then
    timeout 15 "$STRATA" launch --stop --no-tui --yes >/dev/null 2>&1 || true
    STARTED=0
  fi
  # Scoped fallback (never a node-wide pkill of the engine): only processes whose
  # argv names this job's staged pack or private data dir, i.e. its engine and copies.
  local pat
  for pat in "$JOB_LOCAL/" "$JOB_DATA_DIR/"; do
    pkill -TERM -f "$pat" 2>/dev/null || true
  done
  sleep 1
  for pat in "$JOB_LOCAL/" "$JOB_DATA_DIR/"; do
    pkill -KILL -f "$pat" 2>/dev/null || true
  done
}

cleanup() {
  local rc=$?
  trap - EXIT TERM INT
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null || true
  if [ -n "$GATEWAY_PID" ]; then
    kill -TERM "$GATEWAY_PID" 2>/dev/null || true
    wait "$GATEWAY_PID" 2>/dev/null || true
  fi
  stop_strata
  if [ "$LOCKED" = 1 ]; then flock -u 9 2>/dev/null || true; exec 9>&-; fi
  # Keep the engine logs next to the worker's server.log before scratch goes.
  if [ -d "$XDG_STATE_HOME/strata" ]; then
    mkdir -p "$STATE_DIR/strata-logs/$JOB" 2>/dev/null \
      && cp -a "$XDG_STATE_HOME/strata/." "$STATE_DIR/strata-logs/$JOB/" 2>/dev/null || true
    cp "$JOB_LOCAL/launch.log" "$STATE_DIR/strata-logs/$JOB/" 2>/dev/null || true
  fi
  [ "${KEEP_SCRATCH:-0}" = 1 ] || rm -rf "$JOB_LOCAL" 2>/dev/null || true
  [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$JOB_DATA_DIR" 2>/dev/null || true
  log "stopped (exit $rc)"
  exit "$rc"
}
trap cleanup EXIT
trap 'log "SIGTERM: stopping (preemption or down)"; exit 143' TERM
trap 'exit 130' INT

# A sleep the TERM trap can interrupt immediately (bash runs traps only between commands).
nap() { sleep "$1" & SLEEP_PID=$!; wait "$SLEEP_PID" 2>/dev/null || true; SLEEP_PID=""; }

http_json() { # url timeout_s -> body on 2xx, else exit 1
  python3 - "$1" "$2" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as r:
        sys.stdout.write(r.read(1 << 20).decode("utf-8", "replace"))
except Exception as exc:
    print("http-error: %s" % exc, file=sys.stderr); raise SystemExit(1)
PY
}

model_ids() { # port -> the ids /v1/models lists (empty while Strata is loading)
  http_json "http://127.0.0.1:$1/v1/models" 10 2>/dev/null | python3 -c '
import json, sys
try:
    print(" ".join(m["id"] for m in json.load(sys.stdin).get("data") or [] if isinstance(m, dict) and m.get("id")))
except ValueError:
    pass' 2>/dev/null || true
}

# ---- libstdc++ floor (bench-matrix.sh precheck_engine_libs) -------------------
precheck_engine_libs() {
  local cap variant bin need lib
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  [ -n "$cap" ] || cap=7.0
  case "$cap" in 7.*) variant=cuda12 ;; *) variant=cuda13 ;; esac
  bin=$(ls -1d "$STRATA_ROOT"/components/engine/*cuda-${variant}*/strata 2>/dev/null | head -1) || true
  [ -n "$bin" ] || { log "precheck: no $variant engine binary found; leaving the choice to strata launch"; return 0; }
  need=$(strings "$bin" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1 || true)
  [ -n "$need" ] || return 0
  # awk exits at the first match, so ldconfig can die of SIGPIPE; under pipefail + set -e that
  # killed the backend with exit 141 on volta05 (job 38585702). Guard it like the lines above.
  lib=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}' || true)
  has_symbol() { strings "$1" 2>/dev/null | awk -v n="$need" '$0==n{f=1} END{exit !f}'; }
  if [ -n "$lib" ] && has_symbol "$lib"; then
    log "precheck: engine $variant needs $need; node libstdc++ has it"
    return 0
  fi
  if [ -f "$CXX_LIB_DIR/libstdc++.so.6" ] && has_symbol "$CXX_LIB_DIR/libstdc++.so.6"; then
    export LD_LIBRARY_PATH="$CXX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "precheck: engine $variant needs $need; prepended $CXX_LIB_DIR to LD_LIBRARY_PATH"
    return 0
  fi
  die "engine $variant requires $need but the node libstdc++ (${lib:-not found}) lacks it; set CXX_LIB_DIR to a dir with a newer libstdc++.so.6" 5
}

# ---- 1. stage the pack -----------------------------------------------------------
stage_pack() {
  local src="" cand f sz need=0 avail missing=0
  for cand in "$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S"; do
    local ok=1
    for f in "${SHARDS[@]}"; do
      sz=$(stat -c %s "$cand/$f" 2>/dev/null || echo 0)
      [ "$sz" = "$(size_of "$f")" ] || { ok=0; break; }
    done
    if [ "$ok" = 1 ]; then src=$cand; break; fi
    log "IQ3_S pack incomplete or absent under $cand"
  done
  [ -n "$src" ] || die "no complete IQ3_S pack (exact pinned sizes) under $MODELS_ROOT; run llm/strata/fetch-models.sh --only iq3s" 3
  mkdir -p "$PACK_DIR"
  for f in "${SHARDS[@]}"; do
    sz=$(stat -c %s "$PACK_DIR/$f" 2>/dev/null || echo 0)
    [ "$sz" = "$(size_of "$f")" ] || { need=$((need + $(size_of "$f") - sz)); missing=1; }
  done
  if [ "$missing" = 1 ]; then
    avail=$(df -B1 --output=avail "$SCRATCH_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
    if [ "${avail:-0}" -lt $((need + need / 20 + 2147483648)) ]; then
      die "$SCRATCH_ROOT has ${avail:-?} bytes free, need $need (+5% +2 GiB) for the IQ3_S pack" 4
    fi
    log "staging $((need / 1073741824)) GiB of IQ3_S shards from $src to $PACK_DIR (parallel: shards are on different OSTs)"
    local -a pids=()
    for f in "${SHARDS[@]}"; do
      sz=$(stat -c %s "$PACK_DIR/$f" 2>/dev/null || echo 0)
      [ "$sz" = "$(size_of "$f")" ] && continue
      rm -f "$PACK_DIR/$f"
      cp "$src/$f" "$PACK_DIR/$f.part" && mv "$PACK_DIR/$f.part" "$PACK_DIR/$f" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || die "shard copy to $PACK_DIR failed" 4; done
  else
    log "IQ3_S shards already staged in $PACK_DIR"
  fi
  for f in "${SHARDS[@]}"; do
    sz=$(stat -c %s "$PACK_DIR/$f" 2>/dev/null || echo 0)
    [ "$sz" = "$(size_of "$f")" ] || die "staged $f has $sz bytes, pinned $(size_of "$f")" 4
    if [ "${STRATA_VERIFY_SHA:-0}" = 1 ]; then
      [ "$(sha256sum "$PACK_DIR/$f" | cut -d' ' -f1)" = "$(sha_of "$f")" ] || die "staged $f fails its pinned sha256" 4
    fi
  done
  log "staged: $(du -sh "$PACK_DIR" | cut -f1) in $PACK_DIR; $(df -h "$SCRATCH_ROOT" | tail -1)"
}

# ---- 2. private data dir (bench-matrix.sh seed_private_data_dir) ----------------
seed_private_data_dir() {
  rm -rf "$JOB_DATA_DIR"
  mkdir -p "$JOB_DATA_DIR" || return 1
  [ -d "$SHARED_DATA_DIR/mtp" ] || return 1
  cp -al "$SHARED_DATA_DIR/mtp" "$JOB_DATA_DIR/mtp" 2>/dev/null \
    || cp -a "$SHARED_DATA_DIR/mtp" "$JOB_DATA_DIR/mtp" 2>/dev/null || return 1
  if [ -d "$SHARED_DATA_DIR/mtp/rt" ]; then   # the only tree the engine may rewrite: a real copy
    rm -rf "$JOB_DATA_DIR/mtp/rt"
    cp -a "$SHARED_DATA_DIR/mtp/rt" "$JOB_DATA_DIR/mtp/rt" 2>/dev/null || return 1
  fi
  if [ -d "$SHARED_DATA_DIR/packs" ]; then
    cp -al "$SHARED_DATA_DIR/packs" "$JOB_DATA_DIR/packs" 2>/dev/null \
      || cp -a "$SHARED_DATA_DIR/packs" "$JOB_DATA_DIR/packs" 2>/dev/null || return 1
  fi
  [ -f "$JOB_DATA_DIR/mtp/rt/dense.bin" ] && [ -f "$JOB_DATA_DIR/mtp/rt/experts.bin" ] \
    && [ -f "$JOB_DATA_DIR/mtp/rt/draft_vocab.bin" ]
}

# ---- 3. launch + readiness + recipe check ----------------------------------------
verify_engine_recipe() {
  local elog line want
  elog=$(ls -1t "$XDG_STATE_HOME"/strata/launch/*.engine.log 2>/dev/null | head -1 || true)
  [ -n "$elog" ] || { log "WARN: no engine log; cannot verify the engine argv"; return 0; }
  line=$(grep -m1 "engine started:" "$elog" || true)
  [ -n "$line" ] || { log "WARN: engine log has no 'engine started:' line; argv unverified"; return 0; }
  log "engine argv: $line"
  for want in "--mtp " "--kv $KV" "--max-context $CONTEXT"; do
    case "$line" in
      *"$want"*) ;;
      *) log "engine argv is missing '$want' (shared install config drift?)"; return 1 ;;
    esac
  done
}

mkdir -p "$STRATA_PROFILE_DIR" "$STRATA_RUNTIME_DIR" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME"
log "job=$JOB node=${SLURMD_NODENAME:-?} gpus=$GPU_LIST context=$CONTEXT parallel=$PARALLEL kv=$KV port=$PORT strata_port=$UPORT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
if command -v nvidia-smi >/dev/null 2>&1; then
  n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  want=$(awk -F, '{print NF}' <<<"$GPU_LIST")
  [ "${n:-0}" -ge "$want" ] || die "need $want GPUs ($GPU_LIST), $n visible" 4
fi
precheck_engine_libs
stage_pack
seed_private_data_dir || die "cannot seed a private data dir from $SHARED_DATA_DIR (needs mtp/rt/{dense,experts,draft_vocab}.bin; see models/strata/DRAFTER-READY)" 6
log "private engine data dir: $JOB_DATA_DIR"

# strata launch rewrites the shared $STRATA_ROOT/install/strata-iq3_s.json; serialise
# this deployment's launches until the engine has read its argv.
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -w 1800 9 || die "could not take $LOCK within 30 min" 6
LOCKED=1

mapfile -t ARGS < <(launch_args "$JOB_DATA_DIR")
LAUNCH_LOG=$JOB_LOCAL/launch.log
log "launch:$(printf ' %q' "$STRATA" "${ARGS[@]}")"
STARTED=1
if ! "$STRATA" "${ARGS[@]}" >"$LAUNCH_LOG" 2>&1; then
  tail -n 30 "$LAUNCH_LOG" || true
  die "strata launch failed (log above)" 7
fi
cat "$LAUNCH_LOG" || true

t0=$(date +%s)
IDS=""
while :; do
  IDS=$(model_ids "$UPORT")
  [ -n "$IDS" ] && break
  [ $(( $(date +%s) - t0 )) -lt "$READY_TIMEOUT_S" ] || die "Strata not serving /v1/models after ${READY_TIMEOUT_S}s on 127.0.0.1:$UPORT" 8
  nap 10
done
log "Strata ready on 127.0.0.1:$UPORT after $(( $(date +%s) - t0 ))s; /v1/models ids: $IDS"
verify_engine_recipe || die "engine argv does not match the pinned recipe; refusing to serve" 9
flock -u 9; exec 9>&-; LOCKED=0

# ---- 4. node gateway (rewrite rules only; no bad_words for Strata) ----------------
GW_STATE=$JOB_LOCAL/gateway
mkdir -p "$GW_STATE"
GW=(--upstream "http://127.0.0.1:$UPORT/v1" --listen-host 0.0.0.0 --listen-port "$PORT"
    --max-active -1 --timeout 3600 --key-env SLURMHELPS_GATEWAY_NO_KEY)
if [ -n "$REWRITE_RULES" ]; then
  mkdir -p "$(dirname "$REWRITE_LOG")"
  GW+=(--rewrite-rules "$REWRITE_RULES" --rewrite-log "$REWRITE_LOG")
fi
SLURMHELPS_SERVE_DIR=$GW_STATE "$ROOT/bin/slurmhelps-serve" --config "$HELPER/config-strata-volta.json" \
  gateway "${GW[@]}" &
GATEWAY_PID=$!
log "gateway 0.0.0.0:$PORT -> 127.0.0.1:$UPORT (pid $GATEWAY_PID, rewrite=${REWRITE_RULES:-none}, bad_words=none)"

# ---- 5. watch ----------------------------------------------------------------------
fails=0
while :; do
  nap 30
  kill -0 "$GATEWAY_PID" 2>/dev/null || { wait "$GATEWAY_PID" || true; GATEWAY_PID=""; die "node gateway exited" 10; }
  if [ -n "$(model_ids "$UPORT")" ]; then
    fails=0
  else
    fails=$((fails + 1))
    log "Strata /v1/models check failed ($fails/4)"
    [ "$fails" -lt 4 ] || die "Strata stopped answering on 127.0.0.1:$UPORT" 11
  fi
done
