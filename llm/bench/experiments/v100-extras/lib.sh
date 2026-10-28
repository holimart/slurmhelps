#!/usr/bin/env bash
# lib.sh — shared server lifecycle for the 2026-10-27 V100 extras.
#
# These helpers are deliberate copies of the canonical runner blocks in
# llm/strata/{bench-matrix,warm-cache,chain-cache}.sh: identical launch recipe
# and verification, identical private --data-dir seeding, identical engine-log
# archive.  They exist because H1/H3/H6/M11 need a server that stays up across
# several probe programs, which no single canonical runner offers.  Nothing
# here changes the measurement contract.
#
# Usage:
#   source lib.sh
#   start_server <pack_dir> <window> <parallel> <gpus> <phase> <outdir>
#   ... run probe.py / niah.py / multiconv.py against $SERVER_PORT ...
#   stop_server
set -uo pipefail

STRATA_ROOT=${STRATA_ROOT:-/work/ffa/holecem/strata-engine}
DATA_ROOT=${DATA_ROOT:-/work/ffa/holecem/models/strata/_strata-data}
CXX_LIB_DIR=${CXX_LIB_DIR:-/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib}
export LD_LIBRARY_PATH="$CXX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SERVER_PORT=""
SERVER_PHASE=""
SERVER_STATE=""
SERVER_DATA=""
SERVER_OUT=""
SERVER_PID=""

vlog() { printf '[v100-extras] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

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

wait_ready() { # port timeout_s
  local port=$1 deadline=$(( $(date +%s) + $2 ))
  while :; do
    if http_json "http://127.0.0.1:$port/v1/models" 10 2>/dev/null | grep -q '"id"'; then
      return 0
    fi
    [ "$(date +%s)" -ge "$deadline" ] && return 1
    sleep 10
  done
}

seed_private_data() { # shared private
  local shared=$1 private=$2
  rm -rf "$private"; mkdir -p "$private" || return 1
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
  [ -f "$private/mtp/rt/dense.bin" ] && [ -f "$private/mtp/rt/experts.bin" ] \
    && [ -f "$private/mtp/rt/draft_vocab.bin" ]
}

verify_recipe() { # engine log -> 0/1
  local elog line want
  elog=$(ls -1t "$SERVER_STATE"/strata/launch/*.engine.log 2>/dev/null | head -1 || true)
  [ -n "$elog" ] || { vlog "WARN: no engine log yet; recipe unverified"; return 0; }
  line=$(grep -m1 "engine started:" "$elog" || true)
  [ -n "$line" ] || { vlog "WARN: no 'engine started:' line"; return 0; }
  # A parallel=1 server omits --batch entirely (1 is the engine default), so
  # requiring "--batch 1" rejected a perfectly healthy server (job 38249188).
  local wanted=("--mtp " "--kv int8" "--kv-resident 32768" \
                "--expert-cache auto" "--spec 4" "--spec-min-p 0.5")
  [ "$SERVER_PAR" != 1 ] && wanted+=("--batch $SERVER_PAR")
  for want in "${wanted[@]}"; do
    case "$line" in
      *"$want"*) ;;
      *) vlog "FATAL: engine argv missing '$want'"; vlog "  $line"; return 1 ;;
    esac
  done
  vlog "engine recipe verified ($SERVER_PHASE)"
}

start_server() { # pack_dir window parallel gpus phase outdir [extra launch args...]
  local pack=$1 window=$2 par=$3 gpus=$4 phase=$5 out=$6
  shift 6
  local extra=("$@")
  SERVER_PHASE=$phase SERVER_PAR=$par SERVER_OUT=$out
  # Control's runtime state is a Unix-domain socket.  Lustre (/work) cannot
  # bind one: a server whose STRATA_RUNTIME_DIR lived under the results tree
  # died with "runtime operation wait expired" and never wrote an engine log
  # (2026-10-27, job 38243607).  Keep every runtime tree on node-local /tmp and
  # archive the logs to $out at stop.
  local state_root=${STATE_ROOT:-/tmp/v100-extras-state-$JOB}
  SERVER_STATE="$state_root/$phase"
  mkdir -p "$out" "$SERVER_STATE"
  export XDG_STATE_HOME="$SERVER_STATE"
  export XDG_CONFIG_HOME="$state_root/$phase-config"
  export STRATA_RUNTIME_DIR="$state_root/$phase-runtime"
  export STRATA_PROFILE_DIR="$state_root/$phase-profile"
  export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
  mkdir -p "$XDG_CONFIG_HOME" "$STRATA_RUNTIME_DIR" "$STRATA_PROFILE_DIR"

  SERVER_DATA="$out/data-$phase"
  if ! seed_private_data "$DATA_ROOT" "$SERVER_DATA"; then
    vlog "FATAL: cannot seed a private data dir from $DATA_ROOT"
    return 1
  fi
  SERVER_PORT=$(free_port)
  local logf=$out/launch-$(date -u +%H%M%S).log
  vlog "launch $phase: pack=$pack window=$window parallel=$par gpus=$gpus port=$SERVER_PORT"
  if ! "$STRATA_ROOT/bin/strata" launch \
      --model-path "$pack" --context "$window" --parallel "$par" \
      --mtp on --gpu "$gpus" --kv int8 --vision off --access local \
      --port "$SERVER_PORT" --no-api-key --data-dir "$SERVER_DATA" \
      --yes --no-tui --restart \
      --set kv-resident=32768 --set expert-cache=auto --set prefill=auto \
      --set ple-io=ram --set spec=4 --set spec-min-p=0.5 \
      --set "mtp=$SERVER_DATA/mtp/rt" "${extra[@]}" >"$logf" 2>&1; then
    vlog "FATAL: launch failed; tail of $logf:"; tail -n 20 "$logf" || true
    return 1
  fi
  if ! wait_ready "$SERVER_PORT" 2700; then
    vlog "FATAL: server not ready; tail of $logf:"; tail -n 30 "$logf" || true
    return 1
  fi
  SERVER_MODEL=$(http_json "http://127.0.0.1:$SERVER_PORT/v1/models" 10 \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)
  [ -n "$SERVER_MODEL" ] || SERVER_MODEL=strata
  vlog "ready $phase: 127.0.0.1:$SERVER_PORT model=$SERVER_MODEL"
  # A failed recipe check is an audit flag, not a reason to throw away a
  # server that is already answering (the argv is archived either way).
  if ! verify_recipe; then
    vlog "WARN: engine recipe unverified; continuing ($phase)"
    printf 'unverified at %s\n' "$(date -u +%FT%TZ)" >> "$out/recipe-warning.txt"
  fi
  vlog "$SERVER_PORT" > "$out/port.txt"
}

archive_engine() { # outdir
  local out=$1
  local dest=$out/engine-logs
  mkdir -p "$dest"
  [ -d "$SERVER_STATE/strata" ] && cp -a "$SERVER_STATE/strata/." "$dest/" 2>/dev/null || true
  local elog; elog=$(ls -1t "$SERVER_STATE"/strata/launch/*.engine.log 2>/dev/null | head -1 || true)
  if [ -n "$elog" ]; then
    grep -m1 "engine started:" "$elog" > "$out/engine-argv.txt" 2>/dev/null || true
    grep -E "layer split|experts resident|expert cache [0-9]+ slots|KV streaming:|host->device|VRAM free|GPU [0-9]+:|prompt chunk|CPU prefill" \
      "$elog" | head -120 > "$out/engine-facts.txt" 2>/dev/null || true
    local cfg; cfg=$(ls -1t "$dest"/launch/*.json 2>/dev/null | head -1 || true)
    [ -n "$cfg" ] && cp "$cfg" "$out/launch-config.json" 2>/dev/null || true
  fi
  vlog "archived engine logs -> $dest"
}

stop_server() {
  "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
  [ -n "$SERVER_DATA" ] && pkill -f "$SERVER_DATA/mtp" 2>/dev/null || true
  # The seeded private data dir is a per-phase copy (~1-2 GB).  /tmp on
  # volta05 is a shared 206 GB filesystem, so reclaim it as soon as the
  # server is gone; the next phase re-seeds its own.
  [ -n "$SERVER_DATA" ] && rm -rf "$SERVER_DATA" 2>/dev/null || true
  SERVER_PORT=""
}

server_base() { echo "http://127.0.0.1:$SERVER_PORT"; }
