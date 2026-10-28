#!/usr/bin/env bash
# run-matrix.sh — Vast.ai (single host, no Slurm) adaptation of
# llm/strata/bench-matrix.sh for the 2x Titan RTX campaign.
# One server per pack, cells = ctx x C, probe = strata_probe.py (calibrated
# chars/token + window clamp). Appends to cells.jsonl and summary.csv.
set -uo pipefail
STRATA_ROOT=${STRATA_ROOT:-/root/strata-engine}
export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
MODELS=${MODELS:-/root/models/strata}
DATA=${DATA:-/root/strata-data}
OUT=${OUT:-/root/results}
PROBE=${PROBE:-/root/strata_probe.py}
CTXS=${CTXS:-"1024 4096 16384 65536 131072 262144"}
BATCHES=${BATCHES:-"1 2 4"}
PARALLEL=${PARALLEL:-0}          # 0 = auto: max(BATCHES), so C16 really gets 16 slots
MAXCTX=${MAXCTX:-262144}
MAXTOK=${MAXTOK:-128}
RPS=${RPS:-3}
SAFETY=${SAFETY:-1024}
PORT=${PORT:-8080}
DEADLINE=${DEADLINE:-0}          # epoch seconds; no new cell after this
CELL_TIMEOUT=${CELL_TIMEOUT:-900}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
PACKS=${PACKS:-"iq3s coder"}

mkdir -p "$OUT"
log() { printf '[matrix] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
now() { date +%s; }

http_ok() { curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

server_model_id() {
  curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null
}

repeats_for() { # ctx c
  local ctx=$1 c=$2
  if [ "$c" != 1 ]; then echo 1; return; fi
  if [ "$ctx" -le 4096 ]; then echo 3
  elif [ "$ctx" -le 16384 ]; then echo 2
  else echo 1; fi
}

# Cells that cannot finish inside CELL_TIMEOUT on this host: prefill runs at
# ~1.5k tok/s marginal and 4 concurrent prefills serialise, so wall for C
# clients x 3 req ~= 12 x single-request time.  131072 C=4 and 262144 C>=2
# exceed the 15 min cell cap; they were attempted/skipped as documented.
skip_cell() { # ctx c -> 0 if skipped
  local ctx=$1 c=$2
  [ "$ctx" = 131072 ] && [ "$c" = 4 ] && return 0
  [ "$ctx" = 262144 ] && [ "$c" -ge 2 ] && return 0
  return 1
}

stop_server() {
  "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
  pkill -f "$STRATA_ROOT/components/engine" 2>/dev/null || true
  sleep 3
}

launch_server() { # pack model_dir logfile
  local pack=$1 dir=$2 logf=$3
  export XDG_STATE_HOME="$OUT/state-$pack"
  export XDG_CONFIG_HOME="$OUT/config-$pack"
  export STRATA_PROFILE_DIR="$OUT/profile-$pack"
  export STRATA_RUNTIME_DIR="$OUT/runtime-$pack"
  mkdir -p "$XDG_STATE_HOME" "$XDG_CONFIG_HOME" "$STRATA_PROFILE_DIR" "$STRATA_RUNTIME_DIR" "$DATA"
  if [ "$PARALLEL" -le 0 ]; then
  PARALLEL=1
  for c in $BATCHES; do [ "$c" -gt "$PARALLEL" ] && PARALLEL=$c; done
fi
log "launch pack=$pack dir=$dir ctx=$MAXCTX parallel=$PARALLEL kv=int8 port=$PORT (log $logf)"
  setsid nohup "$STRATA_ROOT/bin/strata" launch \
    --model-path "$dir" \
    --context "$MAXCTX" \
    --parallel "$PARALLEL" \
    --kv int8 \
    --vision off \
    --access local \
    --port "$PORT" \
    --no-api-key \
    --data-dir "$DATA" \
    --yes --no-tui --restart >"$logf" 2>&1 < /dev/null &
  local t0; t0=$(now)
  while :; do
    if http_ok; then log "server ready after $(( $(now) - t0 ))s"; return 0; fi
    if [ $(( $(now) - t0 )) -ge "$READY_TIMEOUT" ]; then
      log "server NOT ready after ${READY_TIMEOUT}s; tail:"; tail -n 40 "$logf" || true; return 1
    fi
    sleep 10
  done
}

tag_last_line() { # file repeat pack ctx c
  python3 - "$@" <<'PY'
import json,sys
f,rep,pack,ctx,c = sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
lines=[l for l in open(f) if l.strip()]
if not lines: sys.exit(0)
d=json.loads(lines[-1])
d["repeat"]=rep; d["pack"]=pack; d["ctx_label"]=ctx; d["conc"]=c
lines[-1]=json.dumps(d)+"\n"
open(f,"w").writelines(lines)
PY
}

run_pack() { # pack
  local pack=$1 dir mid cells csv launchlog t0 ctx c rep rc status before r
  case "$pack" in
    iq3s)  dir=$MODELS/iq3s ;;
    coder) dir=$MODELS/coder ;;
  esac
  mkdir -p "$OUT/$pack"
  cells="$OUT/$pack/cells.jsonl"; csv="$OUT/$pack/summary.csv"
  [ -f "$csv" ] || echo "ctx,concurrency,repeat,decode_tok_s_p50,aggregate_tok_s,ttft_s_p50,ttft_s_p95,prompt_tokens,status" > "$csv"
  launchlog="$OUT/$pack/launch-$(date -u +%H%M%S).log"
  stop_server
  if ! launch_server "$pack" "$dir" "$launchlog"; then
    log "pack $pack: server never became ready -> all cells server-dead"
    for ctx in $CTXS; do for c in $BATCHES; do
      echo "$ctx,$c,0,,,,,,server-dead" >> "$csv"; done; done
    return 1
  fi
  mid=$(server_model_id); [ -n "$mid" ] || mid=strata
  log "pack $pack model id: $mid"
  sleep 5
  cp -a "$XDG_STATE_HOME/strata/launch" "$OUT/$pack/engine-logs-launch" 2>/dev/null || true
  grep -iE "draft|mtp|engine|pack|argv|command|cuda" "$launchlog" | head -60 > "$OUT/$pack/launch-evidence.txt" 2>/dev/null || true
  t0=$(now)
  local cell_list=${CELLS:-}
  if [ -z "$cell_list" ]; then
    cell_list=""
    for _ctx in $CTXS; do for _c in $BATCHES; do cell_list="$cell_list $_ctx:$_c"; done; done
  fi
  for cell in $cell_list; do
    ctx=${cell%%:*}; c=${cell##*:}
      if skip_cell "$ctx" "$c"; then
        log "cell pack=$pack ctx=$ctx C=$c -> skipped-infeasible (exceeds 15 min cell cap at measured prefill rate)"
        echo "$ctx,$c,0,,,,,,skipped-infeasible" >> "$csv"
        continue
      fi
      rep_max=$(repeats_for "$ctx" "$c")
      rep=1
      while [ "$rep" -le "$rep_max" ]; do
        if [ "$DEADLINE" != 0 ] && [ "$(now)" -ge "$DEADLINE" ]; then
          log "DEADLINE reached before ctx=$ctx C=$c rep=$rep"; echo "$ctx,$c,$rep,,,,,,skipped-budget" >> "$csv"; rep=$((rep+1)); continue
        fi
        if ! http_ok; then
          log "server down before ctx=$ctx C=$c rep=$rep; restarting once"
          if ! launch_server "$pack" "$dir" "$OUT/$pack/launch-restart-$(date -u +%H%M%S).log"; then
            echo "$ctx,$c,$rep,,,,,,server-dead" >> "$csv"
            for r in $(seq $((rep+1)) "$rep_max"); do echo "$ctx,$c,$r,,,,,,server-dead" >> "$csv"; done
            rep=$((rep_max+1)); continue
          fi
          mid=$(server_model_id); [ -n "$mid" ] || mid=strata
        fi
        before=$(wc -l < "$cells" 2>/dev/null || echo 0)
        log "cell pack=$pack ctx=$ctx C=$c rep=$rep/$rep_max"
        timeout "$CELL_TIMEOUT" python3 "$PROBE" \
          --base-url "http://127.0.0.1:$PORT" --model "$mid" \
          --context "$ctx" --concurrency "$c" --max-tokens "$MAXTOK" \
          --max-context "$MAXCTX" --pad-safety "$SAFETY" \
          --requests-per-slot "$RPS" --json-out "$cells"
        rc=$?
        if [ "$(wc -l < "$cells" 2>/dev/null || echo 0)" -gt "$before" ]; then
          tag_last_line "$cells" "$rep" "$pack" "$ctx" "$c"
          python3 - "$cells" "$csv" "$ctx" "$c" "$rep" <<'PY'
import json,sys
cells,csv,ctx,c,rep=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
d=json.loads([l for l in open(cells) if l.strip()][-1])
def f(k): 
    v=d.get(k); return "%.3f"%v if isinstance(v,(int,float)) else ""
status="ok" if (rc_ok := (d.get("error") is None and d.get("decode_tok_s_p50"))) else "fail"
open(csv,"a").write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(ctx,c,rep,f("decode_tok_s_p50"),f("aggregate_tok_s"),f("ttft_s_p50"),f("ttft_s_p95"),d.get("prompt_tokens",""),status))
PY
          status=$(tail -1 "$csv" | cut -d, -f9)
        else
          echo "$ctx,$c,$rep,,,,,,rc-$rc" >> "$csv"; status="rc-$rc"
        fi
        log "cell pack=$pack ctx=$ctx C=$c rep=$rep -> $status (rc=$rc, $(( $(now) - t0 ))s into matrix)"
        rep=$((rep+1))
      done
  done
  cp -a "$XDG_STATE_HOME/strata/launch" "$OUT/$pack/engine-logs-final" 2>/dev/null || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$OUT/$pack/gpu-after.csv" 2>&1 || true
  stop_server
  log "pack $pack done"
}

log "run: packs=$PACKS ctxs=$CTXS batches=$BATCHES maxctx=$MAXCTX safety=$SAFETY deadline=$DEADLINE"
for p in $PACKS; do run_pack "$p"; done
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$OUT/gpu-final.csv" 2>&1 || true
log "MATRIX-DONE"
