#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-knobs
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:50:00
#SBATCH --output=/tmp/amd-mi210-knobs-%j.out
#
# Step 3b — decode knobs, the 4096-boundary bug, and the 262k fit check.
# All logs node-local; only the small facts file goes to Lustre at the end.
set -uo pipefail

JOB=${SLURM_JOB_ID:?}
ROOT=${SLURMHELPS_ROOT:-/work/ffa/holecem/slurmhelps-fleet}
EXTRAS=${AMD_MI210_DIR:-$ROOT/llm/bench/experiments/amd-mi210}
MODELS=/work/ffa/holecem/models/strata
PREFIX=/work/ffa/holecem/build/llama-hip-b11785-gfx90a
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/knobs
LOCAL=/dev/shm/amd-mi210-knobs-$JOB
BIN=$LOCAL/bin
mkdir -p "$LOCAL" "$OUT" "$BIN" "$LOCAL/models/_shared" "$LOCAL/models/iq3s"
cp -a "$PREFIX/bin/." "$BIN/"
export LD_LIBRARY_PATH="$BIN:$BIN/lib:${LD_LIBRARY_PATH:-}"
exec > >(tee "$LOCAL/knobs.log") 2>&1
log() { printf '[knobs] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf
PORT=8099

log "job=$JOB node=$(hostname)"
"$BIN/llama-server" --version 2>&1 | head -3

t0=$(date +%s)
log "staging IQ3_S"
cp "$MODELS/iq3s/$I3S2" "$LOCAL/models/_shared/$SH2" && \
ln -f "$LOCAL/models/_shared/$SH2" "$LOCAL/models/iq3s/$I3S2" && \
cp "$MODELS/iq3s/$I3S1" "$LOCAL/models/iq3s/$I3S1"
log "stage_s=$(( $(date +%s) - t0 ))"
MODEL=$LOCAL/models/iq3s/$I3S1

bench_tg() { # tag depth extra...
  local tag=$1 depth=$2; shift 2
  log "tg depth=$depth tag=$tag $*"
  timeout 480 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 \
      -p 0 -n 64 -d "$depth" -r 2 -o json "$@" \
      > "$LOCAL/tg-$tag.json" 2> "$LOCAL/tg-$tag.err"
  local rc=$?
  echo "== $tag rc=$rc =="
  python3 - "$LOCAL/tg-$tag.json" <<'PY'
import json,sys
try:
    for x in json.load(open(sys.argv[1])):
        print('  n_gen=%s depth=%s avg_ts=%.3f' % (x.get('n_gen'),x.get('n_depth'),x.get('avg_ts') or 0))
except Exception as e: print('  parse failed',e)
PY
  grep -E "Memory access fault|does not support split buffers|failed to load|offloaded|buffer size|KV self" "$LOCAL/tg-$tag.err" | head -10
}

# exact boundary (known failure) vs one short of it: give the fault a short leash
log "tg depth=4096 boundary (expected fault)"
timeout 90 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 \
    -p 0 -n 64 -d 4096 -r 1 -o json > "$LOCAL/tg-boundary-4096.json" 2> "$LOCAL/tg-boundary-4096.err"
echo "boundary_rc=$?"
grep -E "Memory access fault|does not support split buffers" "$LOCAL/tg-boundary-4096.err" | head -3
bench_tg baseline-4095 4095
bench_tg ple-gpu-4095 4095 -ot "per_layer_token_embd=ROCm0"
bench_tg row-split-4095 4095 -sm row
bench_tg fa-off-4095 4095 -fa off
bench_tg ts-even-4095 4095 -ts 0.5/0.5

# 262k fit check + full load facts
log "== server 262k fit =="
"$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c 262144 -np 1 -ctk f16 -ctv f16 \
    --host 127.0.0.1 --port $PORT --metrics > "$LOCAL/server-262k.log" 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 150); do
  kill -0 $SPID 2>/dev/null || break
  python3 -c "import json,urllib.request;exit(0 if json.loads(urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read())['status'] in ('ok','no slot available') else 1)" 2>/dev/null && { ok=1; break; }
  sleep 3
done
echo "server_262k_ready=$ok"
if [ $ok = 1 ]; then
  rocm-smi --showmeminfo vram 2>&1 | tee "$LOCAL/vram-262k.txt"
  ( for i in $(seq 1 20); do
      echo "--- $(date +%s.%N)"; rocm-smi --showuse --showpower --showclocks 2>/dev/null | grep -E "GPU use|Graphics Package Power|sclk clock level";
      top -b -n1 | head -5 | tail -3;
      sleep 1;
    done ) > "$LOCAL/load-sample.txt" 2>&1 &
  MON=$!
  timeout 900 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 4096 \
      --n-predict 512 --label knobs-load --out "$LOCAL/cell-load.jsonl" --timeout 900 || true
  wait $MON 2>/dev/null
  cat "$LOCAL/load-sample.txt" | head -40
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

{
  echo "job=$JOB stage_s=$(( $(date +%s) - t0 )) server_262k_ready=$ok"
  for f in "$LOCAL"/tg-*.json; do
    [ -s "$f" ] || continue
    echo "-- $(basename "$f")"
    python3 - "$f" <<'PY'
import json,sys
try:
    for x in json.load(open(sys.argv[1])):
        print('  n_gen=%s depth=%s avg_ts=%.3f test_time=%s' % (x.get('n_gen'),x.get('n_depth'),x.get('avg_ts') or 0,x.get('test_time')))
except Exception as e: print('  parse failed',e)
PY
  done
  echo "-- errors --"
  for f in "$LOCAL"/tg-*.err; do
    echo "## $(basename "$f")"
    grep -E "Memory access fault|does not support split buffers|failed to load|error" "$f" | head -5
  done
  echo "-- 262k vram --"; cat "$LOCAL/vram-262k.txt" 2>/dev/null
  echo "-- 262k load head --"; head -80 "$LOCAL/server-262k.log" 2>/dev/null
  echo "-- cell --"; cat "$LOCAL/cell-load.jsonl" 2>/dev/null
  echo "-- cpu/gpu sample --"; head -60 "$LOCAL/load-sample.txt" 2>/dev/null
} > "$OUT/knobs-facts.txt"
cp "$LOCAL"/tg-*.json "$LOCAL"/tg-*.err "$LOCAL/knobs.log" "$LOCAL/cell-load.jsonl" "$OUT/" 2>/dev/null
log "done"
