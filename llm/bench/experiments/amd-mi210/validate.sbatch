#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-validate
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:50:00
#SBATCH --output=/tmp/amd-mi210-validate-%j.out
#
# Step 2b — smoke the HIP build and the qwen4exp GGUF before the 11 h run.
# Stages IQ3_S to /dev/shm, checks llama-bench pp/tg at depth, llama-server
# /health + one measured prompt, q8_0 KV start, and the float4 HBM baseline.
set -uo pipefail

JOB=${SLURM_JOB_ID:?}
ROOT=${SLURMHELPS_ROOT:-/work/ffa/holecem/slurmhelps-fleet}
EXTRAS=${AMD_MI210_DIR:-$ROOT/llm/bench/experiments/amd-mi210}
MODELS=/work/ffa/holecem/models/strata
PREFIX=/work/ffa/holecem/build/llama-hip-b11785-gfx90a
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210/validate
LOCAL=/dev/shm/amd-mi210-validate-$JOB
mkdir -p "$LOCAL" "$OUT"
exec > >(tee "$LOCAL/validate.log") 2>&1

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf
PORT=8099
BIN=$LOCAL/bin
mkdir -p "$BIN" "$LOCAL/models/_shared" "$LOCAL/models/iq3s" "$LOCAL/logs"
cp -a "$PREFIX/bin/." "$BIN/"
export LD_LIBRARY_PATH="$BIN:$BIN/lib:${LD_LIBRARY_PATH:-}"

log() { printf '[validate] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "job=$JOB node=$(hostname)"
"$BIN/llama-server" --version 2>&1 | head -5
rocm-smi --showmeminfo vram 2>&1 | head -12

# HBM float4 baseline first (device idle)
hipcc -O3 --offload-arch=gfx90a "$EXTRAS/hbm-bw.cpp" -o "$LOCAL/hbm-bw" || log "hbm build failed"
for d in 0 1; do
  HIP_VISIBLE_DEVICES=$d "$LOCAL/hbm-bw" $((1<<28)) 20 2>&1 | tee -a "$LOCAL/hbm-bw.txt"
done
cp "$LOCAL/hbm-bw.txt" "$OUT/hbm-bw.txt" 2>/dev/null

# stage IQ3_S
t0=$(date +%s)
log "staging IQ3_S (84 GB) to /dev/shm"
cp "$MODELS/iq3s/$I3S2" "$LOCAL/models/_shared/$SH2" && \
ln -f "$LOCAL/models/_shared/$SH2" "$LOCAL/models/iq3s/$I3S2" && \
cp "$MODELS/iq3s/$I3S1" "$LOCAL/models/iq3s/$I3S1"
log "stage_s=$(( $(date +%s) - t0 ))"; df -h /dev/shm | tail -1
MODEL=$LOCAL/models/iq3s/$I3S1

# llama-bench syntax checks
log "== bench pp 1024 =="
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 -p 1024 -n 0 -r 1 -o json \
    > "$LOCAL/bench-pp.json" 2> "$LOCAL/logs/bench-pp.err"
echo "pp_rc=$?"; cat "$LOCAL/bench-pp.json"; tail -5 "$LOCAL/logs/bench-pp.err"

log "== bench tg depth 1024 =="
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 -p 0 -n 32 -d 1024 -r 1 -o json \
    > "$LOCAL/bench-tg.json" 2> "$LOCAL/logs/bench-tg.err"
echo "tg_rc=$?"; cat "$LOCAL/bench-tg.json"; tail -5 "$LOCAL/logs/bench-tg.err"

# residency facts from the bench load
grep -E "offloaded|offloading|model buffer size|KV self size|n_ctx|ROCm[0-9]|CPU_Mapped" "$LOCAL/logs/bench-pp.err" | head -30

# server f16
log "== server f16 16k =="
"$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c 16384 -np 1 -ctk f16 -ctv f16 \
    --host 127.0.0.1 --port $PORT --metrics > "$LOCAL/logs/server-f16.log" 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 120); do
  kill -0 $SPID 2>/dev/null || break
  python3 -c "import json,urllib.request;exit(0 if json.loads(urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read())['status'] in ('ok','no slot available') else 1)" 2>/dev/null && { ok=1; break; }
  sleep 3
done
ok_f16=$ok
echo "server_f16_ready=$ok pid=$SPID"
if [ $ok = 1 ]; then
  python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 4096 --n-predict 64 \
      --label validate-f16 --out "$LOCAL/cells-f16.jsonl" --timeout 900
  rocm-smi --showmeminfo vram 2>&1 | head -12
  grep -E "offloaded|offloading|model buffer size|KV self size|n_ctx|ROCm[0-9]|CPU_Mapped" "$LOCAL/logs/server-f16.log" | head -30
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

# server q8_0 KV (does the hybrid model accept it?)
log "== server q8_0 KV 16k =="
"$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c 16384 -np 1 -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port $PORT --metrics > "$LOCAL/logs/server-q8kv.log" 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 120); do
  kill -0 $SPID 2>/dev/null || break
  python3 -c "import json,urllib.request;exit(0 if json.loads(urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2).read())['status'] in ('ok','no slot available') else 1)" 2>/dev/null && { ok=1; break; }
  sleep 3
done
echo "server_q8kv_ready=$ok"
if [ $ok = 1 ]; then
  python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 4096 --n-predict 64 \
      --label validate-q8kv --out "$LOCAL/cells-q8kv.jsonl" --timeout 900
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

# small bundle only
{
  echo "== validate facts =="
  echo "server_f16_ready=$ok_f16"
  echo "server_q8kv_ready=$ok"
  echo "stage_s=$(( $(date +%s) - t0 ))"
  echo "-- bench pp --"; cat "$LOCAL/bench-pp.json" 2>/dev/null
  echo "-- bench tg --"; cat "$LOCAL/bench-tg.json" 2>/dev/null
  echo "-- cells --"; cat "$LOCAL/cells-f16.jsonl" "$LOCAL/cells-q8kv.jsonl" 2>/dev/null
  echo "-- q8kv log tail --"; tail -30 "$LOCAL/logs/server-q8kv.log" 2>/dev/null
} > "$OUT/validate-facts.txt"
cp "$LOCAL/bench-pp.json" "$LOCAL/bench-tg.json" "$LOCAL/cells-f16.jsonl" "$LOCAL/cells-q8kv.jsonl" "$OUT/" 2>/dev/null
cp "$LOCAL/validate.log" "$OUT/" 2>/dev/null
log "done"
