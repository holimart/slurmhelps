#!/usr/bin/env bash
#SBATCH --job-name=strata-r2-fixup
#SBATCH --partition=ffa-gpulab
#SBATCH --nodelist=volta05
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=00:45:00
#SBATCH --nice=100
#SBATCH --output=/work/ffa/holecem/logs/strata-r2-fixup-%j.out
#
# R2 fixup: the first R2b cells asked for a 32,768-token prompt but passed
# --max-context 262144, so the probe did not clamp and three of four cells were
# rejected with HTTP 400 before any measurement.  This short job reuses the
# IQ3_S pack left in /tmp by the main run and repeats the three cells with the
# server's real window (32,768 → target 32,512), now also recording the device
# of every copy's source/destination buffer (memcpy-trace v2).
#
# Cells: default (no shim) | traced | STRATA_PEER_NO_P2P=1 (no shim)
set -uo pipefail

ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
EXTRAS=${V100_EXTRAS_DIR:-$ROOT/llm/bench/experiments/v100-extras}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
JOB=${SLURM_JOB_ID:?must run inside an allocation}
MODELS=/work/ffa/holecem/models/strata
RUN=/tmp/v100-rem-$JOB
PACKS=$RUN/packs
OUT=/work/ffa/holecem/strata-runs/2026-10-28-v100-remaining
LOG=$RUN/session.log

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf

mkdir -p "$RUN"
# Reuse the newest staged packs (the main run deletes coder/Q2_0 and keeps
# IQ3_S + the shared PLE table); size is checked before trusting each file.
BEST=$(ls -dt /tmp/v100-rem-*/packs 2>/dev/null | head -1 || true)
if [ -n "$BEST" ] && [ ! -e "$PACKS" ]; then
  mv "$BEST" "$PACKS" 2>/dev/null && echo "[r2-fixup] reused packs from $BEST"
  rmdir "$(dirname "$BEST")" 2>/dev/null || true
fi
for d in /tmp/v100-rem-*; do
  [ -d "$d" ] || continue
  [ "$d" = "$RUN" ] && continue
  [ -O "$d" ] || continue
  rm -rf "$d" && echo "[r2-fixup] removed stale $d"
done
for d in /tmp/v100-extras-state-*; do
  [ -d "$d" ] && [ -O "$d" ] && rm -rf "$d"
done
mkdir -p "$PACKS" "$RUN/bin" "$RUN/r2-engine-fixup" "$OUT"

exec >>"$RUN/stdout.log" 2>&1
log() { printf '[r2-fixup] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

source "$EXTRAS/lib.sh"

log "job=$JOB node=$(hostname) start=$(date -u +%FT%TZ)"
nvidia-smi -L 2>&1 | tee -a "$LOG"
df -h /tmp | tail -1 | tee -a "$LOG"

# ---- packs ----------------------------------------------------------------
if [ -f "$PACKS/iq3s/$I3S1" ] && [ "$(stat -c %s "$PACKS/iq3s/$I3S1")" = "$(stat -c %s "$MODELS/iq3s/$I3S1")" ]; then
  log "IQ3_S shard 1 already staged"
else
  log "staging IQ3_S shard 1"
  cp "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1" 2>/dev/null || { mkdir -p "$PACKS/iq3s"; cp "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1"; }
fi
if [ -f "$PACKS/_shared/$SH2" ] && [ "$(stat -c %s "$PACKS/_shared/$SH2")" = "$(stat -c %s "$MODELS/iq3s/$I3S2")" ]; then
  log "shared PLE table already staged"
else
  mkdir -p "$PACKS/_shared"
  cp "$MODELS/iq3s/$I3S2" "$PACKS/_shared/$SH2"
fi
ln -f "$PACKS/_shared/$SH2" "$PACKS/iq3s/$I3S2" || { echo "FATAL: cannot link PLE table"; exit 1; }

if gcc -O2 -shared -fPIC -o "$RUN/bin/memcpy-trace.so" "$EXTRAS/memcpy-trace.c" \
     -ldl -lpthread >"$RUN/build-trace.log" 2>&1; then
  log "memcpy-trace v2 built"
else
  log "FATAL: shim did not build"; tail -5 "$RUN/build-trace.log"; exit 1
fi

probe_cell() { # outdir label
  local out=$1
  python3 "$HELPER/strata_probe.py" \
    --base-url "http://127.0.0.1:$SERVER_PORT" --model "$SERVER_MODEL" \
    --context 32768 --concurrency 1 --max-context 32768 \
    --max-tokens 128 --requests-per-slot 2 --json-out "$out/cells.jsonl"
}

run_cell() { # outdir label
  local out=$1 label=$2 t0 t1
  mkdir -p "$out"
  t0=$(date +%s%N)
  probe_cell "$out" >>"$out/run.log" 2>&1
  t1=$(date +%s%N)
  printf '%s %s\n' "$t0" "$t1" > "$out/cell-window.txt"
  log "cell $label done ($(( (t1-t0)/1000000 )) ms)"
}

# ---- default --------------------------------------------------------------
if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2f-default "$RUN/r2-engine-fixup/default"; then
  run_cell "$RUN/r2-engine-fixup/default" default
  stop_server; archive_engine "$RUN/r2-engine-fixup/default"
else
  log "default server failed"; stop_server
fi

# ---- traced (device-attributed) ------------------------------------------
export LD_PRELOAD="$RUN/bin/memcpy-trace.so"
export STRATA_MEMCPY_TRACE="$RUN/r2-engine-fixup/trace.tsv"
if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2f-trace "$RUN/r2-engine-fixup/trace"; then
  run_cell "$RUN/r2-engine-fixup/trace" trace
  read -r c0 c1 < "$RUN/r2-engine-fixup/trace/cell-window.txt"
  python3 "$EXTRAS/trace-summary.py" "$STRATA_MEMCPY_TRACE" --since "$c0" --until "$c1" \
    --json-out "$RUN/r2-engine-fixup/trace-summary.json" \
    > "$RUN/r2-engine-fixup/trace-summary.txt" 2>&1
  # peer-access calls happen at load, outside the cell window; keep them too
  python3 "$EXTRAS/trace-summary.py" "$STRATA_MEMCPY_TRACE" --since 0 \
    --json-out "$RUN/r2-engine-fixup/trace-whole-run.json" \
    > "$RUN/r2-engine-fixup/trace-whole-run.txt" 2>&1
  stop_server; archive_engine "$RUN/r2-engine-fixup/trace"
else
  log "trace server failed"; stop_server
fi
unset LD_PRELOAD STRATA_MEMCPY_TRACE

# ---- no-P2P env -----------------------------------------------------------
export STRATA_PEER_NO_P2P=1
if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2f-nop2p "$RUN/r2-engine-fixup/nop2p"; then
  run_cell "$RUN/r2-engine-fixup/nop2p" nop2p
  stop_server; archive_engine "$RUN/r2-engine-fixup/nop2p"
else
  log "no-P2P server failed"; stop_server
fi
unset STRATA_PEER_NO_P2P

# ---- publish --------------------------------------------------------------
( cd "$RUN" && find r2-engine-fixup -type f ! -path '*/data-*' \
    ! -name '*.engine.log' ! -size +8M -exec cp --parents {} "$OUT/" \; ) 2>/dev/null || true
cp "$LOG" "$OUT/r2-engine-fixup/session.log" 2>/dev/null || true
cp "$RUN/stdout.log" "$OUT/r2-engine-fixup/stdout.log" 2>/dev/null || true
echo "[r2-fixup] done"
