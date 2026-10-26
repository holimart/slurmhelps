#!/usr/bin/env bash
# 2026-10-26 V100 campaign, one allocation on volta05 (4x V100-32G).
#
# Why one job: every pack's 28.8 GB PLE shard is the same single-striped file
# on one OST, and on this day it read at ~4-5 MB/s.  Each separate benchmark
# job would pay that 2 h read again.  This wrapper stages the shared shard
# ONCE into node-local /tmp, hard-links it into every pack dir, and then runs
# the three experiments back to back with the proven runners:
#
#   A  2x V100, Q2_0          bench-matrix.sh --gpus 0,1
#   B  4x V100, IQ3_S + warm  bench-matrix.sh --gpus 0,1,2,3 --warm-cells
#   E  1x V100, Coder knobs   knob-sweep.sh  --gpus 0
#
# Each runner gets its own SCRATCH_ROOT so their state/log trees never mix;
# the staged packs live outside those trees and survive between phases.  The
# Q2_0 shard1 is deleted after A to keep /tmp under its 206 GB.
#
# Submit (never without --nice=100; one job owns all four cards):
#   sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta05 \
#     --gres=gpu:4 --cpus-per-task=32 --mem=340G --time=11:30:00 \
#     llm/strata/experiments/2026-10-26-v100-run.sh
set -uo pipefail

ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
JOB=${SLURM_JOB_ID:?this wrapper belongs inside an sbatch job}
MODELS=/work/ffa/holecem/models/strata
DATA=$MODELS/_strata-data
PACKS=/tmp/v100-packs-$JOB
RES=/work/ffa/holecem/strata-runs/2026-10-26-v100-resident

SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf
Q2S1=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q2S2=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
COS1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
COS2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf

log() { printf '[v100-session] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

mkdir -p "$PACKS/q2_0" "$PACKS/iq3s" "$PACKS/coder" "$PACKS/_shared" "$RES"

# ---- stage the shards once -------------------------------------------------
# shard1 for Q2_0 (OST1) and IQ3_S (OST0) in parallel; the shared table (OST2)
# once.  The shared file is stored under each pack's own filename via hardlink.
pids=()
log "staging $Q2S1 from $MODELS/q2_0"
cp "$MODELS/q2_0/$Q2S1" "$PACKS/q2_0/$Q2S1" & pids+=($!)
log "staging $I3S1 from $MODELS/iq3s"
cp "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1" & pids+=($!)
log "staging the shared 28.8 GB table from $MODELS/iq3s (single OST, ~2 h on this day)"
cp "$MODELS/iq3s/$I3S2" "$PACKS/_shared/$SH2" & pids+=($!)
for p in "${pids[@]}"; do
  if ! wait "$p"; then log "FATAL: staging failed"; exit 4; fi
done
ln -f "$PACKS/_shared/$SH2" "$PACKS/q2_0/$Q2S2"
ln -f "$PACKS/_shared/$SH2" "$PACKS/iq3s/$I3S2"
log "staged: $(du -sh "$PACKS" | cut -f1) under $PACKS; $(df -h /tmp | tail -1)"

common_env=(
  MODELS_ROOT="$PACKS"
  DATA_DIR="$DATA"
  COPY_MODELS=0
  CXX_LIB_DIR=/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib
)

# ---- A. 2x V100, Q2_0 ------------------------------------------------------
log "=== A: 2x V100, Q2_0, window 131072 ==="
env "${common_env[@]}" SCRATCH_ROOT="/tmp/v100-a-$JOB" \
  bash "$HELPER/bench-matrix.sh" \
    --model q2_0 --gpus 0,1 --max-tokens 128 --requests-per-slot 3 \
    --cells "1024:1,1024:2,1024:4,4096:1,4096:2,4096:4,16384:1,16384:2,16384:4,65536:1,65536:2,65536:4,131072:1" \
    --outdir "$RES/a-2x-q2_0" --budget-min 120 \
  || log "A (Q2_0 2x) exited non-zero; see $RES/a-2x-q2_0"

# free the Q2_0 shard1 before the 4x pack needs the room
rm -f "$PACKS/q2_0/$Q2S1"

# ---- B. 4x V100, IQ3_S, cold + warm at max context -------------------------
log "=== B: 4x V100, IQ3_S, window 262144, cell budget 420 min ==="
env "${common_env[@]}" SCRATCH_ROOT="/tmp/v100-b-$JOB" \
  bash "$HELPER/bench-matrix.sh" \
    --model iq3s --gpus 0,1,2,3 --max-tokens 128 --requests-per-slot 3 \
    --cells "1024:1,1024:2,1024:4,4096:1,4096:2,4096:4,16384:1,16384:2,16384:4,65536:1,65536:2,131072:1,131072:2,262144:1,262144:2" \
    --warm-cells "65536:1,131072:1,262144:1" --warm-repeats 3 \
    --outdir "$RES/b-4x-iq3s" --budget-min 300 \
  || log "B (IQ3_S 4x) exited non-zero; see $RES/b-4x-iq3s"

# ---- E. Volta kernels, 1x V100, Coder IQ1_M --------------------------------
# Replace the IQ3_S shard1 with Coder's (IQ3_S is no longer needed; the shared
# table is already local and hard-linked under the Coder name).
rm -f "$PACKS/iq3s/$I3S1"
log "staging $COS1 for the kernel sweep"
if cp "$MODELS/coder/$COS1" "$PACKS/coder/$COS1"; then
  ln -f "$PACKS/_shared/$SH2" "$PACKS/coder/$COS2"
  log "=== E: 1x V100, Coder IQ1_M, knobs (window 65536) ==="
  env "${common_env[@]}" SCRATCH_ROOT="/tmp/v100-e-$JOB" \
    bash "$HELPER/knob-sweep.sh" \
      --model coder --node volta05 --gpus 0 --max-context 65536 \
      --contexts 1024,4096,16384 --batches 1,4 \
      --variants "control,basefull,spec_pure,mtpoff,kvfp16,kvres16k,kvres64k,batch8,sm70table,fusedecode,pfwmma,argmaxrows,grnormfast" \
      --budget-min 180 \
      --outdir "$RES/e-knobs-coder" \
    || log "E (knob sweep) exited non-zero; see $RES/e-knobs-coder"
else
  log "FATAL: could not stage Coder shard1; E skipped"
fi

log "session done; results under $RES"
df -h /tmp | tail -1
