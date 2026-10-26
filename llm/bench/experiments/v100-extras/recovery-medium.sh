#!/usr/bin/env bash
#SBATCH --job-name=strata-v100-extras-medium
#SBATCH --partition=ffa-gpulab
#SBATCH --nodelist=volta05
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=04:00:00
# Job log is node-local while the job runs (Lustre is not to be tailed during an
# allocation); it is copied to the results directory at exit.
#SBATCH --output=/tmp/strata-v100-extras-medium-%j.out
#
# 2026-10-27 V100 extras — medium-value recovery.
#
# In the main job (38246490) M7/M8/M9 failed before launching because the
# orchestrator pre-created their output directories (and put runner.log in
# them); knob-sweep refuses a non-empty outdir.  H3's first run also shared
# one filler body across conversations, so a "cold" turn reused a suffix.
# This job re-runs M7, M8, M9 with the runners owning their outdirs, and H3
# with a distinct filler per conversation.  It also stages each pack once.
set -uo pipefail

ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
EXTRAS=$ROOT/llm/bench/experiments/v100-extras
HELPER=$ROOT/llm/strata
JOB=${SLURM_JOB_ID:?}
MODELS=/work/ffa/holecem/models/strata
DATA=$MODELS/_strata-data
PACKS=/tmp/v100-medium-$JOB
LOG_LOCAL=/tmp/v100-medium-$JOB-logs
OUT=/work/ffa/holecem/strata-runs/2026-10-27-v100-extras
I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
COS1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
COS2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf
mkdir -p "$PACKS/iq3s" "$PACKS/coder" "$PACKS/_shared" "$LOG_LOCAL"
log() { printf '[medium] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
copy_logs() {
  local rc=$?
  mkdir -p "$OUT/_logs-medium-$JOB"
  cp -f "/tmp/strata-v100-extras-medium-$JOB.out" \
        "$OUT/_logs-medium-$JOB/sbatch.out" 2>/dev/null || true
  cp -f "$LOG_LOCAL"/*.log "$OUT/_logs-medium-$JOB/" 2>/dev/null || true
  if [ -d "$LOG_LOCAL/h3b-server" ]; then
    mkdir -p "$OUT/h3b-server"
    cp -a "$LOG_LOCAL/h3b-server/engine-logs" "$OUT/h3b-server/" 2>/dev/null || true
    for f in engine-argv.txt engine-facts.txt launch-config.json port.txt; do
      cp -f "$LOG_LOCAL/h3b-server/$f" "$OUT/h3b-server/" 2>/dev/null || true
    done
    cp -f "$LOG_LOCAL/h3b-server"/launch-*.log "$OUT/h3b-server/" 2>/dev/null || true
  fi
  return $rc
}
trap copy_logs EXIT
log "job=$JOB node=$(hostname) packs=$PACKS logs=$LOG_LOCAL"

# move the failed/partial dirs out of the way, keeping them as evidence
mkdir -p "$OUT/_attempts/run-38246490"
for d in m7-kv-262k m8-prefill-131k m9-batchmtp-lookup-coder; do
  [ -e "$OUT/$d" ] && mv "$OUT/$d" "$OUT/_attempts/run-38246490/$d"
done

# ---- stage IQ3_S + the shared table (parallel) -----------------------------
pids=()
log "staging IQ3_S shard1 + shared PLE table"
cp "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1" & pids+=($!)
cp "$MODELS/iq3s/$I3S2" "$PACKS/_shared/$SH2" & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || { log "FATAL iq3s staging"; exit 4; }; done
ln -f "$PACKS/_shared/$SH2" "$PACKS/iq3s/$I3S2"

# ---- M7 KV dtypes at 262k (runner owns the outdir) -------------------------
log "M7 KV dtypes (int8/q4_0/k8v4) at 262k"
MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 SCRATCH_ROOT="/tmp/v100-medium-$JOB-m7" \
  bash "$HELPER/knob-sweep.sh" --model iq3s --node volta05 --gpus 0,1,2,3 \
    --contexts 262144 --batches 1 --max-context 262144 --max-tokens 128 \
    --variants basefull,kvq4_0,kvk8v4 --budget-min 45 \
    --outdir "$OUT/m7-kv-262k" > "$LOG_LOCAL/m7-kv-262k.runner.log" 2>&1 \
  || log "M7 runner exited non-zero"

# ---- M8 prefill chunks at 131k ---------------------------------------------
log "M8 prefill chunks at 131k"
MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 SCRATCH_ROOT="/tmp/v100-medium-$JOB-m8" \
  bash "$HELPER/knob-sweep.sh" --model iq3s --node volta05 --gpus 0,1,2,3 \
    --contexts 131072 --batches 1 --max-context 131072 --max-tokens 128 \
    --variants basefull,pf4096,pf8192,pf16384,cpushare_auto --budget-min 35 \
    --outdir "$OUT/m8-prefill-131k" > "$LOG_LOCAL/m8-prefill-131k.runner.log" 2>&1 \
  || log "M8 runner exited non-zero"

# ---- stage Coder for M9 and H1-quality (if the main job skipped it) --------
log "staging Coder shard1"
if cp "$MODELS/coder/$COS1" "$PACKS/coder/$COS1"; then
  ln -f "$PACKS/_shared/$SH2" "$PACKS/coder/$COS2"
  log "M9 batch-MTP / prompt lookup (Coder 1x)"
  MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 SCRATCH_ROOT="/tmp/v100-medium-$JOB-m9" \
    bash "$HELPER/knob-sweep.sh" --model coder --node volta05 --gpus 0 \
      --contexts 16384,65536 --batches 1,4 --max-context 65536 --max-tokens 128 \
      --variants basefull,batchmtp,lookup4 --budget-min 40 \
      --outdir "$OUT/m9-batchmtp-lookup-coder" > "$LOG_LOCAL/m9-batchmtp-lookup-coder.runner.log" 2>&1 \
    || log "M9 runner exited non-zero"
else
  log "FATAL: Coder staging failed; M9 skipped"
fi

# ---- H3 rerun: distinct conversation bodies --------------------------------
log "H3 multi-conversation warm (distinct fillers) at 64k/131k/262k"
source "$EXTRAS/lib.sh"
if start_server "$PACKS/iq3s" 262144 4 0,1,2,3 h3b "$LOG_LOCAL/h3b-server"; then
  for pair in "65536 4" "131072 2" "262144 2"; do
    read -r ctx conv <<< "$pair"
    mkdir -p "$OUT/h3-multiconv-distinct/$ctx"
    log "H3b ctx=$ctx conversations=$conv"
    python3 "$EXTRAS/multiconv.py" --base-url "$(server_base)" --model "$SERVER_MODEL" \
      --context "$ctx" --conversations "$conv" --warm-repeats 3 \
      --max-context 262144 --max-tokens 32 \
      --json-out "$OUT/h3-multiconv-distinct/$ctx/cells.jsonl" \
      --summary-out "$OUT/h3-multiconv-distinct/$ctx/summary.csv" \
      --cache-out "$OUT/h3-multiconv-distinct/$ctx/cache.json" \
      >"$OUT/h3-multiconv-distinct/$ctx/run.log" 2>&1 || log "H3b ctx=$ctx failed"
  done
  stop_server; archive_engine "$LOG_LOCAL/h3b-server"
else
  log "H3b server failed"; stop_server
fi
log "done"
