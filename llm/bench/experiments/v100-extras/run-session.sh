#!/usr/bin/env bash
#SBATCH --job-name=strata-v100-extras
#SBATCH --partition=ffa-gpulab
#SBATCH --nodelist=volta05
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=11:00:00
#SBATCH --output=/work/ffa/holecem/logs/strata-v100-extras-%j.out
#
# 2026-10-27 V100 extras — one 11 h allocation on volta05, sequential.
#
# Stages the packs once (the shared 28.8 GB PLE shard is single-striped and
# reads at 4-14 MB/s), then runs:
#   H1 quality (one-off niah.py)      H2 sustained (canonical probe)
#   H3 multi-conversation warm        H4 2x IQ3_S (bench-matrix recipe)
#   H5 chain (chain-cache recipe)     H6 power (sampler around H2)
#   M7 KV dtypes (knob-sweep)         M8 prefill chunks (knob-sweep)
#   M9 batch-MTP/lookup (Coder)       M10 topology (p2p.sh)
#   M11 stability (canonical probe loop, last)
# Results: /work/ffa/holecem/strata-runs/2026-10-27-v100-extras/<item>/
set -uo pipefail

ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
EXTRAS=${V100_EXTRAS_DIR:-$ROOT/llm/bench/experiments/v100-extras}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
JOB=${SLURM_JOB_ID:?must run inside an allocation}
MODELS=/work/ffa/holecem/models/strata
DATA=$MODELS/_strata-data
PACKS=/tmp/v100-extras-$JOB
OUT=/work/ffa/holecem/strata-runs/2026-10-27-v100-extras
LOG=$OUT/session.log

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
COS1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
COS2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
Q2S1=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q2S2=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf

mkdir -p "$PACKS/iq3s" "$PACKS/coder" "$PACKS/q2_0" "$PACKS/_shared" "$OUT"
START=$(date +%s)
DEADLINE=$((START + 10*3600))

log() { printf '[extras] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
now() { date +%s; }
remaining_min() { echo $(( (DEADLINE - $(now)) / 60 )); }
need() { # min
  local r; r=$(remaining_min)
  [ "$r" -ge "$1" ] || { log "SKIP (only ${r} min left, need $1): $2"; \
    printf '%s\tskipped-deadline\n' "$2" >> "$OUT/skipped.tsv"; return 1; }
  return 0
}
phase() { log "=== START $* ==="; }
phase_done() { log "=== DONE $* ($(remaining_min) min left) ==="; }
skip() { log "SKIP $*"; printf '%s\tskipped\n' "$*" >> "$OUT/skipped.tsv"; }
record() { # csv line
  printf '%s\n' "$*" >> "$OUT/timing.csv"
}

source "$EXTRAS/lib.sh"

# ---- gate ------------------------------------------------------------------
log "job=$JOB node=$(hostname) start=$(date -u +%FT%TZ) deadline=$DEADLINE"
nvidia-smi -L 2>&1 | tee -a "$LOG"
others=$(squeue -w "$(hostname)" -h -o '%u %i %j' 2>/dev/null | awk -v me="$USER" '$1 != me')
[ -n "$others" ] && log "WARN other jobs on $(hostname): $others"
df -h /tmp | tail -1 | tee -a "$LOG"
echo "phase,stage_s,load_s,wall_s" > "$OUT/timing.csv"
echo "item,reason" > "$OUT/skipped.tsv"

# ---- staging ---------------------------------------------------------------
stage_iq3s() {
  local t0 t1 pids=()
  t0=$(now)
  log "staging $I3S1 and the shared PLE table"
  cp "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1" & pids+=($!)
  cp "$MODELS/iq3s/$I3S2" "$PACKS/_shared/$SH2" & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { log "FATAL: iq3s staging failed"; return 1; }; done
  ln -f "$PACKS/_shared/$SH2" "$PACKS/iq3s/$I3S2"
  t1=$(now)
  log "staged iq3s in $((t1-t0))s: $(du -sh "$PACKS/iq3s" | cut -f1)"
  record "stage-iq3s,$((t1-t0)),,"
}
stage_coder() {
  local t0 t1
  t0=$(now)
  log "staging $COS1"
  cp "$MODELS/coder/$COS1" "$PACKS/coder/$COS1" || return 1
  ln -f "$PACKS/_shared/$SH2" "$PACKS/coder/$COS2" || return 1
  t1=$(now)
  log "staged coder in $((t1-t0))s"
  record "stage-coder,$((t1-t0)),,"
}
stage_q2() {
  local t0 t1
  t0=$(now)
  log "staging $Q2S1"
  cp "$MODELS/q2_0/$Q2S1" "$PACKS/q2_0/$Q2S1" || return 1
  ln -f "$PACKS/_shared/$SH2" "$PACKS/q2_0/$Q2S2" || return 1
  t1=$(now)
  log "staged q2_0 in $((t1-t0))s"
  record "stage-q2_0,$((t1-t0)),,"
}

# ---- probe wrapper ---------------------------------------------------------
probe_cell() { # outjson ctx c max_tokens rps
  python3 "$HELPER/strata_probe.py" \
    --base-url "http://127.0.0.1:$SERVER_PORT" --model "$SERVER_MODEL" \
    --context "$2" --concurrency "$3" --max-context 262144 \
    --max-tokens "$4" --requests-per-slot "$5" --json-out "$1"
}

# ============================ H1/H2/H3 =====================================
stage_iq3s || skip "all IQ3_S phases (staging failed)"
if [ -f "$PACKS/iq3s/$I3S1" ] && [ -f "$PACKS/iq3s/$I3S2" ]; then

  # H6 idle baseline: 30 s before any server exists
  mkdir -p "$OUT/h6-power"
  log "H6 idle power baseline (30 s)"
  bash "$EXTRAS/power-sampler.sh" "$OUT/h6-power/idle.csv" 30 || true

  phase "long-context 4x resident server + H1/H2/H3"
  t_load0=$(now)
  if start_server "$PACKS/iq3s" 262144 4 0,1,2,3 longctx "$OUT/longctx-4x"; then
    load_s=$(( $(now) - t_load0 ))
    record "load-longctx-4x,,$load_s,"
    log "long-context server ready in ${load_s}s"

    # ---- H1 quality ------------------------------------------------------
    if need 25 "H1 quality IQ3_S"; then
      mkdir -p "$OUT/h1-quality/iq3s"
      log "H1 needle + tool canary: 64k/131k/262k, 10/50/90 %, 2 cold repeats"
      python3 "$EXTRAS/niah.py" \
        --base-url "$(server_base)" --model "$SERVER_MODEL" \
        --contexts 65536,131072,262144 --positions 10,50,90 --repeats 2 \
        --max-context 262144 --max-tokens 32 --tool-canary \
        --json-out "$OUT/h1-quality/iq3s/cells.jsonl" \
        --summary-out "$OUT/h1-quality/iq3s/summary.csv" \
        --metrics-out "$OUT/h1-quality/iq3s/metrics.jsonl" \
        >"$OUT/h1-quality/iq3s/run.log" 2>&1 \
        || log "H1 exited non-zero (real errors are in cells.jsonl)"
      phase_done "H1 quality IQ3_S"
    fi

    # ---- H2 sustained + H6 power ----------------------------------------
    if need 15 "H2 sustained + H6 power"; then
      mkdir -p "$OUT/h2-sustained"
      log "H2 sustained: 64k C1 then 262k C1, 1024 tokens, 3 requests"
      probe_cell "$OUT/h2-sustained/cells.jsonl" 65536 1 1024 3 \
        >"$OUT/h2-sustained/run-64k.log" 2>&1 || log "H2 64k failed"
      t0=$(date +%s.%N)
      bash "$EXTRAS/power-sampler.sh" "$OUT/h6-power/cell-262k.csv" & samp=$!
      probe_cell "$OUT/h2-sustained/cells.jsonl" 262144 1 1024 3 \
        >"$OUT/h2-sustained/run-262k.log" 2>&1 || log "H2 262k failed"
      t1=$(date +%s.%N)
      kill "$samp" 2>/dev/null || true
      printf '%s %s\n' "$t0" "$t1" > "$OUT/h6-power/bracket-262k.txt"
      python3 - "$OUT/h2-sustained/cells.jsonl" > "$OUT/h6-power/tokens-262k.txt" <<'PY'
import json,sys
n=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("context")==262144:
        n += d.get("completion_tokens") or 0
print(n)
PY
      cp "$OUT/h2-sustained/cells.jsonl" "$OUT/h6-power/h2-cells.jsonl" 2>/dev/null || true
      phase_done "H2"
    fi

    # ---- H3 multi-conversation warm -------------------------------------
    if need 15 "H3 multi-conversation warm"; then
      mkdir -p "$OUT/h3-multiconv"
      for pair in "65536 4" "131072 2" "262144 2"; do
        read -r ctx conv <<< "$pair"
        mkdir -p "$OUT/h3-multiconv/$ctx"
        log "H3 ctx=$ctx conversations=$conv warm-repeat=3 (interleaved)"
        python3 "$EXTRAS/multiconv.py" \
          --base-url "$(server_base)" --model "$SERVER_MODEL" \
          --context "$ctx" --conversations "$conv" --warm-repeats 3 \
          --max-context 262144 --max-tokens 32 \
          --json-out "$OUT/h3-multiconv/$ctx/cells.jsonl" \
          --summary-out "$OUT/h3-multiconv/$ctx/summary.csv" \
          --cache-out "$OUT/h3-multiconv/$ctx/cache.json" \
          >"$OUT/h3-multiconv/$ctx/run.log" 2>&1 \
          || log "H3 ctx=$ctx exited non-zero"
      done
      phase_done "H3"
    fi
    stop_server
    archive_engine "$OUT/longctx-4x"
  else
    log "long-context server failed; H1/H2/H3/H6 skipped"
    skip "H1/H2/H3/H6 (long-context server failed)"
    stop_server
  fi
fi

# ============================ H4: 2x IQ3_S =================================
if need 25 "H4 2x IQ3_S"; then
  phase "H4 2x IQ3_S residency/crossover"
  mkdir -p "$OUT/h4-2x-iq3s"
  if MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 \
     SCRATCH_ROOT="/tmp/v100-x-$JOB-h4" \
     bash "$HELPER/bench-matrix.sh" --model iq3s --gpus 0,1 \
       --cells "1024:1,1024:2,4096:1,4096:2,16384:1,16384:2,65536:1,65536:2,131072:1,131072:2" \
       --max-tokens 128 --requests-per-slot 3 \
       --outdir "$OUT/h4-2x-iq3s" --budget-min 45 \
       >"$OUT/h4-2x-iq3s/runner.log" 2>&1; then
    phase_done "H4"
  else
    log "H4 exited non-zero; see $OUT/h4-2x-iq3s/runner.log"
    skip "H4 (runner failed; real error in runner.log)"
  fi
fi

# ============================ H5: chain ====================================
if need 20 "H5 chain 4x"; then
  phase "H5 checkpoint chain 4x"
  mkdir -p "$OUT/h5-chain-4x"
  if MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 \
     SCRATCH_ROOT="/tmp/v100-x-$JOB-h5" \
     bash "$HELPER/chain-cache.sh" --model iq3s --node "$(hostname)" \
       --gpus 0,1,2,3 --stops 1024,100000,200000,262144 \
       --tokens-per-turn 512 --chain-last-repeats 1 --max-context 262144 \
       --outdir "$OUT/h5-chain-4x" --budget-min 50 \
       >"$OUT/h5-chain-4x/runner.log" 2>&1; then
    phase_done "H5"
  else
    log "H5 exited non-zero; see $OUT/h5-chain-4x/runner.log"
    skip "H5 (runner failed; real error in runner.log)"
  fi
fi

# ============================ H6 power report ==============================
phase "H6 power / tokens-per-joule (from the H2 bracket)"
if [ -f "$OUT/h6-power/bracket-262k.txt" ]; then
  read -r t0 t1 < "$OUT/h6-power/bracket-262k.txt"
  toks=$(cat "$OUT/h6-power/tokens-262k.txt" 2>/dev/null || echo 0)
  python3 "$EXTRAS/power-report.py" --samples "$OUT/h6-power/cell-262k.csv" \
    --start "$t0" --end "$t1" --tokens "${toks:-0}" \
    --idle "$OUT/h6-power/idle.csv" --json-out "$OUT/h6-power/power-262k.json" \
    >"$OUT/h6-power/report.txt" 2>&1 || log "H6 power report failed"
  phase_done "H6"
else
  skip "H6 (no H2 bracket; H2 did not run)"
fi

# ============================ M7 KV dtypes =================================
if need 30 "M7 KV dtypes 262k"; then
  phase "M7 KV dtypes at 262k"
  mkdir -p "$OUT/m7-kv-262k"
  if MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 \
     SCRATCH_ROOT="/tmp/v100-x-$JOB-m7" \
     bash "$HELPER/knob-sweep.sh" --model iq3s --node "$(hostname)" \
       --gpus 0,1,2,3 --contexts 262144 --batches 1 --max-context 262144 \
       --max-tokens 128 --variants basefull,kvq4_0,kvk8v4 \
       --outdir "$OUT/m7-kv-262k" --budget-min 45 \
       >"$OUT/m7-kv-262k/runner.log" 2>&1; then
    phase_done "M7"
  else
    log "M7 exited non-zero; see $OUT/m7-kv-262k/runner.log"
    skip "M7 (runner failed; real error in runner.log)"
  fi
fi

# ============================ M8 prefill chunks ============================
if need 30 "M8 prefill chunks 131k"; then
  phase "M8 prefill chunk sizes at 131k"
  mkdir -p "$OUT/m8-prefill-131k"
  if MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 \
     SCRATCH_ROOT="/tmp/v100-x-$JOB-m8" \
     bash "$HELPER/knob-sweep.sh" --model iq3s --node "$(hostname)" \
       --gpus 0,1,2,3 --contexts 131072 --batches 1 --max-context 131072 \
       --max-tokens 128 --variants basefull,pf4096,pf8192,pf16384,cpushare_auto \
       --outdir "$OUT/m8-prefill-131k" --budget-min 35 \
       >"$OUT/m8-prefill-131k/runner.log" 2>&1; then
    phase_done "M8"
  else
    log "M8 exited non-zero; see $OUT/m8-prefill-131k/runner.log"
    skip "M8 (runner failed; real error in runner.log)"
  fi
fi

# ============================ M9 Coder =====================================
CODER_OK=0
if need 35 "M9 Coder batch-MTP/lookup"; then
  phase "M9 batch-MTP / prompt lookup (Coder 1x)"
  if stage_coder; then
    CODER_OK=1
    mkdir -p "$OUT/m9-batchmtp-lookup-coder"
    if MODELS_ROOT="$PACKS" DATA_DIR="$DATA" COPY_MODELS=0 \
       SCRATCH_ROOT="/tmp/v100-x-$JOB-m9" \
       bash "$HELPER/knob-sweep.sh" --model coder --node "$(hostname)" \
         --gpus 0 --contexts 16384,65536 --batches 1,4 --max-context 65536 \
         --max-tokens 128 --variants basefull,batchmtp,lookup4 \
         --outdir "$OUT/m9-batchmtp-lookup-coder" --budget-min 40 \
         >"$OUT/m9-batchmtp-lookup-coder/runner.log" 2>&1; then
      phase_done "M9"
    else
      log "M9 exited non-zero; see $OUT/m9-batchmtp-lookup-coder/runner.log"
      skip "M9 (runner failed)"
    fi
  else
    skip "M9 (Coder staging failed)"
  fi
fi

# ============================ M10 topology =================================
phase "M10 topology/NVLink/P2P"
bash "$EXTRAS/p2p.sh" "$OUT/m10-topology" 2>&1 | tee -a "$LOG"
phase_done "M10"

# ============================ H1 Coder (if time) ===========================
if [ "$CODER_OK" = 1 ] && need 20 "H1 Coder quality"; then
  phase "H1 Coder quality 64k/131k"
  mkdir -p "$OUT/h1-quality/coder"
  if start_server "$PACKS/coder" 131072 4 0,1 coder-longctx "$OUT/h1-quality/coder/server"; then
    python3 "$EXTRAS/niah.py" \
      --base-url "$(server_base)" --model "$SERVER_MODEL" \
      --contexts 65536,131072 --positions 10,50,90 --repeats 2 \
      --max-context 131072 --max-tokens 32 --tool-canary \
      --json-out "$OUT/h1-quality/coder/cells.jsonl" \
      --summary-out "$OUT/h1-quality/coder/summary.csv" \
      --metrics-out "$OUT/h1-quality/coder/metrics.jsonl" \
      >"$OUT/h1-quality/coder/run.log" 2>&1 || log "H1 Coder exited non-zero"
    stop_server; archive_engine "$OUT/h1-quality/coder/server"
    phase_done "H1 Coder"
  else
    skip "H1 Coder (server failed)"; stop_server
  fi
fi

# ============================ H1 Q2_0 (if time) ============================
if [ "$CODER_OK" = 1 ] && need 20 "H1 Q2_0 quality"; then
  phase "H1 Q2_0 quality 64k/131k"
  rm -f "$PACKS/coder/$COS1"   # free 29.6 GB for Q2_0
  if stage_q2; then
    mkdir -p "$OUT/h1-quality/q2_0"
    if start_server "$PACKS/q2_0" 131072 4 0,1 q2-longctx "$OUT/h1-quality/q2_0/server"; then
      python3 "$EXTRAS/niah.py" \
        --base-url "$(server_base)" --model "$SERVER_MODEL" \
        --contexts 65536,131072 --positions 10,50,90 --repeats 2 \
        --max-context 131072 --max-tokens 32 --tool-canary \
        --json-out "$OUT/h1-quality/q2_0/cells.jsonl" \
        --summary-out "$OUT/h1-quality/q2_0/summary.csv" \
        --metrics-out "$OUT/h1-quality/q2_0/metrics.jsonl" \
        >"$OUT/h1-quality/q2_0/run.log" 2>&1 || log "H1 Q2_0 exited non-zero"
      stop_server; archive_engine "$OUT/h1-quality/q2_0/server"
      phase_done "H1 Q2_0"
    else
      skip "H1 Q2_0 (server failed)"; stop_server
    fi
    rm -f "$PACKS/q2_0/$Q2S1"
  else
    skip "H1 Q2_0 (staging failed)"
  fi
fi

# ============================ M11 stability (last) =========================
if need 45 "M11 stability"; then
  phase "M11 long-run stability 131k/262k alternating cold/warm"
  mkdir -p "$OUT/m11-stability"
  if start_server "$PACKS/iq3s" 262144 4 0,1,2,3 m11 "$OUT/m11-stability/server"; then
    bash "$EXTRAS/power-sampler.sh" "$OUT/m11-stability/power.csv" & samp=$!
    echo "ts,mem_total_mib,mem_available_mib" > "$OUT/m11-stability/ram.csv"
    i=0
    while [ "$(remaining_min)" -ge 20 ]; do
      i=$((i+1))
      for mode in cold warm; do
        for ctx in 131072 262144; do
          if [ "$mode" = cold ]; then
            probe_cell "$OUT/m11-stability/cells.jsonl" "$ctx" 1 128 1 \
              >>"$OUT/m11-stability/run.log" 2>&1 || true
          else
            python3 "$HELPER/strata_probe.py" \
              --base-url "$(server_base)" --model "$SERVER_MODEL" \
              --context "$ctx" --concurrency 1 --max-context 262144 \
              --max-tokens 128 --mode warm --warm-repeats 2 \
              --json-out "$OUT/m11-stability/cells.jsonl" \
              >>"$OUT/m11-stability/run.log" 2>&1 || true
          fi
          printf '%s,%s,%s\n' "$(date -u +%FT%TZ)" \
            "$(free -m | awk 'NR==2{print $2}')" \
            "$(free -m | awk 'NR==2{print $7}')" >> "$OUT/m11-stability/ram.csv"
        done
      done
      log "M11 iteration $i done ($(remaining_min) min left)"
      [ "$i" -ge 24 ] && break
    done
    kill "$samp" 2>/dev/null || true
    stop_server; archive_engine "$OUT/m11-stability/server"
    phase_done "M11"
  else
    skip "M11 (server failed)"; stop_server
  fi
fi

# ---- final timing ----------------------------------------------------------
TOTAL=$(( $(now) - START ))
log "session done in $((TOTAL/60)) min; results under $OUT"
printf 'session-total,,,%s\n' "$TOTAL" >> "$OUT/timing.csv"
df -h /tmp | tail -1 | tee -a "$LOG"
