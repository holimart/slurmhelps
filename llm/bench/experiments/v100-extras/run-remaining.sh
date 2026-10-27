#!/usr/bin/env bash
#SBATCH --job-name=strata-v100-rem
#SBATCH --partition=ffa-gpulab
#SBATCH --nodelist=volta05
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=05:00:00
#SBATCH --nice=100
#SBATCH --output=/tmp/strata-v100-rem-%j.out
#
# 2026-10-28 remaining V100 experiments (R1-R3), one allocation on volta05.
#
#   R1  262,144-token NIAH + tool canary for Coder IQ1_M and Q2_0 on 2x V100
#       (falls back to 4x and records the exact launch error if a pack does
#       not fit)
#   R2  NVLink/P2P: nvidia-smi topology/NVLink/counter support, a driver-API
#       peer-bandwidth microbenchmark (256 MiB, GPU0->GPU1 and GPU0->GPU2),
#       and three 4x layer-split cells: default, LD_PRELOAD copy trace, and
#       STRATA_PEER_NO_P2P=1 -- to decide which transport the stage hand-off
#       actually uses.
#   R3  45 min 262k C1 stability loop (cold + warm alternating) with the
#       calibrated/clamped probe; per-request errors, TTFT drift, VRAM, RAM,
#       power.
#
# Node-local discipline (previous runner was cancelled for tailing Lustre):
# every log, engine log, pack and intermediate file lives under /tmp; only
# small JSONL/CSV/JSON/TXT summaries are copied to the results tree after a
# phase or at the end.  This runner never reads a live file on /work.
#
# Results: /work/ffa/holecem/strata-runs/2026-10-28-v100-remaining/
set -uo pipefail

ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
EXTRAS=${V100_EXTRAS_DIR:-$ROOT/llm/bench/experiments/v100-extras}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
JOB=${SLURM_JOB_ID:?must run inside an allocation}
NODE_SHORT=${NODE_SHORT:-volta05}
MODELS=/work/ffa/holecem/models/strata
DATA=$MODELS/_strata-data
RUN=/tmp/v100-rem-$JOB
PACKS=$RUN/packs
OUT=/work/ffa/holecem/strata-runs/2026-10-28-v100-remaining
LOG=$RUN/session.log

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
COS1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
COS2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
Q2S1=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q2S2=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf

mkdir -p "$PACKS"/{iq3s,coder,q2_0,_shared} "$RUN/bin" "$RUN/logs" "$OUT"
START=$(date +%s)
DEADLINE=$((START + 4*3600 + 20*60))   # 5 h allocation, 40 min of slack

log() { printf '[v100-rem] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
now() { date +%s; }
remaining_min() { echo $(( (DEADLINE - $(now)) / 60 )); }
need() { # min item
  local r; r=$(remaining_min)
  [ "$r" -ge "$1" ] || { log "SKIP (only ${r} min left, need $1): $2";
    printf '%s\tskipped-deadline\n' "$2" >> "$OUT/skipped.tsv"; return 1; }
  return 0
}
phase() { log "=== START $* ==="; }
phase_done() { log "=== DONE $* ($(remaining_min) min left) ==="; }

# Copy only small text results to Lustre; never the packs, data dirs or full
# engine logs.  Runs after a phase and at the very end.
publish() { # relative dir under $RUN
  local rel=${1:-.}
  ( cd "$RUN" && find "$rel" -type f \
      ! -path '*/data-*' ! -path '*/packs/*' \
      ! -name '*.engine.log' ! -name '*.gguf' ! -name '*.so' \
      -size -8M -exec cp --parents {} "$OUT/" \; ) 2>/dev/null || true
}

source "$EXTRAS/lib.sh"

# ---- gate ------------------------------------------------------------------
log "job=$JOB node=$(hostname) start=$(date -u +%FT%TZ) deadline=$DEADLINE"
log "R0=${SLURM_NODELIST:-?} gpus=$(nvidia-smi -L 2>/dev/null | wc -l) cpus=${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi -L 2>&1 | tee -a "$LOG"
others=$(squeue -w "$(hostname)" -h -o '%u %i %j' 2>/dev/null | awk -v me="$USER" '$1 != me')
[ -n "$others" ] && log "WARN other jobs on $(hostname): $others"
df -h /tmp | tail -1 | tee -a "$LOG"
printf "phase,start_utc,end_utc,wall_s,note\n" > "$OUT/timing.csv"
printf "item\treason\n" > "$OUT/skipped.tsv"

record() { # phase note
  printf '%s,%s,%s,%s,%s\n' "$1" "${2:-}" "${3:-}" "${4:-}" "${5:-}" >> "$OUT/timing.csv"
}

# ---- staging (once; all packs in parallel, node-local /tmp) ----------------
declare -A STAGE_PID=() STAGE_T0=() STAGE_DONE=()
start_stage() { # name src dst
  STAGE_T0[$1]=$(now)
  cp "$2" "$3" & STAGE_PID[$1]=$!
  log "staging $1 -> $3 (pid ${STAGE_PID[$1]})"
}
finish_stage() { # name
  local name=$1 pid=${STAGE_PID[$1]:-}
  [ -n "${STAGE_DONE[$1]:-}" ] && return 0
  [ -n "$pid" ] || { log "FATAL: no staging job for $name"; return 1; }
  if wait "$pid"; then
    STAGE_DONE[$1]=1
    log "staged $name in $(( $(now) - STAGE_T0[$1] ))s"
    return 0
  fi
  log "FATAL: staging $name failed"
  return 1
}
link_shared() { # packdir shard2name
  ln -f "$PACKS/_shared/$SH2" "$1/$2" || return 1
  log "linked shared PLE table into $1/$2"
}

start_stage ple "$MODELS/iq3s/$I3S2" "$PACKS/_shared/$SH2"
start_stage iq3s "$MODELS/iq3s/$I3S1" "$PACKS/iq3s/$I3S1"
start_stage coder "$MODELS/coder/$COS1" "$PACKS/coder/$COS1"
start_stage q2_0 "$MODELS/q2_0/$Q2S1" "$PACKS/q2_0/$Q2S1"

# ---- helpers ---------------------------------------------------------------
probe_cell() { # outjson ctx c max_tokens rps
  python3 "$HELPER/strata_probe.py" \
    --base-url "http://127.0.0.1:$SERVER_PORT" --model "$SERVER_MODEL" \
    --context "$2" --concurrency "$3" --max-context 262144 \
    --max-tokens "$4" --requests-per-slot "$5" --json-out "$1"
}

r2_cell() { # outdir ctx
  local out=$1 ctx=$2 t0 t1
  t0=$(date +%s%N)
  probe_cell "$out/cells.jsonl" "$ctx" 1 128 2 >>"$out/run.log" 2>&1
  t1=$(date +%s%N)
  printf '%s %s\n' "$t0" "$t1" > "$out/cell-window.txt"
  log "R2 cell $(basename "$out") done (window $(( (t1-t0)/1000000 )) ms)"
}

# ============================ R2a: hardware ================================
phase "R2a hardware topology / NVLink / peer bandwidth"
mkdir -p "$RUN/r2-hardware"
bash "$EXTRAS/p2p.sh" "$RUN/r2-hardware" 2>&1 | tee -a "$LOG"
# Document what the node's nvidia-smi can and cannot report about NVLink.
{
  echo "== nvidia-smi nvlink --help"
  nvidia-smi nvlink --help 2>&1 || true
  echo
  echo "== nvidia-smi nvlink -s (all)"
  nvidia-smi nvlink -s 2>&1 || true
  echo
  echo "== nvidia-smi nvlink -c (all)"
  nvidia-smi nvlink -c 2>&1 || true
} > "$RUN/r2-hardware/nvlink-cli.txt" 2>&1

if gcc -O2 -o "$RUN/bin/p2pbench" "$EXTRAS/p2pbench.c" -ldl >"$RUN/r2-hardware/build-p2pbench.log" 2>&1; then
  "$RUN/bin/p2pbench" --mib 256 --iters 20 \
    --json "$RUN/r2-hardware/p2p-bench.json" \
    > "$RUN/r2-hardware/p2p-bench.txt" 2>&1
  log "R2a peer bandwidth: $(tr '\n' ' ' < "$RUN/r2-hardware/p2p-bench.txt" | sed 's/  */ /g')"
else
  log "R2a FATAL: p2pbench did not build; see build-p2pbench.log"
fi

if gcc -O2 -shared -fPIC -o "$RUN/bin/memcpy-trace.so" "$EXTRAS/memcpy-trace.c" \
     -ldl -lpthread >"$RUN/r2-hardware/build-trace.log" 2>&1; then
  log "R2a memcpy-trace shim built"
else
  log "R2a FATAL: memcpy-trace shim did not build; see build-trace.log"
fi
record "R2a-hardware" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" "$(( $(now) - START ))" "topology+bench"
publish "r2-hardware"

# ============================ R1: 262k quality =============================
run_niah_262k() { # packdir label gpus
  local pack=$1 label=$2 gpus=$3 out="$RUN/r1-quality/$label"
  mkdir -p "$out"
  if start_server "$pack" 262144 1 "$gpus" "$label" "$out/server"; then
    log "R1 $label: server up on $gpus; 262k needle 3x2 + tool canary"
    python3 "$EXTRAS/niah.py" \
      --base-url "$(server_base)" --model "$SERVER_MODEL" \
      --contexts 262144 --positions 10,50,90 --repeats 2 \
      --max-context 262144 --max-tokens 512 --tool-canary \
      --json-out "$out/cells.jsonl" --summary-out "$out/summary.csv" \
      --metrics-out "$out/metrics.jsonl" \
      >"$out/run.log" 2>&1 || log "R1 $label exited non-zero (real errors in cells.jsonl)"
    stop_server
    archive_engine "$out/server"
    return 0
  fi
  # Launch failed: keep the exact error (node-local log), then fall back to 4x.
  local lf; lf=$(ls -1t "$out/server"/launch-*.log 2>/dev/null | head -1 || true)
  {
    echo "start_server failed on gpus=$gpus at $(date -u +%FT%TZ)"
    [ -n "$lf" ] && grep -iE "error|out of memory|no room|insufficient|failed|not enough|cannot|KV" "$lf" | tail -40
  } > "$out/launch-error.txt" 2>&1
  log "R1 $label: launch failed on $gpus; error captured in $out/launch-error.txt"
  stop_server
  if [ "$gpus" != "0,1,2,3" ]; then
    log "R1 $label: retrying on 4x V100"
    if start_server "$pack" 262144 1 "0,1,2,3" "$label-4x" "$out/server-4x"; then
      python3 "$EXTRAS/niah.py" \
        --base-url "$(server_base)" --model "$SERVER_MODEL" \
        --contexts 262144 --positions 10,50,90 --repeats 2 \
        --max-context 262144 --max-tokens 512 --tool-canary \
        --json-out "$out/cells-4x.jsonl" --summary-out "$out/summary-4x.csv" \
        --metrics-out "$out/metrics-4x.jsonl" \
        >"$out/run-4x.log" 2>&1 || log "R1 $label 4x exited non-zero"
      stop_server
      archive_engine "$out/server-4x"
      return 0
    fi
    log "R1 $label: 4x launch also failed"
    stop_server
  fi
  return 1
}

if need 35 "R1 Coder IQ1_M 262k"; then
  if finish_stage coder && finish_stage ple && link_shared "$PACKS/coder" "$COS2"; then
    phase "R1 Coder IQ1_M at 262,144 (2x first)"
    t0=$(now)
    run_niah_262k "$PACKS/coder" coder-262k "0,1" || log "R1 Coder ended with a recorded failure"
    record "R1-coder-262k" "" "" "$(( $(now) - t0 ))" "2x-then-4x"
    publish "r1-quality/coder-262k"
    phase_done "R1 Coder"
  else
    log "SKIP R1 Coder (staging failed)"
    printf 'R1 Coder 262k\tstaging-failed\n' >> "$OUT/skipped.tsv"
  fi
fi

if need 30 "R1 Q2_0 262k"; then
  if finish_stage q2_0 && finish_stage ple && link_shared "$PACKS/q2_0" "$Q2S2"; then
    phase "R1 Q2_0 at 262,144 (2x first)"
    t0=$(now)
    run_niah_262k "$PACKS/q2_0" q2_0-262k "0,1" || log "R1 Q2_0 ended with a recorded failure"
    record "R1-q2_0-262k" "" "" "$(( $(now) - t0 ))" "2x-then-4x"
    publish "r1-quality/q2_0-262k"
    phase_done "R1 Q2_0"
  else
    log "SKIP R1 Q2_0 (staging failed)"
    printf 'R1 Q2_0 262k\tstaging-failed\n' >> "$OUT/skipped.tsv"
  fi
fi

# ============================ R2b: engine transport ========================
if need 25 "R2b engine layer-split transport"; then
  if finish_stage iq3s && finish_stage ple && link_shared "$PACKS/iq3s" "$I3S2"; then
    phase "R2b 4x layer-split transport (32k cell, 3 runs)"
    mkdir -p "$RUN/r2-engine"

    # 1) clean baseline ----------------------------------------------------
    if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2-default "$RUN/r2-engine/default"; then
      r2_cell "$RUN/r2-engine/default" 32768
      stop_server; archive_engine "$RUN/r2-engine/default"
    else
      log "R2b default server failed"; stop_server
    fi

    # 2) copy trace --------------------------------------------------------
    if [ -x "$RUN/bin/memcpy-trace.so" ] || [ -f "$RUN/bin/memcpy-trace.so" ]; then
      export LD_PRELOAD="$RUN/bin/memcpy-trace.so"
      export STRATA_MEMCPY_TRACE="$RUN/r2-engine/trace-default.tsv"
      if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2-trace "$RUN/r2-engine/trace-server"; then
        r2_cell "$RUN/r2-engine/trace-server" 32768
        read -r c0 c1 < "$RUN/r2-engine/trace-server/cell-window.txt"
        python3 "$EXTRAS/trace-summary.py" "$STRATA_MEMCPY_TRACE" \
          --since "$c0" --until "$c1" \
          --json-out "$RUN/r2-engine/trace-default-summary.json" \
          > "$RUN/r2-engine/trace-default-summary.txt" 2>&1
        stop_server; archive_engine "$RUN/r2-engine/trace-server"
      else
        log "R2b trace server failed"; stop_server
      fi
      unset LD_PRELOAD STRATA_MEMCPY_TRACE
    fi

    # 3) no-P2P engine switch ---------------------------------------------
    export STRATA_PEER_NO_P2P=1
    if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2-nop2p "$RUN/r2-engine/nop2p"; then
      r2_cell "$RUN/r2-engine/nop2p" 32768
      stop_server; archive_engine "$RUN/r2-engine/nop2p"
    else
      log "R2b no-P2P server failed"; stop_server
    fi

    # 4) no-P2P + trace, if there is time: shows whether the copy path changes
    if need 15 "R2b no-P2P trace"; then
      export LD_PRELOAD="$RUN/bin/memcpy-trace.so"
      export STRATA_MEMCPY_TRACE="$RUN/r2-engine/trace-nop2p.tsv"
      if start_server "$PACKS/iq3s" 32768 1 0,1,2,3 r2-nop2p-trace "$RUN/r2-engine/nop2p-trace-server"; then
        r2_cell "$RUN/r2-engine/nop2p-trace-server" 32768
        read -r c0 c1 < "$RUN/r2-engine/nop2p-trace-server/cell-window.txt"
        python3 "$EXTRAS/trace-summary.py" "$STRATA_MEMCPY_TRACE" \
          --since "$c0" --until "$c1" \
          --json-out "$RUN/r2-engine/trace-nop2p-summary.json" \
          > "$RUN/r2-engine/trace-nop2p-summary.txt" 2>&1
        stop_server; archive_engine "$RUN/r2-engine/nop2p-trace-server"
      else
        log "R2b no-P2P trace server failed"; stop_server
      fi
      unset LD_PRELOAD STRATA_MEMCPY_TRACE
    fi
    unset STRATA_PEER_NO_P2P
    publish "r2-engine"
    phase_done "R2b"
  else
    log "SKIP R2b (IQ3_S staging failed)"
    printf 'R2b engine transport\tstaging-failed\n' >> "$OUT/skipped.tsv"
  fi
fi

# ============================ R3: 262k stability ===========================
if need 60 "R3 262k stability"; then
  mkdir -p "$RUN/r3-stability"
  phase "R3 45 min 262k stability (cold + warm, C1)"
  if start_server "$PACKS/iq3s" 262144 4 0,1,2,3 r3 "$RUN/r3-stability/server"; then
    bash "$EXTRAS/power-sampler.sh" "$RUN/r3-stability/power.csv" & samp=$!
    python3 "$EXTRAS/stability.py" \
      --base-url "$(server_base)" --model "$SERVER_MODEL" \
      --context 262144 --max-context 262144 --max-tokens 128 \
      --warm-repeats 2 --duration-min 45 \
      --json-out "$RUN/r3-stability/cells.jsonl" \
      --summary-out "$RUN/r3-stability/summary.json" \
      >"$RUN/r3-stability/run.log" 2>&1 || log "R3 stability loop exited non-zero"
    kill "$samp" 2>/dev/null || true
    stop_server; archive_engine "$RUN/r3-stability/server"
    publish "r3-stability"
    phase_done "R3"
  else
    log "SKIP R3 (server failed)"; stop_server
    printf 'R3 262k stability\tserver-failed\n' >> "$OUT/skipped.tsv"
  fi
fi

# ---- flush -----------------------------------------------------------------
TOTAL=$(( $(now) - START ))
log "session done in $((TOTAL/60)) min; results under $OUT"
record "session-total" "$(date -u +%FT%TZ)" "" "$TOTAL" "minutes=$(($TOTAL/60))"
df -h /tmp | tail -1 | tee -a "$LOG"
publish "."
cp "$LOG" "$OUT/session.log" 2>/dev/null || true
cp "/tmp/strata-v100-rem-$JOB.out" "$OUT/slurm-$JOB.out" 2>/dev/null || true
echo "[v100-rem] done"
