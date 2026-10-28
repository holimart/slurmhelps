#!/usr/bin/env bash
#SBATCH --job-name=amd-mi210-run
#SBATCH --partition=ffa-preempt
#SBATCH --nodelist=r50
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=1200G
#SBATCH --time=11:00:00
#SBATCH --output=/tmp/amd-mi210-run-%j.out
#
# Step 3 — measure Qwen3.8-Flash-Next on 2x AMD MI210 (gfx90a) with the
# b11785 HIP build.  One allocation, packs sequential, fully resident.
#
# Hard rule: job logs live on node-local /dev/shm.  Only small summaries and
# the CSV/JSON cells are copied to Lustre at the end of each phase.  Never
# tail/cat/grep a live log on Lustre (not even the copy target).
set -uo pipefail

JOB=${SLURM_JOB_ID:?inside allocation only}
ROOT=${SLURMHELPS_ROOT:-/work/ffa/holecem/slurmhelps-fleet}
EXTRAS=${AMD_MI210_DIR:-$ROOT/llm/bench/experiments/amd-mi210}
MODELS=/work/ffa/holecem/models/strata
PREFIX=/work/ffa/holecem/build/llama-hip-b11785-gfx90a
OUT=/work/ffa/holecem/strata-runs/2026-10-27-amd-mi210
LOCAL=/dev/shm/amd-mi210-$JOB
BIN=$LOCAL/bin
DATA=$LOCAL/models
LOGS=$LOCAL/logs
CELLS=$OUT/cells
PORT=8099
SERVER_PID=

I3S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
I3S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
COS1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
COS2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
Q2S1=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q2S2=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
SH2=Qwen3.8-Flash-Next-GSQ-RCO-00002-of-00002.gguf

mkdir -p "$LOCAL" "$DATA/_shared" "$LOGS" "$OUT"/{cells,logs,power,rocminfo,build}

exec > >(tee "$LOCAL/session.log") 2>&1
START=$(date +%s)
DEADLINE=$((START + 10*3600))
log() { printf '[mi210] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
now() { date +%s; }
remaining_min() { echo $(( (DEADLINE - $(now)) / 60 )); }
need() { local r; r=$(remaining_min); [ "$r" -ge "$1" ] || { log "SKIP (only ${r} min left, need $1): $2"; \
  printf '%s\tskipped-deadline\n' "$2" >> "$OUT/skipped.tsv"; return 1; }; }
phase() { log "=== START $* ==="; }
phase_done() { log "=== DONE $* ($(remaining_min) min left) ==="; }
skip() { log "SKIP $*"; printf '%s\tskipped\n' "$*" >> "$OUT/skipped.tsv"; }

stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap 'stop_server' EXIT

wait_ready() { # pid logfile timeout_s
  local pid=$1 log=$2 tmo=${3:-600} t=0
  while [ $t -lt "$tmo" ]; do
    kill -0 "$pid" 2>/dev/null || { log "server died during load"; return 1; }
    if python3 - "$PORT" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=2) as r:
        sys.exit(0 if json.loads(r.read().decode()).get("status") in ("ok","no slot available") else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
    sleep 5; t=$((t+5))
  done
  return 1
}

start_server() { # tag extra-args...
  local tag=$1; shift
  local logf=$LOGS/server-$tag.log
  local t0=$SECONDS
  "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on "$@" \
      --host 127.0.0.1 --port "$PORT" --metrics > "$logf" 2>&1 &
  SERVER_PID=$!
  if wait_ready "$SERVER_PID" "$logf" 900; then
    local secs=$((SECONDS-t0))
    log "server $tag ready in ${secs}s (pid $SERVER_PID)"
    echo "$tag,$secs" >> "$OUT/load-times.csv"
    # small warm-up request, not measured
    python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 512 \
        --n-predict 8 --out "$LOCAL/warmup-$tag.jsonl" --timeout 600 >/dev/null 2>&1 || true
    return 0
  fi
  log "server $tag FAILED to become ready; last log lines:"
  tail -40 "$logf" 2>/dev/null || true
  stop_server
  return 1
}

residency() { # tag
  local tag=$1
  {
    echo "== $tag =="
    grep -E "offloaded|offloading|model buffer size|KV self size|n_ctx|n_ctx_per_seq|graph splits|using .* compute|CPU_Mapped|ROCm[0-9]" "$LOGS/server-$tag.log" | head -40
    echo "-- vram --"
    rocm-smi --showmeminfo vram 2>/dev/null
  } > "$OUT/residency-$tag.txt" 2>&1
  cat "$OUT/residency-$tag.txt" >> "$LOCAL/session.log"
}

run_bench() { # pack model ctx...
  local pack=$1 model=$2; shift 2
  for ctx in "$@"; do
    # b11785 cannot restore the hybrid KV state between llama-bench
    # repetitions ("failed to restore kv cache" then a GPU fault), so every
    # bench cell is a single repetition; repeats live in the server cells.
    local r=1
    local outj=$CELLS/bench-$pack-pp$ctx.json
    local outt=$CELLS/bench-$pack-tg$ctx.json
    if [ -s "$outj" ] && [ -s "$outt" ]; then log "bench $pack ctx=$ctx already present"; continue; fi
    [ "$(remaining_min)" -ge 5 ] || { skip "bench $pack ctx=$ctx"; continue; }
    # b11785 QSA faults when the indexer cache ends on a block boundary
    # (n_prompt % 4 == 0).  Use ctx-1 for the actual test length; the JSON
    # reports the real n_prompt/n_depth.
    local pctx=$((ctx-1))
    log "llama-bench $pack pp ctx=$ctx (actual $pctx) r=$r"
    timeout 1800 "$BIN/llama-bench" -m "$model" -ngl 99 -fa on -ctk f16 -ctv f16 \
        -p "$pctx" -n 0 -r "$r" -o json > "$outj" 2> "$LOGS/bench-$pack-pp$ctx.err"
    local rc=$?
    [ $rc -ne 0 ] && { log "bench pp $pack ctx=$ctx rc=$rc"; tail -20 "$LOGS/bench-$pack-pp$ctx.err" || true; }
    log "llama-bench $pack tg ctx=$ctx (actual $pctx) r=$r"
    timeout 1800 "$BIN/llama-bench" -m "$model" -ngl 99 -fa on -ctk f16 -ctv f16 \
        -p 0 -n 128 -d "$pctx" -r "$r" -o json > "$outt" 2> "$LOGS/bench-$pack-tg$ctx.err"
    rc=$?
    [ $rc -ne 0 ] && { log "bench tg $pack ctx=$ctx rc=$rc"; tail -20 "$LOGS/bench-$pack-tg$ctx.err" || true; }
    cp "$outj" "$outt" "$OUT/logs/" 2>/dev/null
  done
}

run_cells() { # pack model
  local pack=$1 model=$2
  local tag=$pack-f16
  if [ -s "$CELLS/server-$pack-f16.jsonl" ]; then log "server cells $pack already present"; else
    phase "server C1 $pack f16"
    if start_server "$tag" -c 262144 -np 1 -ctk f16 -ctv f16; then
      residency "$tag"
      local first=1
      for ctx in 4096 16384 65536 131072 261888; do
        [ "$(remaining_min)" -ge 5 ] || { skip "server $pack ctx=$ctx"; break; }
        local reps=2; [ "$ctx" -ge 131072 ] && reps=1
        log "probe $pack C1 ctx=$ctx reps=$reps"
        timeout 2400 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx "$ctx" \
            --n-predict 128 --repeats "$reps" --label "$pack-f16-c1" \
            --out "$CELLS/server-$pack-f16.jsonl" --timeout 3600 || log "probe $pack ctx=$ctx FAILED"
        if [ "$first" = 1 ]; then
          # idle window is sampled before the first measured prompt of the first server
          first=0
        fi
      done
    fi
    stop_server
    phase_done "server C1 $pack f16"
  fi

  # ---- power cells (IQ3_S only; the other packs reuse the same shape) -------
  if [ "$pack" = iq3s ] && [ ! -s "$OUT/power/cell-16k.json" ]; then
    phase "power $pack"
    if start_server "$tag-power" -c 262144 -np 1 -ctk f16 -ctv f16; then
      residency "$tag-power"
      local t0 t1
      t0=$(date +%s.%N)
      timeout 2400 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 16384 \
          --n-predict 512 --repeats 1 --label "$pack-power-16k" \
          --out "$LOCAL/power-cell-16k.jsonl" --timeout 3600 >/dev/null || true
      t1=$(date +%s.%N)
      cp "$LOCAL/power-cell-16k.jsonl" "$OUT/power/"
      python3 "$EXTRAS/power-report.py" --csv "$LOCAL/power-$JOB.csv" --start "$t0" --end "$t1" \
          --tokens "$(python3 -c "import json,sys;print(sum(json.loads(l)['predicted_n'] for l in open('$LOCAL/power-cell-16k.jsonl')))" 2>/dev/null || echo 0)" \
          --decode-tps "$(python3 -c "import json;d=json.loads(open('$LOCAL/power-cell-16k.jsonl').readline());print(d['decode_tps'])" 2>/dev/null || echo 0)" \
          --idle-start "$IDLE_T0" --idle-end "$IDLE_T1" \
          --json "$OUT/power/cell-16k.json" > "$LOGS/power-cell-16k.txt" 2>&1

      t0=$(date +%s.%N)
      timeout 2400 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx 261888 \
          --n-predict 256 --repeats 1 --label "$pack-power-262k" \
          --out "$LOCAL/power-cell-262k.jsonl" --timeout 3600 >/dev/null || true
      t1=$(date +%s.%N)
      cp "$LOCAL/power-cell-262k.jsonl" "$OUT/power/"
      python3 "$EXTRAS/power-report.py" --csv "$LOCAL/power-$JOB.csv" --start "$t0" --end "$t1" \
          --tokens "$(python3 -c "import json;print(sum(json.loads(l)['predicted_n'] for l in open('$LOCAL/power-cell-262k.jsonl')))" 2>/dev/null || echo 0)" \
          --decode-tps "$(python3 -c "import json;d=json.loads(open('$LOCAL/power-cell-262k.jsonl').readline());print(d['decode_tps'])" 2>/dev/null || echo 0)" \
          --idle-start "$IDLE_T0" --idle-end "$IDLE_T1" \
          --json "$OUT/power/cell-262k.json" > "$LOGS/power-cell-262k.txt" 2>&1
      cp "$LOGS"/power-cell-*.txt "$OUT/power/" 2>/dev/null
    fi
    stop_server
    phase_done "power $pack"
  fi

  # ---- q8_0 KV variant (IQ3_S only) ---------------------------------------
  if [ "$pack" = iq3s ] && [ ! -s "$CELLS/server-$pack-q8kv.jsonl" ]; then
    phase "q8_0 KV $pack"
    if start_server "$pack-q8kv" -c 262144 -np 1 -ctk q8_0 -ctv q8_0; then
      residency "$pack-q8kv"
      for ctx in 65536 261888; do
        log "probe $pack q8_0 KV ctx=$ctx"
        timeout 2400 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx "$ctx" \
            --n-predict 128 --repeats 1 --label "$pack-q8kv-c1" \
            --out "$CELLS/server-$pack-q8kv.jsonl" --timeout 3600 || log "q8kv $ctx FAILED"
      done
    else
      printf 'q8_0 KV: server did not start\n' > "$OUT/q8kv-failure.txt"
    fi
    stop_server
    phase_done "q8_0 KV $pack"
  fi

  # ---- concurrency at 4k/16k (IQ3_S) --------------------------------------
  if [ "$pack" = iq3s ] && [ ! -s "$CELLS/server-$pack-np2.jsonl" ]; then
    phase "concurrency $pack -np 2"
    if start_server "$pack-np2" -c 262144 -np 2 -ctk f16 -ctv f16; then
      residency "$pack-np2"
      for ctx in 4096 16384; do
        log "probe $pack C2 ctx=$ctx"
        timeout 2400 python3 "$EXTRAS/probe_client.py" --url "http://127.0.0.1:$PORT" --ctx "$ctx" \
            --n-predict 128 --parallel 2 --repeats 2 --label "$pack-f16-c2" \
            --out "$CELLS/server-$pack-np2.jsonl" --timeout 3600 || log "C2 $ctx FAILED"
      done
    fi
    stop_server
    phase_done "concurrency $pack"
  fi
}

# ---------------------------------------------------------------------------
log "job=$JOB node=$(hostname) start=$(date -u +%FT%TZ) deadline=$DEADLINE"
printf "phase,stage_s,load_s,wall_s\n" > "$OUT/timing.csv"
printf "item\treason\n" > "$OUT/skipped.tsv"
printf "tag,load_s\n" > "$OUT/load-times.csv"

echo "== gate ==" | tee "$LOGS/gate.txt"
hipcc --version | head -2 | tee -a "$LOGS/gate.txt"
rocm-smi --showproductname --showhw 2>&1 | tee -a "$LOGS/gate.txt" | head -30
[ -x "$PREFIX/bin/llama-server" ] || { log "FATAL build missing at $PREFIX"; exit 2; }
[ -x "$PREFIX/bin/llama-bench" ] || { log "FATAL llama-bench missing"; exit 2; }

# binaries to node-local (rpath in the build tree is /tmp/amd-mi210-build-<job>,
# so always run with the copied libs on LD_LIBRARY_PATH)
mkdir -p "$BIN"
cp -a "$PREFIX/bin/." "$BIN/"
[ -d "$PREFIX/lib" ] && cp -a "$PREFIX/lib/." "$BIN/"
export LD_LIBRARY_PATH="$BIN:$BIN/lib:${LD_LIBRARY_PATH:-}"
"$BIN/llama-server" --version 2>&1 | head -10 | tee -a "$LOGS/gate.txt"
"$BIN/llama-bench" --help 2>&1 | grep -E "^\s+-[pndrco]," | head -20 | tee -a "$LOGS/gate.txt"
cp "$LOGS/gate.txt" "$OUT/gate.txt"

# power sampler self-test: prove hwmon/rocm-smi gives rows before the long run
python3 "$EXTRAS/power_sampler.py" --out "$LOCAL/power-selftest.csv" --duration 3 >/dev/null 2>&1
if [ "$(wc -l < "$LOCAL/power-selftest.csv" 2>/dev/null || echo 0)" -lt 3 ]; then
  log "WARN power sampler produced no useful rows"
else
  log "power sampler self-test OK: $(tail -n +2 "$LOCAL/power-selftest.csv" | head -2 | tr '\n' ' ')"
fi

# pin check: exact sizes from the manifest, no full sha256 (Lustre read)
check_size() { local f=$1 want=$2; local got; got=$(stat -c %s "$f" 2>/dev/null || echo 0); \
  [ "$got" = "$want" ] || { log "FATAL size mismatch $f got=$got want=$want"; exit 2; }; }
check_size "$MODELS/iq3s/$I3S1" 54817524224
check_size "$MODELS/iq3s/$I3S2" 28800138432
check_size "$MODELS/coder/$COS1" 29608446496
check_size "$MODELS/coder/$COS2" 28800138432
check_size "$MODELS/q2_0/$Q2S1" 37623740192
check_size "$MODELS/q2_0/$Q2S2" 28800138432
log "size pins OK"

# rocminfo snapshot in the result dir (small)
rocminfo > "$OUT/rocminfo/rocminfo.txt" 2>&1 || true
rocm-smi --showtopo > "$OUT/rocminfo/topo.txt" 2>&1 || true
rocm-smi --showhw > "$OUT/rocminfo/hw.txt" 2>&1 || true
cp "$LOCAL/session.log" "$OUT/logs/"

# ---- staging (Lustre -> /dev/shm, node-local) -----------------------------
stage_one() { # src dst
  local src=$1 dst=$2 t0 t1
  [ -s "$dst" ] && { log "already staged $dst"; return 0; }
  t0=$(now)
  log "staging $(basename "$src") -> $dst"
  cp "$src" "$dst" || { log "STAGING FAILED $src"; return 1; }
  t1=$(now)
  echo "stage-$(basename "$dst"),$((t1-t0)),," >> "$OUT/timing.csv"
}
phase "staging"
mkdir -p "$DATA/iq3s" "$DATA/coder" "$DATA/q2_0"
stage_one "$MODELS/iq3s/$I3S2" "$DATA/_shared/$SH2" || exit 2
stage_one "$MODELS/iq3s/$I3S1" "$DATA/iq3s/$I3S1"
ln -f "$DATA/_shared/$SH2" "$DATA/iq3s/$I3S2" 2>/dev/null || cp "$DATA/_shared/$SH2" "$DATA/iq3s/$I3S2"
stage_one "$MODELS/coder/$COS1" "$DATA/coder/$COS1"
ln -f "$DATA/_shared/$SH2" "$DATA/coder/$COS2" 2>/dev/null || cp "$DATA/_shared/$SH2" "$DATA/coder/$COS2"
stage_one "$MODELS/q2_0/$Q2S1" "$DATA/q2_0/$Q2S1"
ln -f "$DATA/_shared/$SH2" "$DATA/q2_0/$Q2S2" 2>/dev/null || cp "$DATA/_shared/$SH2" "$DATA/q2_0/$Q2S2"
df -h /dev/shm | tail -1
phase_done "staging"

# power sampler for the whole run: start it first, then take the idle window
python3 "$EXTRAS/power_sampler.py" --out "$LOCAL/power-$JOB.csv" --duration - > "$LOGS/power-sampler.log" 2>&1 &
POWER_PID=$!
IDLE_T0=$(date +%s.%N)
sleep 30
IDLE_T1=$(date +%s.%N)
log "power sampler pid=$POWER_PID idle_window=$IDLE_T0..$IDLE_T1"
cp "$LOCAL/power-$JOB.csv" "$OUT/power/" 2>/dev/null || true

# ---- pack 1: IQ3_S --------------------------------------------------------
MODEL=$DATA/iq3s/$I3S1
run_bench iq3s "$MODEL" 4096 16384 65536 131072 261888
run_cells iq3s "$MODEL"

# ---- pack 2: Coder IQ1_M --------------------------------------------------
if need 45 "coder pack"; then
  MODEL=$DATA/coder/$COS1
  run_bench coder "$MODEL" 4096 16384 65536 131072 261888
  run_cells coder "$MODEL"
fi

# ---- pack 3: Q2_0 ---------------------------------------------------------
if need 30 "q2_0 pack"; then
  MODEL=$DATA/q2_0/$Q2S1
  run_bench q2_0 "$MODEL" 4096 16384 65536 131072 261888
  run_cells q2_0 "$MODEL"
fi

# ---- summary --------------------------------------------------------------
kill "$POWER_PID" 2>/dev/null || true
wait "$POWER_PID" 2>/dev/null || true
cp "$LOCAL/power-$JOB.csv" "$OUT/power/" 2>/dev/null || true

python3 "$EXTRAS/summarize.py" --cells "$CELLS" --power "$OUT/power" --out "$OUT/summary.csv" \
    >> "$LOGS/summary.log" 2>&1 || log "summarize failed"
cp "$OUT/summary.csv" "$OUT/logs/summary.log" "$LOGS/" 2>/dev/null || true

# small logs only: server logs can be large, keep the head/tail facts
for f in "$LOGS"/server-*.log; do
  [ -s "$f" ] || continue
  b=$(basename "$f")
  { head -25 "$f"; echo "..."; tail -25 "$f"; } > "$OUT/logs/$b"
done
for f in "$LOGS"/bench-*.err; do
  [ -s "$f" ] || continue
  b=$(basename "$f")
  { head -10 "$f"; echo "..."; tail -20 "$f"; } > "$OUT/logs/$b"
done
cp "$LOCAL/session.log" "$OUT/logs/"
cp "$OUT"/cells/*.json "$OUT/logs/" 2>/dev/null || true

log "done ($(remaining_min) min left); results $OUT"
