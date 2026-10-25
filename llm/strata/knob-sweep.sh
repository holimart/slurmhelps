#!/usr/bin/env bash
# knob-sweep.sh — sweep the Strata engine knobs that the 2026-10-25 smoke
# matrix never varied, at SMALL contexts only.
#
# Why: the stored matrix (strata-runs/2026-10-25-l40x1) pins one baseline
# recipe (`--kv int8 --kv-resident 32768 --expert-cache auto --prefill auto
# --spec 4 --spec-min-p 0.5 --batch 4`).  This runner reuses the stored ladder
# (1024/4096/16384) and only changes ONE knob per variant, so the deltas are
# attributable.  It never re-measures the context×concurrency matrix, 64k,
# 131k, V100 or 2-GPU configs.
#
# Per job:
#   1. copies the pinned GGUF shards to node-local scratch ONCE (all variants
#      read the same staged copy; nothing is recopied per variant),
#   2. per variant: seeds a PRIVATE engine data dir from the shared `_strata-data`
#      tree, starts one headless server with that variant's flags, waits for
#      /v1/models, runs strata_probe.py per (context, concurrency) cell,
#      archives the engine log + the launch config, then stops the server.
#   3. writes OUTDIR/<model>/<variant>/{summary.csv,cells.jsonl,engine-logs/}
#      and rebuilds OUTDIR/knobs.csv (variant,ctx,concurrency,decode_p50,
#      aggregate,ttft_p50,ttft_p95,status,error) plus OUTDIR/variants.csv
#      (the exact --set/env each variant actually used).
#
# Every cell still starts from the exact launch the baseline used:
#   strata launch --model-path <staged> --context 131072 --parallel 4
#     --kv int8 --vision off --access local --port <free> --no-api-key
#     --data-dir <private> --yes --no-tui --restart [--set k=v ...]
# `--context 131072` is the SERVER window (the baseline's), not the probe
# ladder: at a 16k window `--kv-resident` would stop streaming entirely and the
# kvres variants would be no-ops.  `--contexts` selects the probe ladder.
#
# Usage:
#   knob-sweep.sh [--model iq3s|coder] [--node uk-g6-07]
#                 [--contexts 1024,4096,16384] [--batches 1,4]
#                 [--variants all|control,nospec,...] [--budget-min 240]
#                 [--outdir DIR] [--max-context 131072]
#                 [--max-tokens 128] [--requests-per-slot 3] [--gpus 0]
#                 [--plan] [--resume] [--force] [--self-test]
#
#   --model X            pack to sweep (default iq3s)
#   --node NAME          expected Slurm node; the run aborts when the job did
#                        not land there (pass --nodelist=NAME to sbatch too)
#   --contexts LIST      probe prompt sizes (default 1024,4096,16384)
#   --batches LIST       concurrent clients (default 1,4; batch8 adds C=8)
#   --variants LIST      all | comma list (default all).  Names:
#                        all,control,nospec,spec_pure,kvbf16,kvfp16,kvres16k,
#                        kvres64k,batch8,groups1,mtpserial,expert_cache_max
#   --budget-min MIN     stop opening cells after this wall time (default 240;
#                        keep it well inside the 11.5 h Slurm limit)
#   --outdir DIR         results root (default
#                        $RESULTS_ROOT/strata-knobs-<job>-<ts>)
#   --max-context N      server window passed to strata launch (default 131072,
#                        the baseline's; use 65536 only for a V100-style run)
#   --max-tokens N       completion length per request (default 128)
#   --requests-per-slot N  sequential requests per client (default 3)
#   --gpus LIST          GPU numbers for `strata launch --gpu` (default 0)
#   --plan               print the variant/cell plan, launch nothing (login OK)
#   --resume             keep variants whose summary.csv is complete, redo the
#                        rest (their old dir is renamed *.bak-<ts>)
#   --force              delete this model's variant dirs in --outdir first
#   --self-test          stub server + 1 MB dummy gguf, no Slurm/GPU/Strata
#
# Variant definitions (one knob each; baseline otherwise):
#   control       baseline recipe, ONE cell (1024, C1): host/engine variance
#                 check against the stored 141.69 tok/s (iq3s) / 140.88 (coder)
#   nospec        --spec removed: IMPOSSIBLE on a native IQ pack (the engine
#                 refuses to start: it needs --spec T (T >= 2)); recorded as
#                 launch-fail with the real error
#   mtpoff        `strata launch --mtp off` (no draft layer at all; --spec 4
#                 stays in the config because the native pack requires it) —
#                 the workable "no drafts" arm
#   spec_pure     --spec 4 --spec-min-p 0 (extend every verify window)
#   kvbf16        SKIPPED: engine 0.1.50 only has --kv fp16|int8|q4_0|k8v4;
#                 `--kv bf16` answers "--kv must be fp16, int8, q4_0 or k8v4"
#   kvfp16        --kv fp16 (the real lossless-KV mode; substitute for kvbf16)
#   kvres16k      --kv-resident 16384 (engine clamps to its 20480 floor)
#   kvres64k      --kv-resident 65536
#   batch8        --parallel 8 and C=8 cells added to the ladder
#   groups1       --batch-mtp --batch-groups 1 (serial windows; on one GPU
#                 --batch-groups 1 is the default, so this measures batch-MTP)
#   mtpserial     STRATA_BATCH_MTP=1 STRATA_BATCH_MTP_SERIAL=1 (what the docs
#                 call serial MTP windows; the SERIAL var alone does NOT enable
#                 --batch-mtp, so both are needed)
#   vramreserve   --vram-reserve-mib 1024: the auto expert cache otherwise
#                 leaves ~430 MiB free and the engine warns about #2065 paging
#   expert_cache_max  SKIPPED: --expert-cache takes N or auto; "max" is not a
#                 mode (atoi("max") = 0 = cache off) and auto is the baseline
#
# Environment overrides (CLI wins):
#   STRATA_ROOT (default /work/ffa/holecem/strata-engine)
#   MODELS_ROOT (default /work/ffa/holecem/models/strata)
#   DATA_DIR    shared engine data root, seeded per variant from mtp/ + packs/
#   RESULTS_ROOT / SCRATCH_ROOT / COPY_MODELS / READY_TIMEOUT_S / CXX_LIB_DIR
#   KEEP_DATA_DIR=1 / KEEP_SCRATCH=1 keep the private data dir / node scratch
#
# Slurm: submit through knob-sweep.sbatch (pins gpu-ffa + one L40).  Always
# `sbatch --nice=100`; never run the real sweep on the login node.  Comma lists
# belong in trailing CLI args, never in --export (Slurm splits them).
#
# Status vocabulary in summary.csv/knobs.csv:
#   ok | fail | server-dead | launch-fail | skipped-flag | skipped-budget |
#   skipped-done   (the last only on --resume)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STRATA_ROOT=${STRATA_ROOT:-/work/ffa/holecem/strata-engine}
MODELS_ROOT=${MODELS_ROOT:-/work/ffa/holecem/models/strata}
DATA_DIR=${DATA_DIR:-$MODELS_ROOT/_strata-data}
RESULTS_ROOT=${RESULTS_ROOT:-/work/ffa/holecem/results}
SCRATCH_ROOT=${SCRATCH_ROOT:-/tmp}
COPY_MODELS=${COPY_MODELS:-1}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2700}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-1200}
SELFTEST_DIR=${SELFTEST_DIR:-/work/ffa/holecem/tmp/strata-knob-selftest}
CXX_LIB_DIR=${CXX_LIB_DIR:-/work/ffa/holecem/strata-runs/2026-10-25-l40x1/deps/cxx/lib}

MODEL=iq3s
NODE=""
CTXS=1024,4096,16384
BATCHES=1,4
VARIANTS=all
MAX_CONTEXT=131072
MAX_TOKENS=128
RPS=3
BUDGET_MIN=240
OUTDIR=""
GPU_LIST=${GPU_LIST:-0}
PLAN=0
RESUME=0
FORCE=0
SELF_TEST=0
CONTROL_CTX=1024

ALL_VARIANTS="control nospec mtpoff spec_pure kvbf16 kvfp16 kvres16k kvres64k batch8 groups1 mtpserial vramreserve expert_cache_max"

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=${2:?--model needs iq3s|coder}; shift 2 ;;
    --model=*) MODEL=${1#*=}; shift ;;
    --node) NODE=${2:?--node needs e.g. uk-g6-07}; shift 2 ;;
    --node=*) NODE=${1#*=}; shift ;;
    --contexts) CTXS=${2:?--contexts needs e.g. 1024,4096}; shift 2 ;;
    --contexts=*) CTXS=${1#*=}; shift ;;
    --batches) BATCHES=${2:?--batches needs e.g. 1,4}; shift 2 ;;
    --batches=*) BATCHES=${1#*=}; shift ;;
    --variants) VARIANTS=${2:?--variants needs all|name,...}; shift 2 ;;
    --variants=*) VARIANTS=${1#*=}; shift ;;
    --budget-min) BUDGET_MIN=${2:?}; shift 2 ;;
    --budget-min=*) BUDGET_MIN=${1#*=}; shift ;;
    --outdir) OUTDIR=${2:?}; shift 2 ;;
    --outdir=*) OUTDIR=${1#*=}; shift ;;
    --max-context) MAX_CONTEXT=${2:?}; shift 2 ;;
    --max-context=*) MAX_CONTEXT=${1#*=}; shift ;;
    --max-tokens) MAX_TOKENS=${2:?}; shift 2 ;;
    --max-tokens=*) MAX_TOKENS=${1#*=}; shift ;;
    --requests-per-slot) RPS=${2:?}; shift 2 ;;
    --requests-per-slot=*) RPS=${1#*=}; shift ;;
    --gpus) GPU_LIST=${2:?--gpus needs e.g. 0}; shift 2 ;;
    --gpus=*) GPU_LIST=${1#*=}; shift ;;
    --plan) PLAN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --force) FORCE=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) sed -n '2,110p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "knob-sweep: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODEL,,}" in
  iq3s|iq3_s) MODEL=iq3s ;;
  coder|iq1_m) MODEL=coder ;;
  *) echo "knob-sweep: --model must be iq3s|coder (got '$MODEL')" >&2; exit 2 ;;
esac
if ! [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "knob-sweep: --gpus must look like 0 or 0,1 (got '$GPU_LIST')" >&2; exit 2
fi
if ! [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] || [ "$MAX_CONTEXT" -lt 1024 ]; then
  echo "knob-sweep: --max-context must be an integer >= 1024 (got '$MAX_CONTEXT')" >&2; exit 2
fi
if ! [[ "$BUDGET_MIN" =~ ^[0-9]+$ ]]; then
  echo "knob-sweep: --budget-min must be an integer (got '$BUDGET_MIN')" >&2; exit 2
fi
list_max() { printf '%s\n' "${1//,/ }" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1; }
MAXPAR=$(list_max "$BATCHES")
MAXCTX=$(list_max "$CTXS")
[ -n "$MAXPAR" ] && [ -n "$MAXCTX" ] || { echo "knob-sweep: --contexts/--batches need numbers" >&2; exit 2; }
if [ "$MAXPAR" -gt 8 ]; then
  echo "knob-sweep: --batches capped at 8 (strata launch --parallel range)" >&2
  MAXPAR=8
fi
if [ "$MAXCTX" -gt "$MAX_CONTEXT" ]; then
  echo "knob-sweep: --contexts cell $MAXCTX exceeds the server window --max-context $MAX_CONTEXT" >&2
  exit 2
fi

variant_known() {
  case "$1" in
    control|nospec|mtpoff|spec_pure|kvbf16|kvfp16|kvres16k|kvres64k|batch8|groups1|mtpserial|vramreserve|expert_cache_max) return 0 ;;
    *) return 1 ;;
  esac
}
case "$VARIANTS" in
  all|ALL) VARIANTS=$ALL_VARIANTS ;;
  *) VARIANTS=${VARIANTS//,/ } ;;
esac
for v in $VARIANTS; do
  variant_known "$v" || { echo "knob-sweep: unknown variant '$v' (known: $ALL_VARIANTS)" >&2; exit 2; }
done

# ---- variant definitions -----------------------------------------------------
# Each helper is a pure function of the name + the CLI ladder, so --plan prints
# exactly what a real run would pass.
variant_skip_reason() {
  case "$1" in
    kvbf16) printf '%s' "flag value does not exist: engine 0.1.50 --help lists --kv fp16|int8|q4_0|k8v4; running 'strata --kv bf16' answers 'strata generate: --kv must be fp16, int8, q4_0 or k8v4' (0.1.50 cuda13, login node) -> skipped; the real lossless mode is measured as variant kvfp16" ;;
    expert_cache_max) printf '%s' "flag mode does not exist: --expert-cache takes N or auto (engine treats any non-numeric value as atoi(\"max\")=0 = cache off), and auto is already the baseline -> no 'max' knob to vary, skipped" ;;
    *) printf '' ;;
  esac
}
variant_set_flags() {
  case "$1" in
    nospec) printf '%s' "--set spec=off" ;;
    spec_pure) printf '%s' "--set spec=4 --set spec-min-p=0" ;;
    kvfp16) printf '%s' "--set kv=fp16" ;;
    kvres16k) printf '%s' "--set kv-resident=16384" ;;
    kvres64k) printf '%s' "--set kv-resident=65536" ;;
    groups1) printf '%s' "--set batch-mtp=1 --set batch-groups=1" ;;
    vramreserve) printf '%s' "--set vram-reserve-mib=1024" ;;
    *) printf '' ;;
  esac
}
variant_launch_flags() {
  # launch-level flags (not engine --set): only --mtp off needs one
  case "$1" in
    mtpoff) printf '%s' "--mtp off" ;;
    *) printf '' ;;
  esac
}
variant_all_flags() { printf '%s %s' "$(variant_launch_flags "$1")" "$(variant_set_flags "$1")" | sed 's/^ *//;s/ *$//'; }
variant_env_pairs() {
  case "$1" in
    mtpserial) printf '%s' "STRATA_BATCH_MTP=1 STRATA_BATCH_MTP_SERIAL=1" ;;
    *) printf '' ;;
  esac
}
variant_parallel() {
  case "$1" in batch8) printf '%s' 8 ;; *) printf '%s' "$MAXPAR" ;; esac
}
variant_ctxs() {
  case "$1" in control) printf '%s' "$CONTROL_CTX" ;; *) printf '%s' "${CTXS//,/ }" ;; esac
}
variant_batches() {
  case "$1" in
    control) printf '%s' 1 ;;
    batch8) printf '%s' "$(printf '%s\n' ${BATCHES//,/ } 8 | sort -nu | tr '\n' ' ')" ;;
    *) printf '%s' "${BATCHES//,/ }" ;;
  esac
}
# env keys that must never leak from one variant into the next
clear_variant_env() {
  unset STRATA_BATCH_MTP STRATA_BATCH_MTP_SERIAL 2>/dev/null || true
}

# ---- logging / CSV helpers ---------------------------------------------------
log() { printf '[strata-knobs] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
csv_escape() { local s=${1:-}; s=${s//$'\n'/ }; s=${s//$'\r'/ }; s=${s//\"/\"\"}; printf '"%s"' "$s"; }

jsonl_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# last JSONL cell -> decode,aggregate,ttft50,ttft95,error (tab-separated; the
# error field is last so a comma inside it cannot break the split).
cell_fields() { # cells.jsonl lineno_before
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, before = sys.argv[1], int(sys.argv[2])
lines = [l for l in open(path) if l.strip()] if os.path.exists(path) else []
d = {}
if len(lines) > before:
    try:
        d = json.loads(lines[-1])
    except Exception:
        d = {}
def f(k):
    v = d.get(k)
    return "%.2f" % v if isinstance(v, (int, float)) else ""
err = d.get("error") or ""
print("\t".join([f("decode_tok_s_p50"), f("aggregate_tok_s"),
                 f("ttft_s_p50"), f("ttft_s_p95"), str(err).replace("\t", " ")]))
PY
}

annotate_last_cell() { # cells.jsonl variant
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, variant = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    raise SystemExit(0)
lines = open(path).read().splitlines()
if not lines:
    raise SystemExit(0)
try:
    d = json.loads(lines[-1])
except Exception:
    raise SystemExit(0)
d["variant"] = variant
lines[-1] = json.dumps(d)
open(path, "w").write("\n".join(lines) + "\n")
PY
}

CSV_HEADER='variant,ctx,concurrency,decode_p50,aggregate,ttft_p50,ttft_p95,status,error'
csv_row() { # file variant ctx c decode agg t50 t95 status error
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$2" "$3" "$4" "${5:-}" "${6:-}" "${7:-}" "${8:-}" \
    "$9" "$(csv_escape "${10:-}")" >> "$1"
}

# knobs.csv is derived: header + every variant's summary.csv, in the canonical
# ALL_VARIANTS order (so a --resume job that runs two new variants keeps the
# rows from the earlier job) plus any unknown dirs last.
rebuild_knobs_csv() {
  local v f tmp=$OUTDIR/knobs.csv.tmp d seen=" "
  [ -d "$OUTDIR/$MODEL" ] || return 0
  printf '%s\n' "$CSV_HEADER" > "$tmp"
  for v in $ALL_VARIANTS; do
    f=$OUTDIR/$MODEL/$v/summary.csv
    [ -f "$f" ] || continue
    grep -v '^variant,' "$f" >> "$tmp" || true
    seen="$seen$v "
  done
  for d in "$OUTDIR/$MODEL"/*/; do
    [ -d "$d" ] || continue
    v=$(basename "$d")
    case "$v" in *.bak-*) continue ;; esac
    case "$seen" in *" $v "*) continue ;; esac
    f=$d/summary.csv
    [ -f "$f" ] || continue
    grep -v '^variant,' "$f" >> "$tmp" || true
  done
  mv "$tmp" "$OUTDIR/knobs.csv"
}

# ---- per-job isolation (strata_control/paths.py honours these) ---------------
JOB=${SLURM_JOB_ID:-local}
TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ -z "$OUTDIR" ]; then OUTDIR=$RESULTS_ROOT/strata-knobs-$JOB-$TS; fi
PROBE=$here/strata_probe.py
PROBE_SRC=$here/strata_probe.py
STUB=$here/tests/stub_server.py

JOB_LOCAL=$SCRATCH_ROOT/strata-knobs-$JOB
JOB_SCRATCH=$JOB_LOCAL/model
export STRATA_PROFILE_DIR="$JOB_LOCAL/profile"
export STRATA_RUNTIME_DIR="$JOB_LOCAL/runtime"
export XDG_CONFIG_HOME="$JOB_LOCAL/config"
export STRATA_COMPONENTS_DIR="$STRATA_ROOT/components"
mkdir -p "$STRATA_PROFILE_DIR" "$STRATA_RUNTIME_DIR" "$XDG_CONFIG_HOME"

SERVER=none          # none | strata | stub
SERVER_PID=""
SERVER_MODEL=""
CUR_MODEL=""
CUR_VARIANT=""
CUR_DIR=""
VARIANT_DATA_DIR=""
STARTED_AT=$(date +%s)
PROCESSED_ORDER=""

free_port() {
  python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
}
http_json() {
  python3 - "$1" "$2" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as r:
        sys.stdout.write(r.read(1 << 20).decode("utf-8", "replace"))
except Exception as exc:
    print("http-error: %s" % exc, file=sys.stderr); raise SystemExit(1)
PY
}
server_alive() { http_json "http://127.0.0.1:$1/v1/models" 10 >/dev/null 2>&1; }
wait_ready() { # port
  local port=$1 t0 now
  t0=$(date +%s)
  while :; do
    if http_json "http://127.0.0.1:$port/v1/models" 10 | grep -q '"id"'; then
      return 0
    fi
    now=$(date +%s)
    if [ $((now - t0)) -ge "$READY_TIMEOUT_S" ]; then return 1; fi
    sleep 10
  done
}

# ---- libstdc++ floor (cuda13 engine on EL9) ----------------------------------
precheck_engine_libs() {
  local cap variant bin need lib
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ')
  if [ -z "$cap" ]; then
    if nvidia-smi -L 2>/dev/null | grep -qi 'V100'; then cap=7.0; else cap=8.9; fi
  fi
  case "$cap" in 7.*) variant=cuda12 ;; *) variant=cuda13 ;; esac
  bin=$(ls -1d "$STRATA_ROOT"/components/engine/*cuda-${variant}*/strata 2>/dev/null | head -1)
  [ -n "$bin" ] || return 0
  need=$(strings "$bin" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1)
  [ -n "$need" ] || return 0
  lib=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}')
  # No `strings | grep -q`: grep -q exits early, strings takes SIGPIPE and
  # `set -o pipefail` turns a successful match into status 141.
  has_symbol() { strings "$1" 2>/dev/null | awk -v n="$need" '$0==n{f=1} END{exit !f}'; }
  if [ -n "$lib" ] && has_symbol "$lib"; then return 0; fi
  if [ -f "$CXX_LIB_DIR/libstdc++.so.6" ] && has_symbol "$CXX_LIB_DIR/libstdc++.so.6"; then
    export LD_LIBRARY_PATH="$CXX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "precheck: engine $variant needs $need; prepended $CXX_LIB_DIR to LD_LIBRARY_PATH"
    return 0
  fi
  log "FATAL: engine $variant requires $need but the node libstdc++ (${lib:-not found}) lacks it."
  log "       Stage a newer libstdc++ and set CXX_LIB_DIR to its directory."
  return 5
}

# ---- private per-variant engine data dir -------------------------------------
seed_private_data_dir() { # shared private
  local shared=$1 private=$2
  rm -rf "$private"
  mkdir -p "$private" || return 1
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
  [ -f "$private/mtp/rt/dense.bin" ] \
    && [ -f "$private/mtp/rt/experts.bin" ] \
    && [ -f "$private/mtp/rt/draft_vocab.bin" ] || return 1
  return 0
}

# A shared flock makes the hardlink seed atomic with respect to another job's
# first-launch builder (bench-matrix builds the tree under the same lock).
seed_variant_data() { # shared private
  mkdir -p "$1" || return 1
  ( flock -w 300 9 || exit 1
    seed_private_data_dir "$1" "$2"
  ) 9>"$1/.prepare.lock"
}

# ---- model staging (once per job) --------------------------------------------
size_of() { case "$1" in
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf) echo 54817524224 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf) echo 28800138432 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf) echo 29608446496 ;;
  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf) echo 28800138432 ;;
  *) echo 0 ;;
esac }

prepare_model_dir() { # model -> sets CUR_DIR; copies ONCE for all variants
  CUR_MODEL=$1
  local cands=() want=() dir f ok sz need avail
  case "$1" in
    iq3s) want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf)
          cands=("$MODELS_ROOT/iq3s" "$MODELS_ROOT/models/IQ3_S") ;;
    coder) want=(Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
                  Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf)
          cands=("$MODELS_ROOT/coder" "$MODELS_ROOT/models/coder-IQ1_M") ;;
    *) return 2 ;;
  esac
  dir=""
  for cand in "${cands[@]}"; do
    ok=1
    for f in "${want[@]}"; do
      sz=$(stat -c %s "$cand/$f" 2>/dev/null || echo 0)
      [ "$sz" = "$(size_of "$f")" ] || { ok=0; break; }
    done
    if [ "$ok" = 1 ]; then dir=$cand; log "pack $1 complete in $dir"; break; fi
    log "pack $1 incomplete or absent under $cand"
  done
  if [ -z "$dir" ]; then
    log "FATAL: no complete pack for $1; finish fetch-models.sh --only $1 first"
    return 3
  fi
  if [ "$COPY_MODELS" = 1 ]; then
    need=0
    for f in "${want[@]}"; do need=$((need + $(stat -c %s "$dir/$f"))); done
    avail=$(df -B1 --output=avail "$SCRATCH_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
    if [ "${avail:-0}" -lt $((need + (need / 20) + 2147483648)) ]; then
      log "FATAL: $SCRATCH_ROOT has ${avail:-?} bytes free, need $((need + need / 20)) (+5%)."
      log "       set SCRATCH_ROOT to a bigger node-local disk, or COPY_MODELS=0."
      return 4
    fi
    rm -rf "$JOB_SCRATCH"; mkdir -p "$JOB_SCRATCH"
    log "staging $((need / 1073741824)) GiB of shards ONCE to $JOB_SCRATCH (reused by every variant)"
    for f in "${want[@]}"; do cp "$dir/$f" "$JOB_SCRATCH/$f"; done
    CUR_DIR=$JOB_SCRATCH
  else
    CUR_DIR=$dir
    log "COPY_MODELS=0: reading shards straight from $dir"
  fi
}

# ---- server control ----------------------------------------------------------
launch_variant_server() { # variant port datadir
  local variant=$1 port=$2 datadir=$3 logf par
  logf=$OUTDIR/$MODEL/$variant/launch-$(date -u +%H%M%S).log
  par=$(variant_parallel "$variant")
  local -a setargs=()
  local flags; flags=$(variant_set_flags "$variant")
  if [ -n "$flags" ]; then
    # shellcheck disable=SC2206  # intentional word split: "--set k=v --set k=v"
    setargs=($flags)
  fi
  local -a lflags=()
  local lf; lf=$(variant_launch_flags "$variant")
  if [ -n "$lf" ]; then
    # shellcheck disable=SC2206
    lflags=($lf)
  fi
  local -a envs=()
  local ep; ep=$(variant_env_pairs "$variant")
  if [ -n "$ep" ]; then
    # shellcheck disable=SC2206
    envs=($ep)
    local pair; for pair in "${envs[@]}"; do export "${pair?}"; done
  fi
  local -a gpu_args=()
  [ -n "$GPU_LIST" ] && gpu_args=(--gpu "$GPU_LIST")
  mkdir -p "$OUTDIR/$MODEL/$variant"
  log "launch variant=$variant: strata launch --model-path $CUR_DIR --context $MAX_CONTEXT --parallel $par${GPU_LIST:+ --gpu $GPU_LIST} --kv int8 --vision off --access local --port $port --no-api-key --data-dir $datadir ${lf} ${flags} ${ep} (log $logf)"
  if ! "$STRATA_ROOT/bin/strata" launch \
      --model-path "$CUR_DIR" \
      --context "$MAX_CONTEXT" \
      --parallel "$par" \
      ${gpu_args[@]+"${gpu_args[@]}"} \
      --kv int8 --vision off --access local --port "$port" --no-api-key \
      --data-dir "$datadir" --yes --no-tui --restart \
      ${lflags[@]+"${lflags[@]}"} \
      ${setargs[@]+"${setargs[@]}"} >"$logf" 2>&1; then
    log "launch failed for variant=$variant; tail of $logf:"
    tail -n 20 "$logf" || true
    return 1
  fi
  SERVER=strata
  if ! wait_ready "$port"; then
    log "server not ready after ${READY_TIMEOUT_S}s for variant=$variant; tail of $logf:"
    tail -n 30 "$logf" || true
    return 1
  fi
  SERVER_MODEL=$(http_json "http://127.0.0.1:$port/v1/models" 10 \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)
  [ -n "$SERVER_MODEL" ] || SERVER_MODEL=strata
  log "server ready on 127.0.0.1:$port, model id: $SERVER_MODEL"
}

stop_variant_server() {
  "$STRATA_ROOT/bin/strata" launch --stop --no-tui --yes >/dev/null 2>&1 || true
  # Precise fallback: only an engine reading THIS variant's private data dir.
  [ -n "$VARIANT_DATA_DIR" ] && pkill -f "$VARIANT_DATA_DIR/mtp" 2>/dev/null || true
  SERVER=none
}

start_stub() {
  python3 "$STUB" --port "$1" >"$OUTDIR/selftest/stub.log" 2>&1 &
  SERVER_PID=$!
  SERVER=stub
  sleep 0.5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stub died:"; cat "$OUTDIR/selftest/stub.log" || true; return 1
  fi
  if ! wait_ready "$1"; then log "stub never became ready"; return 1; fi
  SERVER_MODEL=strata-selftest
}

# ---- engine evidence ---------------------------------------------------------
archive_variant_logs() { # variant
  local v=$1 src=$XDG_STATE_HOME/strata dest=$OUTDIR/$MODEL/$v/engine-logs
  [ -d "$src" ] || return 0
  mkdir -p "$dest" 2>/dev/null || return 0
  cp -a "$src/." "$dest/" 2>/dev/null || true
  local cfg; cfg=$(ls -1 "$dest"/launch/*.json 2>/dev/null | head -1 || true)
  if [ -n "$cfg" ]; then cp "$cfg" "$OUTDIR/$MODEL/$v/launch-config.json" 2>/dev/null || true; fi
  local elog; elog=$(ls -1 "$dest"/launch/*.engine.log 2>/dev/null | tail -1 || true)
  if [ -n "$elog" ]; then
    { grep -E 'INFO batch_slots=|INFO batch_groups=|KV streaming:|expert cache [0-9]+ slots|draft layer loaded|spec_min_p=|WARNING: --batch-mtp is off|WARNING: .*slots' "$elog" || true; } \
      | head -30 > "$OUTDIR/$MODEL/$v/variant-facts.txt"
  fi
  log "archived engine log/config for variant=$v -> $dest"
}

# ---- cells -------------------------------------------------------------------
run_cell() { # variant port ctx c -> status; appends summary/cells
  local v=$1 port=$2 ctx=$3 c=$4
  local cells=$OUTDIR/$MODEL/$v/cells.jsonl
  local csv=$OUTDIR/$MODEL/$v/summary.csv
  local before rc status dec agg t50 t95 err
  before=$(jsonl_lines "$cells")
  log "  cell ctx=$ctx C=$c (rps=$RPS max_tokens=$MAX_TOKENS)"
  rc=0
  timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
    --base-url "http://127.0.0.1:$port" --model "$SERVER_MODEL" \
    --context "$ctx" --concurrency "$c" --max-context "$MAX_CONTEXT" \
    --max-tokens "$MAX_TOKENS" --requests-per-slot "$RPS" \
    --json-out "$cells" || rc=$?
  if [ "$rc" = 1 ] && [ "$VAR_RESTARTS" -lt 2 ] && [ "$SERVER" != stub ] \
     && ! server_alive "$port"; then
    VAR_RESTARTS=$((VAR_RESTARTS + 1))
    log "  cell lost its server — relaunch $VAR_RESTARTS/2 for variant=$v"
    stop_variant_server
    if launch_variant_server "$v" "$port" "$VARIANT_DATA_DIR"; then
      before=$(jsonl_lines "$cells")
      rc=0
      timeout "$CELL_TIMEOUT_S" python3 "$PROBE" \
        --base-url "http://127.0.0.1:$port" --model "$SERVER_MODEL" \
        --context "$ctx" --concurrency "$c" --max-context "$MAX_CONTEXT" \
        --max-tokens "$MAX_TOKENS" --requests-per-slot "$RPS" \
        --json-out "$cells" || rc=$?
    else
      rc=7
    fi
  fi
  case $rc in
    0) status=ok ;;
    1) status=fail ;;
    7) status=server-dead ;;
    124) status=fail ;;
    *) status=fail ;;
  esac
  if [ "$SERVER" != stub ] && ! server_alive "$port"; then status=server-dead; fi
  annotate_last_cell "$cells" "$v" 2>/dev/null || true
  IFS=$'\t' read -r dec agg t50 t95 err < <(cell_fields "$cells" "$before") || true
  if [ "$rc" = 124 ]; then err="probe timeout after ${CELL_TIMEOUT_S}s"; fi
  if [ -z "$err" ] && [ "$status" != ok ]; then err="probe exit $rc"; fi
  csv_row "$csv" "$v" "$ctx" "$c" "$dec" "$agg" "$t50" "$t95" "$status" "$err"
  log "  cell ctx=$ctx C=$c -> $status${err:+ ($err)}"
}

variant_complete() { # variant -> 0 when every expected cell has a final row
  local v=$1 f=$OUTDIR/$MODEL/$1/summary.csv ctx c found
  [ -f "$f" ] || return 1
  for ctx in $(variant_ctxs "$v"); do
    for c in $(variant_batches "$v"); do
      found=$(awk -F, -v ctx="$ctx" -v c="$c" \
        '$2==ctx && $3==c && ($8=="ok"||$8=="fail"||$8=="skipped-flag"){n++} END{print n+0}' "$f")
      [ "${found:-0}" -ge 1 ] || return 1
    done
  done
  return 0
}

run_variant() { # variant
  local v=$1 ctxs batches ctx c port skip reason csv cells
  PROCESSED_ORDER="$PROCESSED_ORDER $v"
  ctxs=$(variant_ctxs "$v")
  batches=$(variant_batches "$v")
  csv=$OUTDIR/$MODEL/$v/summary.csv
  cells=$OUTDIR/$MODEL/$v/cells.jsonl
  mkdir -p "$OUTDIR/$MODEL/$v"

  reason=$(variant_skip_reason "$v")
  if [ -n "$reason" ]; then
    log "variant=$v SKIPPED-FLAG: $reason"
    [ -f "$csv" ] || printf '%s\n' "$CSV_HEADER" > "$csv"
    for ctx in $ctxs; do for c in $batches; do
      csv_row "$csv" "$v" "$ctx" "$c" "" "" "" "" "skipped-flag" "$reason"
    done; done
    csv_row_variant "$v" "skipped-flag" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "$reason"
    return 0
  fi

  if [ "$RESUME" = 1 ] && variant_complete "$v"; then
    log "variant=$v already complete — resume skips it"
    csv_row_variant "$v" "skipped-done" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "complete summary.csv reused"
    return 0
  fi
  if [ -f "$csv" ] || [ -f "$cells" ]; then
    local bak=$OUTDIR/$MODEL/$v.bak-$(date -u +%H%M%S)
    log "variant=$v has partial results; moving the old dir to $bak"
    mv "$OUTDIR/$MODEL/$v" "$bak"
    mkdir -p "$OUTDIR/$MODEL/$v"
  fi
  printf '%s\n' "$CSV_HEADER" > "$csv"

  if [ "$BUDGET_EXCEEDED" = 1 ]; then
    for ctx in $ctxs; do for c in $batches; do
      csv_row "$csv" "$v" "$ctx" "$c" "" "" "" "" "skipped-budget" "wall budget ${BUDGET_MIN}min exhausted before this variant"
    done; done
    csv_row_variant "$v" "skipped-budget" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "budget exhausted"
    return 0
  fi

  # fresh engine-log state AND a fresh profile/config per variant: launch-level
  # choices (notably --mtp off) are written into the model's saved settings and
  # would otherwise leak into the next variant.  The runtime dir stays shared so
  # `launch --stop` can always find this job's server.
  export XDG_STATE_HOME="$JOB_LOCAL/state/$v"
  export STRATA_PROFILE_DIR="$JOB_LOCAL/profile/$v"
  export XDG_CONFIG_HOME="$JOB_LOCAL/config/$v"
  mkdir -p "$XDG_STATE_HOME" "$STRATA_PROFILE_DIR" "$XDG_CONFIG_HOME"
  clear_variant_env
  VAR_RESTARTS=0

  VARIANT_DATA_DIR=""
  if seed_variant_data "$DATA_DIR" "$DATA_DIR/jobs/$JOB-$MODEL-$v"; then
    VARIANT_DATA_DIR=$DATA_DIR/jobs/$JOB-$MODEL-$v
    log "private engine data dir for $v: $VARIANT_DATA_DIR"
  else
    log "FATAL: shared engine data dir $DATA_DIR has no complete drafter tree; run bench-matrix once first"
    for ctx in $ctxs; do for c in $batches; do
      csv_row "$csv" "$v" "$ctx" "$c" "" "" "" "" "launch-fail" "shared DATA_DIR $DATA_DIR incomplete (mtp/rt missing)"
    done; done
    csv_row_variant "$v" "launch-fail" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "shared data dir incomplete"
    return 0
  fi

  port=$(free_port)
  if ! launch_variant_server "$v" "$port" "$VARIANT_DATA_DIR"; then
    local lasterr
    lasterr=$(ls -1t "$OUTDIR/$MODEL/$v"/launch-*.log 2>/dev/null | head -1 || true)
    lasterr=$([ -n "$lasterr" ] && tail -n 3 "$lasterr" | tr '\n' ' ' || echo "strata launch failed")
    for ctx in $ctxs; do for c in $batches; do
      csv_row "$csv" "$v" "$ctx" "$c" "" "" "" "" "launch-fail" "$lasterr"
    done; done
    csv_row_variant "$v" "launch-fail" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "$lasterr"
    archive_variant_logs "$v" || true
    stop_variant_server
    [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$VARIANT_DATA_DIR" 2>/dev/null || true
    VARIANT_DATA_DIR=""
    return 0
  fi

  for ctx in $ctxs; do
    for c in $batches; do
      if [ $((( $(date +%s) - STARTED_AT) / 60)) -ge "$BUDGET_MIN" ]; then
        log "budget ${BUDGET_MIN}min reached — rest of variant=$v skipped"
        csv_row "$csv" "$v" "$ctx" "$c" "" "" "" "" "skipped-budget" "wall budget ${BUDGET_MIN}min exhausted"
        BUDGET_EXCEEDED=1
        continue
      fi
      run_cell "$v" "$port" "$ctx" "$c" 0
    done
  done
  stop_variant_server
  archive_variant_logs "$v" || true
  [ "${KEEP_DATA_DIR:-0}" = 1 ] || rm -rf "$VARIANT_DATA_DIR" 2>/dev/null || true
  VARIANT_DATA_DIR=""
  csv_row_variant "$v" "done" "$(variant_parallel "$v")" "$(variant_all_flags "$v")" "$(variant_env_pairs "$v")" "see summary.csv"
  rebuild_knobs_csv
}

csv_row_variant() { # variant status parallel flags env note
  local f=$OUTDIR/variants.csv
  [ -f "$f" ] || printf 'variant,status,parallel,flags,env,note\n' > "$f"
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$(csv_escape "$1")" "$(csv_escape "$2")" "$(csv_escape "$3")" \
    "$(csv_escape "$4")" "$(csv_escape "$5")" "$(csv_escape "$6")" >> "$f"
}

# ---- plan --------------------------------------------------------------------
print_plan() {
  local v ctx c
  echo "model=$MODEL node=${NODE:-any} server_window=$MAX_CONTEXT probe_contexts=$CTXS batches=$BATCHES"
  echo "outdir=$OUTDIR budget=${BUDGET_MIN}min max_tokens=$MAX_TOKENS rps=$RPS gpus=$GPU_LIST"
  printf 'launch: strata launch --model-path <staged> --context %s --parallel <N>%s --kv int8 --vision off --access local --port <free> --no-api-key --data-dir <private> --yes --no-tui --restart [--set ...]\n' \
    "$MAX_CONTEXT" "${GPU_LIST:+ --gpu $GPU_LIST}"
  for v in $VARIANTS; do
    printf '\n%-16s parallel=%s cells=%s x %s\n' "$v" "$(variant_parallel "$v")" \
      "$(variant_ctxs "$v" | tr ' ' ',')" "$(variant_batches "$v" | tr ' ' ',')"
    local r; r=$(variant_skip_reason "$v")
    if [ -n "$r" ]; then printf '  SKIP: %s\n' "$r"; continue; fi
    printf '  flags: %s\n' "$(variant_all_flags "$v")"
    printf '  env: %s\n' "$(variant_env_pairs "$v")"
  done
  echo
  local n=0
  for v in $VARIANTS; do
    [ -n "$(variant_skip_reason "$v")" ] && continue
    for ctx in $(variant_ctxs "$v"); do for c in $(variant_batches "$v"); do n=$((n + 1)); done; done
  done
  echo "planned cells: $n"
}

# ---- self test ---------------------------------------------------------------
self_test() {
  log "SELF-TEST: stub server + 1 MB dummy gguf, no Slurm/GPU/Strata needed"
  [ -f "$STUB" ] || { log "FATAL: $STUB missing"; exit 2; }
  rm -rf "$SELFTEST_DIR"; mkdir -p "$SELFTEST_DIR/model"
  head -c 1048576 /dev/urandom > "$SELFTEST_DIR/model/dummy-00001-of-00001.gguf"
  OUTDIR=$SELFTEST_DIR/out
  mkdir -p "$OUTDIR"
  VARIANTS="control nospec"
  CTXS=512,1024
  BATCHES=1,2
  CONTROL_CTX=512
  MAXCTX=1024
  MAXPAR=2
  MAX_TOKENS=32
  RPS=2
  CELL_TIMEOUT_S=120
  PROCESSED_ORDER=""
  export XDG_STATE_HOME="$SELFTEST_DIR/state"
  local port rc=0 v n
  port=$(free_port)
  mkdir -p "$OUTDIR/selftest"
  start_stub "$port" || exit 2
  for v in $VARIANTS; do
    CUR_VARIANT=$v
    VAR_RESTARTS=0
    mkdir -p "$OUTDIR/$MODEL/$v"
    printf '%s\n' "$CSV_HEADER" > "$OUTDIR/$MODEL/$v/summary.csv"
    for ctx in $(variant_ctxs "$v"); do
      for c in $(variant_batches "$v"); do
        run_cell "$v" "$port" "$ctx" "$c"
      done
    done
    # exercise the log-archiving path with a fake engine log
    mkdir -p "$XDG_STATE_HOME/strata/launch"
    echo "INFO batch_slots=4" > "$XDG_STATE_HOME/strata/launch/selftest.engine.log"
    archive_variant_logs "$v"
    csv_row_variant "$v" "done" "2" "" "" "selftest"
    PROCESSED_ORDER="$PROCESSED_ORDER $v"
  done
  rebuild_knobs_csv
  kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER=none

  n=$(grep -vc '^variant,' "$OUTDIR/knobs.csv" 2>/dev/null || true)
  log "knobs.csv rows: $n (expected 5: control 1 + nospec 4)"
  [ "${n:-0}" = 5 ] || { log "FAIL: expected 5 knobs.csv data rows"; rc=1; }
  head -1 "$OUTDIR/knobs.csv" | grep -q '^variant,ctx,concurrency,decode_p50,aggregate,ttft_p50,ttft_p95,status,error$' \
    || { log "FAIL: knobs.csv header wrong"; rc=1; }
  head -1 "$OUTDIR/$MODEL/nospec/summary.csv" | grep -q '^variant,' \
    || { log "FAIL: summary.csv has no variant column"; rc=1; }
  python3 - "$OUTDIR" "$MODEL" <<'PY' || rc=1
import json, os, sys
out, model = sys.argv[1], sys.argv[2]
rows = []
for line in open(os.path.join(out, "knobs.csv")):
    if line.startswith("variant,"):
        continue
    rows.append(line.rstrip("\n").split(","))
assert rows, "no knobs.csv rows"
for r in rows:
    assert r[0] in ("control", "nospec"), "variant column wrong: %s" % (r,)
    assert r[7] in ("ok", "fail", "server-dead"), "status not a real cell status: %s" % (r,)
    if r[7] == "ok":
        assert 90 <= float(r[3]) <= 110, "stub decode outside 90..110: %s" % (r,)
for v in ("control", "nospec"):
    cells = os.path.join(out, model, v, "cells.jsonl")
    got = [json.loads(l) for l in open(cells) if l.strip()]
    assert got, "no JSONL cells for %s" % v
    assert all(c.get("variant") == v for c in got), "JSONL variant field missing in %s" % v
    assert os.path.exists(os.path.join(out, model, v, "engine-logs", "launch", "selftest.engine.log")), \
        "engine log not archived for %s" % v
print("  knobs.csv schema, variant columns, JSONL variant field, log archive: ok")
PY
  if [ "$rc" = 0 ]; then
    log "SELF-TEST PASS — results under $OUTDIR (self-test scratch kept)"
  else
    log "SELF-TEST FAIL — see $OUTDIR"
  fi
  exit "$rc"
}

if [ "$SELF_TEST" = 1 ]; then self_test; fi

OUTDIR_PREEXISTING=0
[ -d "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ] && OUTDIR_PREEXISTING=1

if [ "$PLAN" = 1 ]; then print_plan; exit 0; fi

mkdir -p "$OUTDIR"
log "run: job=$JOB model=$MODEL contexts=$CTXS batches=$BATCHES variants=$(printf '%s' "$VARIANTS" | tr ' ' ',') server_window=$MAX_CONTEXT max_tokens=$MAX_TOKENS rps=$RPS budget=${BUDGET_MIN}min outdir=$OUTDIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  log "FATAL: no SLURM_JOB_ID — the sweep belongs on a GPU node (sbatch knob-sweep.sbatch)."
  log "       Use --plan or --self-test for the login-node checks."
  exit 2
fi
[ -x "$STRATA_ROOT/bin/strata" ] || { log "FATAL: $STRATA_ROOT/bin/strata missing — run install-engine.sh first"; exit 3; }

if [ -n "$NODE" ]; then
  host=$(hostname -s)
  nodestr="${SLURM_JOB_NODELIST:-}"
  case "$host" in "$NODE"|"$NODE".*) ok=1 ;; *) ok=0 ;; esac
  case "$nodestr" in *"$NODE"*) ok=1 ;; esac
  if [ "$ok" != 1 ]; then
    log "FATAL: expected node $NODE but this job is on host=$host nodelist=$nodestr"
    log "       submit with --nodelist=$NODE (and keep --node in sync)"
    exit 2
  fi
  log "node check: $NODE ok (host=$host nodelist=$nodestr)"
fi

if [ "$OUTDIR_PREEXISTING" = 1 ] && [ "$RESUME" != 1 ] && [ "$FORCE" != 1 ]; then
  log "FATAL: $OUTDIR is not empty; pass --resume (keep complete variants), --force (start over)"
  log "       or use a fresh --outdir"
  exit 2
fi
if [ "$FORCE" = 1 ]; then
  log "--force: removing $OUTDIR/$MODEL and the derived CSVs"
  rm -rf "$OUTDIR/$MODEL" "$OUTDIR/knobs.csv" "$OUTDIR/variants.csv" 2>/dev/null || true
fi
if [ "$RESUME" != 1 ]; then
  rm -f "$OUTDIR/knobs.csv" "$OUTDIR/variants.csv" 2>/dev/null || true
fi

precheck_engine_libs || exit $?
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  if [ "${visible_gpus:-0}" -lt 1 ]; then
    log "FATAL: no GPU visible on this node"; exit 4
  fi
  log "GPUs visible: $visible_gpus; using --gpu ${GPU_LIST:-auto}"
fi

BUDGET_EXCEEDED=0
prepare_model_dir "$MODEL" || { log "FATAL: model staging failed; nothing swept"; exit 3; }
CUR_MODEL=$MODEL

# Snapshot the probe for the whole job: a sweep must not change measurement
# code between variants because another session edited the checkout.
cp "$PROBE_SRC" "$JOB_LOCAL/strata_probe.py"
PROBE=$JOB_LOCAL/strata_probe.py
log "probe snapshot: $PROBE ($(sha256sum "$PROBE" | cut -c1-16))"

cleanup() {
  local rc=$?
  if [ "$SERVER" = strata ]; then stop_variant_server || true; fi
  if [ "$SERVER" = stub ] && [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$VARIANT_DATA_DIR" ] && [ "${KEEP_DATA_DIR:-0}" != 1 ]; then
    rm -rf "$VARIANT_DATA_DIR" 2>/dev/null || true
  fi
  [ "${KEEP_SCRATCH:-0}" = 1 ] || rm -rf "$JOB_LOCAL" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT

for v in $VARIANTS; do
  run_variant "$v"
done
rebuild_knobs_csv

elapsed=$(( ($(date +%s) - STARTED_AT) / 60 ))
log "done in ${elapsed}min. results: $OUTDIR (knobs.csv, variants.csv, $MODEL/<variant>/)"
