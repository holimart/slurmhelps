#!/usr/bin/env bash
# Long-context batched benchmark: one server per context point, because
# llama.cpp's -c is shared by -np slots and the slot count must change with the
# context size.
#
# Point syntax: CTX:CONCURRENCY, separated by '+' (Slurm --export cannot carry
# spaces or commas). Default:
#   4096:8 16384:4 65536:2 262144:1 524288:1 1048576:1
#
# The 1M point is attempted last and with a hard timeout: on llama.cpp's dense
# DSA fallback it is expected to be compute-bound and may not finish. Earlier
# points are kept even if it times out.
#
# usage: context-sweep.sh [--engine llamacpp] [--variant iq4_xs]
#                         [--contexts "4096:8 16384:4 ..."] [--osl 128]
#                         [--budget-min 55] [--result-dir DIR] [--dry-run]
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
# The long-context sweep defaults to the gentle llama.cpp/GGUF path; override
# --engine/--variant to run it on SGLang or vLLM instead.
: "${ENGINE:=llamacpp}"
: "${VARIANT:=iq4_xs}"
# shellcheck source=config.env
source "$HELPER/config.env"

DRY_RUN=0
CONTEXTS=${CONTEXTS:-4096:8+16384:4+65536:2+262144:1+524288:1+1048576:1}
CONTEXTS=${CONTEXTS//+/ }
SWEEP_OSL=${SWEEP_OSL:-128}
SWEEP_BUDGET_MIN=${SWEEP_BUDGET_MIN:-55}
SWEEP_TIMEOUT_MULT=${SWEEP_TIMEOUT_MULT:-1}
RESULT_DIR=${RESULT_DIR:-$RUN_DIR/context-sweep-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL=${MODEL:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-reap50}

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)      ENGINE=$2; shift 2 ;;
    --variant)     VARIANT=$2; shift 2 ;;
    --model)       MODEL=$2; shift 2 ;;
    --contexts)    CONTEXTS=$2; CONTEXTS=${CONTEXTS//+/ }; shift 2 ;;
    --osl)         SWEEP_OSL=$2; shift 2 ;;
    --budget-min)  SWEEP_BUDGET_MIN=$2; shift 2 ;;
    --timeout-mult) SWEEP_TIMEOUT_MULT=$2; shift 2 ;;
    --result-dir)  RESULT_DIR=$2; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) printf '[sweep] unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# Keep the server pid/log beside this job's results, not in the shared run dir.
RUN_DIR="$RESULT_DIR/runtime"
mkdir -p "$RESULT_DIR" "$RUN_DIR"
log() { printf '[sweep] %s\n' "$*"; }
stop_server() { "$HELPER/serve.sh" --stop >/dev/null 2>&1 || true; }
trap stop_server EXIT

START=$(date +%s)
DEADLINE=$((START + SWEEP_BUDGET_MIN * 60))
log "engine=$ENGINE variant=$VARIANT osl=$SWEEP_OSL budget=${SWEEP_BUDGET_MIN}min"
log "contexts: $CONTEXTS"
log "result dir: $RESULT_DIR"

base_timeout_for() {
  local ctx=$1
  if [ "$ctx" -ge 1048576 ]; then echo 1500
  elif [ "$ctx" -ge 524288 ]; then echo 1200
  elif [ "$ctx" -ge 262144 ]; then echo 900
  elif [ "$ctx" -ge 65536 ]; then echo 600
  else echo 420
  fi
}

first=1
for spec in $CONTEXTS; do
  ctx=${spec%%:*}
  conc=${spec##*:}
  if [ "$ctx" = "$spec" ]; then conc=1; fi
  case "$ctx" in ''|*[!0-9]*) log "bad context: $spec"; continue ;; esac
  case "$conc" in ''|*[!0-9]*) log "bad concurrency: $spec"; continue ;; esac

  now=$(date +%s)
  remaining=$((DEADLINE - now))
  if [ "$remaining" -lt 90 ]; then
    log "budget exhausted before context $ctx (${remaining}s left)"
    break
  fi
  timeout_s=$(( $(base_timeout_for "$ctx") * SWEEP_TIMEOUT_MULT ))
  if [ "$timeout_s" -gt $((remaining - 30)) ]; then timeout_s=$((remaining - 30)); fi
  if [ "$timeout_s" -lt 60 ]; then
    log "not enough time for context $ctx (${remaining}s left)"
    break
  fi

  # Leave room for the generated tokens inside each slot. At 1M the model's
  # own maximum is the cap, so trim the prompt instead of growing the slot.
  if [ "$ctx" -ge 1048576 ]; then
    slot_ctx=$ctx
  else
    slot_ctx=$((ctx + SWEEP_OSL + 64))
  fi
  # Bench prompts are word-based and tokenize ~5% longer; leave 10% slack so
  # the request fits in its slot. The actual token count is in results.json.
  prompt_ctx=$((ctx * 90 / 100))

  point="$RESULT_DIR/ctx-${ctx}-c${conc}"
  mkdir -p "$point"
  log "context=$ctx concurrency=$conc prompt=$prompt_ctx slot=$slot_ctx timeout=${timeout_s}s"
  printf '%s\n' "$ctx" > "$point/context"
  printf '%s\n' "$conc" > "$point/concurrency"

  if [ "$DRY_RUN" = 1 ]; then
    printf '+ CTX_TOTAL=%s MAX_RUNNING=%s serve.sh --engine %s --variant %s --background --wait\n' \
      "$((slot_ctx * conc))" "$conc" "$ENGINE" "$VARIANT"
    printf '+ bench.sh --isl %s --osl %s --concurrency %s --min-prompts %s --max-prompts %s --runs 1\n' \
      "$prompt_ctx" "$SWEEP_OSL" "$conc" "$conc" "$conc"
    continue
  fi

  stop_server
  serve_args=(--engine "$ENGINE" --variant "$VARIANT" --background --wait)
  if [ -n "$MODEL" ]; then serve_args+=(--model "$MODEL"); fi
  kv_type=$LLAMA_KV_TYPE
  batch=$LLAMA_BATCH
  ubatch=$LLAMA_UBATCH
  if [ "$ctx" -ge 1048576 ]; then
    kv_type=q4_0
    batch=512
    ubatch=128
  elif [ "$ctx" -ge 524288 ]; then
    batch=1024
    ubatch=256
  fi
  if ! CTX_TOTAL=$((slot_ctx * conc)) MAX_RUNNING=$conc PORT=$PORT \
      LLAMA_KV_TYPE=$kv_type LLAMA_BATCH=$batch LLAMA_UBATCH=$ubatch \
      timeout 1800 "$HELPER/serve.sh" "${serve_args[@]}"; then
    log "server failed for context $ctx"
    printf 'server_failed\n' > "$point/status"
    cp -f "$RUN_DIR/server.log" "$point/server.log" 2>/dev/null || true
    stop_server
    if [ "$first" = 1 ]; then
      log "first point failed to start; aborting the sweep"
      break
    fi
    continue
  fi
  first=0
  cp -f "$RUN_DIR/server.log" "$point/server.log" 2>/dev/null || true

  # A 256K/1M warmup would double an already huge prefill; skip it there.
  if [ "$ctx" -ge 262144 ]; then warmup=-1; else warmup=1; fi

  rc=0
  timeout --signal=TERM "$timeout_s" "$HELPER/bench.sh" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "$SERVED_MODEL_NAME" \
    --isl "$prompt_ctx" --osl "$SWEEP_OSL" \
    --concurrency "$conc" --min-prompts "$conc" --max-prompts "$conc" \
    --runs 1 --warmup="$warmup" --timeout "$timeout_s" \
    --output-dir "$point" || rc=$?
  case "$rc" in
    0)   status=ok ;;
    124|137) status=timeout ;;
    *)   status=failed ;;
  esac
  printf '%s\n' "$status" > "$point/status"
  printf '%s\n' "$rc" > "$point/rc"
  log "context=$ctx concurrency=$conc status=$status rc=$rc"
  stop_server
done

python3 - "$RESULT_DIR" <<'PY'
import csv, datetime, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
rows = []
for point in sorted(root.glob("ctx-*-c*"), key=lambda p: int(p.name.split("-")[1])):
    try:
        ctx = int((point / "context").read_text().strip())
        conc = int((point / "concurrency").read_text().strip())
    except (OSError, ValueError):
        continue
    status = (point / "status").read_text().strip() if (point / "status").exists() else "no_status"
    row = {"context": ctx, "concurrency": conc, "status": status}
    results = point / "results.json"
    if results.exists():
        data = json.loads(results.read_text())
        run = data["points"][0]["runs"][0]
        row.update({
            "completed": run.get("completed"),
            "failed": run.get("failed"),
            "ttft_s": round(run.get("ttft_ms", {}).get("p50", 0) / 1000.0, 4),
            "prefill_tokens_per_s": run.get("prefill_tokens_per_s"),
            "decode_tokens_per_s": run.get("decode_tokens_per_s"),
            "output_tokens_per_s": run.get("output_tokens_per_s"),
            "total_tokens_per_s": run.get("total_tokens_per_s"),
            "tpot_ms": run.get("tpot_ms", {}).get("p50"),
        })
    rows.append(row)

(root / "context-results.json").write_text(json.dumps({
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "result_dir": str(root),
    "rows": rows,
}, indent=2) + "\n")

fields = list(rows[0].keys()) if rows else ["context", "concurrency", "status"]
with (root / "context-results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"{'ctx':>10} {'C':>3} {'status':<14} {'prefill tok/s':>14} {'decode tok/s':>13} {'TTFT s':>9}")
for row in rows:
    print(f"{row['context']:>10} {row['concurrency']:>3} {row['status']:<14} "
          f"{str(row.get('prefill_tokens_per_s', '-')):>14} "
          f"{str(row.get('decode_tokens_per_s', '-')):>13} "
          f"{str(row.get('ttft_s', '-')):>9}")
PY

log "wrote $RESULT_DIR/context-results.json and .csv"
