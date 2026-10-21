#!/usr/bin/env bash
# Cache-reuse probe for llama.cpp on one node.
#
# Fills a prefix ladder (100K, 200K, ... up to ~950K), measures the hit TTFT
# and active-slot repeat at each point, and infers the miss time from the
# cumulative fill. It never issues a cold miss request.
#
# It prints the unmeasured gaps explicitly: llama.cpp has one active KV slot
# and a host prompt cache, not a GPU multi-prefix tree, so multi-branch GPU
# reuse, RAM-vs-GPU KV tiers and multi-agent GPU capacity are not measured.
#
# usage: cache-sweep.sh [--engine llamacpp] [--variant iq4_xs]
#                       [--prefixes "100000 200000 ... 950000"]
#                       [--suffix 128] [--osl 128] [--ctx 1048576]
#                       [--result-dir DIR] [--dry-run]
set -euo pipefail

HELPER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HELPER/../.." && pwd)
if [ -z "${WORK_ROOT:-}" ]; then
  WORK_ROOT=$(python3 "$ROOT/lib/slurmhelps/config.py" get site.work_root 2>/dev/null || true)
fi
: "${WORK_ROOT:=$HOME}"
export WORK_ROOT
# The cache probe is llama.cpp-only: it measures the host prompt cache and one
# active slot. Set the defaults before sourcing config.env so they win.
: "${ENGINE:=llamacpp}"
: "${VARIANT:=iq4_xs}"
# shellcheck source=config.env
source "$HELPER/config.env"
PREFIXES=${CACHE_PREFIXES:-"100000 200000 300000 400000 500000 600000 700000 800000 900000 950000"}
PREFIXES=${PREFIXES//+/ }
SUFFIX=${CACHE_SUFFIX:-128}
OSL=${CACHE_OSL:-128}
CTX=${CACHE_CTX:-1048576}
TIMEOUT=${CACHE_TIMEOUT:-3600}
KV_TYPE=${CACHE_KV_TYPE:-q4_0}
BATCH=${CACHE_BATCH:-512}
UBATCH=${CACHE_UBATCH:-128}
RESULT_DIR=${CACHE_RESULT_DIR:-$RUN_DIR/cache-probe-$(date -u +%Y%m%dT%H%M%SZ)}
DRY_RUN=0

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    ENGINE=$2; shift 2 ;;
    --variant)   VARIANT=$2; shift 2 ;;
    --prefixes)  PREFIXES=$2; PREFIXES=${PREFIXES//+/ }; shift 2 ;;
    --suffix)    SUFFIX=$2; shift 2 ;;
    --osl)       OSL=$2; shift 2 ;;
    --ctx)       CTX=$2; shift 2 ;;
    --result-dir) RESULT_DIR=$2; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) printf '[cache] unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# Per-job runtime files, not the shared run dir.
RUN_DIR="$RESULT_DIR/runtime"
mkdir -p "$RESULT_DIR" "$RUN_DIR"
MAX_PREFIX=$(printf '%s\n' $PREFIXES | sort -n | tail -1)
# The actual tokenizer count is ~5% above the word count; use the full 1M
# context so the largest ladder point still fits.
if [ "$CTX" -lt $((MAX_PREFIX + SUFFIX + 50000)) ]; then
  CTX=$((MAX_PREFIX + SUFFIX + 50000))
fi

log() { printf '[cache] %s\n' "$*"; }
stop_server() { "$HELPER/serve.sh" --stop >/dev/null 2>&1 || true; }
trap stop_server EXIT

log "engine=$ENGINE variant=$VARIANT prefixes=$PREFIXES suffix=$SUFFIX ctx=$CTX"
log "kv=$KV_TYPE batch=$BATCH ubatch=$UBATCH (q4_0 + small batch for 1M on 96 GB)"
log "result dir: $RESULT_DIR"
cat <<'EOF'
[cache] GAPS (deliberately not measured):
[cache]   GPU reuse: one active sequence only, not a multi-branch prefix tree.
[cache]   RAM vs GPU KV tier: unmeasured (llama.cpp prompt cache is host state).
[cache]   Multi-agent GPU cache capacity: unmeasured.
[cache]   Miss: inferred as cumulative fill-to-P + suffix TTFT; no cold miss request.
EOF

if [ "$DRY_RUN" = 1 ]; then
  printf '+ CTX_TOTAL=%s MAX_RUNNING=1 serve.sh --engine %s --variant %s --background --wait\n' "$CTX" "$ENGINE" "$VARIANT"
  printf '+ cache_probe.py --prefixes %q --suffix %s --osl %s --ctx %s\n' "$PREFIXES" "$SUFFIX" "$OSL" "$CTX"
  exit 0
fi

stop_server
if ! CTX_TOTAL=$CTX MAX_RUNNING=1 PORT=$PORT LLAMA_PROMPT_CACHE=1 \
    LLAMA_KV_TYPE=$KV_TYPE LLAMA_BATCH=$BATCH LLAMA_UBATCH=$UBATCH \
    timeout 1800 "$HELPER/serve.sh" --engine "$ENGINE" --variant "$VARIANT" --background --wait; then
  log "server failed to start; see $RUN_DIR/server.log"
  tail -40 "$RUN_DIR/server.log" >&2 || true
  exit 1
fi

# Parse llama.cpp's real KV allocation and prompt-cache limit.
KV_CELLS=$(grep -oE 'llama_kv_cache: size =[^()]*\([[:space:]]*[0-9]+ cells' "$RUN_DIR/server.log" 2>/dev/null | grep -oE '[0-9]+ cells' | tail -1 | awk '{print $1}' || true)
KV_MIB=$(grep -oE 'llama_kv_cache: size =[[:space:]]*[0-9.]+ MiB' "$RUN_DIR/server.log" 2>/dev/null | grep -oE '[0-9.]+' | tail -1 || true)
HOST_MIB=$(grep -oE 'prompt cache is enabled, size limit: [0-9]+ MiB' "$RUN_DIR/server.log" 2>/dev/null | grep -oE '[0-9]+' | tail -1 || true)
KV_CELLS=${KV_CELLS:-0}
KV_MIB=${KV_MIB:-0}
HOST_MIB=${HOST_MIB:-0}
log "GPU KV: cells=${KV_CELLS} MiB=${KV_MIB}; host prompt cache: ${HOST_MIB} MiB"

python3 "$HELPER/cache_probe.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$SERVED_MODEL_NAME" \
  --prefixes "$PREFIXES" --suffix "$SUFFIX" --osl "$OSL" \
  --timeout "$TIMEOUT" \
  --kv-cells "$KV_CELLS" --kv-mib "$KV_MIB" --host-cache-mib "$HOST_MIB" \
  --output-dir "$RESULT_DIR"

cp -f "$RUN_DIR/server.log" "$RESULT_DIR/server.log" 2>/dev/null || true
stop_server

cat <<EOF

[cache] Done: $RESULT_DIR/cache-probe.json and .csv
[cache] Miss times are INFERRED from cumulative_fill_s, not measured.
[cache] GPU reuse is a single active sequence; GPU tree/RAM tier/multi-agent
[cache] capacity are UNMEASURED and printed in the JSON caveats block.
EOF
