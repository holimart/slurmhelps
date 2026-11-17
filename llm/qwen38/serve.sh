#!/usr/bin/env bash
# Qwen3.8-Flash-Next launcher.
#
#   llm/qwen38/serve.sh [--target bw|hopper|strata-volta] up|status|wait|endpoint|tunnel|relay|down|doctor
#
# --target bw (default) uses config-bw.json (2x BW6000 on bw01, port 8010).
# --target hopper uses config-hopper.json (2x H100 on hopper01, port 8020,
# own state dir .slurmhelps-serve/qwen38-flash-next-hopper) with
# serve-container-hopper.sh: the upstream vLLM nightly container (Marlin
# W4A16 MoE for the CSF experts, PLE table offloaded to host RAM).
# --target strata-volta uses config-strata-volta.json (4x V100-32G on volta05,
# ffa-gpulab, --nice=100000, port 8030, state dir
# .slurmhelps-serve/qwen38-flash-next-strata-volta) with serve-strata.sh: the
# IQ3_S GGUF on the Strata engine, the lowest-priority fallback replica.
# QWEN38_TARGET=hopper is the same as --target hopper.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

target=${QWEN38_TARGET:-bw}
if [ "${1:-}" = "--target" ]; then
  target=${2:?--target needs bw, hopper or strata-volta}
  shift 2
fi
case "$target" in
  bw)
    CONFIG="$HERE/config-bw.json"
    BACKEND="$HERE/serve-container.sh"
    ;;
  hopper)
    CONFIG="$HERE/config-hopper.json"
    BACKEND="$HERE/serve-container-hopper.sh"
    ;;
  strata-volta)
    CONFIG="$HERE/config-strata-volta.json"
    BACKEND="$HERE/serve-strata.sh"
    ;;
  *) echo "unknown target: $target (bw|hopper|strata-volta)" >&2; exit 2 ;;
esac

cmd=${1:-up}
[ $# -gt 0 ] && shift
case "$cmd" in
  up)
    exec "$CLI" --config "$CONFIG" up --yes "$@" \
      --backend-command bash "$BACKEND"
    ;;
  status|wait|endpoint|tunnel|relay|down|doctor)
    exec "$CLI" --config "$CONFIG" "$cmd" "$@"
    ;;
  -h|--help|help) usage ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
