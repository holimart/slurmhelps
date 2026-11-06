#!/usr/bin/env bash
# Qwen3.8-Flash-Next launcher.
#
#   llm/qwen38/serve.sh [--target bw|hopper] up|status|wait|endpoint|tunnel|relay|down|doctor
#
# --target bw (default) uses config-bw.json (2x BW6000 on bw01, port 8010).
# --target hopper uses config-hopper.json (2x H100 on hopper01, port 8020,
# own state dir .slurmhelps-serve/qwen38-flash-next-hopper). Hopper `up` is
# refused: the community container cannot run this checkpoint on SM90 (see
# README "Two deployments"); status/down/doctor still work.
# QWEN38_TARGET=hopper is the same as --target hopper.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

target=${QWEN38_TARGET:-bw}
if [ "${1:-}" = "--target" ]; then
  target=${2:?--target needs bw or hopper}
  shift 2
fi
case "$target" in
  bw)
    CONFIG="$HERE/config-bw.json"
    BACKEND="$HERE/serve-container.sh"
    ;;
  hopper)
    CONFIG="$HERE/config-hopper.json"
    BACKEND=""
    ;;
  *) echo "unknown target: $target (bw|hopper)" >&2; exit 2 ;;
esac

cmd=${1:-up}
[ $# -gt 0 ] && shift
case "$cmd" in
  up)
    if [ -z "$BACKEND" ]; then
      echo "qwen38 hopper: blocked -- the qwen38 vLLM container has no SM90 path for this" >&2
      echo "checkpoint (NVFP4-CSF experts and the NVFP4 PLE table run only on b12x, SM120/121)." >&2
      echo "See llm/qwen38/README.md and experiments/2026-11-06-qwen38-hopper.md. Not submitting." >&2
      exit 3
    fi
    exec "$CLI" --config "$CONFIG" up --yes "$@" \
      --backend-command bash "$BACKEND"
    ;;
  status|wait|endpoint|tunnel|relay|down|doctor)
    exec "$CLI" --config "$CONFIG" "$cmd" "$@"
    ;;
  -h|--help|help) usage ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
