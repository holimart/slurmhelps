#!/usr/bin/env bash
# cloud_vast.sh — thin adapter over llm/cloud/vast/vast.py.
#
# The Python client is the canonical Vast adapter: offer filter (driver,
# internet_down_cost_per_tb), create, upload, run, pull, destroy. This shell
# wrapper only exists so every backend has the same `adapters/*.sh --dry-run`
# contract.
#
# usage: cloud_vast.sh [--dry-run] plan|rent|stage|run|pull|destroy|offers|list|balance
#                      [--recipe-file FILE] [--outdir DIR] [--yes] [vast.py flags...]
#
# Safety: rent/destroy/run/stage never happen without --yes; --dry-run prints
# the equivalent request or rsync/ssh command.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
VAST="$ROOT/llm/cloud/vast/vast.py"

DRY=0
JSON=0
OUTDIR=""
RECIPE=""
YES=0
SUB=""
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --json) JSON=1; shift ;;
    --outdir) OUTDIR=${2:?--outdir needs a directory}; shift 2 ;;
    --recipe-file) RECIPE=${2:?--recipe-file needs a path}; shift 2 ;;
    --yes) YES=1; shift ;;
    plan|rent|stage|run|pull|destroy|offers|list|balance)
      if [ -z "$SUB" ]; then SUB=$1; shift; else PASSTHROUGH+=("$1"); shift; fi ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

[ -n "$SUB" ] || { echo "cloud_vast.sh: need a subcommand (plan|rent|stage|run|pull|destroy)" >&2; exit 2; }
[ -f "$VAST" ] || { echo "cloud_vast.sh: missing $VAST" >&2; exit 2; }

cmd=(python3 "$VAST")
[ "$DRY" = 1 ] && cmd+=(--dry-run)
[ "$JSON" = 1 ] && cmd+=(--json)
cmd+=("$SUB")
[ -n "$RECIPE" ] && cmd+=(--recipe-file "$RECIPE")
[ -n "$OUTDIR" ] && cmd+=(--outdir "$OUTDIR")
[ "$YES" = 1 ] && cmd+=(--yes)
cmd+=("${PASSTHROUGH[@]}")

if [ "$DRY" = 1 ]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
exec "${cmd[@]}"
