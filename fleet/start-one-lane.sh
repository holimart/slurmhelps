#!/usr/bin/env bash
# =============================================================================
#  start-one-lane.sh [--abandon-stale] <lane> — start exactly one lane
# =============================================================================
#  Same path as `start-lanes.sh <lane>` (kept for existing callers). Run inside
#  the allocation, or via: ly start-lane <lane>. Source of truth: slurmhelps fleet/start-one-lane.sh.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  echo "usage: start-one-lane.sh [--abandon-stale] <lane>" >&2
  exit 2
fi
exec bash "$HERE/start-lanes.sh" "$@"
