#!/usr/bin/env bash
# =============================================================================
#  start-all-lanes.sh [--abandon-stale] [lane ...] — start the worker lanes
# =============================================================================
#  Starts every worker lane (all lanes in .langywrap/lanes.json except LANE_EXCLUDE, default `steward`)
#  INSIDE a Slurm allocation. The work is done by start-lanes.sh, which is the
#  one implementation every launcher shares. The steward is NOT started here:
#    ly start-lane steward        (from the login node)
#    start-lanes.sh steward       (inside the allocation)
#  Run from the login node via: ly start-lane <lanes...> or ly exec "$SLURMHELPS_HOME/fleet/start-all-lanes.sh"
#  Source of truth: slurmhelps fleet/start-all-lanes.sh.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! bash "$HERE/login-node-guard.sh"; then
  echo "start-all-lanes: refused by login-node-guard.sh; not starting any lane." >&2
  exit 1
fi
exec bash "$HERE/start-lanes.sh" "$@"
