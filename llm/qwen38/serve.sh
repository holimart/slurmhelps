#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"
CONFIG="$HERE/config-bw.json"
BACKEND="$HERE/serve-container.sh"

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

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
