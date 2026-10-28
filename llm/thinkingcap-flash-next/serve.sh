#!/usr/bin/env bash
# ThinkingCap Qwen3.8-Flash-Next NVFP4 on 2x A100 80 GB.
#
# Usage:
#   llm/thinkingcap-flash-next/serve.sh up --yes
#   llm/thinkingcap-flash-next/serve.sh status --json
#   llm/thinkingcap-flash-next/serve.sh wait
#   llm/thinkingcap-flash-next/serve.sh endpoint
#   llm/thinkingcap-flash-next/serve.sh down
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"
CONFIG="$HERE/config-a100.json"
BACKEND="$HERE/serve-server.sh"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

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
