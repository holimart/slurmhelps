#!/usr/bin/env bash
# Default Qwen3.5-122B-A10B launcher for the A100 nodes.
#
# This is the "just run the model" entry point. It fixes the 122B config and
# backend so the caller does not have to remember the paths:
#
#   llm/qwen35/serve.sh up --yes
#   llm/qwen35/serve.sh status --json
#   llm/qwen35/serve.sh wait
#   llm/qwen35/serve.sh endpoint
#   llm/qwen35/serve.sh down
#
# `up` defaults to --yes and the 122B backend; extra flags (for example
# --force or --dry-run) are passed through before the backend.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"
CONFIG="$HERE/config-122b.json"
BACKEND="$HERE/serve-122b.sh"

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

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
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
