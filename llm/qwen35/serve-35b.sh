#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export QWEN_VARIANT=35b
exec bash "$HERE/serve-server.sh" "$@"
