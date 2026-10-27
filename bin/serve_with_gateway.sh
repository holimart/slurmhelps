#!/usr/bin/env bash
# Run a model backend behind the slurmhelps gateway on the compute node.
#
#   serve_with_gateway.sh --config CFG --port PORT --upstream-port UPORT \
#       [--bad-words-file FILE] -- BACKEND_CMD...
#
# The backend must listen on 127.0.0.1:UPORT. The gateway is started on
# 0.0.0.0:PORT only after the backend's /health answers, so the worker's
# readiness probe on PORT keeps its meaning. If either process exits, both
# are stopped and the script exits with that status; SIGTERM stops both.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CONFIG="" PORT="" UPORT="" BAD_WORDS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --upstream-port) UPORT=$2; shift 2 ;;
    --bad-words-file) BAD_WORDS=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "serve_with_gateway: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$CONFIG" ] && [ -n "$PORT" ] && [ -n "$UPORT" ] && [ $# -gt 0 ] || {
  echo "serve_with_gateway: need --config, --port, --upstream-port and a backend command" >&2; exit 2; }

PY=${SLURMHELPS_PYTHON:-python3.12}
command -v "$PY" >/dev/null 2>&1 || PY=python3

"$@" &
BACKEND=$!
GATEWAY=""
stop() {
  [ -n "$GATEWAY" ] && kill -TERM "$GATEWAY" 2>/dev/null || true
  kill -TERM "$BACKEND" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap 'stop; exit 143' TERM INT

until "$PY" -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
    "http://127.0.0.1:$UPORT/health" 2>/dev/null; do
  if ! kill -0 "$BACKEND" 2>/dev/null; then
    wait "$BACKEND"; exit $?
  fi
  sleep 2
done

GW_ARGS=(--upstream "http://127.0.0.1:$UPORT/v1" --listen-host 0.0.0.0 --listen-port "$PORT"
  --max-active -1 --timeout 3600
  # No bearer auth on the node: same as the bare vLLM it replaces.
  --key-env SLURMHELPS_GATEWAY_NO_KEY)
[ -n "$BAD_WORDS" ] && GW_ARGS+=(--bad-words-file "$BAD_WORDS")
printf '[serve-with-gateway] backend=127.0.0.1:%s gateway=0.0.0.0:%s bad_words=%s\n' \
  "$UPORT" "$PORT" "${BAD_WORDS:-none}"
"$HERE/slurmhelps-serve" --config "$CONFIG" gateway "${GW_ARGS[@]}" &
GATEWAY=$!

set +e
wait -n "$BACKEND" "$GATEWAY"
rc=$?
set -e
stop
exit "$rc"
