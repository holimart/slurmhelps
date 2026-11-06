#!/usr/bin/env bash
# Login-node router for the qwen38 deployments (bw first, then hopper).
#
#   llm/qwen38/router.sh start    # background; pid + log under <state root>/router/
#   llm/qwen38/router.sh status   # which deployment is active (exit 0 only if one is ready)
#   llm/qwen38/router.sh stop
#   llm/qwen38/router.sh run      # foreground
#
# Clients use http://127.0.0.1:${QWEN38_ROUTER_PORT:-18000}/v1 on the login node
# (from a laptop: ssh -L 18000:127.0.0.1:18000 login). Loopback only.
# SLURMHELPS_SERVE_ROOT: the .slurmhelps-serve directory holding the
# deployments' state (default: <this checkout>/.slurmhelps-serve).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CLI="$ROOT/bin/slurmhelps-serve"
PORT=${QWEN38_ROUTER_PORT:-18000}
STATE_ROOT=${SLURMHELPS_SERVE_ROOT:-$ROOT/.slurmhelps-serve}
RUN_DIR="$STATE_ROOT/router"
PIDFILE="$RUN_DIR/router-$PORT.pid"
LOG="$RUN_DIR/router-$PORT.log"
ARGS=(router --state-root "$STATE_ROOT" --listen-port "$PORT"
  --deployment "bw=$HERE/config-bw.json"
  --deployment "hopper=$HERE/config-hopper.json")

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "router already running (pid $(cat "$PIDFILE"), port $PORT)"; exit 0; fi
    mkdir -p "$RUN_DIR"
    nohup "$CLI" "${ARGS[@]}" >>"$LOG" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    echo "router started pid $! port $PORT log $LOG"
    ;;
  stop)
    if running; then kill -TERM "$(cat "$PIDFILE")"; echo "router stopped (pid $(cat "$PIDFILE"))"; else echo "router not running"; fi
    rm -f "$PIDFILE"
    ;;
  status)
    if running; then echo "pid $(cat "$PIDFILE") log $LOG"; else echo "router process not running (pidfile $PIDFILE)"; fi
    exec "$CLI" router --status --listen-port "$PORT" "${@:2}"
    ;;
  run) exec "$CLI" "${ARGS[@]}" ;;
  -h|--help|help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "usage: $0 start|stop|status|run" >&2; exit 2 ;;
esac
