# shellcheck shell=bash
# =============================================================================
#  lane-common.sh — shared environment and lane-release helpers for the lane
#  launchers (start-lanes.sh, start-all-lanes.sh, start-one-lane.sh,
#  label-lanes.sh). SOURCED, never executed. Source of truth: slurmhelps
#  fleet/lane-common.sh.
#
#  Caller must set PROJECT (absolute path of the project checkout) before
#  calling lane_env_setup / lane_release. Paths come from lib/common.sh
#  (project env $SLURMHELPS_PROJECT_ENV + site profile).
# =============================================================================

# shellcheck source=../lib/common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
sh_load || return 1
LANE_AGENTS="$AGENTS_DIR"
LANE_ALLOC_ENV="$LANE_AGENTS/state/alloc.env"
LANE_LW_CHECKOUT="$LANGYWRAP_CHECKOUT"

# alloc_value KEY — the value recorded for KEY in state/alloc.env (empty if absent).
alloc_value() { sh_alloc_value "$LANE_ALLOC_ENV" "$1"; }

# lane_in_workset — true when this process sees the allocation's workset mount
# namespace (or the allocation does not run a workset).
lane_in_workset() {
  [ "$(alloc_value WORKSET)" = 1 ] || return 0
  local pid; pid="$(alloc_value WORKSET_NS_PID)"
  [ -n "$pid" ] && [ "$(readlink /proc/self/ns/mnt)" = "$(readlink "/proc/$pid/ns/mnt" 2>/dev/null)" ]
}

# lane_env_setup — make this shell match the daemon's environment. Refuses
# outside a Slurm allocation: the session store is node-local to the compute node.
# `ly exec` / `ly start-lane` already run the same prelude as this function (see
# REMOTE_PRELUDE in ly); this only enforces the values, so a login-shell HOME or
# a stale store path cannot leak in when a launcher is run directly.
lane_env_setup() {
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "lane-env: not inside a Slurm allocation (SLURM_JOB_ID unset); refusing" >&2
    return 1
  fi
  # Never the login home: /home is not writable from compute nodes and must not
  # receive lane state. Same values as ly's REMOTE_PRELUDE and supervisor.sbatch.
  sh_fleet_env || return 1

  # alloc.env is the daemon's own record; it must be for THIS allocation.
  local recorded_job v
  recorded_job="$(alloc_value SLURM_JOB_ID)"
  if [ -n "$recorded_job" ] && [ "$recorded_job" != "$SLURM_JOB_ID" ]; then
    echo "lane-env: $LANE_ALLOC_ENV records job $recorded_job but this is job $SLURM_JOB_ID; stale record, refusing" >&2
    return 1
  fi
  # A workset allocation serves the project from node-local tmp bound over its
  # /work path, visible only inside the workset mount namespace. Outside it a
  # lane would work on the stale Lustre copy.
  if ! lane_in_workset; then
    echo "lane-env: this allocation runs a workset (fleet.workset) but this shell is outside its namespace;" >&2
    echo "lane-env:   run through ly exec / ly start-lane (or: workset.sh exec -- ...); refusing" >&2
    return 1
  fi
  v="$(alloc_value XDG_RUNTIME_DIR)"
  [ -n "$v" ] && export XDG_RUNTIME_DIR="$v"
  # The CLI must open the same node-local session DB as the daemon.
  v="$(alloc_value LANGYWRAP_SESSIONS_DB_DIR)"
  if [ -n "$v" ]; then
    export LANGYWRAP_SESSIONS_DB_DIR="$v"
  elif [ -z "${LANGYWRAP_SESSIONS_DB_DIR:-}" ]; then
    echo "lane-env: no LANGYWRAP_SESSIONS_DB_DIR in the environment or $LANE_ALLOC_ENV" >&2
    echo "lane-env: the allocation predates the node-local store; restart it with ly renew" >&2
    return 1
  fi
  return 0
}

# lane_active_turn LANE — prints the active turn id of LANE, if any.
lane_active_turn() {
  langywrap lanes status "$1" --project . --json 2>/dev/null \
    | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("active_turn") or "")
except ValueError:
    pass'
}

# lane_release LANE ABANDON_STALE(0|1)
#   Detach any idle attachment so `lanes start` can attach a fresh session.
#   Returns 0 when the lane is free to start, 1 otherwise (diagnostic on stderr).
#   A delivered turn blocking the detach is abandoned only with ABANDON_STALE=1,
#   and only if abandon-turn accepts it (it refuses while any attempt is live).
lane_release() {
  local lane="$1" abandon="$2" out turn ab
  out="$(langywrap lanes detach "$lane" --project . 2>&1)" && return 0
  case "$out" in
    *"no attached session"*|*"lease is missing"*) return 0 ;;
    *"active turn"*) ;;
    *) echo "$lane: detach failed: $(printf '%s' "$out" | tail -1)" >&2; return 1 ;;
  esac

  turn="$(lane_active_turn "$lane")"
  if [ -z "$turn" ]; then
    echo "$lane: detach refused ($(printf '%s' "$out" | tail -1)) but status reports no active turn" >&2
    return 1
  fi
  if [ "$abandon" != 1 ]; then
    echo "$lane: detach refused: delivered turn $turn is still open." >&2
    echo "$lane:   if its attempt is terminal, abandon it with:" >&2
    echo "    langywrap lanes abandon-turn $lane --project . --turn $turn --reason '<why>'" >&2
    echo "$lane:   or re-run with --abandon-stale (automatic only when no attempt is live)." >&2
    return 1
  fi
  if ! ab="$(langywrap lanes abandon-turn "$lane" --project . --turn "$turn" \
      --reason "start-lanes --abandon-stale: detach refused on open turn $turn" 2>&1)"; then
    echo "$lane: abandon-turn $turn refused: $(printf '%s' "$ab" | tail -1)" >&2
    return 1
  fi
  echo "$lane: abandoned open turn $turn (its items are now deferred debt)" >&2
  if out="$(langywrap lanes detach "$lane" --project . 2>&1)"; then
    return 0
  fi
  echo "$lane: detach still failed after abandon: $(printf '%s' "$out" | tail -1)" >&2
  return 1
}
