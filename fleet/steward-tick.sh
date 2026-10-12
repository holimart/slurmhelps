#!/usr/bin/env bash
# One auditable steward tick. Run via `ly exec 'bash "$SLURMHELPS_HOME/fleet/steward-tick.sh"'`.
# Source of truth: slurmhelps fleet/steward-tick.sh.
set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "steward-tick: refusing to run outside a Slurm allocation" >&2
  exit 2
fi

if ! command -v langywrap >/dev/null 2>&1; then
  echo "steward-tick: langywrap is not on PATH inside allocation" >&2
  exit 2
fi

failures=0
retry_base="${STEWARD_TICK_RETRY_BASE_SECONDS:-2}"
retry_max="${STEWARD_TICK_RETRY_MAX_SECONDS:-30}"
retry_limit="${STEWARD_TICK_RETRIES:-0}"
# Slurm/transport errors only; the same set as ly's is_slurm_transport_error.
# Ordinary failures (semantic mismatch, bad arguments, "timeout" in a report)
# are never retried.
transient_re='Error creating slurm stream socket: (Connection refused|Connection timed out|Resource temporarily unavailable)|Unable to contact slurm controller|Unable to confirm allocation|slurm_load_jobs error: (Connection|Unable to contact|Socket|Timed out|Timeout|Resource temporarily unavailable)|slurm_recv_timeout|Connection timed out|Socket timed out|Resource temporarily unavailable'
[[ "$retry_base" =~ ^[1-9][0-9]*$ && "$retry_base" -le 30 ]] || { echo "steward-tick: invalid retry base" >&2; exit 2; }
[[ "$retry_max" =~ ^[1-9][0-9]*$ && "$retry_max" -le 60 ]] || { echo "steward-tick: invalid retry maximum" >&2; exit 2; }
[[ "$retry_limit" =~ ^(0|[1-9][0-9]*)$ && "$retry_limit" -le 12 ]] || { echo "steward-tick: retries must be 0 (unlimited) or 1..12" >&2; exit 2; }
log_dir="${STEWARD_TICK_LOG_DIR:-${AGENTS_DIR:?AGENTS_DIR unset; run through ly exec}/logs}"
mkdir -p "$log_dir" || {
  echo "steward-tick: cannot create durable log directory $log_dir" >&2
  exit 2
}
log_file="$log_dir/steward-tick-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
printf 'steward tick start=%s job=%s host=%s\n' "$(date -u +%FT%TZ)" \
  "$SLURM_JOB_ID" "$(hostname)" | tee -a "$log_file"

run_probe() {
  local title="$1" limit="$2" retry_safe="$3"; shift 3
  local output rc attempt=1 delay="$retry_base" before_tick_state="" after_tick_state="" tick_summary=""
  output="$(mktemp "${TMPDIR:-/tmp}/steward-tick.XXXXXX")" || {
    echo "steward-tick: cannot create temporary output for $title" >&2
    failures=$((failures + 1))
    return
  }
  echo "########## $title ##########"
  printf '\n########## %s ##########\n' "$title" >>"$log_file"
  if [ "$retry_safe" = "tick" ]; then
    before_tick_state="$(cat .cache/steward/tick-state.json 2>/dev/null || true)"
  fi
  while :; do
    : >"$output"
    if "$@" >"$output" 2>&1; then rc=0; else rc=$?; fi
    [ "$rc" -eq 0 ] && break
    # A tick writes its state, so retry it only if the durable state is
    # unchanged and the error is a recognized transient. The tick's configured
    # commands are checked into the registry and must remain read-only here.
    if [ "$retry_safe" = "tick" ]; then
      after_tick_state="$(cat .cache/steward/tick-state.json 2>/dev/null || true)"
      if [ -n "$after_tick_state" ] && [ "$after_tick_state" != "$before_tick_state" ]; then
        echo "steward-tick: tick state advanced despite exit $rc; not repeating the tick" >&2
        break
      fi
    fi
    if { [ "$retry_safe" = "yes" ] || [ "$retry_safe" = "tick" ]; } \
       && { [ "$retry_limit" -eq 0 ] || [ "$attempt" -le "$retry_limit" ]; } \
       && grep -Eiq "$transient_re" "$output"; then
      if [ "$retry_limit" -eq 0 ]; then
        echo "steward-tick: $title transient failure (retry $attempt; unlimited); waiting ${delay}s" >&2
      else
        echo "steward-tick: $title transient failure (retry $attempt/$retry_limit); waiting ${delay}s" >&2
      fi
      sleep "$delay"
      attempt=$((attempt + 1))
      delay=$((delay * 2)); [ "$delay" -le "$retry_max" ] || delay="$retry_max"
      continue
    fi
    break
  done
  if [ "$retry_safe" = "tick" ]; then
    tick_summary="$(python3 - "$output" <<'PY'
import json
import sys
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
start = raw.find("{")
try:
    report, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else ({}, 0)
except (json.JSONDecodeError, IndexError):
    report = {}
if report.get("schema") == "fleet-tick/v1":
    print(
        "CHANGED this tick: fleet report generated_at={} ok={} needs_action={}".format(
            report.get("generated_at", "unknown"),
            report.get("ok", "unknown"),
            len(report.get("needs_action", [])),
        )
    )
PY
    )"
    if [ -n "$tick_summary" ]; then
      printf '%s\n' "$tick_summary"
      printf '%s\n' "$tick_summary" >>"$log_file"
    fi
  fi
  cat "$output" >>"$log_file"
  tail -n "$limit" "$output"
  # Every nonzero exit is a failure, even when the tick advanced its state:
  # the advance only stops a retry, it does not excuse the exit code.
  if [ "$rc" -ne 0 ]; then
    echo "steward-tick: $title exited $rc" >&2
    echo "steward-tick: $title exited $rc" >>"$log_file"
    failures=$((failures + 1))
  fi
  rm -f "$output"
}

# Retry recognizable Slurm transport failures indefinitely by
# default (STEWARD_TICK_RETRIES=N caps retries after the first attempt) on read-only
# checks. The tick is repeated only when its durable state is unchanged; a
# semantic mismatch or stale live-session setting is evidence to record, not a
# transient condition that another identical probe can heal.

run_probe "STEWARD TICK" 35 tick langywrap steward tick run --text
run_probe "VERIFY DISPATCH" 20 yes langywrap steward verify-dispatch --project .
run_probe "LANE MAP" 20 yes langywrap steward lane-map --project .

if [ "$failures" -gt 0 ]; then
  echo "steward-tick: $failures probe(s) failed" >&2
  echo "steward tick complete=$(date -u +%FT%TZ) failures=$failures log=$log_file" | tee -a "$log_file"
  exit 1
fi
echo "steward-tick: all probes exited 0"
echo "steward tick complete=$(date -u +%FT%TZ) failures=0 log=$log_file" | tee -a "$log_file"
