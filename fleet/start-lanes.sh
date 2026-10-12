#!/usr/bin/env bash
# =============================================================================
#  start-lanes.sh [--abandon-stale] [lane ...] — start each lane as one turn
# =============================================================================
#  The single implementation behind every lane launcher (start-all-lanes.sh,
#  start-one-lane.sh, `ly start-lane`). Run it INSIDE a Slurm allocation, e.g.
#  `ly start-lane harden`. Source of truth: slurmhelps fleet/start-lanes.sh.
#
#  Per lane: detach any idle attachment, then `langywrap lanes start` with ONE
#  explicit prompt (no continuation loop armed). Two phases: every lane is
#  released first so no stale lease can block an attach, then every lane is
#  started. A lane whose release failed is not started. Exit 1 if any lane
#  failed; a summary table is printed at the end.
#
#  Options:
#    --abandon-stale  when a detach is refused because a delivered turn is still
#                     open, abandon that turn (`langywrap lanes abandon-turn`)
#                     and detach again. abandon-turn itself refuses while any
#                     attempt of the lane is live, so a live lane is never
#                     touched. Without this flag the remedy command is printed.
#
#  Usage:
#    start-lanes.sh                       # all worker lanes (not `steward`)
#    start-lanes.sh harden counters       # a chosen subset
#    start-lanes.sh steward               # the steward lane (never in the default set)
#    start-lanes.sh --abandon-stale       # all workers, auto-abandon stale turns
#    MODEL=<provider/model> start-lanes.sh
#
#  Project configuration (slurmhelps config "fleet" section, docs/config.md):
#    lane_model        default model (MODEL in the environment wins)
#    lane_prompt_file  the one-turn prompt; {lane} and {model} are replaced.
#                      Without it a generic prompt is used.
#    lane_exclude      lanes never in the default set (default: steward)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-}"   # default: PROJECT from state/alloc.env (the node-local root when enabled)
ABANDON_STALE=0
LANE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --abandon-stale) ABANDON_STALE=1 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "start-lanes: unknown option $arg" >&2; exit 2 ;;
    *) LANE_ARGS+=("$arg") ;;
  esac
done

# Refuse on a Slurm login node (see login-node-guard.sh). Checked before any
# environment work: outside an allocation the session DB path is meaningless.
if ! bash "$HERE/login-node-guard.sh"; then
  echo "start-lanes: refused by login-node-guard.sh; not starting any lane." >&2
  exit 1
fi

# shellcheck source=lane-common.sh
. "$HERE/lane-common.sh"
PROJECT="${PROJECT:-$(alloc_value PROJECT)}"
PROJECT="${PROJECT:-${AGENT_PROJECT:?no PROJECT in alloc.env and AGENT_PROJECT unset}}"
MODEL="${MODEL:-${LANE_MODEL:?set fleet.lane_model in the slurmhelps config or MODEL in the environment}}"
LANE_EXCLUDE="${LANE_EXCLUDE:-steward}"
cd "$PROJECT" || exit 1
lane_env_setup || { echo "start-lanes: environment not valid; not starting any lane." >&2; exit 1; }

LWPY="$LANE_LW_CHECKOUT/.venv/bin/python"
LOG="$LANE_AGENTS/logs/start-lanes.log"
mkdir -p "$(dirname "$LOG")"

if [ "${#LANE_ARGS[@]}" -gt 0 ]; then
  LANES=("${LANE_ARGS[@]}")
else
  mapfile -t LANES < <("$LWPY" - "$PROJECT/.langywrap/lanes.json" "$LANE_EXCLUDE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
skip=set(sys.argv[2].split())
for l in d["lanes"]:
    if l["id"] not in skip:
        print(l["id"])
PY
)
fi
if [ "${#LANES[@]}" -eq 0 ]; then
  echo "start-lanes: no lanes to start" >&2
  exit 1
fi

say() { echo "$*" | tee -a "$LOG"; }

# lane_prompt LANE — the one-turn start prompt for LANE.
lane_prompt() {
  local t
  if [ -n "${LANE_PROMPT_FILE:-}" ]; then
    case "$LANE_PROMPT_FILE" in /*) ;; *) LANE_PROMPT_FILE="$PROJECT/$LANE_PROMPT_FILE" ;; esac
    t="$(cat "$LANE_PROMPT_FILE")" || { echo "start-lanes: cannot read LANE_PROMPT_FILE $LANE_PROMPT_FILE" >&2; return 1; }
  else
    t='You are the `{lane}` lane (model {model}). Read your brief and backlog as declared for lane {lane} in .langywrap/lanes.json, then do the FIRST actionable item of your backlog. Never edit a file owned by another live lane. Record your turn and commit only files you own. End your turn with a one-line status: lane, item, outcome.'
  fi
  t="${t//\{lane\}/$1}"
  printf '%s' "${t//\{model\}/$MODEL}"
}
say "starting ${#LANES[@]} lane(s) on $MODEL in $PROJECT (abandon-stale=$ABANDON_STALE)"

declare -A RELEASE START
for lane in "${LANES[@]}"; do
  # diagnostics from lane_release go straight to the console
  if lane_release "$lane" "$ABANDON_STALE"; then
    RELEASE[$lane]=ok
  else
    RELEASE[$lane]=FAILED
  fi
done

for lane in "${LANES[@]}"; do
  if [ "${RELEASE[$lane]}" != ok ]; then
    START[$lane]="skipped (release failed)"
    continue
  fi
  if ! prompt="$(lane_prompt "$lane")" || [ -z "$prompt" ]; then
    START[$lane]="FAILED (no prompt)"
    continue
  fi
  say "--- starting $lane ---"
  out="$(langywrap lanes start "$lane" \
      --project "$PROJECT" \
      --prompt "$prompt" \
      --request-key "$lane-$(date +%s%N)-$RANDOM" 2>&1)"
  rc=$?
  printf '%s\n' "$out" | tail -n 3 | tee -a "$LOG"
  if [ "$rc" -eq 0 ]; then
    START[$lane]=ok
  else
    START[$lane]="FAILED (exit $rc)"
  fi
  sleep 1
done

echo
printf '%-14s %-10s %s\n' LANE RELEASE START | tee -a "$LOG"
failed=0
for lane in "${LANES[@]}"; do
  printf '%-14s %-10s %s\n' "$lane" "${RELEASE[$lane]}" "${START[$lane]}" | tee -a "$LOG"
  case "${RELEASE[$lane]} ${START[$lane]}" in *FAILED*|*skipped*) failed=$((failed + 1)) ;; esac
done
echo

if [ "$failed" -gt 0 ]; then
  say "start-lanes: $failed of ${#LANES[@]} lane(s) did not start"
  exit 1
fi
say "start-lanes: all ${#LANES[@]} lane(s) started"

langywrap session list --path "$PROJECT" 2>/dev/null | "$LWPY" -c "
import sys,json
want=set('''${LANES[*]}'''.split())
for s in json.load(sys.stdin):
    if s.get('label') in want:
        print(s['id'][:8], s['state'], s['model'], s['label'])
" || true
