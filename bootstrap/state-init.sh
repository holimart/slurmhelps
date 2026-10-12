#!/usr/bin/env bash
# =============================================================================
#  state-init.sh — create the fleet's runtime skeleton under a work root
# =============================================================================
#  Usage: state-init.sh [--prefix DIR] [--config F] [--dry-run]
#
#  Creates (idempotent, never overwrites a file):
#    $SH_AGENTS_DIR/{logs,state}                        fleet logs + alloc.env
#    $SH_AGENTS_DIR/state/runtime/$PROJECT_NAME/store   0700, EMPTY: the session
#        store is ephemeral (lanes.json fleet.session_store.persist=false); the
#        daemon creates sessions.sqlite3 on node-local tmp and store-sync only
#        persists langypi-mail.sqlite3 here
#    $SH_FLEET_HOME/{.config,.local/bin,.local/share,.local/state}   HOME of
#        every fleet process
#    $SH_PI_AGENT_DIR (0700 parents)   pi agent dir: auth.json is NOT created
#    $SH_CODEX_HOME                   codex home: auth.json is NOT created
#    $SH_XDG_CACHE_HOME, $SH_IMAGES_DIR, $WORK_ROOT/{tmp,logs}
#  Then prints the MANUAL steps no script can do (credentials).
#  Secrets are never copied by any bootstrap script, into the repo or images.
# =============================================================================
set -euo pipefail
BS_PROG=state-init
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
bs_parse_common "$@"
set -- ${BS_ARGS[@]+"${BS_ARGS[@]}"}
case "${1:-}" in -h|--help) sed -n '2,21p' "$0"; exit 0 ;; "") ;; *) bs_die "unknown argument $1" ;; esac
bs_load
trap bs_cleanup EXIT
: "${PROJECT_NAME:?the config must set bootstrap.project_name}"
A="${SH_AGENTS_DIR:-$PREFIX/agents}"

mk() {  # mk MODE DIR...
  local mode="$1" d; shift
  for d in "$@"; do
    [ -n "$d" ] || continue
    if [ -d "$d" ]; then continue; fi
    bs_run install -d -m "$mode" "$d"
    [ "$BS_DRY" = 1 ] || bs_log "created $d"
  done
}
mk 0755 "$A" "$A/logs" "$A/state" "$A/state/runtime" "$A/state/runtime/$PROJECT_NAME"
mk 0700 "$A/state/runtime/$PROJECT_NAME/store"
mk 0755 "$SH_FLEET_HOME" "$SH_FLEET_HOME/.config" "$SH_FLEET_HOME/.local" "$SH_FLEET_HOME/.local/bin" \
  "$SH_FLEET_HOME/.local/share" "$SH_FLEET_HOME/.local/state"
# pi: the parent of .pi is private (auth.json lives below it)
if [ -n "${SH_PI_AGENT_DIR:-}" ]; then
  pi_home="$(dirname "$(dirname "$SH_PI_AGENT_DIR")")"
  mk 0700 "$pi_home" "$(dirname "$SH_PI_AGENT_DIR")"
  mk 0755 "$SH_PI_AGENT_DIR"
fi
mk 0700 "${SH_CODEX_HOME:-}"
mk 0755 "${SH_XDG_CACHE_HOME:-}" "${SH_IMAGES_DIR:-}" "$PREFIX/tmp" "$PREFIX/logs"

if [ -e "$A/state/runtime/$PROJECT_NAME/store/sessions.sqlite3" ]; then
  bs_warn "$A/state/runtime/$PROJECT_NAME/store/sessions.sqlite3 exists; the ephemeral store never restores it"
fi

cat <<EOF

== MANUAL steps (cannot be scripted; never copy credentials into a repo or an image) ==
 1. GitHub SSH key (clone the PRIVATE project + langywrap repos, push):
      ssh-keygen -t ed25519      # login node, login \$HOME/.ssh; add the .pub to
      # GitHub (and GitLab MFF for the upstream remote), then: ssh -T git@github.com
    checkouts.sh --https only works for public repos (holimart/pi is public;
    holimart/adgfem and holimart/langywrap are private).
 2. Pi / openai-codex OAuth (interactive, login node, once):
      source $PREFIX/config/env.sh
      export HOME=$SH_FLEET_HOME PI_CODING_AGENT_DIR=$SH_PI_AGENT_DIR
      langypi                      # then inside Pi:  /login openai-codex
      langypi auth check --provider openai-codex --json --no-refresh
    This writes $SH_PI_AGENT_DIR/auth.json (mode 0600). It is excluded from
    every workset image and staged in from /work at stage-in.
 3. Codex CLI (only if a lane uses it): CODEX_HOME=$SH_CODEX_HOME codex login
 4. API keys come from the environment, never files in a repo: e.g.
      export OLLAMA_API_KEY=...   (Ollama Cloud) in a private 0600 file sourced
      from $PREFIX/config/env.local.sh (not in an image).
 5. Claude Code steward login (login node): CLAUDE_CONFIG_DIR=$PREFIX/home/claude claude
 6. Verify: ly ensure; ly doctor   (langypi doctor --strict inside the allocation)
EOF
bs_log "done"
