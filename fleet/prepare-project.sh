#!/usr/bin/env bash
# Verify a project's runtime is directly on the shared work filesystem
# ($SH_WORK_ROOT from the site profile) before starting its fleet.
# Source of truth: slurmhelps fleet/prepare-project.sh.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
sh_load

PROJ="${1:-${AGENT_PROJECT:-}}"
[ -n "$PROJ" ] || { echo "usage: prepare-project.sh <project-dir> (project must be under \$SH_WORK_ROOT)" >&2; exit 2; }
[ -n "${SH_WORK_ROOT:-}" ] || { echo "refusing: SH_WORK_ROOT unset (set site.work_root in the project's slurmhelps config; SLURMHELPS_PROJECT)" >&2; exit 2; }
PROJ="$(cd "$PROJ" && pwd -P)"
case "$PROJ" in
  "$SH_WORK_ROOT"/*) ;;
  *)
    echo "refusing: move the project checkout under $SH_WORK_ROOT first; Langywrap requires real runtime directories and cannot use symlinked private paths" >&2
    exit 2
    ;;
esac

LW="$PROJ/.langywrap"
GIT_COMMON="$(git -C "$PROJ" rev-parse --git-common-dir)"
if [[ "$GIT_COMMON" != /* ]]; then GIT_COMMON="$PROJ/$GIT_COMMON"; fi
GIT_COMMON="$(cd "$GIT_COMMON" && pwd -P)"

entries=(
  sessions.sqlite3 sessions.sqlite3-wal sessions.sqlite3-shm
  langypi-mail.sqlite3 langypi-mail.sqlite3-wal langypi-mail.sqlite3-shm
  session-supervisor.lock .sessions.sqlite3.execution-schema.lock
  .sessions.sqlite3.migration.lock pi-sessions activity-spool artifacts
  lane-reports turn-gate-reports turn-gates session-supervisor.log steward
)
for entry in "${entries[@]}"; do
  path="$LW/$entry"
  if [ -L "$path" ]; then
    echo "refusing: runtime path must be a real file/directory under $SH_WORK_ROOT: $path -> $(readlink "$path")" >&2
    exit 2
  fi
done
for path in "$LW"/.langypi-session.*; do
  [ -e "$path" ] || continue
  if [ -L "$path" ]; then
    echo "refusing: runtime path must be a real file under $SH_WORK_ROOT: $path -> $(readlink "$path")" >&2
    exit 2
  fi
done
if [ -L "$GIT_COMMON/langywrap" ]; then
  echo "refusing: .git/langywrap must be a real directory inside the work checkout" >&2
  exit 2
fi

echo "runtime paths are real and project-local under $SH_WORK_ROOT: $PROJ"
