#!/bin/bash
# =============================================================================
#  trash-bulky.sh [--apply] [-l LABEL] PATH... — move bulky items into the
#  trash with ONE rename each (cheap on Lustre: no copy, no recursive delete),
#  then prune git worktree metadata of the repos they belonged to.
# =============================================================================
#  Source of truth: slurmhelps maint/trash-bulky.sh (generalized from the
#  2026-10-12 one-off). Destination: $SH_TRASH_DIR/<LABEL>/<path relative to
#  SH_WORK_ROOT> (config site.work_root; trash = <work_root>/.trash). LABEL
#  defaults to bulky-<UTC date>. Dry run unless --apply. Only move things that
#  are merged, on a pushed branch, regenerable, or half-written copies; the
#  trash is emptied by a separate, deliberate step (never by this script).
#  Branches are never touched.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/common.sh"; sh_load >/dev/null || exit 1
APPLY=0; LABEL="bulky-$(date -u +%F)"
while [ $# -gt 0 ]; do
  case "$1" in --apply) APPLY=1; shift ;; -l) LABEL="$2"; shift 2 ;; -h|--help) sed -n '2,15p' "$0"; exit 0 ;; *) break ;; esac
done
WR="${SH_WORK_ROOT:?site.work_root unset}"
TR="${SH_TRASH_DIR:-$WR/.trash}/$LABEL"
repos=()
for p in "$@"; do
  p="$(realpath -s "$p")"
  case "$p" in "$WR"/*) ;; *) echo "skip (outside $WR) $p"; continue ;; esac
  [ -e "$p" ] || { echo "skip (missing) $p"; continue; }
  dst="$TR/${p#"$WR"/}"
  if [ "$APPLY" = 1 ]; then
    mkdir -p "$(dirname "$dst")" && mv "$p" "$dst" && echo "trashed $p -> $dst"
  else
    echo "would trash $p -> $dst"
  fi
  r="$(git -C "$(dirname "$p")" rev-parse --show-toplevel 2>/dev/null)" && repos+=("$r")
done
[ "$APPLY" = 1 ] || { echo "dry run only; re-run with --apply"; exit 0; }
for r in $(printf '%s\n' ${repos[@]+"${repos[@]}"} | sort -u); do
  ( cd "$r" && git worktree prune -v 2>&1 | tail -3 )
done
