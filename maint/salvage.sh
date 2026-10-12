#!/bin/bash
# =============================================================================
#  salvage.sh [-C REPO] [-w WT_DIR] WORKTREE... — commit each worktree's
#  uncommitted work (tracked + untracked, not ignored, not *.log) onto a
#  salvage/<name> branch at that worktree's HEAD; print the resulting ref.
# =============================================================================
#  Source of truth: slurmhelps maint/salvage.sh. A WORKTREE that is not an
#  absolute path is taken relative to WT_DIR (default <REPO>/.lane-worktrees).
#  Commit trailer: $SALVAGE_TRAILER (default none).
# =============================================================================
REPO=.; WTD=""
while getopts "C:w:h" o; do
  case "$o" in C) REPO="$OPTARG" ;; w) WTD="$OPTARG" ;; *) sed -n '2,10p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1))
REPO="$(cd "$REPO" && pwd)" || exit 2
WTD="${WTD:-$REPO/.lane-worktrees}"
S="$(mktemp)"; trap 'rm -f "$S"' EXIT
for wt in "$@"; do
  echo "=== $wt"
  case "$wt" in /*) d=$wt ;; *) d=$WTD/$wt ;; esac
  ( cd "$d" || exit 0
    name=$(basename "$wt")
    git status --porcelain --untracked-files=all | grep -v '^!!' | grep -vE "\.log$" > "$S"
    n=$(wc -l < "$S"); echo "dirty_paths=$n"
    [ "$n" -eq 0 ] && exit 0
    git checkout -q -B "salvage/$name"
    cut -c4- "$S" | sed 's/^"//;s/"$//' | while read -r p; do git add -A -- "$p"; done
    msg="salvage: uncommitted work from worktree $name ($n paths, $(date -u +%F))"
    [ -n "${SALVAGE_TRAILER:-}" ] && msg="$msg

$SALVAGE_TRAILER"
    git commit -q -m "$msg" && echo "salvaged $(git rev-parse --short HEAD) on salvage/$name"
    git diff --stat HEAD~1 HEAD | tail -1 )
done
