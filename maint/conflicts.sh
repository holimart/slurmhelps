#!/bin/bash
# =============================================================================
#  conflicts.sh [-C WORKTREE] REF... — for each ref: trial merge, dump conflict
#  hunk counts and which side changed what, then abort. Read-only in effect.
# =============================================================================
#  Source of truth: slurmhelps maint/conflicts.sh. Append-only logs
#  (*.md, *.ndjson, *.jsonl) are skipped (land.sh unions them).
# =============================================================================
WT=.
while getopts "C:h" o; do case "$o" in C) WT="$OPTARG" ;; *) sed -n '2,8p' "$0"; exit 2 ;; esac; done
shift $((OPTIND - 1))
cd "$WT" || exit 2
for ref in "$@"; do
  echo "##### $ref  (mb=$(git merge-base HEAD "$ref" | cut -c1-9))"
  git merge --no-ff --no-commit "$ref" >/dev/null 2>&1
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in *.md|*.ndjson|*.jsonl) continue ;; esac
    mb=$(git merge-base HEAD MERGE_HEAD)
    printf '%s  conflict_hunks=%s  ours_since_mb=%s  theirs_since_mb=%s\n' "$f" \
      "$(grep -c '^<<<<<<<' "$f")" \
      "$(git log --oneline "$mb"..HEAD -- "$f" | wc -l)" "$(git log --oneline "$mb"..MERGE_HEAD -- "$f" | wc -l)"
    echo "   ours:   $(git log --format='%h %s' -3 "$mb"..HEAD -- "$f" | tr '\n' '|')"
    echo "   theirs: $(git log --format='%h %s' -3 "$mb"..MERGE_HEAD -- "$f" | tr '\n' '|')"
  done < <(git diff --name-only --diff-filter=U)
  git merge --abort
done
