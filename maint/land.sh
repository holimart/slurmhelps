#!/bin/bash
# =============================================================================
#  land.sh [-C WORKTREE] [-b BRANCH] [-k KEEP_PATHSPEC] REF... — merge lane refs
#  one at a time into a landing worktree.
# =============================================================================
#  Source of truth: slurmhelps maint/land.sh (was a one-off landing script of
#  2026-10-12). Append-only logs (*.md, *.ndjson, *.jsonl) are union-merged;
#  paths under KEEP_PATHSPEC (default SRC_MPI) keep the landing side ("ours",
#  recorded in the commit message: they must land through the lane with their
#  own gate); any other conflict aborts that merge and is reported.
#    -C WORKTREE  the landing worktree (default: cwd)
#    -b BRANCH    name used in the commit message (default: the worktree's branch)
#    -k PATHSPEC  "keep ours" pathspec (repeatable; default SRC_MPI)
#  Commit trailer: $LAND_TRAILER (default none).
# =============================================================================
WT=.; BR=""; KEEP=()
while getopts "C:b:k:h" o; do
  case "$o" in C) WT="$OPTARG" ;; b) BR="$OPTARG" ;; k) KEEP+=("$OPTARG") ;; *) sed -n '2,16p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1))
[ "${#KEEP[@]}" -gt 0 ] || KEEP=(SRC_MPI)
cd "$WT" || exit 2
BR="${BR:-$(git symbolic-ref -q --short HEAD || echo HEAD)}"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
in_keep() { local k; for k in "${KEEP[@]}"; do case "$1" in "$k"|"$k"/*) return 0 ;; esac; done; return 1; }
for ref in "$@"; do
  echo "=== $ref"
  if git merge-base --is-ancestor "$ref" HEAD; then echo "already contained"; continue; fi
  git merge --no-ff --no-commit "$ref" > "$T/m.out" 2>&1
  bad=""
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in
      *.md|*.ndjson|*.jsonl)
        git show :1:"$f" > "$T/b" 2>/dev/null || : > "$T/b"
        git show :2:"$f" > "$T/o" 2>/dev/null || : > "$T/o"
        git show :3:"$f" > "$T/t" 2>/dev/null || : > "$T/t"
        git merge-file --union -p "$T/o" "$T/b" "$T/t" > "$f"; git add "$f" ;;
      *) in_keep "$f" || bad="$bad $f" ;;
    esac
  done < <(git diff --name-only --diff-filter=U)
  if [ -n "$bad" ]; then
    echo "ABORT: non-log conflicts:$bad"; git merge --abort; continue
  fi
  kept=$( { git diff --cached --name-only HEAD -- "${KEEP[@]}"; git diff --name-only --diff-filter=U -- "${KEEP[@]}"; } | sort -u | grep .)
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if git cat-file -e HEAD:"$f" 2>/dev/null; then git checkout HEAD -- "$f"; else git rm -q --cached "$f"; rm -f "$f"; fi
  done <<< "$kept"
  n=$(git diff --cached --name-only HEAD | wc -l)
  msg="land: merge $ref into $BR ($n files; append-only logs unioned)"
  [ -n "$kept" ] && msg="$msg

Paths kept at $BR (land them through their lane with its gate): $(echo $kept | tr '\n' ' ')"
  [ -n "${LAND_TRAILER:-}" ] && msg="$msg

$LAND_TRAILER"
  git commit -q -m "$msg" && echo "landed $(git log --oneline -1 | cut -c1-9) files=$n kept=[$(echo $kept)]"
done
