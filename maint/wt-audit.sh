#!/bin/bash
# =============================================================================
#  wt-audit.sh [-C REPO] [-b BRANCH] WORKTREE... — read-only, one worktree at a
#  time: what in each worktree is not in BRANCH.
# =============================================================================
#  Source of truth: slurmhelps maint/wt-audit.sh.
#   ahead   = commits on HEAD not reachable from BRANCH (ancestry)
#   content = files touched by those commits that still differ from BRANCH and
#             whose last branch-side touch is not older than BRANCH's last touch
#             since the fork (conflict-resolved landings and superseded edits
#             do not count)
#   dirty   = uncommitted tracked changes + untracked (ignored not counted)
#  BRANCH default: the main checkout's current branch.
# =============================================================================
export GIT_OPTIONAL_LOCKS=0
REPO=.; I=""
while getopts "C:b:h" o; do
  case "$o" in C) REPO="$OPTARG" ;; b) I="$OPTARG" ;; *) sed -n '2,14p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1))
cd "$REPO" || exit 2
I="${I:-$(git symbolic-ref -q --short HEAD)}"
for wt in "$@"; do
  wt=${wt%/}
  br=$(git -C "$wt" symbolic-ref -q --short HEAD || echo "detached@$(git -C "$wt" rev-parse --short HEAD)")
  tip_age=$(( ( $(date +%s) - $(git -C "$wt" log -1 --format=%at HEAD) ) / 3600 ))
  ahead=$(git -C "$wt" rev-list --count --no-merges "$I"..HEAD)
  content=0; cfiles=""
  if [ "$ahead" -gt 0 ]; then
    B=$(mktemp); M=$(mktemp); D=$(mktemp)
    git -C "$wt" log --no-merges --format=@%at --name-only "$I"..HEAD | awk '/^@/{t=substr($0,2);next} NF&&!($0 in s){s[$0]=t; print $0"\t"t}' > "$B"
    git -C "$wt" log --format=@%at --name-only HEAD.."$I" | awk '/^@/{t=substr($0,2);next} NF&&!($0 in s){s[$0]=t; print $0"\t"t}' > "$M"
    git -C "$wt" diff --name-only "$I" HEAD > "$D"
    cfiles=$(awk -F'\t' 'FILENAME==ARGV[1]{d[$0]=1;next} FILENAME==ARGV[2]{m[$1]=$2;next} ($1 in d) && ($2+0 >= m[$1]+0){print $1}' "$D" "$M" "$B")
    content=$(printf '%s' "$cfiles" | grep -c .)
    rm -f "$B" "$M" "$D"
  fi
  dirty=$(git -C "$wt" status --porcelain --untracked-files=normal | grep -vc '^!!')
  printf '%-45s %-40s tip=%4sh ahead=%4s content_files=%4s dirty=%4s\n' "$wt" "$br" "$tip_age" "$ahead" "$content" "$dirty"
  [ "$content" -gt 0 ] && echo "    areas: $(printf '%s\n' "$cfiles" | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn | head -5 | awk '{printf "%s(%s) ", $2, $1}')"
  [ "$dirty" -gt 0 ] && git -C "$wt" status --porcelain --untracked-files=normal | grep -v '^!!' | head -4 | sed 's/^/    dirty: /'
done
