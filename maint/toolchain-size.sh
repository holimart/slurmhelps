#!/bin/bash
# =============================================================================
#  toolchain-size.sh [DIR...] — file count and GB per toolchain/cache dir
#  (what a toolchain image would hold). Run in a Slurm job: it walks trees.
# =============================================================================
#  Source of truth: slurmhelps maint/toolchain-size.sh. Default dirs: the
#  config's site.toolchain_dirs under site.work_root, plus uv/cache. Lustre
#  paths are walked with `lfs find` (cheaper than find), others with find -xdev.
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../lib/common.sh"; sh_load >/dev/null || exit 1
one() {
  if command -v lfs >/dev/null 2>&1 && [ "$(stat -f -c %T "$1" 2>/dev/null)" = lustre ]; then
    lfs find "$1" -type f -printf '%s\n' 2>/dev/null
  else
    find "$1" -xdev -type f -printf '%s\n' 2>/dev/null
  fi | awk -v d="$1" '{n++; s+=$1} END{printf "%8d files %8.2f GB  %s\n", n, s/1e9, d}'
}
export -f one
if [ $# -eq 0 ]; then
  set -- $(for d in ${SH_TOOLCHAIN_DIRS:-} uv/cache; do echo "$SH_WORK_ROOT/$d"; done)
fi
for d in "$@"; do [ -e "$d" ] && echo "$d"; done | xargs -P4 -I{} bash -c 'one "{}"'
