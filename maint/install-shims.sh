#!/usr/bin/env bash
# =============================================================================
#  install-shims.sh [--apply] [--project DIR] — point the old per-user helper
#  paths at their slurmhelps copies (run once after adopting slurmhelps).
# =============================================================================
#  Source of truth: slurmhelps maint/install-shims.sh. Dry run unless --apply.
#  Every replaced file is kept next to it as <file>.pre-slurmhelps (once).
#
#  What it writes (all paths from the project's slurmhelps config, "site" and
#  "bootstrap" sections; the same files bootstrap/toolchain-install.sh writes,
#  so the result is byte-identical to a fresh rebuild):
#    $WORK_ROOT/config/env.sh         shim -> slurmhelps env/env.sh (other
#                                     projects source this path: keep it)
#    $WORK_ROOT/config/toolchain.sh   shim -> site.toolchain_profile
#    $WORK_ROOT/config/env.local.sh   bootstrap.env_local (only if absent)
#    $WORK_ROOT/bin/<name>            bootstrap.bin_shims exec shims
#  and, by this script only (old paths that callers may still use):
#    $WORK_ROOT/config/toolchain-mffsrv.sh
#                                     shim -> site.toolchain_profile
#    $AGENTS_DIR/scripts/{label-lanes,start-all-lanes,start-one-lane,steward-tick}.sh
#                                     exec shims -> slurmhelps fleet/ (these old
#                                     copies predate the repo ones). Other files
#                                     in agents/scripts are one-off scripts tied
#                                     to a retired /home checkout: left alone.
#
#  Run it on the login node (small writes only) or in a short job.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH="$(cd "$HERE/.." && pwd)"
APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --project) export SLURMHELPS_PROJECT="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "install-shims: unknown option $1" >&2; exit 2 ;;
  esac
done
# shellcheck source=../lib/common.sh
. "$SH/lib/common.sh"
sh_load || exit 1
WR="${SH_WORK_ROOT:?site.work_root unset in the slurmhelps config}"
PROJ="${SLURMHELPS_PROJECT:-$SH_PROJECT}"

say() { echo "install-shims: $*"; }
# shim PATH CONTENT [MODE] — replace PATH with CONTENT (backup once).
shim() {
  local p="$1" body="$2" mode="${3:-644}"
  if [ -f "$p" ] && [ ! -L "$p" ] && [ "$(cat "$p")" = "$body" ]; then
    say "unchanged $p"; return 0
  fi
  if [ "$APPLY" = 0 ]; then
    say "would write $p$( [ -e "$p" ] || [ -L "$p" ] && echo " (existing kept as $p.pre-slurmhelps)")"
    return 0
  fi
  if { [ -e "$p" ] || [ -L "$p" ]; } && [ ! -e "$p.pre-slurmhelps" ] && [ ! -L "$p.pre-slurmhelps" ]; then
    mv "$p" "$p.pre-slurmhelps"
  fi
  rm -f "$p"
  mkdir -p "$(dirname "$p")"
  printf '%s\n' "$body" > "$p.tmp$$" && chmod "$mode" "$p.tmp$$" && mv -f "$p.tmp$$" "$p"
  say "wrote $p"
}

# 1. config/env.sh, toolchain.sh, env.local.sh, bin/<shims>: the bootstrap's own steps
args=(--only config,helpers)
[ "$APPLY" = 0 ] && args+=(--dry-run)
for f in "$WR/config/env.sh" "$WR/config/toolchain.sh"; do
  if [ "$APPLY" = 1 ] && [ -e "$f" ] && [ ! -e "$f.pre-slurmhelps" ]; then cp -p "$f" "$f.pre-slurmhelps"; fi
done
for n in $("$(sh_python)" "$SH/lib/slurmhelps/config.py" get bootstrap.bin_shims 2>/dev/null \
           | "$(sh_python)" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))' 2>/dev/null); do
  f="$WR/bin/$n"
  if [ "$APPLY" = 1 ] && { [ -e "$f" ] || [ -L "$f" ]; } && [ ! -e "$f.pre-slurmhelps" ]; then mv "$f" "$f.pre-slurmhelps"; fi
done
bash "$SH/bootstrap/toolchain-install.sh" "${args[@]}" || exit 1

# 2. the old toolchain profile path
if [ -n "${SH_TOOLCHAIN_PROFILE:-}" ]; then
  shim "$WR/config/toolchain-mffsrv.sh" "# Shim (slurmhelps maint/install-shims.sh): the site toolchain profile lives in slurmhelps.
. \"$SH_TOOLCHAIN_PROFILE\""
fi

# 3. old fleet launcher copies in agents/scripts
for n in label-lanes.sh start-all-lanes.sh start-one-lane.sh steward-tick.sh; do
  f="$AGENTS_DIR/scripts/$n"
  [ -e "$f" ] || continue
  shim "$f" "#!/usr/bin/env bash
# Shim (slurmhelps maint/install-shims.sh): the script lives in slurmhelps fleet/.
export SLURMHELPS_PROJECT=\"\${SLURMHELPS_PROJECT:-$PROJ}\"
exec bash \"$SH/fleet/$n\" \"\$@\"" 755
done
[ "$APPLY" = 1 ] || say "dry run only; re-run with --apply"
