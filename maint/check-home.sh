#!/usr/bin/env bash
# =============================================================================
#  check-home.sh — verify $HOME is not accumulating cache/pollution
# =============================================================================
#  The cluster gives $HOME only 10 GB soft / 15 GB hard. Every cache, model
#  store and tool database is redirected to /work via symlinks or env vars
#  (see bootstrap/setup-caches.sh and env/env.sh). This script checks that those
#  redirects are intact and flags anything that has started writing real
#  data into $HOME again — i.e. a tool that "installs into our home" and
#  therefore needs a new redirect.
#
#  Usage:  check-home.sh [THRESHOLD]
#            THRESHOLD  report real dirs larger than this (default 50M)
#  Exit:   0 = clean, 1 = something needs attention (or over quota)
#  Source of truth: slurmhelps maint/check-home.sh. WORK_ROOT: the
#  environment, else the slurmhelps config's site.work_root.
# =============================================================================
set -uo pipefail
if [ -z "${WORK_ROOT:-}" ]; then
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh" && sh_load >/dev/null 2>&1 && WORK_ROOT="${SH_WORK_ROOT:-}"
fi

H="$HOME"
WORK_ROOT="${WORK_ROOT:?WORK_ROOT unset and no slurmhelps config found}"
THRESHOLD="${1:-50M}"

warn=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  !! %s\n' "$*"; warn=1; }
good() { printf '  ok %s\n' "$*"; }

echo "== quota =="
if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -u "$(id -u)" /home 2>/dev/null | tail -2 | sed 's/^/  /'
fi
home_size="$(du -sh "$H" 2>/dev/null | cut -f1)"
echo "  \$HOME total: ${home_size:-?}"

echo
echo "== symlinks pointing into $WORK_ROOT =="
mapfile -t links < <(find "$H" -maxdepth 3 -type l -print 2>/dev/null | sort)
if [ "${#links[@]}" -eq 0 ]; then
  bad "no symlinks found — did setup-caches.sh run?"
fi
for l in "${links[@]}"; do
  tgt="$(readlink "$l")"
  case "$tgt" in
    "$WORK_ROOT"/*)
      if [ -e "$l" ]; then good "$l -> $tgt"; else bad "$l -> $tgt (BROKEN target)"; fi ;;
    *)
      note "$l -> $tgt  (not under WORK_ROOT; may be an intentional in-home link)" ;;
  esac
done

echo
echo "== real (non-symlink) directories > $THRESHOLD in \$HOME =="
# Depth 1 and depth 2, skipping anything that is (or is under) a symlink.
found=0
scan_dir() {
  local d="$1"
  [ -e "$d" ] || return 0
  [ -L "$d" ] && return 0            # redirected — good
  local sz
  sz="$(du -sm "$d" 2>/dev/null | cut -f1)"
  [ -z "$sz" ] && return 0
  # convert threshold to MB (supports N, NM, NG)
  local thr_mb
  case "$THRESHOLD" in
    *G|*g) thr_mb=$(( ${THRESHOLD%[Gg]} * 1024 )) ;;
    *M|*m) thr_mb=${THRESHOLD%[Mm]} ;;
    *)     thr_mb=$(( THRESHOLD / 1024 / 1024 )) ;;
  esac
  if [ "$sz" -ge "$thr_mb" ]; then
    bad "$d  (${sz} MB real — redirect this to $WORK_ROOT)"
    found=1
  fi
}
for d in "$H"/* "$H"/.[!.]*/; do
  [ -e "$d" ] || continue
  scan_dir "$d"
  # one level deeper for the usual container dirs
  case "$(basename "$d")" in
    .local|.config|.cache|.local/share|.local/state)
      for c in "$d"/*; do [ -e "$c" ] || continue; scan_dir "$c"; done ;;
  esac
done

# Known-cache names that must never exist as real dirs in $HOME.
echo
echo "== known cache/tool dirs that must be symlinks =="
for name in .cache .config .claude .jupyter .ipython .npm .opencode .vscode-server .langywrap; do
  p="$H/$name"
  [ -e "$p" ] || continue
  if [ -L "$p" ]; then good "$name linked"
  else bad "$name is a real directory (should point to $WORK_ROOT)"; fi
done

echo
if [ "$warn" -eq 0 ]; then
  printf '\033[32mclean\033[0m — $HOME holds no un-redirected cache data over %s\n' "$THRESHOLD"
  exit 0
else
  printf '\033[31mattention\033[0m — run: %s/bin/setup-caches.sh\n' "$WORK_ROOT"
  exit 1
fi
