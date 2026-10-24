#!/usr/bin/env bash
# =============================================================================
#  lock-versions.sh — record (or check) every toolchain/checkout version
# =============================================================================
#  Usage:
#    lock-versions.sh [--prefix DIR] [--config F] [--out FILE]
#        Inspect the install under the work root and write versions.lock
#        (default bootstrap/versions.lock). Run it on the CURRENT install so a
#        rebuild from zero reproduces today's versions.
#    lock-versions.sh --check [--prefix DIR] [--config F] [--lock FILE]
#        Inspect the install under DIR and print a table lock-vs-actual.
#        Exit 0 when every pinned key matches, 1 otherwise.
#
#  Read-only: runs version commands and lists a few directories (no walks).
#  Keys (KEY=VALUE, sourced by toolchain-install.sh / checkouts.sh):
#    UV_VERSION UV_PYTHONS UV_TOOLS NODE_VERSION NPM_VERSION NPM_GLOBALS
#    RUSTUP_VERSION RUST_TOOLCHAIN RUST_PROFILE FLEET_PYTHON FLEET_PYTHON_VERSION
#    LANGYWRAP_COMMIT LANGYWRAP_VENV_REQS PI_COMMIT PI_RELEASE_TAG PI_VERSION
#    RTK_COMMIT RTK_VERSION ; info only (not compared): GIT_VERSION
#    PROJECT_COMMIT GENERATED_FROM GENERATED_AT
# =============================================================================
set -uo pipefail
BS_PROG=lock-versions
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
CHECK=0; OUT=""
bs_parse_common "$@"
set -- ${BS_ARGS[@]+"${BS_ARGS[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --lock) BS_LOCK="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) bs_die "unknown argument $1" ;;
  esac
done
bs_load
trap bs_cleanup EXIT
W="$PREFIX"
LW="${SH_LANGYWRAP_CHECKOUT:-$W/langywrap}"

# --------------------------------------------------------------- inspection
v_uv() { "$W/bin/uv" --version 2>/dev/null | awk '{print $2}'; }
v_uv_pythons() {
  local d out=()
  for d in "$W"/uv/python/cpython-*; do
    [ -d "$d" ] && [ ! -L "$d" ] || continue
    d="$(basename "$d")"; d="${d#cpython-}"; out+=("${d%%-*}")
  done
  printf '%s\n' ${out[@]+"${out[@]}"} | sort -V | tr '\n' ' ' | sed 's/ $//'
}
v_uv_tools() {
  local t dist out=()
  for t in "$W"/uv/tools/*/; do
    [ -d "$t" ] || continue
    for dist in "$t"lib/python*/site-packages/*.dist-info; do
      [ -d "$dist" ] || continue
      local n; n="$(basename "$dist" .dist-info)"
      # the tool's own package: the dist named like the tool dir (- vs _)
      case "$(echo "${n%-*}" | tr '_' '-')" in "$(basename "$t")") out+=("$(basename "$t")==${n##*-}") ;; esac
    done
  done
  printf '%s ' ${out[@]+"${out[@]}"} | sed 's/ $//'
}
v_node() { local d; for d in "$W"/node/v*; do [ -x "$d/bin/node" ] && basename "$d"; done | sort -V | tail -1; }
v_npm() { local n; n="$(v_node)"; [ -n "$n" ] && PATH="$W/node/$n/bin:$PATH" npm --version 2>/dev/null; }
v_npm_globals() {
  local m out=() d
  d="$W/npm-global/lib/node_modules"
  [ -d "$d" ] || { echo ""; return; }
  for m in "$d"/* "$d"/@*/*; do
    [ -f "$m/package.json" ] || continue
    out+=("$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["name"]+"@"+p["version"])' "$m/package.json")")
  done
  printf '%s ' ${out[@]+"${out[@]}"} | sed 's/ $//'
}
rust_env() { env RUSTUP_HOME="$W/rustup" CARGO_HOME="$W/cargo" "$@"; }
v_rustup() { [ -x "$W/cargo/bin/rustup" ] && rust_env "$W/cargo/bin/rustup" --version 2>/dev/null | awk 'NR==1{print $2}'; }
v_rustc() { [ -x "$W/cargo/bin/rustc" ] && rust_env "$W/cargo/bin/rustc" --version 2>/dev/null | awk '{print $2}'; }
v_rust_profile() { [ -f "$W/rustup/settings.toml" ] && awk -F'"' '/^profile/{print $2}' "$W/rustup/settings.toml"; }
v_fleet_python() {
  local cfg="$LW/.venv/pyvenv.cfg" home
  [ -f "$cfg" ] || return 0
  home="$(awk -F' = ' '$1=="home"{print $2}' "$cfg")"
  readlink "$LW/.venv/bin/python" 2>/dev/null || echo "$home/python3"
}
v_fleet_python_version() { [ -f "$LW/.venv/pyvenv.cfg" ] && awk -F' = ' '$1=="version_info"||$1=="version"{print $2; exit}' "$LW/.venv/pyvenv.cfg"; }
v_venv_reqs() {
  local sp out=() dist n
  for sp in "$LW"/.venv/lib/python*/site-packages; do
    for dist in "$sp"/*.dist-info; do
      [ -d "$dist" ] || continue
      n="$(basename "$dist" .dist-info)"
      case "$n" in langywrap-*) continue ;; esac
      out+=("$(echo "${n%-*}" | tr '_' '-')==${n##*-}")
    done
  done
  printf '%s\n' ${out[@]+"${out[@]}"} | sort | tr '\n' ' ' | sed 's/ $//'
}
git_head() { git -C "$1" rev-parse HEAD 2>/dev/null; }
v_pi_tag() { git -C "$LW/pi" describe --tags --abbrev=0 2>/dev/null; }
v_pi_version() {
  [ -f "$LW/pi/packages/coding-agent/package.json" ] || return 0
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$LW/pi/packages/coding-agent/package.json"
}
v_rtk() { [ -x "$LW/.exec/rtk" ] && "$LW/.exec/rtk" --version 2>/dev/null | awk '{print $2}'; }
v_project() {
  local p="${PROJECT_NAME:-}"
  [ -n "$p" ] && git_head "$W/$p"
}

generate() {
  cat <<EOF
# versions.lock — generated by bootstrap/lock-versions.sh; do not hand-edit.
# Regenerate from the live install:  bootstrap/lock-versions.sh [--config F]
GENERATED_AT=$(date -u +%FT%TZ)
GENERATED_FROM=<work-root>
UV_VERSION=$(v_uv)
UV_PYTHONS="$(v_uv_pythons)"
UV_TOOLS="$(v_uv_tools)"
NODE_VERSION=$(v_node)
NPM_VERSION=$(v_npm)
NPM_GLOBALS="$(v_npm_globals)"
RUSTUP_VERSION=$(v_rustup)
RUST_TOOLCHAIN=$(v_rustc)
RUST_PROFILE=$(v_rust_profile)
FLEET_PYTHON=$(v_fleet_python)
FLEET_PYTHON_VERSION=$(v_fleet_python_version)
LANGYWRAP_COMMIT=$(git_head "$LW")
LANGYWRAP_VENV_REQS="$(v_venv_reqs)"
PI_COMMIT=$(git_head "$LW/pi")
PI_RELEASE_TAG=$(v_pi_tag)
PI_VERSION=$(v_pi_version)
RTK_COMMIT=$(git_head "$LW/rtk")
RTK_VERSION=$(v_rtk)
GIT_VERSION=$(git --version | awk '{print $3}')
PROJECT_COMMIT=$(v_project)
EOF
}

if [ "$CHECK" = 0 ]; then
  OUT="${OUT:-$BS_LOCK}"
  tmp="$(mktemp "$OUT.XXXXXX")"
  generate > "$tmp" && mv -f "$tmp" "$OUT"
  bs_log "wrote $OUT"
  cat "$OUT"
  exit 0
fi

bs_lock_load
actual="$(mktemp)"; trap 'rm -f "$actual"; bs_cleanup' EXIT
generate > "$actual"
fails=0
printf '%-22s %-40s %-40s %s\n' KEY LOCK ACTUAL RESULT
for k in UV_VERSION UV_PYTHONS UV_TOOLS NODE_VERSION NPM_VERSION NPM_GLOBALS RUSTUP_VERSION \
         RUST_TOOLCHAIN RUST_PROFILE FLEET_PYTHON FLEET_PYTHON_VERSION LANGYWRAP_COMMIT \
         LANGYWRAP_VENV_REQS PI_COMMIT PI_RELEASE_TAG PI_VERSION RTK_COMMIT RTK_VERSION; do
  want="${!k:-}"
  got="$(awk -v k="$k" 'index($0, k "=") == 1 { v = substr($0, length(k) + 2); gsub(/^"|"$/, "", v); print v; exit }' "$actual")"
  if [ "$want" = "$got" ]; then r=MATCH; else r=DIFF; fails=$((fails + 1)); fi
  printf '%-22s %-40.40s %-40.40s %s\n' "$k" "${want:-<none>}" "${got:-<none>}" "$r"
  if [ "$r" = DIFF ] && { [ "${#want}" -gt 40 ] || [ "${#got}" -gt 40 ]; }; then
    printf '    lock:   %s\n    actual: %s\n' "$want" "$got"
  fi
done
echo "== $fails key(s) differ =="
[ "$fails" -eq 0 ]
