#!/usr/bin/env bash
# =============================================================================
#  toolchain-install.sh — (re)install the user toolchain under a work root
# =============================================================================
#  Usage: toolchain-install.sh [--prefix DIR] [--config F]
#                              [--lock FILE] [--with-rust auto|yes|no]
#                              [--only STEP[,STEP...]] [--dry-run]
#  Steps (in order; each is idempotent and skipped when already at the locked
#  version):
#    dirs      skeleton below the prefix (bin, lib, config, uv/*, node, ...)
#    uv        uv + uvx  -> $PREFIX/bin  (GitHub release tarball, sha256-checked)
#    python    uv python install UV_PYTHONS -> $PREFIX/uv/python
#    uvtools   uv tool install UV_TOOLS     -> $PREFIX/uv/{tools,bin}
#    node      node NODE_VERSION -> $PREFIX/node/<ver>, linked into $PREFIX/bin
#              (nodejs.org tarball, SHASUMS256-checked)
#    npm       npm -g NPM_GLOBALS -> $PREFIX/npm-global (none today)
#    rust      rustup RUSTUP_VERSION + toolchain RUST_TOOLCHAIN (profile minimal)
#              -> $PREFIX/{cargo,rustup}. Used only to (re)build langywrap's rtk
#              submodule (scripts/build_rtk.sh -> .exec/rtk; the binary is also
#              committed in langywrap). --with-rust auto = install when
#              LANGYWRAP_SUBMODULES contains rtk.
#    config    $PREFIX/config/{env.sh,toolchain.sh} shims into slurmhelps
#              env/env.sh + the site toolchain profile, env.local.sh (project
#              extras, only if absent), uv.toml, pip.conf
#    helpers   $PREFIX/bin helper shims (PROJECT_BIN_SHIMS), uv-env.sh, and
#              $PREFIX/lib/libscalapack-openmpi.so (MKL linker script, when the
#              site toolchain has MKLROOT)
#  Writes only under the prefix. Needs network for uv/python/node/rust.
#  Versions come from bootstrap/versions.lock (lock-versions.sh).
# =============================================================================
set -euo pipefail
BS_PROG=toolchain-install
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
WITH_RUST=auto; ONLY=""
bs_parse_common "$@"
set -- ${BS_ARGS[@]+"${BS_ARGS[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --lock) BS_LOCK="$2"; shift 2 ;;
    --with-rust) WITH_RUST="$2"; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) bs_die "unknown argument $1" ;;
  esac
done
bs_load
trap bs_cleanup EXIT
bs_lock_load
bs_toolchain_env
P="$PREFIX"
want() { [ -z "$ONLY" ] || [[ "$ONLY" == *",$1,"* ]]; }
DL="$P/tmp/bootstrap-dl"

step_dirs() {
  bs_log "dirs: skeleton under $P"
  bs_run mkdir -p "$P"/{bin,lib,config,cache,tmp,logs,images,home} "$P"/uv/{cache,python,tools,bin} \
    "$P/pip/cache" "$P"/{node,npm-global,cargo,rustup} "$DL"
  bs_run chmod 700 "$P"
}

step_uv() {
  local have; have="$("$P/bin/uv" --version 2>/dev/null | awk '{print $2}' || true)"
  if [ "$have" = "$UV_VERSION" ]; then bs_log "uv: $have present"; return 0; fi
  local t="uv-x86_64-unknown-linux-gnu" base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  bs_log "uv: installing $UV_VERSION (have ${have:-none})"
  [ "$BS_DRY" = 1 ] && { bs_run bs_fetch "$base/$t.tar.gz" "$DL/$t.tar.gz"; return 0; }
  bs_fetch "$base/$t.tar.gz" "$DL/$t.tar.gz"
  bs_fetch "$base/$t.tar.gz.sha256" "$DL/$t.tar.gz.sha256"
  ( cd "$DL" && sha256sum -c "$t.tar.gz.sha256" >/dev/null ) || bs_die "uv tarball checksum FAILED"
  tar -xzf "$DL/$t.tar.gz" -C "$DL"
  install -m 0755 "$DL/$t/uv" "$DL/$t/uvx" "$P/bin/"
  rm -rf "$DL/$t" "$DL/$t.tar.gz"*
  bs_log "uv: $("$P/bin/uv" --version)"
}

step_python() {
  local v
  for v in $UV_PYTHONS; do
    if ls -d "$P/uv/python/cpython-$v-"* >/dev/null 2>&1; then bs_log "python: $v present"; continue; fi
    bs_log "python: uv python install $v"
    bs_run "$P/bin/uv" python install "$v"
  done
}

step_uvtools() {
  local t name
  for t in $UV_TOOLS; do
    name="${t%%==*}"
    if [ -f "$P/uv/tools/$name/uv-receipt.toml" ] && \
       ls -d "$P/uv/tools/$name"/lib/python*/site-packages/"$(echo "$name" | tr - _)-${t##*==}.dist-info" >/dev/null 2>&1; then
      bs_log "uv tool: $t present"; continue
    fi
    bs_log "uv tool: install $t"
    bs_run "$P/bin/uv" tool install --force "$t"
  done
}

step_node() {
  local ver="$NODE_VERSION" dir="$P/node/$NODE_VERSION" pkg="node-$NODE_VERSION-linux-x64"
  local base="https://nodejs.org/dist/$NODE_VERSION" b
  if [ -x "$dir/bin/node" ]; then
    bs_log "node: $ver present"
  else
    bs_log "node: installing $ver"
    if [ "$BS_DRY" = 1 ]; then bs_run bs_fetch "$base/$pkg.tar.xz" "$DL/$pkg.tar.xz"; else
      bs_fetch "$base/$pkg.tar.xz" "$DL/$pkg.tar.xz"
      bs_fetch "$base/SHASUMS256.txt" "$DL/SHASUMS256.txt"
      ( cd "$DL" && grep " $pkg.tar.xz\$" SHASUMS256.txt | sha256sum -c - >/dev/null ) || bs_die "node checksum FAILED"
      mkdir -p "$dir"
      tar -xJf "$DL/$pkg.tar.xz" -C "$dir" --strip-components=1
      rm -f "$DL/$pkg.tar.xz" "$DL/SHASUMS256.txt"
    fi
  fi
  for b in node npm npx corepack; do bs_run ln -sfn "$dir/bin/$b" "$P/bin/$b"; done
}

step_npm() {
  local g
  [ -n "${NPM_GLOBALS// /}" ] || { bs_log "npm: no global packages in the lock"; return 0; }
  for g in $NPM_GLOBALS; do
    if [ -f "$P/npm-global/lib/node_modules/${g%@*}/package.json" ]; then bs_log "npm: $g present"; continue; fi
    bs_log "npm: install -g $g"
    bs_run npm install -g "$g"
  done
}

step_rust() {
  case "$WITH_RUST" in
    yes) ;; no) bs_log "rust: skipped (--with-rust no)"; return 0 ;;
    auto) case " ${LANGYWRAP_SUBMODULES:-} " in *" rtk "*) ;; *) bs_log "rust: not needed (no rtk submodule)"; return 0 ;; esac ;;
    *) bs_die "--with-rust auto|yes|no" ;;
  esac
  local have
  have="$("$P/cargo/bin/rustc" --version 2>/dev/null | awk '{print $2}' || true)"
  if [ "$have" = "$RUST_TOOLCHAIN" ]; then bs_log "rust: rustc $have present"; return 0; fi
  if [ ! -x "$P/cargo/bin/rustup" ]; then
    local url="https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/x86_64-unknown-linux-gnu/rustup-init"
    bs_log "rust: rustup-init $RUSTUP_VERSION"
    if [ "$BS_DRY" = 1 ]; then bs_run bs_fetch "$url" "$DL/rustup-init"; else
      bs_fetch "$url" "$DL/rustup-init"
      bs_fetch "$url.sha256" "$DL/rustup-init.sha256"
      [ "$(awk '{print $1}' "$DL/rustup-init.sha256")" = "$(sha256sum "$DL/rustup-init" | awk '{print $1}')" ] \
        || bs_die "rustup-init checksum FAILED"
      chmod +x "$DL/rustup-init"
      "$DL/rustup-init" -y --no-modify-path --profile "${RUST_PROFILE:-minimal}" --default-toolchain "$RUST_TOOLCHAIN"
      rm -f "$DL/rustup-init"*
    fi
  else
    bs_run "$P/cargo/bin/rustup" toolchain install --profile "${RUST_PROFILE:-minimal}" "$RUST_TOOLCHAIN"
    bs_run "$P/cargo/bin/rustup" default "$RUST_TOOLCHAIN"
  fi
}

# write_file FILE CONTENT — atomic, only when the content changed
write_file() {
  local f="$1" c="$2"
  if [ "$BS_DRY" = 1 ]; then echo "[dry-run] write $f"; return 0; fi
  if [ -f "$f" ] && [ "$(cat "$f")" = "$c" ]; then return 0; fi
  printf '%s\n' "$c" > "$f.tmp$$" && mv -f "$f.tmp$$" "$f"
  bs_log "wrote $f"
}

step_config() {
  local site_tc="${SH_TOOLCHAIN_PROFILE:-}"
  write_file "$P/config/env.sh" "# Shim written by slurmhelps bootstrap/toolchain-install.sh: the profile lives in
# slurmhelps env/env.sh. Projects source THIS path; keep it.
export WORK_ROOT=\"$P\"
export SH_TOOLCHAIN_PROFILE=\"\${SH_TOOLCHAIN_PROFILE:-$P/config/toolchain.sh}\"
. \"$SLURMHELPS_HOME/env/env.sh\""
  if [ -n "$site_tc" ]; then
    write_file "$P/config/toolchain.sh" "# Shim written by slurmhelps bootstrap: the site toolchain profile.
. \"$site_tc\""
  fi
  if [ ! -f "$P/config/env.local.sh" ] && [ -n "${PROJECT_ENV_LOCAL:-}" ]; then
    write_file "$P/config/env.local.sh" "$PROJECT_ENV_LOCAL"
  fi
  bs_run mkdir -p "$P/config/uv" "$P/config/pip"
  write_file "$P/config/uv/uv.toml" "cache-dir = \"$P/uv/cache\"
link-mode = \"copy\""
  write_file "$P/config/pip/pip.conf" "[global]
cache-dir = $P/pip/cache"
}

step_helpers() {
  local spec name target
  for spec in ${PROJECT_BIN_SHIMS:-}; do
    name="${spec%%=*}"; target="${spec#*=}"
    target="${target//\{slurmhelps\}/$SLURMHELPS_HOME}"
    case "$target" in
      /*) ;;
      *) [ -n "${PROJECT_NAME:-}" ] || { bs_warn "helpers: $name -> $target needs bootstrap.project_name; skipped"; continue; }
         target="$P/$PROJECT_NAME/$target" ;;
    esac
    write_file "$P/bin/$name" "#!/usr/bin/env bash
# shim (slurmhelps bootstrap): the script lives in $target
exec \"$target\" \"\$@\""
    [ "$BS_DRY" = 1 ] || chmod +x "$P/bin/$name"
  done
  write_file "$P/bin/uv-env.sh" "#!/bin/sh
case \":\${PATH}:\" in
    *:\"$P/bin\":*) ;;
    *) export PATH=\"$P/bin:\$PATH\" ;;
esac"
  # -lscalapack-openmpi -> MKL ScaLAPACK (Debian name expected by some builds).
  local mkl
  mkl="$( (WORK_ROOT="$P"; [ -n "${SH_TOOLCHAIN_PROFILE:-}" ] && [ -f "$SH_TOOLCHAIN_PROFILE" ] && . "$SH_TOOLCHAIN_PROFILE" >/dev/null 2>&1; echo "${MKLROOT:-}") )"
  if [ -n "$mkl" ] && [ -d "$mkl/lib/intel64" ]; then
    local k="$mkl/lib/intel64"
    write_file "$P/lib/libscalapack-openmpi.so" "GROUP ( $k/libmkl_scalapack_lp64.so $k/libmkl_blacs_openmpi_lp64.so $k/libmkl_gf_lp64.so $k/libmkl_core.so $k/libmkl_sequential.so )"
  else
    bs_log "helpers: no MKL on this host; libscalapack-openmpi.so not written"
  fi
}

bs_log "prefix $P, lock $BS_LOCK, network: $(bs_net_check)"
for s in dirs uv python uvtools node npm rust config helpers; do
  want "$s" || continue
  "step_$s"
done
bs_log "done"
