#!/usr/bin/env bash
# =============================================================================
#  setup-caches.sh  —  relocate ALL user caches/models/config to /work
# =============================================================================
#  Source of truth: slurmhelps bootstrap/setup-caches.sh (formerly a copy in a
#  project checkout). Root: the site profile's SH_WORK_ROOT (override with
#  WORK_ROOT). env.sh / toolchain / config files are written by
#  bootstrap/toolchain-install.sh (steps dirs,node,config,helpers), which this
#  script calls; this script adds the $HOME redirects and shell hooks.
#
#  Design:
#    * XDG dirs (~/.cache, ~/.config, ~/.local/share, ~/.local/state) and tool
#      home dirs are SYMLINKED into /work, so even tools that hard-code
#      $HOME/... write to /work.
#    * Non-XDG tools (uv, pip, HF, torch, cargo, go, npm, ...) get explicit
#      environment variables, sourced from  $WORK_ROOT/config/env.sh
#      via ~/.bashrc.d, ~/.config/environment.d and ~/.profile.
#    * Existing data is rsync-moved (originals kept under $WORK_ROOT/backup).
#    * Anything with open file handles is SKIPPED (unless --force), so a live
#      VS Code / OpenCode / Jupyter session is never disturbed. Re-run after
#      reconnecting to finish.
#    * NEW installs go to WORK_ROOT from the start: the pinned Node toolchain
#      (WORK_ROOT/node/<ver>, exposed via WORK_ROOT/bin) and, opt-in through
#      BUILD_LINK_REPOS, the heavy build trees of checkouts that still live in
#      $PROJECTS_DIR (WORK_ROOT/build/<repo>/...). Checkouts that live on
#      WORK_ROOT (the normal case now) need no build links.
#      Each link is also excluded from that repo's git status (local
#      info/exclude only), because a symlink is not matched by `dir/` patterns.
#    * --status also reports dangling symlinks (a broken route) under $HOME
#      and ~/Projects, so a moved or deleted target is caught early.
#
#  Usage:  setup-caches.sh [--force] [--dry-run] [--status]
#  Env:    SKIP_MIGRATE=1  keep existing real directories where they are
#          SLURMHELPS_PROJECT  project whose slurmhelps config holds the
#                          "site" section (default: found from the cwd)
#          BUILD_LINK_REPOS="langywrap ..."  repos under PROJECTS_DIR
#                          (default ~/Projects) whose build trees go to /work
#  Re-run after every npm ci / uv sync / build that (re)creates a build dir,
#  because those tools may replace a symlink with a real directory.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SLURMHELPS_HOME="${SLURMHELPS_HOME:-$(cd "$HERE/.." && pwd)}"
export SLURMHELPS_HOME
if [ -z "${WORK_ROOT:-}" ]; then
  # shellcheck source=../lib/common.sh
  . "$SLURMHELPS_HOME/lib/common.sh"
  sh_load
  WORK_ROOT="${SH_WORK_ROOT:?the slurmhelps config sets site.work_root (or export WORK_ROOT)}"
fi
H="$HOME"
case "$H" in "$WORK_ROOT"/*) echo "HOME is already under WORK_ROOT ($H); nothing to redirect" >&2; exit 0 ;; esac
PROJECTS_DIR="${PROJECTS_DIR:-$H/Projects}"
MODE=run
case "${1:-}" in
  --force)   MODE=force ;;
  --dry-run) MODE=dry ;;
  --status)  MODE=status ;;
  --help|-h) sed -n '2,27p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

say()  { printf '%s\n' "$*"; }
info() { printf '  ..   %s\n' "$*"; }
ok()   { printf '  ok   %s\n' "$*"; }
warn() { printf '  skip %s\n' "$*"; }
act()  { [ "$MODE" = run ] || [ "$MODE" = force ]; }

# ----------------------------------------------------------------------------
# 0. skeleton
# ----------------------------------------------------------------------------
if act; then
  mkdir -p "$WORK_ROOT"/{cache,config,share,state,tmp,logs,backup,work,runtime}
  mkdir -p "$WORK_ROOT"/home
  mkdir -p "$WORK_ROOT"/uv/{cache,python,tools,bin}
  mkdir -p "$WORK_ROOT"/pip/cache
  mkdir -p "$WORK_ROOT"/models/{huggingface,torch,keras,ollama,llama,whisper}
  mkdir -p "$WORK_ROOT"/datasets/huggingface
  mkdir -p "$WORK_ROOT"/{cargo,rustup,go,gradle,julia,conda-pkgs,poetry,precommit}
  mkdir -p "$WORK_ROOT"/{electron,electron-builder,puppeteer,playwright}
  mkdir -p "$WORK_ROOT"/venvs "$WORK_ROOT"/lib
  mkdir -p "$WORK_ROOT"/{node,npm-global,build}
  chmod 700 "$WORK_ROOT" 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
is_live() {  # 0 => something has an open file below $1
  if command -v lsof >/dev/null 2>&1; then
    lsof +D "$1" >/dev/null 2>&1
  else
    return 1
  fi
}

# link_dir SRC ABS_TARGET LABEL
link_dir() {
  local src="$1" dst="$2" label="$3"
  if [ -L "$src" ]; then
    local cur; cur="$(readlink "$src")"
    if [ "$cur" = "$dst" ]; then
      if [ "$MODE" = status ]; then ok "$label -> $dst"; fi
    else
      warn "$label is already a symlink -> $cur (left as-is)"
    fi
    return 0
  fi
  if [ ! -e "$src" ]; then
    if [ "$MODE" = status ]; then warn "$label : absent"; return 0; fi
    if [ "$MODE" = dry ]; then info "would link $label -> $dst"; return 0; fi
    mkdir -p "$dst"; ln -s "$dst" "$src"; ok "$label -> $dst (new)"
    return 0
  fi
  if [ ! -d "$src" ]; then
    warn "$label exists but is not a directory (left as-is)"; return 0
  fi
  if [ "$MODE" = status ]; then info "$label : real dir, not yet linked"; return 0; fi
  if [ "$MODE" = dry ]; then info "would move $label -> $dst"; return 0; fi
  if [ "$MODE" != force ] && is_live "$src"; then
    warn "$label : IN USE by a running process (re-run after closing it)"
    return 0
  fi
  mkdir -p "$dst"
  rsync -a "$src"/ "$dst"/
  mv "$src" "$WORK_ROOT/backup/${label//\//_}.$(date +%s)"
  ln -s "$dst" "$src"
  ok "$label -> $dst (migrated, original in backup/)"
}

# link_file SRC ABS_TARGET LABEL
link_file() {
  local src="$1" dst="$2" label="$3"
  if [ -L "$src" ]; then
    if [ "$(readlink "$src")" != "$dst" ]; then
      warn "$label already symlink -> $(readlink "$src")"
    fi
    return 0
  fi
  if [ ! -e "$src" ]; then
    if [ "$MODE" = status ]; then warn "$label : absent"; fi
    return 0
  fi
  if [ "$MODE" = dry ]; then info "would link file $label -> $dst"; return 0; fi
  mkdir -p "$(dirname "$dst")"
  if [ ! -e "$dst" ]; then cp -a "$src" "$dst"; fi
  mv "$src" "$WORK_ROOT/backup/${label//\//_}.$(date +%s)"
  ln -s "$dst" "$src"; ok "$label -> $dst"
}

# install_node: the pinned Node (versions.lock) and the env/config shims via
# toolchain-install.sh.
install_node() {
  case "$MODE" in
    dry)    info "would run toolchain-install.sh --only dirs,node,config,helpers" ;;
    status) if [ -x "$WORK_ROOT/bin/node" ]; then ok "node $("$WORK_ROOT/bin/node" --version) active via $WORK_ROOT/bin"; else warn "node: not installed"; fi ;;
    *)      bash "$HERE/toolchain-install.sh" --prefix "$WORK_ROOT" --only dirs,node,config,helpers ;;
  esac
}

# link_build REPO REL : heavy build output of a sibling checkout lives on /work.
#   ~/Projects/REPO/REL  ->  WORK_ROOT/build/REPO/REL   (same nested layout)
# The nested layout matters: Node only searches directories whose basename is
# exactly `node_modules`, so a flattened name such as pi_node_modules breaks
# package resolution. Works before the directory exists (pre-link), and
# migrates it if it is already a real directory (same rules as link_dir).
# Skipped if the checkout or the parent of REL is missing.
link_build() {
  local repo="$1" rel="$2"
  local src="$PROJECTS_DIR/$repo/$rel"
  [ -d "$PROJECTS_DIR/$repo" ] || return 0
  [ -d "$(dirname "$src")" ] || return 0
  link_dir "$src" "$WORK_ROOT/build/$repo/$rel" "build:$repo/$rel"
  if act; then exclude_from_git "$src"; fi
}

# A symlink is not matched by a directory pattern such as `node_modules/`, so
# git would list the link as untracked and block builds that need a clean tree
# (langywrap's build_pi.sh does). Add an anchored, no-trailing-slash entry to the
# NEAREST repo's LOCAL exclude file (never committed; works for submodules too).
exclude_from_git() {
  local path="$1" top rel ex
  top="$(git -C "$(dirname "$path")" rev-parse --show-toplevel 2>/dev/null)" || return 0
  rel="${path#"$top"/}"
  ex="$(git -C "$top" rev-parse --git-path info/exclude)"
  case "$ex" in /*) ;; *) ex="$top/$ex" ;; esac
  mkdir -p "$(dirname "$ex")"
  grep -qxF "/$rel" "$ex" 2>/dev/null || echo "/$rel" >> "$ex"
}

# Known build trees per checkout. Add a line here when a new repo needs one.
# Do NOT link a build OUTPUT whose bundler resolves imports from its real path
# (pi/packages/coding-agent/dist): esbuild then looks for node_modules next to
# the /work target, finds none, and the pi build fails. node_modules is fine,
# because it is reached through the repo path.
BUILD_LINKS_langywrap=(
  "pi/node_modules"
  "pi/packages/ai/node_modules"
  "pi/packages/coding-agent/node_modules"
  "pi/.artifacts"
  "pi/.langywrap"      # langypi runtime state (sessions.sqlite3, logs), not source
  "rtk/target"
  ".venv"
)

link_all_builds() {
  local rel repo
  for repo in ${BUILD_LINK_REPOS:-}; do
    case "$repo" in
      langywrap)
        for rel in "${BUILD_LINKS_langywrap[@]}"; do link_build langywrap "$rel"; done
        if act; then fix_pi_workspace_links; fi ;;
      *) warn "no build-link table for $repo" ;;
    esac
  done
  [ -n "${BUILD_LINK_REPOS:-}" ] || info "no BUILD_LINK_REPOS: checkouts live on WORK_ROOT, nothing to link"
}

# npm writes workspace links as RELATIVE symlinks (pi-ai -> ../../packages/ai).
# Once pi/node_modules lives on /work those targets no longer exist, so the
# workspace packages cannot be resolved. Rewrite only the links that point into
# the repo's pi/packages/ as ABSOLUTE repo paths. Lexical (realpath -s), so the
# repo path is kept even though node_modules is a symlink. Idempotent.
fix_pi_workspace_links() {
  local pi="$PROJECTS_DIR/langywrap/pi" nm="$PROJECTS_DIR/langywrap/pi/node_modules"
  local l rel t abs
  [ -d "$nm" ] || return 0
  while IFS= read -r l; do
    t="$(readlink "$l")"
    case "$t" in ../*) ;; *) continue ;; esac
    rel="${l#"$nm"/}"
    abs="$(realpath -sm "$nm/$(dirname "$rel")/$t")"
    case "$abs" in "$pi/packages/"*) ;; *) continue ;; esac
    ln -sfn "$abs" "$l"
  done < <(find -H "$nm" -maxdepth 2 -type l 2>/dev/null)
}

# ----------------------------------------------------------------------------
# 1. environment + toolchain files
# ----------------------------------------------------------------------------
ENV_FILE="$WORK_ROOT/config/env.sh"
if act; then

  # env.sh / toolchain.sh shims, env.local.sh, uv.toml, pip.conf, bin helpers
  # and the MKL linker script: toolchain-install.sh (the files a rebuild writes).
  install_node

  # --- bashrc.d hook (interactive shells) ---
  mkdir -p "$H/.bashrc.d"
  ln -sf "$ENV_FILE" "$H/.bashrc.d/00-work-caches.sh"

  # --- login profile hook ---
  PROF="$H/.bash_profile"
  if ! grep -q 'work-caches' "$PROF" 2>/dev/null; then
    printf '\n# work-caches\n[ -f "%s" ] && . "%s"\n' "$ENV_FILE" "$ENV_FILE" >> "$PROF"
  fi

  # --- systemd user environment (VS Code server / non-login spawns) ---
  mkdir -p "$H/.config/environment.d"
  cat > "$H/.config/environment.d/90-work-caches.conf" <<EOF
WORK_ROOT=$WORK_ROOT
TMPDIR=/tmp
UV_CACHE_DIR=$WORK_ROOT/uv/cache
UV_PYTHON_INSTALL_DIR=$WORK_ROOT/uv/python
UV_TOOL_DIR=$WORK_ROOT/uv/tools
UV_TOOL_BIN_DIR=$WORK_ROOT/uv/bin
PIP_CACHE_DIR=$WORK_ROOT/pip/cache
HF_HOME=$WORK_ROOT/models/huggingface
HF_HUB_CACHE=$WORK_ROOT/models/huggingface/hub
HUGGINGFACE_HUB_CACHE=$WORK_ROOT/models/huggingface/hub
HF_DATASETS_CACHE=$WORK_ROOT/datasets/huggingface
TORCH_HOME=$WORK_ROOT/models/torch
TRITON_CACHE_DIR=$WORK_ROOT/cache/triton
MPLCONFIGDIR=$WORK_ROOT/cache/matplotlib
WANDB_DIR=$WORK_ROOT/cache/wandb
NPM_CONFIG_CACHE=$WORK_ROOT/cache/npm
NPM_CONFIG_PREFIX=$WORK_ROOT/npm-global
CARGO_HOME=$WORK_ROOT/cargo
RUSTUP_HOME=$WORK_ROOT/rustup
GOPATH=$WORK_ROOT/go
GOMODCACHE=$WORK_ROOT/go/pkg/mod
GOCACHE=$WORK_ROOT/go/build-cache
CONDA_PKGS_DIRS=$WORK_ROOT/conda-pkgs
JULIA_DEPOT_PATH=$WORK_ROOT/julia
CLAUDE_CONFIG_DIR=$WORK_ROOT/home/claude
EOF

  # --- conda ---
  if [ -e "$H/.condarc" ] || command -v conda >/dev/null 2>&1; then
    cat > "$H/.condarc" <<EOF
pkgs_dirs:
  - $WORK_ROOT/conda-pkgs
envs_dirs:
  - $WORK_ROOT/work/conda-envs
EOF
  fi
fi

# ----------------------------------------------------------------------------
# 2. symlink migration
# ----------------------------------------------------------------------------
say ""; say "== toolchain =="
if ! act; then install_node; fi

say ""; say "== relocating directories =="
if [ "$MODE" = status ]; then say "   (status mode)"; fi
# SKIP_MIGRATE=1 keeps existing real directories where they are (they are
# still reported by --status); new installs and build links still apply.
if [ "${SKIP_MIGRATE:-0}" = 1 ]; then
  warn "existing directories left in place (SKIP_MIGRATE=1)"
else

link_dir "$H/.cache"          "$WORK_ROOT/cache"   ".cache"
link_dir "$H/.config"         "$WORK_ROOT/config"  ".config"
link_dir "$H/.ipython"        "$WORK_ROOT/home/ipython" ".ipython"
link_dir "$H/.jupyter"        "$WORK_ROOT/home/jupyter" ".jupyter"
link_dir "$H/.npm"            "$WORK_ROOT/home/npm"     ".npm"
link_dir "$H/.claude"         "$WORK_ROOT/home/claude"  ".claude"
link_file "$H/.claude.json"   "$WORK_ROOT/home/claude.json" ".claude.json"
link_dir "$H/.langywrap"      "$WORK_ROOT/home/langywrap" ".langywrap"
link_dir "$H/.cursor-server"  "$WORK_ROOT/home/cursor-server" ".cursor-server"
link_dir "$H/.code-server"    "$WORK_ROOT/home/code-server"   ".code-server"
link_dir "$H/.vscode-server"  "$WORK_ROOT/home/vscode-server" ".vscode-server"
link_dir "$H/.opencode"       "$WORK_ROOT/home/opencode"      ".opencode"

# children of XDG data/state that are individually migratable
for base in share state; do
  d="$H/.local/$base"
  if [ ! -d "$d" ]; then continue; fi
  for child in "$d"/*; do
    if [ ! -e "$child" ]; then continue; fi
    name="$(basename "$child")"
    case "$name" in
      opencode) warn ".local/$base/opencode : IN USE (deferred)"; continue ;;
    esac
    link_dir "$child" "$WORK_ROOT/$base/$name" ".local/$base/$name"
  done
done
fi

say ""; say "== project build dirs on /work =="
link_all_builds

# ----------------------------------------------------------------------------
# 3. summary
# ----------------------------------------------------------------------------
say ""; say "== dangling symlinks under \$HOME and ~/Projects (broken routes) =="
broken="$(find "$H" -maxdepth 6 \( -path "$H/.vscode-server" -o -path "$H/.cursor-server" \
    -o -path "$H/.opencode" \) -prune \
    -o -xtype l -print 2>/dev/null || true)"
if [ -z "$broken" ]; then ok "none"; else printf '%s\n' "$broken" | sed 's/^/  BROKEN /'; fi

if act; then
  say ""; say "== done =="
  say "Home usage:"; du -sh "$H" 2>/dev/null | sed 's/^/   /' || true
  say "Work root usage:"; du -sh "$WORK_ROOT" 2>/dev/null | sed 's/^/   /' || true
  say ""
  say "Activate in the CURRENT shell with:"
  say "   source $ENV_FILE"
  say "Re-run this script after reconnecting to finish any 'IN USE' items."
fi
