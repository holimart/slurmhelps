# shellcheck shell=bash
# =============================================================================
#  bootstrap/_lib.sh — shared helpers for the disaster-recovery scripts.
#  SOURCED, never executed.
# =============================================================================
#  Every bootstrap script accepts:
#    --config FILE        a slurmhelps config (slurmhelps.json, or a lanes.json
#                         with a "slurmhelps" block) holding the "site" and
#                         "bootstrap" sections. Needed when the project checkout
#                         does not exist yet (examples/adgfem.bootstrap.json).
#                         Default: the project's own config ($SLURMHELPS_PROJECT,
#                         else found from the cwd).
#    --prefix DIR         per-user root to (re)create instead of site.work_root;
#                         every site path follows it. A scratch prefix makes the
#                         pipeline a rehearsal that touches nothing live.
#    --dry-run            print what would be done, change nothing
#  Writes only below the prefix.
# =============================================================================

BS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURMHELPS_HOME="${SLURMHELPS_HOME:-$(cd "$BS_DIR/.." && pwd)}"
export SLURMHELPS_HOME

BS_PREFIX=""
BS_CONFIG=""
BS_CFGDIR=""
BS_DRY=0
BS_ARGS=()

bs_log()  { printf '%s [%s] %s\n' "$(date -u +%H:%M:%S)" "${BS_PROG:-bootstrap}" "$*"; }
bs_warn() { bs_log "WARN $*" >&2; }
bs_die()  { bs_log "FATAL $*" >&2; exit 1; }

# bs_run CMD... — run, or print under --dry-run.
bs_run() {
  if [ "$BS_DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# bs_parse_common "$@" — consume the common options; leftovers in BS_ARGS.
bs_parse_common() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) BS_PREFIX="$2"; shift 2 ;;
      --prefix=*) BS_PREFIX="${1#*=}"; shift ;;
      --config) BS_CONFIG="$2"; shift 2 ;;
      --config=*) BS_CONFIG="${1#*=}"; shift ;;
      --dry-run) BS_DRY=1; shift ;;
      *) BS_ARGS+=("$1"); shift ;;
    esac
  done
}

# bs_cleanup — remove the temporary config dir (scripts call it from their EXIT trap).
bs_cleanup() { [ -n "$BS_CFGDIR" ] && rm -rf "$BS_CFGDIR"; return 0; }

# bs_prefix_config SRC_JSON PREFIX OUT_JSON — a copy of a slurmhelps config with
# site.work_root = PREFIX and every derived site path back on its default
# ({work_root}/...), so nothing points at the live tree.
bs_prefix_config() {
  "$(sh_python 2>/dev/null || echo python3)" - "$1" "$2" "$3" <<'PY'
import json, sys
src, prefix, out = sys.argv[1:4]
cfg = {}
if src:
    d = json.load(open(src))
    if isinstance(d.get("slurmhelps"), dict):
        d = d["slurmhelps"]
    cfg = d
if prefix:
    site = cfg.setdefault("site", {})
    for k in ("agents_dir", "langywrap_checkout", "env_sh", "user_bin", "images_dir",
              "fleet_home", "pi_agent_dir", "codex_home", "xdg_cache_home"):
        site.pop(k, None)
    site["work_root"] = prefix
json.dump(cfg, open(out, "w"), indent=1)
PY
}

# bs_load — config (+ prefix override), site variables (lib/common.sh sh_load),
# then the "bootstrap" section as shell variables:
#   PROJECT_NAME PROJECT_ORIGIN PROJECT_UPSTREAM PROJECT_BRANCH
#   PROJECT_LANE_WORKTREE_DIR PROJECT_WORKTREES LANGYWRAP_ORIGIN LANGYWRAP_BRANCH
#   LANGYWRAP_SUBMODULES PI_UPSTREAM PROJECT_ENV_LOCAL PROJECT_BIN_SHIMS BS_LOCK
bs_load() {
  # shellcheck source=../lib/common.sh
  . "$SLURMHELPS_HOME/lib/common.sh"
  local orig_root=""
  if [ -n "$BS_CONFIG" ] || [ -n "$BS_PREFIX" ]; then
    if [ -n "$BS_CONFIG" ]; then
      [ -f "$BS_CONFIG" ] || bs_die "config $BS_CONFIG not found"
      BS_CONFIG="$(cd "$(dirname "$BS_CONFIG")" && pwd)/$(basename "$BS_CONFIG")"
    else
      # the project's own config file, found the way config.py finds it
      orig_root="$("$(sh_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); import config; print(config.project_root())' "$SLURMHELPS_HOME/lib/slurmhelps")"
      if [ -f "$orig_root/slurmhelps.json" ]; then BS_CONFIG="$orig_root/slurmhelps.json"
      elif [ -f "$orig_root/.langywrap/lanes.json" ]; then BS_CONFIG="$orig_root/.langywrap/lanes.json"; fi
    fi
    if [ -n "$BS_PREFIX" ]; then
      case "$BS_PREFIX" in /home/*) bs_die "refusing a prefix under /home ($BS_PREFIX)" ;; esac
      mkdir -p "$BS_PREFIX" 2>/dev/null || [ "$BS_DRY" = 1 ] || bs_die "cannot create $BS_PREFIX"
      BS_PREFIX="$(cd "$BS_PREFIX" 2>/dev/null && pwd -P || echo "$BS_PREFIX")"
    fi
    BS_CFGDIR="$(mktemp -d "${TMPDIR:-/tmp}/bs-cfg.XXXXXX")"
    bs_prefix_config "$BS_CONFIG" "$BS_PREFIX" "$BS_CFGDIR/slurmhelps.json" || bs_die "cannot read $BS_CONFIG"
    export SLURMHELPS_PROJECT="$BS_CFGDIR"
  fi
  sh_load || bs_die "cannot load the slurmhelps config"
  [ -n "${SH_WORK_ROOT:-}" ] || bs_die "no work root: set site.work_root in the config or pass --prefix"
  PREFIX="$SH_WORK_ROOT"
  case "$PREFIX" in /home/*) bs_die "refusing a work root under /home ($PREFIX)" ;; esac
  export PREFIX
  local vars
  vars="$("$(sh_python)" "$SLURMHELPS_HOME/lib/slurmhelps/config.py" get bootstrap | "$(sh_python)" -c '
import json, os, shlex, sys
b = json.load(sys.stdin); home = sys.argv[1]; root = sys.argv[2]
x = lambda v: str(v).replace("{slurmhelps}", home)
name = b.get("project_name") or (os.path.basename(root.rstrip("/")) if root else "")
wt = x(b.get("worktrees_file") or "")
if wt and not os.path.isabs(wt) and root:
    wt = os.path.join(root, wt)
out = {
    "PROJECT_NAME": name, "PROJECT_ORIGIN": b.get("origin", ""), "PROJECT_UPSTREAM": b.get("upstream", ""),
    "PROJECT_BRANCH": b.get("branch", ""), "PROJECT_LANE_WORKTREE_DIR": b.get("lane_worktree_dir") or ".lane-worktrees",
    "PROJECT_WORKTREES": wt, "LANGYWRAP_ORIGIN": b.get("langywrap_origin", ""),
    "LANGYWRAP_BRANCH": b.get("langywrap_branch") or "master",
    "LANGYWRAP_SUBMODULES": " ".join(b.get("langywrap_submodules") or []),
    "PI_UPSTREAM": b.get("pi_upstream", ""), "PROJECT_ENV_LOCAL": "\n".join(b.get("env_local") or []),
    "PROJECT_BIN_SHIMS": " ".join(f"{k}={x(v)}" for k, v in (b.get("bin_shims") or {}).items()),
    "BS_LOCK_CFG": x(b.get("lock_file") or ""),
}
for k, v in out.items():
    print(f"{k}={shlex.quote(v)}")
' "$SLURMHELPS_HOME" "$orig_root")" || bs_die "cannot read the bootstrap section"
  eval "$vars"
  BS_LOCK="${BS_LOCK:-${BS_LOCK_CFG:-$BS_DIR/versions.lock}}"
}

# bs_lock_load — source versions.lock (KEY=VALUE lines).
bs_lock_load() {
  [ -f "$BS_LOCK" ] || bs_die "lock file $BS_LOCK missing (generate it with bootstrap/lock-versions.sh)"
  # shellcheck disable=SC1090
  . "$BS_LOCK"
}

# bs_toolchain_env — the env.sh variables for PREFIX, without sourcing a shim
# (the shim may not exist yet).
bs_toolchain_env() {
  export WORK_ROOT="$PREFIX"
  export UV_CACHE_DIR="$PREFIX/uv/cache" UV_PYTHON_INSTALL_DIR="$PREFIX/uv/python"
  export UV_TOOL_DIR="$PREFIX/uv/tools" UV_TOOL_BIN_DIR="$PREFIX/uv/bin"
  export CARGO_HOME="$PREFIX/cargo" RUSTUP_HOME="$PREFIX/rustup"
  export NPM_CONFIG_PREFIX="$PREFIX/npm-global" NPM_CONFIG_CACHE="$PREFIX/cache/npm"
  export npm_config_cache="$NPM_CONFIG_CACHE"
  local nv="${NODE_VERSION:-${SH_NODE_VERSION:-}}"
  export PATH="$PREFIX/bin:$PREFIX/uv/bin:$PREFIX/npm-global/bin:$PREFIX/cargo/bin${nv:+:$PREFIX/node/$nv/bin}:$PATH"
}

# bs_fetch URL OUT — download (curl, fail on HTTP errors, retries).
bs_fetch() {
  curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

# bs_net_check — "ok" when https to github.com works from this host.
bs_net_check() {
  if curl -fsS --max-time 15 -o /dev/null https://github.com 2>/dev/null; then echo ok; else echo none; fi
}
