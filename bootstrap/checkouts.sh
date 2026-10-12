#!/usr/bin/env bash
# =============================================================================
#  checkouts.sh — recreate the project checkout, its lane worktrees, and the
#  shared langywrap checkout (+ venv, pi build, rtk build) from zero
# =============================================================================
#  Usage:
#    checkouts.sh [--prefix DIR] [--config F] [--dry-run]
#                 [--only STEP[,STEP...]] [--skip STEP[,STEP...]]
#                 [--https] [--worktrees FILE]
#    checkouts.sh snapshot [--config F] [--prefix DIR] [--out FILE]
#        Record `git worktree list --porcelain` of the EXISTING checkout (read-
#        only) into FILE (default: $PROJECT_WORKTREES) for a later restore.
#
#  Steps (idempotent; an existing clone is left as it is):
#    project    git clone -b PROJECT_BRANCH PROJECT_ORIGIN $WORK_ROOT/$PROJECT_NAME,
#               remote `upstream` = PROJECT_UPSTREAM, gc.auto=0 (the node-local
#               clones borrow its objects through alternates)
#    worktrees  for each worktree in the snapshot under the lane worktree dir:
#               branch <b> -> `git worktree add DIR <b>` when origin has <b>;
#               detached <sha> -> `--detach DIR <sha>` when the commit exists.
#               Without a snapshot: lane/<id> for each lane id in lanes.json
#               (except steward) whose branch exists on origin. Lanes create
#               any other worktree themselves at runtime.
#    langywrap  git clone -b LANGYWRAP_BRANCH LANGYWRAP_ORIGIN $SH_LANGYWRAP_CHECKOUT,
#               `git submodule update --init LANGYWRAP_SUBMODULES` (pi rtk), the
#               pi upstream release tag (PI_RELEASE_TAG from PI_UPSTREAM)
#    venv       $LW/.venv from FLEET_PYTHON (uv venv), `uv pip install -e $LW`
#               with LANGYWRAP_VENV_REQS as constraints (the fleet uses only the
#               base dependencies: no knowledge-graph extra)
#    pi         node build of the pi submodule (langywrap scripts/build_pi.sh)
#               with HOME=$SH_FLEET_HOME so `langypi` lands in
#               $SH_FLEET_HOME/.local/bin like the live install
#    rtk        cargo build of the rtk submodule (scripts/build_rtk.sh) ->
#               $LW/.exec/rtk; skipped when the committed binary is present
#  --https rewrites github SSH URLs to https (no SSH key needed for PUBLIC
#  repos only). Needs network. Writes only under the prefix.
# =============================================================================
set -euo pipefail
BS_PROG=checkouts
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODE=restore; ONLY=""; SKIP=""; HTTPS=0; WT_FILE=""; OUT=""
bs_parse_common "$@"
set -- ${BS_ARGS[@]+"${BS_ARGS[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    snapshot) MODE=snapshot; shift ;;
    --only) ONLY=",$2,"; shift 2 ;;
    --skip) SKIP=",$2,"; shift 2 ;;
    --https) HTTPS=1; shift ;;
    --worktrees) WT_FILE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --lock) BS_LOCK="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) bs_die "unknown argument $1" ;;
  esac
done
bs_load
trap bs_cleanup EXIT
: "${PROJECT_NAME:?the config must set bootstrap.project_name}"
PROJ="$PREFIX/$PROJECT_NAME"
LW="${SH_LANGYWRAP_CHECKOUT:-$PREFIX/langywrap}"
LANEDIR="${PROJECT_LANE_WORKTREE_DIR:-.lane-worktrees}"

if [ "$MODE" = snapshot ]; then
  OUT="${OUT:-${PROJECT_WORKTREES:?}}"
  [ -d "$PROJ/.git" ] || bs_die "no checkout at $PROJ"
  {
    echo "# git worktree list --porcelain of $PROJ at $(date -u +%FT%TZ) (checkouts.sh snapshot)"
    git -C "$PROJ" worktree list --porcelain
  } > "$OUT.tmp$$" && mv -f "$OUT.tmp$$" "$OUT"
  bs_log "wrote $OUT ($(grep -c '^worktree ' "$OUT") worktrees)"
  exit 0
fi

bs_lock_load
bs_toolchain_env
want() { { [ -z "$ONLY" ] || [[ "$ONLY" == *",$1,"* ]]; } && [[ "$SKIP" != *",$1,"* ]]; }
url() {  # url URL — optionally rewrite git@github.com:x to https
  if [ "$HTTPS" = 1 ]; then echo "$1" | sed -E 's#^git@github.com:#https://github.com/#'; else echo "$1"; fi
}
git_q() { bs_run git "$@"; }

step_project() {
  if [ -d "$PROJ/.git" ]; then bs_log "project: $PROJ exists (left as is)"; return 0; fi
  bs_log "project: clone $(url "$PROJECT_ORIGIN") ($PROJECT_BRANCH) -> $PROJ"
  git_q clone -b "$PROJECT_BRANCH" "$(url "$PROJECT_ORIGIN")" "$PROJ"
  [ "$BS_DRY" = 1 ] && return 0
  git -C "$PROJ" config gc.auto 0
  if [ -n "${PROJECT_UPSTREAM:-}" ]; then
    git -C "$PROJ" remote add upstream "$PROJECT_UPSTREAM" 2>/dev/null || git -C "$PROJ" remote set-url upstream "$PROJECT_UPSTREAM"
  fi
  # keep the canonical (SSH) origin URL unless cloned over https on purpose
  if [ "$HTTPS" = 1 ]; then git -C "$PROJ" remote set-url origin "$(url "$PROJECT_ORIGIN")"; else git -C "$PROJ" remote set-url origin "$PROJECT_ORIGIN"; fi
  return 0
}

# origin_has BRANCH — the clone's origin has refs/remotes/origin/BRANCH
origin_has() { git -C "$PROJ" rev-parse -q --verify "refs/remotes/origin/$1" >/dev/null; }

add_worktree() {  # add_worktree REL BRANCH|"" SHA|""
  local rel="$1" br="$2" sha="$3" dir="$PROJ/$1"
  if [ -e "$dir" ]; then bs_log "worktree: $rel exists"; return 0; fi
  if [ -n "$br" ]; then
    if git -C "$PROJ" rev-parse -q --verify "refs/heads/$br" >/dev/null || origin_has "$br"; then
      bs_log "worktree: $rel on $br"
      git_q -C "$PROJ" worktree add "$dir" "$br"
    else
      bs_warn "worktree: $rel: branch $br is not on origin (unpushed at the snapshot?); skipped"
    fi
  elif [ -n "$sha" ]; then
    if git -C "$PROJ" cat-file -e "$sha^{commit}" 2>/dev/null; then
      bs_log "worktree: $rel detached at ${sha:0:10}"
      git_q -C "$PROJ" worktree add --detach "$dir" "$sha"
    else
      bs_warn "worktree: $rel: commit ${sha:0:10} not on origin; skipped"
    fi
  fi
}

step_worktrees() {
  if [ ! -d "$PROJ/.git" ]; then
    [ "$BS_DRY" = 1 ] && { bs_log "worktrees: (dry-run; no clone yet)"; return 0; }
    bs_die "no clone at $PROJ"
  fi
  local f="${WT_FILE:-${PROJECT_WORKTREES:-}}" path="" br="" sha="" line rel n=0
  if [ -n "$f" ] && [ -f "$f" ]; then
    bs_log "worktrees: from snapshot $f"
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        "worktree "*) path="${line#worktree }"; br=""; sha="" ;;
        "HEAD "*) sha="${line#HEAD }" ;;
        "branch refs/heads/"*) br="${line#branch refs/heads/}" ;;
        "")
          if [ -n "$path" ]; then
            # only worktrees INSIDE the checkout's lane dir; re-rooted under
            # the new prefix by their relative part
            case "$path" in */"$LANEDIR"/*) rel="$LANEDIR/${path##*/"$LANEDIR"/}"; add_worktree "$rel" "$br" "$sha"; n=$((n + 1)) ;; esac
          fi
          path="" ;;
      esac
    done < <(grep -v '^#' "$f"; echo)
    bs_log "worktrees: $n lane worktree(s) in the snapshot"
  else
    local lanes="$PROJ/.langywrap/lanes.json" id
    [ -f "$lanes" ] || { bs_log "worktrees: no snapshot and no lanes.json; none created"; return 0; }
    bs_log "worktrees: no snapshot; lane/<id> branches present on origin"
    for id in $(python3 -c 'import json,sys; print(" ".join(l["id"] for l in json.load(open(sys.argv[1]))["lanes"] if l["id"] != "steward"))' "$lanes"); do
      origin_has "lane/$id" && add_worktree "$LANEDIR/$id" "lane/$id" ""
    done
  fi
}

step_langywrap() {
  if [ -d "$LW/.git" ]; then
    bs_log "langywrap: $LW exists (left as is)"
  else
    bs_log "langywrap: clone $(url "$LANGYWRAP_ORIGIN") ($LANGYWRAP_BRANCH) -> $LW"
    git_q clone -b "$LANGYWRAP_BRANCH" "$(url "$LANGYWRAP_ORIGIN")" "$LW"
  fi
  [ "$BS_DRY" = 1 ] && return 0
  local sm u
  if [ "$HTTPS" = 1 ]; then
    for sm in ${LANGYWRAP_SUBMODULES:-}; do
      u="$(git -C "$LW" config -f .gitmodules --get "submodule.$sm.url")"
      git -C "$LW" config "submodule.$sm.url" "$(url "$u")"
    done
  fi
  # shellcheck disable=SC2086
  [ -z "${LANGYWRAP_SUBMODULES:-}" ] || git -C "$LW" submodule update --init ${LANGYWRAP_SUBMODULES}
  if [ -n "${PI_RELEASE_TAG:-}" ] && [ -d "$LW/pi" ] && ! git -C "$LW/pi" rev-parse -q --verify "refs/tags/$PI_RELEASE_TAG" >/dev/null; then
    bs_log "langywrap: fetch pi release tag $PI_RELEASE_TAG from $PI_UPSTREAM"
    git -C "$LW/pi" fetch "$PI_UPSTREAM" "refs/tags/$PI_RELEASE_TAG:refs/tags/$PI_RELEASE_TAG"
  fi
  [ -z "${LANGYWRAP_COMMIT:-}" ] || [ "$(git -C "$LW" rev-parse HEAD)" = "$LANGYWRAP_COMMIT" ] \
    || bs_log "langywrap: HEAD $(git -C "$LW" rev-parse --short HEAD) != lock ${LANGYWRAP_COMMIT:0:10} (branch tip moved since the lock; fine)"
}

step_venv() {
  local py="${FLEET_PYTHON:-python3}"
  [ -x "$py" ] || bs_die "venv: FLEET_PYTHON $py missing on this host"
  if [ -x "$LW/.venv/bin/python" ] && [ -x "$LW/.venv/bin/langywrap" ]; then bs_log "venv: $LW/.venv present"; return 0; fi
  bs_log "venv: $LW/.venv from $py"
  bs_run "$PREFIX/bin/uv" venv --python "$py" "$LW/.venv"
  local c="$DLDIR/langywrap-constraints.txt"
  # shellcheck disable=SC2086
  if [ "$BS_DRY" != 1 ]; then mkdir -p "$DLDIR"; printf '%s\n' $LANGYWRAP_VENV_REQS > "$c"; fi
  bs_run "$PREFIX/bin/uv" pip install --python "$LW/.venv/bin/python" -c "$c" -e "$LW"
}

step_pi() {
  [ -d "$LW/pi" ] || { bs_log "pi: submodule absent; skipped"; return 0; }
  if [ -x "$LW/pi/packages/coding-agent/dist/cli.js" ] && [ -L "$SH_FLEET_HOME/.local/bin/langypi" ]; then bs_log "pi: built"; return 0; fi
  bs_log "pi: build (npm ci + offline build) with HOME=$SH_FLEET_HOME"
  bs_run mkdir -p "$SH_FLEET_HOME"
  bs_run env HOME="$SH_FLEET_HOME" bash "$LW/scripts/build_pi.sh"
}

step_rtk() {
  [ -f "$LW/rtk/Cargo.toml" ] || { bs_log "rtk: submodule absent; skipped"; return 0; }
  if [ -x "$LW/.exec/rtk" ]; then bs_log "rtk: $LW/.exec/rtk present"; return 0; fi
  command -v cargo >/dev/null || bs_die "rtk: cargo missing (toolchain-install.sh --with-rust yes)"
  bs_log "rtk: cargo build --release"
  bs_run mkdir -p "$SH_FLEET_HOME"
  bs_run env HOME="$SH_FLEET_HOME" bash "$LW/scripts/build_rtk.sh"
}

DLDIR="$PREFIX/tmp/bootstrap-dl"
bs_log "prefix $PREFIX project $PROJ langywrap $LW network: $(bs_net_check)"
for s in project worktrees langywrap venv pi rtk; do
  want "$s" || continue
  "step_$s"
done
bs_log "done"
