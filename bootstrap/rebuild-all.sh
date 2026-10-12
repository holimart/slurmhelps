#!/usr/bin/env bash
# =============================================================================
#  rebuild-all.sh — the whole "lost everything" pipeline, in order
# =============================================================================
#  Usage: rebuild-all.sh [--config F] [--prefix DIR]
#                        [--dry-run] [--https] [--from STEP] [--skip STEP[,..]]
#                        [--lanes LANES_JSON] [--no-check]
#  Steps:
#    1 toolchain   toolchain-install.sh         uv, pythons, node, rust, config
#    2 checkouts   checkouts.sh                 project + lane worktrees,
#                                               langywrap + venv + pi + rtk
#    3 state       state-init.sh                agents/, homes, empty store
#    4 tcimage     image/toolchain.sh build     toolchain squashfs image
#    5 wsimage     image/workset.sh build --from-lustre   working-set image
#    6 check       lock-versions.sh --check     installed versions vs the lock
#  --prefix DIR rebuilds into DIR instead of the site's work root (a rehearsal;
#  DIR may be node-local /tmp inside a Slurm job). Run the whole thing in a
#  Slurm job (never on a login node), e.g.:
#    srun -p <short> -c 8 --mem=32G --time=110 env -u BASH_ENV \
#      slurmhelps/bootstrap/rebuild-all.sh --config slurmhelps/examples/adgfem.bootstrap.json
#  With --prefix, the image steps run against the rehearsal clone with a
#  <clone>/slurmhelps.json whose site.work_root is the prefix (written here,
#  git-excluded), so the image tools never read or write the live tree.
#  Exit status: 0 all ran (skipped steps are reported), 1 a step failed.
#  A step whose script exits 3 (not configured, e.g. no fleet.workset block, or
#  the image tools absent) is reported as SKIPPED; a version DIFF in `check` is
#  reported but not fatal (commits move between the lock and a rebuild).
# =============================================================================
set -uo pipefail
BS_PROG=rebuild-all
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
FROM=""; SKIP=""; HTTPS=""; LANES=""; CHECK=1
bs_parse_common "$@"
set -- ${BS_ARGS[@]+"${BS_ARGS[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --skip) SKIP=",$2,"; shift 2 ;;
    --https) HTTPS=--https; shift ;;
    --lanes) LANES="$2"; shift 2 ;;
    --no-check) CHECK=0; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) bs_die "unknown argument $1" ;;
  esac
done
bs_load
trap bs_cleanup EXIT
[ "${ALLOW_LOGIN_NODE:-0}" = 1 ] || [ -n "${SLURM_JOB_ID:-}" ] || [ "$BS_DRY" = 1 ] \
  || bs_die "run inside a Slurm job (or --dry-run); ALLOW_LOGIN_NODE=1 overrides"

common=()
[ -n "$BS_CONFIG" ] && common+=(--config "$BS_CONFIG")
[ -n "$BS_PREFIX" ] && common+=(--prefix "$PREFIX")
[ "$BS_DRY" = 1 ] && common+=(--dry-run)
proj="$PREFIX/${PROJECT_NAME:?}"
lanes="${LANES:-$proj/.langywrap/lanes.json}"
img_env=(env SLURMHELPS_PROJECT="$proj")

declare -a ORDER=(toolchain checkouts state tcimage wsimage check)
declare -A RES DUR
started=0; [ -z "$FROM" ] && started=1
run_step() {
  local s="$1" t0 rc=0; shift
  if [ "$started" = 0 ]; then
    if [ "$s" = "$FROM" ]; then started=1; else RES[$s]="not run (--from $FROM)"; return 0; fi
  fi
  if [[ "$SKIP" == *",$s,"* ]]; then RES[$s]="skipped (--skip)"; return 0; fi
  bs_log "==== step $s: $*"
  t0="$(date +%s)"
  "$@" || rc=$?
  DUR[$s]=$(( $(date +%s) - t0 ))
  case "$rc" in
    0) RES[$s]=ok ;;
    1) if [ "$s" = check ]; then RES[$s]="DIFF vs versions.lock (table above)"; else RES[$s]="FAILED rc=1"; fi ;;
    3) RES[$s]="SKIPPED (not configured, rc=3)" ;;
    *) RES[$s]="FAILED rc=$rc" ;;
  esac
  bs_log "==== step $s: ${RES[$s]} in ${DUR[$s]}s"
  return 0
}
# prefix_project_config — rehearsal only: <clone>/slurmhelps.json = the clone's
# own config (lanes.json "slurmhelps" block) with site paths under the prefix.
prefix_project_config() {
  [ -n "$BS_PREFIX" ] && [ -d "$proj/.git" ] || return 0
  [ -f "$proj/slurmhelps.json" ] && return 0
  local src=""
  if python3 -c 'import json,sys; sys.exit(0 if isinstance(json.load(open(sys.argv[1])).get("slurmhelps"), dict) else 1)' "$lanes" 2>/dev/null; then
    src="$lanes"
  else
    src="$BS_CFGDIR/slurmhelps.json"
  fi
  bs_prefix_config "$src" "$PREFIX" "$proj/slurmhelps.json"
  grep -qx '/slurmhelps.json' "$proj/.git/info/exclude" 2>/dev/null || echo '/slurmhelps.json' >> "$proj/.git/info/exclude"
  bs_log "rehearsal: wrote $proj/slurmhelps.json (site.work_root=$PREFIX) for the image tools"
}
image_step() {  # image_step SCRIPT ARGS... — a missing script = not configured
  local sc="$SLURMHELPS_HOME/image/$1"; shift
  [ -f "$sc" ] || { bs_warn "$sc not present"; return 3; }
  if [ "$BS_DRY" = 1 ]; then echo "[dry-run] ${img_env[*]} bash $sc $*"; return 0; fi
  prefix_project_config
  "${img_env[@]}" bash "$sc" "$@"
}
ws_step() {
  if [ ! -f "$lanes" ]; then
    [ "$BS_DRY" = 1 ] && { echo "[dry-run] workset build (lanes $lanes)"; return 0; }
    bs_warn "no $lanes"; return 3
  fi
  python3 -c 'import json,sys; sys.exit(0 if "workset" in (json.load(open(sys.argv[1])).get("fleet") or {}) else 3)' "$lanes" \
    || { bs_warn "$lanes has no fleet.workset block"; return 3; }
  image_step workset.sh --project "$proj" --lanes "$lanes" build --from-lustre
}

run_step toolchain bash "$BS_DIR/toolchain-install.sh" "${common[@]}"
# shellcheck disable=SC2086
run_step checkouts bash "$BS_DIR/checkouts.sh" "${common[@]}" $HTTPS
run_step state     bash "$BS_DIR/state-init.sh" "${common[@]}"
run_step tcimage   image_step toolchain.sh --project "$proj" --lanes "$lanes" build
run_step wsimage   ws_step
if [ "$CHECK" = 1 ] && [ "$BS_DRY" != 1 ]; then
  run_step check bash "$BS_DIR/lock-versions.sh" --check "${common[@]}"
else
  RES[check]="not run"
fi

echo
printf '%-10s %-8s %s\n' STEP SECONDS RESULT
fail=0
for s in "${ORDER[@]}"; do
  printf '%-10s %-8s %s\n' "$s" "${DUR[$s]:--}" "${RES[$s]:-not run}"
  case "${RES[$s]:-}" in FAILED*) fail=1 ;; esac
done
[ "$fail" = 0 ]
