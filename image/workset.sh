#!/usr/bin/env bash
# =============================================================================
#  workset.sh — squashfs working-set images: build, stage-in, hold/enter/exec
#  (mount namespace), spill, status, view, unpack, writeback.
# =============================================================================
#  Source of truth: slurmhelps image/workset.sh. The engine is workset.py, run
#  with the SYSTEM python3 (the user toolchain may itself live in an image).
#  Configuration: the project's slurmhelps config ("site" paths, "workset"
#  defaults; docs/config.md) and its .langywrap/lanes.json `fleet.workset`
#  block. See image/README.md and `workset.sh --help`.
#
#  Usage: workset.sh [--project DIR] [--lanes FILE] [--kind data|toolchain] CMD ...
#  Heavy commands (build, stage-in, spill, writeback) belong in a Slurm job,
#  never on a login node; `view` and `status` are fine on the login node.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# --project selects the config too (lib/common.sh reads SLURMHELPS_PROJECT)
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --project) [ -z "${SLURMHELPS_PROJECT:-}" ] && export SLURMHELPS_PROJECT="${args[$((i + 1))]:-}"; break ;;
    --project=*) [ -z "${SLURMHELPS_PROJECT:-}" ] && export SLURMHELPS_PROJECT="${args[$i]#--project=}"; break ;;
  esac
done
# shellcheck source=../lib/common.sh
. "$HERE/../lib/common.sh"
sh_load >/dev/null || exit 1
# mksquashfs/unsquashfs live in /usr/sbin on RHEL-likes; batch PATHs may lack it.
case ":$PATH:" in *":/usr/sbin:"*) ;; *) PATH="$PATH:/usr/sbin:/sbin" ;; esac
export PATH
PY="${WORKSET_PYTHON:-$(sh_python)}"
exec "$PY" -I "$HERE/workset.py" "$@"
