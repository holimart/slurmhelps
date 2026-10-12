#!/usr/bin/env bash
# =============================================================================
#  toolchain.sh build|stage-in|status|unpack ... — the user-toolchain image.
# =============================================================================
#  Source of truth: slurmhelps image/toolchain.sh. Packs the user toolchain
#  dirs (config site.toolchain_dirs, or workset.toolchain_dirs; e.g. uv
#  python/tools, node, npm-global, cargo, rustup, bin, lib) into
#  workset.toolchain_image. `workset.sh stage-in` unpacks it next to the data
#  image and `hold`/`enter` bind it over the same absolute paths. The system
#  toolchain (MPI, BLAS, compilers on node-local disks) is never packed.
#  Package caches (uv/npm/pip/pycache) go to node-local tmp inside a workset.
#  Same engine and options as workset.sh, with --kind toolchain.
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pre=()
while [ $# -gt 0 ]; do
  case "$1" in --project|--lanes) pre+=("$1" "$2"); shift 2 ;; *) break ;; esac
done
exec bash "$HERE/workset.sh" ${pre[@]+"${pre[@]}"} --kind toolchain "$@"
