#!/usr/bin/env bash
# =============================================================================
#  tests/run-cluster-tests.sh [--here] — the cluster-side suites, in ONE short
#  Slurm job (never on a login node): image/test-workset.sh,
#  fleet/test-nodelocal.sh, fleet/test_index_hygiene.py, tests/shell/test-remote.sh.
# =============================================================================
#  Scratch: node-local tmp of the job. --here runs in the current shell (use
#  only inside an allocation). Partition/footprint: the project's config
#  site.partition_short / short_* (SLURMHELPS_PROJECT), else SH_TEST_PARTITION
#  or the defaults below.
# =============================================================================
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${1:-}" != --here ]; then
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    part="${SH_TEST_PARTITION:-}"
    if [ -z "$part" ] && [ -n "${SLURMHELPS_PROJECT:-}" ]; then
      part="$(python3 "$R/lib/slurmhelps/config.py" get site.partition_short 2>/dev/null)"
    fi
    exec srun ${part:+-p "$part"} -c "${SH_TEST_CPUS:-8}" --mem="${SH_TEST_MEM:-24G}" --time="${SH_TEST_TIME:-60}" \
      -J slurmhelps-tests env -u BASH_ENV bash "$0" --here
  fi
fi
D="${TMPDIR_TESTS:-/tmp}/slurmhelps-tests-$(id -u)-$$"
rm -rf "$D"; mkdir -p "$D"
total_fail=0
suite() {  # suite NAME CMD...
  local name="$1"; shift
  "$@" > "$D/$name.log" 2>&1; local rc=$?
  local p f
  p="$(grep -c '^PASS' "$D/$name.log")"; f="$(grep -c '^FAIL' "$D/$name.log")"
  printf '%-14s rc=%s PASS=%s FAIL=%s  (%s)\n' "$name" "$rc" "$p" "$f" "$D/$name.log"
  grep '^FAIL' "$D/$name.log"
  [ "$rc" = 0 ] || total_fail=$((total_fail + 1))
}
suite workset bash "$R/image/test-workset.sh" "$D/workset"
suite nodelocal bash "$R/fleet/test-nodelocal.sh" "$D/nodelocal"
suite index-hygiene python3 "$R/fleet/test_index_hygiene.py" -v
suite remote bash "$R/tests/shell/test-remote.sh" "$D/remote"
echo "suites failed: $total_fail"
[ "$total_fail" = 0 ]
