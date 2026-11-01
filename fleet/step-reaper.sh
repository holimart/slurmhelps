#!/usr/bin/env bash
# step-reaper.sh -- cancel hung solver steps of one Slurm allocation.
#
# A solver (Adgfem / mpirun) step that outlives every sanctioned wall limit is a hang
# (2026-11-01: step 38344985.693 spun at 95% CPU for 3 days; the M=1 broadcast_to_inactive_ranks
# deadlock). The sanctioned runners wrap solvers in `timeout` (run-ddm-case.sh 1800 s,
# oracle-common.sh 3600 s), so a step older than SR_MAX_AGE_S (default 7200) is not one of them.
#
# Usage: step-reaper.sh <jobid> [--kill]      (dry run unless --kill)
# Env:   SR_MAX_AGE_S (7200)   SR_NAMES  egrep of step names to police ("Adgfem|mpirun|orted")
# Only steps whose name matches SR_NAMES are touched; batch, extern and `env` steps (ly exec,
# watchers, the endpoint guard) are never cancelled. Exit 0 always; one line per decision.
set -u
job=${1:?usage: step-reaper.sh <jobid> [--kill]}; kill=${2:-}
max=${SR_MAX_AGE_S:-7200}; names=${SR_NAMES:-'Adgfem|mpirun|orted'}
squeue -s -j "$job" -h -o '%i %j %M' 2>/dev/null | while read -r id name elapsed; do
  printf '%s' "$name" | grep -Eq "^($names)$" || continue
  d=0; t=$elapsed; case $t in *-*) d=${t%%-*}; t=${t#*-};; esac
  IFS=: read -r a b c <<<"$t"; if [ -z "${c:-}" ]; then s=$((10#$a*60+10#$b)); else s=$((10#$a*3600+10#$b*60+10#$c)); fi
  s=$((s+d*86400))
  if [ "$s" -gt "$max" ]; then
    if [ "$kill" = "--kill" ]; then scancel "$id" && echo "reaped $id ($name, ${s}s > ${max}s)"; else echo "would reap $id ($name, ${s}s > ${max}s)"; fi
  else echo "ok $id ($name, ${s}s)"; fi
done
