#!/usr/bin/env bash
# Run cluster jobs one after another (one GPU at a time): for each line of a chain file
#   JOB<TAB>MINUTES<TAB>RUNCMD
# wait until none of our <job_prefix>* jobs is queued/running, then submit it with bin/hpc_run.sh --yes
# and wait for it to finish; when the chain is empty it waits for new lines instead of exiting. Blank lines
# and lines starting with # are skipped; a line can be appended to the chain file while the chain runs (it
# is re-read after every job). Configuration: the calling project's slurmhelps.json (see lib/slurmhelps/config.py).
# usage (from the project root):
#   ../slurmhelps/bin/hpc_chain.sh outputs/hpc/chain.tsv   (run in the background; log to <hpc_dir>/chain.log)
#   ../slurmhelps/bin/hpc_chain.sh --dry-run outputs/hpc/chain.tsv   (print each pending line's hpc_run.sh RUN
#                                                                     command; no ssh, no submit, no DONE writes)
# When the chain is empty and <hpc_dir>/bridge.enabled exists, lane gpu_queue jobs run here too
# (bin/hpc_bridge_gpu_queue.py, log <hpc_dir>/bridge.log). Edit this file by write-then-mv:
# a running chain keeps the old inode.
#
# PAIRED lines: RUNCMD = 'GPU_PART &&& CPU_PART' runs both parts CONCURRENTLY inside the same
# single job. GPU_PART must write {output} and runs with OMP_NUM_THREADS=4; CPU_PART must write {output2}
# and runs with OMP_NUM_THREADS=10. The line is translated into ONE RUN command for hpc_run.sh (see
# to_run_cmd). The CPU part runs in its own subshell and its exit code is only collected, never propagated
# to the GPU part. Job exit code: the GPU part's if non-zero, else the CPU part's. Single-command lines are
# passed to hpc_run.sh unchanged.
set -uo pipefail

# to_run_cmd RUNCMD: print the command to hand to hpc_run.sh. Single-command lines are printed unchanged.
# Returns 2 (nothing printed) for a malformed pair: more than one ' &&& ', GPU part without {output}
# or with {output2}, CPU part without {output2} or with {output}.
to_run_cmd() {
  local cmd=$1 gpu cpu
  if [[ "$cmd" != *' &&& '* ]]; then printf '%s' "$cmd"; return 0; fi
  gpu=${cmd%% &&& *}
  cpu=${cmd#* &&& }
  [[ "$cpu" == *' &&& '* ]] && return 2
  [[ "$gpu" == *'{output}'* && "$gpu" != *'{output2}'* ]] || return 2
  [[ "$cpu" == *'{output2}'* && "$cpu" != *'{output}'* ]] || return 2
  printf '%s' "( export OMP_NUM_THREADS=4; $gpu ) & gpu_pid=\$!; "
  printf '%s' "cpu_rc=0; ( export OMP_NUM_THREADS=10; $cpu ) || cpu_rc=\$?; "
  printf '%s' "gpu_rc=0; wait \$gpu_pid || gpu_rc=\$?; "
  printf '%s' 'if [ "$gpu_rc" -ne 0 ]; then exit "$gpu_rc"; fi; exit "$cpu_rc"'
}

main() {
DRY=
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
CHAIN=$1
SLURMHELPS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$CHAIN" in /*) ;; *) CHAIN="$PWD/$CHAIN" ;; esac
eval "$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" shell)"
export SLURMHELPS_PROJECT=$SH_PROJECT
cd "$SH_PROJECT"
H=$SH_HPC_DIR
DONE="$CHAIN.done"
if [ -n "$DRY" ]; then
  grep -v -e '^#' -e '^\s*$' "$CHAIN" | while IFS=$'\t' read -r job mins cmd; do
    grep -qx "$job" "$DONE" 2>/dev/null && continue
    if run_cmd=$(to_run_cmd "$cmd"); then
      printf 'DRY-RUN\t%s\t%s\t%s\n' "$job" "$mins" "$run_cmd"
    else
      printf 'DRY-RUN\t%s\t%s\tMALFORMED-PAIR (not submitted)\n' "$job" "$mins"
    fi
  done
  exit 0
fi
touch "$DONE"
while true; do
  next=$(grep -v -e '^#' -e '^\s*$' "$CHAIN" | while IFS=$'\t' read -r job mins cmd; do
    grep -qx "$job" "$DONE" || { printf '%s\t%s\t%s\n' "$job" "$mins" "$cmd"; break; }
  done)
  if [ -z "$next" ]; then
    # Bridge hook: with <hpc_dir>/bridge.enabled present, an idle chain runs ONE pending lane job from
    # the project's gpu_queue on the cluster (bin/hpc_bridge_gpu_queue.py; exit 0 = ran a job,
    # 10 = nothing to do). Without the file the chain behaves as before.
    if [ -e "$H/bridge.enabled" ]; then
      python3 "$SLURMHELPS_HOME/bin/hpc_bridge_gpu_queue.py" >> "$H/bridge.log" 2>&1
      brc=$?
      if [ "$brc" -eq 0 ]; then echo "$(date +%H:%M) bridge ran a gpu_queue job ($H/bridge.log)"; idle_logged=; continue; fi
      [ "$brc" -ne 10 ] && echo "$(date +%H:%M) bridge exit $brc ($H/bridge.log)"
    fi
    [ -z "${idle_logged:-}" ] && echo "$(date +%H:%M) chain empty; waiting for new lines"; idle_logged=1; sleep 120; continue
  fi; idle_logged=
  IFS=$'\t' read -r job mins cmd <<< "$next"
  if ! run_cmd=$(to_run_cmd "$cmd"); then
    echo "$(date +%H:%M) malformed paired line $job (needs {output} in the GPU part, {output2} in the CPU part, one ' &&& '); skipped"
    echo "$job" >> "$DONE"; continue
  fi
  while [ "$(ssh -o BatchMode=yes "$SH_SSH" "squeue -h -u $SH_USER -o '%j'" | grep -c "^$SH_PREFIX")" -ge "$SH_MAX_JOBS" ]; do sleep 60; done
  # Ship the job's code to the mirror first (2026-10-13: cards built locally failed with FileNotFoundError):
  # scripts/, outputs/hpc/<job>/, and every repo path the command names (+ the directory of each .py, for
  # sibling imports / arm lookup). --update never overwrites newer mirror files; big data files are skipped.
  sync=(); [ -d scripts ] && sync+=(scripts); [ -d "outputs/hpc/$job" ] && sync+=("outputs/hpc/$job")
  for t in $run_cmd; do
    t=${t#*=}
    case "$t" in outputs/*|scripts/*)
      [ -e "$t" ] && sync+=("$t") && case "$t" in *.py) sync+=("$(dirname "$t")") ;; esac ;;
    esac
  done
  rsync -aR --update --max-size=50m --exclude __pycache__ --exclude '*.submit.log' "${sync[@]}" "$SH_SSH:$SH_MIRROR/" \
    || echo "$(date +%H:%M) WARNING code sync for $job failed"
  echo "$(date +%H:%M) submit $job ($mins min)"
  # Parallel up to cluster.max_jobs (2026-10-13): each card runs in its own background subshell,
  # is marked done at submit time (never resubmitted), and logs its own finish line.
  echo "$job" >> "$DONE"
  (
    "$SLURMHELPS_HOME/bin/hpc_run.sh" "$job" "$mins" "$run_cmd" --yes > "$H/$job.submit.log" 2>&1
    rc=$?; st=$(grep -oE "finished: [A-Z_]+" "$H/$job.submit.log" | tail -1 | cut -d' ' -f2)
    echo "$(date +%H:%M) finished $job rc=$rc slurm=${st:-?} ($H/$job.submit.log)"
  ) &
  # let squeue register the new job before the next slot count
  for _ in $(seq 1 18); do
    ssh -o BatchMode=yes "$SH_SSH" "squeue -h -u $SH_USER -o '%j'" | grep -qx "$SH_PREFIX$job" && break
    sleep 10
  done
done
}
# The body is a function called on the last line: bash reads a script incrementally, so an edit
# while the chain runs cannot execute half-old, half-new lines.
main "$@"
