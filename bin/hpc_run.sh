#!/usr/bin/env bash
# One-GPU job on a Slurm cluster, configured by the calling project's slurmhelps.json (cluster.*):
#   cheapest idle GPU among cluster.allowed_partitions, never cluster.exclude_node_prefixes nodes (probe
#   filter AND #SBATCH --exclude), never cluster.exclude_gpus, cluster.gres / cpus / mem caps, and at
#   most cluster.max_jobs of our <job_prefix>* jobs queued or running.
# usage: hpc_run.sh JOB MINUTES 'RUNCMD (writes {output})' [--yes|--dry-run]
# Run from the project (or set SLURMHELPS_PROJECT). Data and env live in cluster.mirror.
#
# Before a real submit (--yes) this slurmhelps checkout is rsynced to cluster.slurmhelps_remote and a
# slurmhelps.pth naming its lib/ is installed into the mirror venv, so job code can `import hpc_partial`
# even when its command sets PYTHONPATH=scripts inline. lib/ is also on the exported PYTHONPATH.
set -euo pipefail
J=$1; M=$2; CMD=$3; shift 3
SLURMHELPS_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eval "$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" shell)"
for v in SH_SSH SH_ACCOUNT SH_MIRROR SH_REMOTE_TMP SH_REMOTE_HOME; do
  [ -n "${!v}" ] || { echo "hpc_run.sh: cluster config incomplete ($v empty) in $SH_PROJECT/slurmhelps.json" >&2; exit 2; }
done
# Troja rule (operator): never fuuk* nodes. Refuse a config that forgot it.
if [[ "$SH_SSH" == *troja* && " $SH_EXCLUDE_PREFIXES " != *" fuuk "* ]]; then
  echo "hpc_run.sh: refusing: a Troja config must list fuuk in cluster.exclude_node_prefixes" >&2; exit 2
fi
SSH=$SH_SSH; PFX=$SH_PREFIX; MIRROR=$SH_MIRROR; SHR=$SH_REMOTE_HOME
cd "$SH_PROJECT"
B="$SH_HPC_DIR/$J/batch"
mkdir -p "$B"
[ -e "$B/README" ] || echo "steward hpc job $J" > "$B/README"
if [[ " $* " == *" --yes "* && " $* " != *" --dry-run "* ]]; then
  ours=$(ssh -o BatchMode=yes "$SSH" "squeue -h -u $SH_USER -o '%j'" | grep -c "^$PFX" || true)
  if [ "$ours" -ge "$SH_MAX_JOBS" ]; then
    echo "refusing: $ours of our ${PFX}* job(s) already queued/running (max $SH_MAX_JOBS)" >&2
    exit 3
  fi
  # Ship slurmhelps (lib/hpc_partial.py and the failure finalizer) and make lib/ importable from the mirror venv.
  rsync -a --delete --exclude .git --exclude __pycache__ --exclude tests "$SLURMHELPS_HOME/" "$SSH:$SHR/"
  ssh -o BatchMode=yes "$SSH" "py=$MIRROR/$SH_VENV/bin/python; if [ -x \$py ]; then sp=\$(\$py -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])') && echo $SHR/lib > \$sp/slurmhelps.pth; fi"
fi
# Node-local writes (operator 2026-10-11): the job writes to RAM-backed /tmp on the node and copies
# to the mirror once at exit, instead of streaming small writes to Lustre while it computes.
#   {output}   -> $JOB_TMP/eval.json, copied to the real {output} at exit
#   $TMPDIR    -> $JOB_TMP (numpy/cupy/python temp files)
#   $WB_STAGE  -> $JOB_TMP/stage: anything written under $WB_STAGE/<repo-relative path> is mirrored into
#                 <mirror>/<repo-relative path> at exit (opt-in for big outputs/caches)
# The copy also runs when Slurm stops the job at its time limit (SIGTERM).
INNER=${CMD//\{output\}/\$JOB_TMP/eval.json}
# Paired jobs (hpc_chain.sh): a second output {output2} is the CPU part's result.
# It is written to $JOB_TMP/eval2.json and copied next to eval.json (same directory as {output}) at exit.
INNER=${INNER//\{output2\}/\$JOB_TMP/eval2.json}
# Failure write-up: the job's combined output is tee'd to $JOB_TMP/job.log. On a non-zero exit
# (or SIGTERM) wb_fail annotates $JOB_TMP/eval.json with exit_code / terminated / log_tail (partial JSON kept,
# or a minimal {"status":"failed"} written when the job left none). wb_fail never fails the wrapper.
# The job script runs under `set -euo pipefail`: every status check is written `cmd || rc=$?` / `if`, so a
# failing job still reaches wb_fail and wb_flush (a bare `wait` used to exit the script first).
RUN="export JOB_TMP=/tmp/${PFX}\${SLURM_JOB_ID:-\$\$}; export TMPDIR=\$JOB_TMP WB_STAGE=\$JOB_TMP/stage; mkdir -p \$WB_STAGE; \
WB_REPO=$MIRROR; \
wb_flush(){ mkdir -p \"\$(dirname {output})\"; [ -f \$JOB_TMP/eval2.json ] && cp -f \$JOB_TMP/eval2.json \"\$(dirname {output})/eval2.json\"; [ -f \$JOB_TMP/eval.json ] && cp -f \$JOB_TMP/eval.json {output}; [ -d \$WB_STAGE ] && rsync -a \$WB_STAGE/ $MIRROR/; }; \
wb_fail(){ python3 $SHR/lib/hpc_partial.py finalize-failure --eval \$JOB_TMP/eval.json --log \$JOB_TMP/job.log --exit-code \$1 --terminated \$2 >&2 || true; }; \
trap 'kill -TERM \$cpid 2>/dev/null || true; wait \$cpid 2>/dev/null || true; wb_fail 143 1; wb_flush || true; exit 143' TERM; \
( ( $INNER ) 2>&1 | tee -a \$JOB_TMP/job.log; exit \${PIPESTATUS[0]} ) & cpid=\$!; rc=0; wait \$cpid || rc=\$?; if [ \$rc -ne 0 ]; then wb_fail \$rc 0; fi; wb_flush || true; exit \$rc"
SETUP="cd $MIRROR && export PYTHONPATH=scripts:$SHR/lib OMP_NUM_THREADS=$SH_THREADS MKL_NUM_THREADS=$SH_THREADS"
if [ -n "$SH_ENV" ]; then SETUP="$SETUP $SH_ENV"; fi
args=()
for p in $SH_PARTITIONS; do args+=(--allowed-partition "$p"); done
for p in $SH_EXCLUDE_PREFIXES; do args+=(--exclude-node-prefix "$p"); done
if [ -n "$SH_EXCLUDE_NODES" ]; then args+=(--exclude "$SH_EXCLUDE_NODES"); fi
for g in $SH_EXCLUDE_GPUS; do args+=(--exclude-gpu "$g"); done
for r in $SH_GPU_RANK; do args+=(--gpu-rank "$r"); done
for r in $SH_PARTITION_RANK; do args+=(--partition-rank "$r"); done
if [ -n "${SH_NICE:-}" ]; then args+=(--nice "$SH_NICE"); fi
exec python3 "$SLURMHELPS_HOME/lib/slurmhelps/gpu_batch.py" run \
  --ssh "$SSH" --account "$SH_ACCOUNT" \
  --workdir "$MIRROR" --remote-tmp "$SH_REMOTE_TMP" \
  --local-batch "$B" --remote-name "$PFX$J" \
  --setup-cmd "$SETUP" \
  --run-cmd "$RUN" \
  --gres "$SH_GRES" --cpus "$SH_CPUS" --mem "$SH_MEM" --minutes "$M" --poll-timeout "$(( M * 60 + SH_POLL_SLACK ))" \
  "${args[@]}" \
  --results-name eval.json --results-local "$SH_HPC_DIR/$J/eval.json" "$@"
