#!/usr/bin/env bash
# Run a command on a LAN GPU host (a workstation reached over ssh), configured by the calling
# project's slurmhelps.json "lan_gpu" block: host, base (host dir), ssh_key, known_hosts, threads, gpu.
# (whiteboxchallenge: martin@192.168.0.106, GTX 1080 Ti 11 GB, base /mnt/512ssd/whitebox-gpu.)
#
# Everything lives under <base> on the host:
#   .venv/        Python + CUDA torch + numpy/scipy (set up by hand once)
#   repo/         rsync mirror of the project's scripts/ and the data directories named with --sync
#   slurmhelps/   rsync of this checkout's lib/ (on PYTHONPATH, so job code can import hpc_partial)
#   runs/         one directory per job (log, exit code, outputs rsynced back here)
#
# Be gentle with the host: it is somebody's workstation. Thread caps (lan_gpu.threads), no more than ONE
# job at a time (the lock below), GPU memory is shared with whatever the owner runs.
#
#   ../slurmhelps/bin/gpu_run.sh --name JOB [--sync DIR ...] [--pull DIR ...] -- <command run inside repo/>
#     --sync DIR   rsync DIR (or a single FILE; relative to the project root) to the host before the run
#     --pull DIR   rsync DIR back from the host after the run (results)
#   ../slurmhelps/bin/gpu_run.sh --status            GPU/CPU/memory snapshot and running jobs
#   ../slurmhelps/bin/gpu_run.sh --log JOB           tail the job log
#
# The command runs under nohup on the host with OMP/MKL threads = lan_gpu.threads and CUDA_VISIBLE_DEVICES=lan_gpu.gpu;
# this script waits for it (use longrun.py around gpu_run.sh for anything over 30 min).
#
# The whole body is a function called on the last line: bash reads a script incrementally, so an
# edit while a job runs could otherwise execute half-old, half-new lines (an exit-2 failure seen
# 2026-10). GPU_RUN_ROOT overrides the project root and SLURMHELPS_HOME this checkout (the queue
# runner executes a per-job copy of this file from outside bin/). A --sync path missing locally exits 3
# before any ssh.
set -euo pipefail
main() {
# Lanes cannot reach the GPU host (their sandbox has no SSH identity); they must use the queue.
# Fail fast with the right instruction instead of an opaque ssh exit 255.
if [[ -z "${GPU_RUN_ROOT:-}" && -n "${LANGYWRAP_LANE_ID:-}" ]] || [[ "${GPU_RUN_FORCE_QUEUE_HINT:-0}" == 1 ]]; then
  echo "gpu_run.sh: lanes do not run this directly. Submit with:" >&2
  echo "  python ../slurmhelps/bin/gpu_queue.py submit --name <job> --lane <lane> --priority high --sync <paths> --pull <dir> -- <command>" >&2
  echo "and read results with: python ../slurmhelps/bin/gpu_queue.py log --name <job>" >&2
  exit 64
fi
SLURMHELPS_HOME=${SLURMHELPS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -n "${GPU_RUN_ROOT:-}" ]]; then export SLURMHELPS_PROJECT=$GPU_RUN_ROOT; fi
eval "$(python3 "$SLURMHELPS_HOME/lib/slurmhelps/config.py" shell)"
HOST=$SH_GPU_HOST
BASE=$SH_GPU_BASE
[[ -n "$HOST" && -n "$BASE" ]] || { echo "gpu_run.sh: set lan_gpu.host and lan_gpu.base in $SH_PROJECT/slurmhelps.json" >&2; exit 2; }
PY="$BASE/.venv/bin/python"
ROOT=${GPU_RUN_ROOT:-$SH_PROJECT}
T=$SH_GPU_THREADS
# Lane sandboxes do not share the steward's accepted host keys: accept the host's key on first
# contact (BatchMode still forbids any password prompt) and use the same transport for rsync.
# Lane processes may run with a sandboxed HOME, where ssh finds neither the key nor known_hosts:
# name both explicitly (override with GPU_SSH_KEY / GPU_KNOWN_HOSTS).
GPU_SSH_KEY=${GPU_SSH_KEY:-$SH_GPU_SSH_KEY}
GPU_KNOWN_HOSTS=${GPU_KNOWN_HOSTS:-$SH_GPU_KNOWN_HOSTS}
SSHOPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
[[ -n "$GPU_SSH_KEY" ]] && SSHOPTS="$SSHOPTS -o IdentitiesOnly=yes -i $GPU_SSH_KEY"
[[ -n "$GPU_KNOWN_HOSTS" ]] && SSHOPTS="$SSHOPTS -o UserKnownHostsFile=$GPU_KNOWN_HOSTS"
SSH=(ssh $SSHOPTS "$HOST")
RSYNC=(rsync -az -e "ssh $SSHOPTS")

if [[ "${1:-}" == "--status" ]]; then
  "${SSH[@]}" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; uptime; free -g | sed -n 2p; ls $BASE/runs 2>/dev/null | tail -5; pgrep -af '$BASE/.venv' | grep -v pgrep | cut -c1-120"
  exit 0
fi
if [[ "${1:-}" == "--log" ]]; then
  "${SSH[@]}" "tail -n 40 $BASE/runs/$2/log.txt"
  exit 0
fi

name=""; syncs=(); pulls=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name=$2; shift 2 ;;
    --sync) syncs+=("$2"); shift 2 ;;
    --pull) pulls+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n "$name" && $# -gt 0 ]] || { echo "usage: gpu_run.sh --name JOB [--sync DIR]... [--pull DIR]... -- cmd" >&2; exit 2; }

# Fail fast, before any ssh, on a --sync path that does not exist locally (was rsync exit 23).
local missing=0
for d in "${syncs[@]}"; do
  if [[ ! -e "$ROOT/$d" ]]; then
    echo "gpu_run.sh: --sync path missing locally: $ROOT/$d" >&2
    missing=1
  fi
done
[[ $missing -eq 0 ]] || exit 3

"${SSH[@]}" "mkdir -p $BASE/repo/scripts $BASE/runs/$name $BASE/slurmhelps/lib"
"${RSYNC[@]}" --delete "$ROOT/scripts/" "$HOST:$BASE/repo/scripts/"
"${RSYNC[@]}" --delete --exclude __pycache__ "$SLURMHELPS_HOME/lib/" "$HOST:$BASE/slurmhelps/lib/"
# Also via a .pth in the host venv, so commands that set PYTHONPATH inline still find slurmhelps' lib/.
"${SSH[@]}" "sp=\$($PY -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])') && echo $BASE/slurmhelps/lib > \$sp/slurmhelps.pth" || true
for d in "${syncs[@]}"; do
  if [[ -f "$ROOT/$d" ]]; then  # a single file (e.g. a cache report json)
    "${SSH[@]}" "mkdir -p $BASE/repo/$(dirname "$d")"
    "${RSYNC[@]}" "$ROOT/$d" "$HOST:$BASE/repo/$d"
    continue
  fi
  "${SSH[@]}" "mkdir -p $BASE/repo/$d"
  "${RSYNC[@]}" "$ROOT/$d/" "$HOST:$BASE/repo/$d/"
done

# Create the pull dirs on the host so commands can write into them.
for d in "${pulls[@]}"; do "${SSH[@]}" "mkdir -p $BASE/repo/$d"; done

# One job at a time on the host (flock on the base dir), thread caps, GPU 0.
cmd=$(printf ' %q' "$@")
"${SSH[@]}" "cd $BASE/repo && flock $BASE/.job.lock env OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T CUDA_VISIBLE_DEVICES=$SH_GPU_DEVICE PATH=$BASE/.venv/bin:\$PATH PYTHONPATH=$BASE/repo/scripts:$BASE/slurmhelps/lib nohup bash -c \"$cmd\" > $BASE/runs/$name/log.txt 2>&1; echo \$? > $BASE/runs/$name/exit.txt; cat $BASE/runs/$name/exit.txt"
# Bring the remote command's own output back so queue users (who cannot ssh) can read it.
echo "# ---- remote log ($BASE/runs/$name/log.txt, last 200 lines) ----"
"${SSH[@]}" "tail -n 200 $BASE/runs/$name/log.txt" || true
echo "# ---- end remote log ----"
# A missing pull dir must not mask the command's own exit code (rsync exits 23).
for d in "${pulls[@]}"; do
  mkdir -p "$ROOT/$d"
  local prc=0
  "${RSYNC[@]}" "$HOST:$BASE/repo/$d/" "$ROOT/$d/" || prc=$?
  if [[ $prc -ne 0 ]]; then
    if "${SSH[@]}" "test -d $BASE/repo/$d"; then
      echo "# pull of $d failed (rsync exit $prc); the dir exists on the host (see the exit code below)"
    else
      echo "# pull of $d failed (rsync exit $prc): $BASE/repo/$d is missing on the host (see the exit code below)"
    fi
  fi
done
rc=$("${SSH[@]}" "cat $BASE/runs/$name/exit.txt")
echo "gpu job $name exit $rc (log: ../slurmhelps/bin/gpu_run.sh --log $name)"
exit "$rc"
}
main "$@"
