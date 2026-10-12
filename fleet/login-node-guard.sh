#!/usr/bin/env bash
# =============================================================================
#  login-node-guard.sh — refuse to run lane work on a Slurm login node
# =============================================================================
#  The lane fleet must run INSIDE a Slurm allocation (see slurmhelps
#  fleet/README.md), never on the login node. A project registers this guard as
#  a fleet command in .langywrap/lanes.json; start-lanes.sh calls it before any
#  lane is started. Source of truth: slurmhelps fleet/login-node-guard.sh.
#
#  Detection (general, not host-specific):
#    * Slurm is present on the host  -> `sbatch` is on PATH, AND
#    * we are NOT inside a job step  -> `SLURM_JOB_ID` is unset.
#  Together these mean: we are on a Slurm login/submit node. Exit 1 (refuse).
#
#  Override (testing only): ALLOW_LOGIN_NODE=1.
#
#  Exit codes: 0 = we are inside an allocation (or Slurm absent), OK to run;
#              1 = login node with Slurm active, REFUSE.
# =============================================================================
set -u

if [ "${ALLOW_LOGIN_NODE:-0}" = "1" ]; then
  echo "login-node-guard: ALLOW_LOGIN_NODE=1 -> allowing $(hostname -f)"
  exit 0
fi

have_slurm=0
command -v sbatch >/dev/null 2>&1 && have_slurm=1
[ -n "${SLURM_JOB_ID:-}" ] && inside_alloc=1 || inside_alloc=0

if [ "$have_slurm" = 1 ] && [ "$inside_alloc" = 0 ]; then
  cat >&2 <<EOF
login-node-guard: REFUSING to run on the Slurm login node.
  host       : $(hostname -f)
  sbatch     : $(command -v sbatch)
  SLURM_JOB_ID: (unset -> not inside an allocation)

Lane work must run inside a Slurm allocation. Bring one up and run lanes there:
  ly ensure project="$PWD"
  # then drive lanes with: ly exec '...'
See slurmhelps fleet/README.md. (Set ALLOW_LOGIN_NODE=1 only to test this guard.)
EOF
  exit 1
fi

if [ "$have_slurm" = 0 ]; then
  echo "login-node-guard: Slurm not present on $(hostname -f) -> not a submit node, OK"
  exit 0
fi

echo "login-node-guard: inside Slurm allocation $(hostname -f) job=${SLURM_JOB_ID} OK"
exit 0
