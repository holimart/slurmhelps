#!/usr/bin/env bash
# Slurm entry point for the GLM-5.3-Flash REAP50 benchmark.
#
# Submit through submit.sh, which fills in the site's partition/account and
# overrides the resources below. The defaults are intentionally conservative:
# a two-GPU Blackwell node, 32 CPUs, 256 GiB host RAM, 12 hours.
#
#SBATCH --job-name=glm53-reap50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=glm53-reap50-%j.out
set -euo pipefail

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${SLURMHELPS_ROOT:-$(cd "$SELF/../.." && pwd)}
exec bash "$ROOT/llm/glm53-reap50/run-job.sh"
