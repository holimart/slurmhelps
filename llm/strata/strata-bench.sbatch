#!/usr/bin/env bash
# Slurm entry point for the Strata Qwen3.8-Flash-Next GGUF benchmark matrix.
#
# Submit with --nice=100 (always — this fleet must never crowd out the
# reservation jobs), e.g.
#
#   sbatch --nice=100 llm/strata/strata-bench.sbatch
#
#   # second L40 node, coder pack only, full V100-legal context ladder:
#   sbatch --nice=100 --nodelist=gpuffaNN llm/strata/strata-bench.sbatch \
#     --model coder --contexts 1024,4096,16384,65536 --batches 1,2,4,8
#
#   # V100 (ffa-gpulab / volta): pick a node, override gres+mem on the
#   # command line — #SBATCH lines do NOT expand shell variables:
#   sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta03 \
#     --gres=gpu:1 --mem=190G --cpus-per-task=24 \
#     llm/strata/strata-bench.sbatch --model coder
#
# Node shape below is the L40 (gpu-ffa) default: one L40 48 GB, enough host
# RAM for the IQ3_S pack (54.8 GB experts in RAM + 28.8 GB n-gram table
# mmap'd + KV), 12 h headroom minus cleanup. --time fits both partitions'
# MaxTime (12 h); keep the --budget-min inside bench-matrix below it.
#
# V100 rules (docs/NVIDIA_V100.md): CUDA 12 engine (install-engine.sh already
# installed both variants; Control auto-picks cuda12 on sm_70), and use ONLY
# contexts 1024,4096,16384,65536.
#
#SBATCH --job-name=strata-bench
#SBATCH --partition=gpu-ffa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=11:30:00
#SBATCH --output=/work/ffa/holecem/logs/strata-bench-%j.out
set -euo pipefail

# Slurm copies this file to /var/spool/slurm/...; its own directory is not the
# checkout. Pass resources through sbatch overrides, matrix knobs here at the end.
ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
exec bash "$HELPER/bench-matrix.sh" "$@"
