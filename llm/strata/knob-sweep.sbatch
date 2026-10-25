#!/usr/bin/env bash
# Slurm entry point for the Strata knob sweep (llm/strata/knob-sweep.sh).
#
# The sweep varies ONE engine knob per variant at SMALL contexts; it is not the
# context×concurrency matrix (that is strata-bench.sbatch / bench-matrix.sh).
#
# Always submit with --nice=100 so this never crowds out the fleet's
# reservation work, and pin the node explicitly:
#
#   sbatch --nice=100 --nodelist=uk-g6-07 --time=11:00:00 \
#     llm/strata/knob-sweep.sbatch \
#     --model iq3s --node uk-g6-07 --gpus 0 --budget-min 240 \
#     --contexts 1024,4096,16384 --batches 1,4 --variants all
#
#   # the fallback pack on a different idle L40:
#   sbatch --nice=100 --nodelist=uk-g6-06 --time=11:00:00 \
#     llm/strata/knob-sweep.sbatch --model coder --node uk-g6-06 --gpus 0
#
# Matrix knobs are trailing CLI args, never --export values: Slurm splits
# comma lists inside --export (CTXS=1024,4096 becomes CTXS=1024 plus a bogus
# 4096 key).  Pass --node in sync with --nodelist; the runner aborts when the
# job landed elsewhere.
#
# Node shape: one L40 48 GB on gpu-ffa, enough host RAM for the IQ3_S pack
# (54.8 GB experts + 28.8 GB n-gram table mmap'd) and node-local scratch for
# the one-time shard copy (~86 GiB under /tmp).  --time stays inside the
# partition's 11.5 h limit; keep --budget-min well below it.
#
#SBATCH --job-name=strata-knobs
#SBATCH --partition=gpu-ffa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=11:30:00
#SBATCH --output=/work/ffa/holecem/logs/strata-knobs-%j.out
set -euo pipefail

# Slurm copies this file to /var/spool/slurm/...; its own directory is not the
# checkout.  Resources pass through sbatch overrides, knobs here at the end.
ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
exec bash "$HELPER/knob-sweep.sh" "$@"
