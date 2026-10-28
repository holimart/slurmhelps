#!/usr/bin/env bash
# Slurm entry point for chain-cache.sh (checkpoint-chain / prefix growth).
#
# Submit with --nice=100 (always — this fleet must never crowd out the
# reservation jobs), and pin a free node distinct from any other run:
#
#   # L40, IQ3_S, 1k -> 100k -> 200k -> 262144, 512 tokens per turn
#   sbatch --nice=100 --nodelist=uk-g6-02 llm/strata/chain-cache.sbatch \
#     --model iq3s --node uk-g6-02 --max-context 262144 \
#     --stops 1024,100000,200000,262144 --tokens-per-turn 512
#
#   # 4x V100-32G resident (ffa-gpulab / volta05); not yet run.
#   sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta05 \
#     --gres=gpu:4 --mem=340G --cpus-per-task=32 \
#     llm/strata/chain-cache.sbatch --model iq3s --node volta05 \
#     --gpus 0,1,2,3 --max-context 262144 --stops 1024,100000,200000,262144
#
# Runner knobs are CLI args, not --export values: Slurm splits comma lists
# inside --export (STOPS=1024,100000 becomes STOPS=1024 plus a bogus key).
#
# Node shape below is the L40 (gpu-ffa) default: one L40 48 GB, enough host
# RAM for the IQ3_S pack and 11.5 h headroom.  --budget-min inside
# chain-cache.sh keeps the job below the limit; the EXIT trap stops the engine
# and removes the node scratch.
#
#SBATCH --job-name=strata-chain
#SBATCH --partition=gpu-ffa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=11:30:00
#SBATCH --output=/work/ffa/holecem/logs/strata-chain-%j.out
set -euo pipefail

# Slurm copies this file to /var/spool/slurm/...; its own directory is not the
# checkout. Pass resources through sbatch overrides, runner knobs at the end.
ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
exec bash "$HELPER/chain-cache.sh" "$@"
