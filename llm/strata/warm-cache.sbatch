#!/usr/bin/env bash
# Slurm entry point for warm-cache.sh (warm vs cold prompt-cache TTFT).
#
# Submit with --nice=100 (always — this fleet must never crowd out the
# reservation jobs), and pin a free node distinct from the knob sweep / any
# other run:
#
#   # L40, IQ3_S, C1 at 1k/4k/16k/64k + C4 at 1k/4k
#   sbatch --nice=100 --nodelist=uk-g6-06 llm/strata/warm-cache.sbatch \
#     --model iq3s --node uk-g6-06 --max-context 131072 \
#     --cells "1024:1,4096:1,16384:1,65536:1,1024:4,4096:4"
#
#   # V100-32G (ffa-gpulab): override partition/gres/mem on the command line
#   # (#SBATCH lines do NOT expand shell variables).  Contexts <= 65536 and the
#   # 65536 server window are the V100 rules (docs/NVIDIA_V100.md).
#   sbatch --nice=100 --partition=ffa-gpulab --nodelist=volta05 \
#     --gres=gpu:1 --mem=190G --cpus-per-task=24 \
#     llm/strata/warm-cache.sbatch \
#     --model iq3s --node volta05 --max-context 65536 \
#     --cells "1024:1,4096:1,16384:1"
#
# Runner knobs are CLI args, not --export values: Slurm splits comma lists
# inside --export (CELLS=1024:1,4096:1 becomes CELLS=1024:1 plus a bogus key).
#
# Node shape below is the L40 (gpu-ffa) default: one L40 48 GB, enough host
# RAM for the IQ3_S pack (54.8 GB experts in RAM + 28.8 GB table mmap'd + KV)
# and 11.5 h headroom.  --budget-min inside warm-cache.sh keeps the job below
# the limit; the EXIT trap stops the engine and removes the node scratch.
#
#SBATCH --job-name=strata-warm
#SBATCH --partition=gpu-ffa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=11:30:00
#SBATCH --output=/work/ffa/holecem/logs/strata-warm-%j.out
set -euo pipefail

# Slurm copies this file to /var/spool/slurm/...; its own directory is not the
# checkout. Pass resources through sbatch overrides, runner knobs at the end.
ROOT=${SLURMHELPS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HELPER=${STRATAHELPER:-$ROOT/llm/strata}
exec bash "$HELPER/warm-cache.sh" "$@"
