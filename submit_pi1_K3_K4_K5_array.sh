#!/bin/bash
#SBATCH --job-name=pi1_K345
#SBATCH --account=def-szepesva
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs pi1_four_gaps_results
module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ARMS_LIST=(3 4 5)
NUM_DELTAS=4

ARM_CASE=$((SLURM_ARRAY_TASK_ID / NUM_DELTAS))
DELTA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_DELTAS))

NUM_ARMS=${ARMS_LIST[$ARM_CASE]}

echo "Running task ${SLURM_ARRAY_TASK_ID}: K=${NUM_ARMS}, delta index=${DELTA_INDEX}"

python plot_pi1_fixed_gap_long_horizon.py run-case \
  --outdir pi1_four_gaps_results \
  --num-arms "${NUM_ARMS}" \
  --delta-index "${DELTA_INDEX}" \
  --num-paths 40 \
  --num-checkpoints 500 \
  --max-mean-change 0.08 \
  --max-noise-change 0.35 \
  --max-block-size 1000000000000000000 \
  --block-quantile 0.995 \
  --exact-small-block-threshold 64
