#!/bin/bash
#SBATCH --job-name=pi1_4gaps
#SBATCH --account=def-szepesva
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs pi1_four_gaps_results
module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python plot_pi1_fixed_gap_long_horizon.py \
  --outdir pi1_four_gaps_results \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --num-arms 7 \
  --num-paths 40 \
  --num-checkpoints 500 \
  --max-mean-change 0.08 \
  --max-noise-change 0.35 \
  --max-block-size 10000000000000 \
  --block-quantile 0.995 \
  --exact-small-block-threshold 64
