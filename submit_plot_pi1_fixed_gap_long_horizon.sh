#!/bin/bash
#SBATCH --job-name=pi1_fixed_gap_long
#SBATCH --account=def-szepesva
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs pi1_fixed_gap_long_horizon_results
module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python plot_pi1_fixed_gap_long_horizon.py \
  --outdir pi1_fixed_gap_long_horizon_results \
  --horizon 100000000000 \
  --delta 0.002 \
  --num-paths 40 \
  --num-checkpoints 400 \
  --arms-list 3,40\
  --max-mean-change 0.08 \
  --max-noise-change 0.35 \
  --max-block-size 1000000000 \
  --block-quantile 0.995 \
  --exact-small-block-threshold 64

