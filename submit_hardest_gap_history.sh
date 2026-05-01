#!/bin/bash
#SBATCH --job-name=pg_hardest_history
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs hardest_gap_results

module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python pg_hardest_gap_regret_history.py \
  --sweep-csv results/combined_gap_sweep.csv \
  --sweep-dir results \
  --horizon 1000000000 \
  --num-arms 40 \
  --trajectories 50000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 2500 \
  --num-checkpoints 250 \
  --outdir hardest_gap_results \
  --max-mean-change 0.20 \
  --max-noise-change 0.80 \
  --block-quantile 0.995 \
  --regret-scale 1000000 \
  --plot-average-per-round
