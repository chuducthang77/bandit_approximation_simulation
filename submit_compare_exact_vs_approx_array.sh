#!/bin/bash
#SBATCH --job-name=pg_cmp_exact_approx
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs compare_results

module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python compare_exact_vs_approx_pg.py \
  --gap-index "${SLURM_ARRAY_TASK_ID}" \
  --num-gaps 21 \
  --gap-start 0.0 \
  --gap-stop 1.0 \
  --horizon 100000 \
  --num-arms 40 \
  --trajectories 50000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 2500 \
  --outdir compare_results \
  --max-mean-change 0.02 \
  --max-noise-change 0.10 \
  --block-quantile 1.0 \
  --exact-eta-sum-threshold 1000000
