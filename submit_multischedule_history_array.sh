#!/bin/bash
#SBATCH --job-name=pg_multi_hist
#SBATCH --account=def-szepesva
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs multi_schedule_results

module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python pg_multischedule_hardest_gap.py history \
  --history-schedule-index "${SLURM_ARRAY_TASK_ID}" \
  --horizon 1000000000 \
  --num-arms 40 \
  --trajectories 50000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 2500 \
  --num-checkpoints 250 \
  --max-mean-change 0.20 \
  --max-noise-change 0.80 \
  --block-quantile 0.995 \
  --outdir multi_schedule_results
