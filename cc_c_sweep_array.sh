#!/bin/bash
#SBATCH --job-name=pg_c_sweep
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-4847%200
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs c_multiplier_results
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCHEDULE_SLUGS="inv_t,inv_t_two_thirds,log_over_t,inv_sqrt_t,sqrt_log_over_t,inv_log_t"
C_VALUES="0.125,0.25,0.5,1,2,4,8,16"
NUM_GAPS=101

python pg_c_multiplier_sweep.py sweep \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --schedule-slugs "${SCHEDULE_SLUGS}" \
  --c-values "${C_VALUES}" \
  --num-gaps "${NUM_GAPS}" \
  --gap-start 0.0 \
  --gap-stop 1.0 \
  --horizon 1000000000 \
  --num-arms 40 \
  --trajectories 50000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 2500 \
  --method approx \
  --regret-mode conditional \
  --outdir c_multiplier_results \
  --max-mean-change 0.20 \
  --max-noise-change 0.80 \
  --block-quantile 0.995 \
  --exact-small-block-threshold 64
