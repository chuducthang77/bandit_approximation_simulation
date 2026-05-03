#!/bin/bash
#SBATCH --job-name=pg_c_validate
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs c_multiplier_results
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCHEDULES=(inv_t inv_t_two_thirds log_over_t inv_sqrt_t sqrt_log_over_t inv_log_t)
SCHEDULE_SLUG=${SCHEDULES[$SLURM_ARRAY_TASK_ID]}

# This validates exact-vs-approx at the hardest gaps found by the large sweep.
# To reduce runtime, it uses 1000 trajectories and T=100000.
python pg_c_multiplier_sweep.py validate \
  --from-hardest \
  --schedule-slugs "${SCHEDULE_SLUG}" \
  --horizon 100000 \
  --num-arms 40 \
  --trajectories 1000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 250 \
  --method approx \
  --regret-mode conditional \
  --num-checkpoints 80 \
  --outdir c_multiplier_results \
  --max-mean-change 0.02 \
  --max-noise-change 0.10 \
  --block-quantile 1.0 \
  --exact-small-block-threshold 64 \
  --plot
