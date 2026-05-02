#!/bin/bash
#SBATCH --job-name=pg_validate
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Exact-vs-approx validation at T=100000, one schedule per array task.
# This is intentionally shorter and stricter than the large 1e9 sweep.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs multi_schedule_results/validation_by_schedule
module load python scipy-stack || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCHEDULES=(inv_t inv_t_two_thirds log_over_t inv_sqrt_t sqrt_log_over_t inv_log_t)
SCHEDULE_SLUG="${SCHEDULES[${SLURM_ARRAY_TASK_ID}]}"

BASE_OUTDIR="multi_schedule_results"
VALIDATION_OUTDIR="${BASE_OUTDIR}/validation_by_schedule/${SCHEDULE_SLUG}"
HARDEST_CSV="${BASE_OUTDIR}/hardest_gaps_by_schedule.csv"

# Exact simulation cost is O(T * trajectories), so keep this modest for debugging.
HORIZON=100000
NUM_ARMS=40
TRAJECTORIES=1000
CHUNK_TRAJECTORIES=250
NUM_CHECKPOINTS=80

# Conservative approximation controls for validation.
MAX_MEAN_CHANGE=0.02
MAX_NOISE_CHANGE=0.10
BLOCK_QUANTILE=1.0
MAX_BLOCK_SIZE=1000000
EXACT_ETA_SUM_THRESHOLD=1000000
EXACT_SMALL_BLOCK_THRESHOLD=64

test -f pg_bandit_core.py
test -f run_pg_bandit_experiments.py
test -f "${HARDEST_CSV}"

python run_pg_bandit_experiments.py validate \
  --outdir "${VALIDATION_OUTDIR}" \
  --hardest-csv "${HARDEST_CSV}" \
  --schedule-slugs "${SCHEDULE_SLUG}" \
  --horizon "${HORIZON}" \
  --num-arms "${NUM_ARMS}" \
  --trajectories "${TRAJECTORIES}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories "${CHUNK_TRAJECTORIES}" \
  --num-checkpoints "${NUM_CHECKPOINTS}" \
  --max-mean-change "${MAX_MEAN_CHANGE}" \
  --max-noise-change "${MAX_NOISE_CHANGE}" \
  --max-block-size "${MAX_BLOCK_SIZE}" \
  --block-quantile "${BLOCK_QUANTILE}" \
  --exact-eta-sum-threshold "${EXACT_ETA_SUM_THRESHOLD}" \
  --exact-small-block-threshold "${EXACT_SMALL_BLOCK_THRESHOLD}" \
  --plot
