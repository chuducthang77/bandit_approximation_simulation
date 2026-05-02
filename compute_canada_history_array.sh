#!/bin/bash
#SBATCH --job-name=pg_history
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Regret-history run at the hardest gap for each schedule.
# Array task 0..5 corresponds to the schedule_index values in hardest_gaps_by_schedule.csv.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs multi_schedule_results/history
module load python scipy-stack || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="multi_schedule_results"
HORIZON=1000000000
NUM_ARMS=40
TRAJECTORIES=50000
CHUNK_TRAJECTORIES=2500
NUM_CHECKPOINTS=250

MAX_MEAN_CHANGE=0.20
MAX_NOISE_CHANGE=0.80
BLOCK_QUANTILE=0.995
MAX_BLOCK_SIZE=50000000
EXACT_ETA_SUM_THRESHOLD=4096
EXACT_SMALL_BLOCK_THRESHOLD=64

test -f pg_bandit_core.py
test -f run_pg_bandit_experiments.py
test -f "${OUTDIR}/hardest_gaps_by_schedule.csv"

python run_pg_bandit_experiments.py history \
  --outdir "${OUTDIR}" \
  --history-schedule-index "${SLURM_ARRAY_TASK_ID}" \
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
  --exact-small-block-threshold "${EXACT_SMALL_BLOCK_THRESHOLD}"
