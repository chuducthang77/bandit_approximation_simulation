#!/bin/bash
#SBATCH --job-name=pg_sweep
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-605%100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Approximate gap sweep for all six stepsize schedules.
# Array range 0-605 corresponds to 6 schedules x 101 gap values.
# If you change NUM_GAPS below, update the #SBATCH --array upper bound to 6*NUM_GAPS - 1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs multi_schedule_results

# Works on most Alliance/Compute Canada clusters. If your cluster uses a
# different Python module, edit this line or activate your own virtualenv.
module load python scipy-stack || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Main experiment parameters.
OUTDIR="multi_schedule_results"
NUM_GAPS=101
HORIZON=1000000000
NUM_ARMS=40
TRAJECTORIES=50000
CHUNK_TRAJECTORIES=2500

# Approximation controls for the large sweep.
MAX_MEAN_CHANGE=0.20
MAX_NOISE_CHANGE=0.80
BLOCK_QUANTILE=0.995
MAX_BLOCK_SIZE=50000000
EXACT_ETA_SUM_THRESHOLD=4096
EXACT_SMALL_BLOCK_THRESHOLD=64

test -f pg_bandit_core.py
test -f run_pg_bandit_experiments.py

python run_pg_bandit_experiments.py sweep \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --outdir "${OUTDIR}" \
  --schedule-slugs all \
  --num-gaps "${NUM_GAPS}" \
  --gap-start 0.0 \
  --gap-stop 1.0 \
  --horizon "${HORIZON}" \
  --num-arms "${NUM_ARMS}" \
  --trajectories "${TRAJECTORIES}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories "${CHUNK_TRAJECTORIES}" \
  --max-mean-change "${MAX_MEAN_CHANGE}" \
  --max-noise-change "${MAX_NOISE_CHANGE}" \
  --max-block-size "${MAX_BLOCK_SIZE}" \
  --block-quantile "${BLOCK_QUANTILE}" \
  --exact-eta-sum-threshold "${EXACT_ETA_SUM_THRESHOLD}" \
  --exact-small-block-threshold "${EXACT_SMALL_BLOCK_THRESHOLD}"
