#!/bin/bash
#SBATCH --job-name=inv_sqrt_env
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-201%100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs inv_sqrt_tplus1_results
module load python scipy-stack || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# K = 1000 is the main large-m experiment.
# K = 3 is the few-arm comparison.
# NUM_GAPS = 101 means Delta = 0.00, 0.01, ..., 1.00.
# Total array tasks = 2 * 101 = 202, indexed 0,...,201.
ARMS_LIST=(1000 3)
NUM_GAPS=101

ARM_CASE=$((SLURM_ARRAY_TASK_ID / NUM_GAPS))
GAP_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_GAPS))

if (( ARM_CASE < 0 || ARM_CASE >= ${#ARMS_LIST[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; ARM_CASE=${ARM_CASE}" >&2
    exit 2
fi

NUM_ARMS=${ARMS_LIST[$ARM_CASE]}

echo "Running eta_t = 1/sqrt(t+1), K=${NUM_ARMS}, gap index ${GAP_INDEX}/${NUM_GAPS}"

python run_inv_sqrt_tplus1_plots.py sweep-one \
  --outdir inv_sqrt_tplus1_results \
  --num-arms "${NUM_ARMS}" \
  --gap-index "${GAP_INDEX}" \
  --num-gaps "${NUM_GAPS}" \
  --gap-start 0.0 \
  --gap-stop 1.0 \
  --horizon 10000000 \
  --num-horizons 21 \
  --trajectories 10000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 2500 \
  --method approx \
  --max-mean-change 0.08 \
  --max-noise-change 0.35 \
  --max-block-size 10000000 \
  --block-quantile 0.995
