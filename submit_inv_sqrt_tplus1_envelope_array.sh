#!/bin/bash
#SBATCH --job-name=inv_sqrt_env
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-101%100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs inv_sqrt_tplus1_results
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ARMS_LIST=(1000 10)
NUM_GAPS=51
ARM_CASE=$((SLURM_ARRAY_TASK_ID / NUM_GAPS))
GAP_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_GAPS))
NUM_ARMS=${ARMS_LIST[$ARM_CASE]}

python run_inv_sqrt_tplus1_plots.py sweep-one   --outdir inv_sqrt_tplus1_results   --num-arms "$NUM_ARMS"   --gap-index "$GAP_INDEX"   --num-gaps 51   --gap-start 0.0   --gap-stop 1.0   --horizon 10000000   --num-horizons 21   --trajectories 10000   --workers "$SLURM_CPUS_PER_TASK"   --chunk-trajectories 2500   --method approx   --max-mean-change 0.08   --max-noise-change 0.35   --max-block-size 10000000   --block-quantile 0.995
