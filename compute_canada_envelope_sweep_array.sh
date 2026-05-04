#!/bin/bash
#SBATCH --job-name=pg_env_sweep
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-305%100
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs envelope_results
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_pg_bandit_experiments.py envelope-sweep \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --outdir envelope_results \
  --schedule-slugs all \
  --num-gaps 51 \
  --gap-start 0.0 \
  --gap-stop 1.0 \
  --horizon 10000000 \
  --num-horizons 21 \
  --num-arms 40 \
  --trajectories 10000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-trajectories 0 \
  --max-mean-change 0.2 \
  --max-noise-change 0.8 \
  --block-quantile 0.995 \
  --exact-small-block-threshold 64
