#!/bin/bash
#SBATCH --job-name=pg_plot
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Plot hardest-gap regret histories in three versions:
#   1. original scale, normal x/y
#   2. normal y with log x
#   3. log-log scale

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs multi_schedule_results
module load python scipy-stack || true
export PYTHONUNBUFFERED=1

test -f pg_bandit_core.py
test -f run_pg_bandit_experiments.py

python run_pg_bandit_experiments.py plot-history \
  --outdir multi_schedule_results \
  --regret-scale 1000000
