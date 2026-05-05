#!/bin/bash
#SBATCH --job-name=pg_env_plot
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs envelope_results
module load python scipy-stack || true

python run_pg_bandit_experiments.py combine-envelope \
  --outdir envelope_results_1000_arms_100_gaps \
  --plot \
  --regret-scale 1000000.0
