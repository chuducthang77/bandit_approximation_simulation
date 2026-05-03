#!/bin/bash
#SBATCH --job-name=pg_c_plot_history
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs c_multiplier_results
module load python scipy-stack || true

python pg_c_multiplier_sweep.py plot-history \
  --outdir c_multiplier_results \
  --schedule-slugs all \
  --regret-scale 1000000 \
  --show-bands
