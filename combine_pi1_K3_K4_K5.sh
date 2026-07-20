#!/bin/bash
#SBATCH --job-name=pi1_K345_plot
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs pi1_four_gaps_results
module load python scipy-stack || true

python plot_pi1_fixed_gap_long_horizon.py plot \
  --outdir pi1_four_gaps_results \
  --arms-list 3,4,5
