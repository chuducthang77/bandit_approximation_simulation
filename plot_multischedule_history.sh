#!/bin/bash
#SBATCH --job-name=pg_multi_plot
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs multi_schedule_results

module load python scipy-stack || true

python pg_multischedule_hardest_gap.py plot-history \
  --outdir multi_schedule_results \
  --regret-scale 1000000 \
  --baseline-mode match-final
