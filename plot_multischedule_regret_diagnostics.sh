#!/bin/bash
#SBATCH --job-name=plot_pg_diag
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs multi_schedule_results

module load python scipy-stack || true

python plot_multischedule_regret_diagnostics.py \
  --outdir multi_schedule_results \
  --regret-scale 1000000 \
  --baseline-mode both \
  --show-bands \
  --prefix diagnostic
