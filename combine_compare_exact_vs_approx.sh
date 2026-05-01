#!/bin/bash
#SBATCH --job-name=pg_cmp_combine
#SBATCH --account=def-szepesva
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module load python scipy-stack || true

python compare_exact_vs_approx_pg.py \
  --combine \
  --outdir compare_results \
  --plot
