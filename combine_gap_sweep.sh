#!/bin/bash
#SBATCH --job-name=pg_gap_combine
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python scipy-stack || true
python pg_gap_sweep_hpc.py --combine --outdir results --plot
