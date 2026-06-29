#!/bin/bash
#SBATCH --job-name=inv_sqrt_plot
#SBATCH --account=def-szepesva
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs inv_sqrt_tplus1_results
module load python scipy-stack || true

python run_inv_sqrt_tplus1_plots.py combine-plot   --outdir inv_sqrt_tplus1_results   --num-arms 40   --num-arms-list 40,10   --horizon 10000000   --regret-scale 1000000.0   --c-values 0.5,1,2,4   --make-sample-path   --sample-paths 40   --sample-checkpoints 250   --method approx   --max-mean-change 0.08   --max-noise-change 0.35   --max-block-size 10000000   --block-quantile 0.995

