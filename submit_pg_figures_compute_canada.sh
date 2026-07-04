#!/bin/bash
#SBATCH --account=def-szepesva
#SBATCH --job-name=pg_lower_bound_figs
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=pg_lower_bound_%j.out
#SBATCH --error=pg_lower_bound_%j.err

set -euo pipefail

module purge
module load python/3.11 scipy-stack

# Numba is strongly recommended. On Compute Canada this often works:
python -m venv "$SLURM_TMPDIR/pg_venv"
source "$SLURM_TMPDIR/pg_venv/bin/activate"
pip install --upgrade pip
pip install --no-index numba tqdm || pip install numba tqdm

mkdir -p results_pg_lower_bound

python plot_pg_lower_bound.py \
  --outdir results_pg_lower_bound \
  --delta 0.002 \
  --fig1-k 40 \
  --fig2-k 3 \
  --ntraj 40 \
  --n-grid 520 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --seed 12345 \
  --method exact_skip

echo "Done. Outputs are in results_pg_lower_bound/"
ls -lh results_pg_lower_bound
