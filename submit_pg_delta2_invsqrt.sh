#!/bin/bash
#SBATCH --account=def-szepesva
#SBATCH --job-name=pg_d2_invsqrt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs
mkdir -p results_pg_delta2_invsqrt

module purge
module load python/3.11 scipy-stack

VENV_DIR="${SLURM_TMPDIR:-/tmp}/pg_venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# On Compute Canada, this usually works from the local wheelhouse.
# If numba is already available from the module stack, this does nothing.
python - <<'PY' || python -m pip install --no-index numba
import numba
print("numba already available:", numba.__version__)
PY

python plot_pg_delta2_invsqrt.py \
  --outdir results_pg_delta2_invsqrt \
  --delta 0.002 \
  --fig1-k 40 \
  --fig2-k 3 \
  --horizon 1000000000 \
  --k3-horizon-mult 3.0 \
  --ntraj-display 40 \
  --ntraj-average 1000 \
  --n-grid 520 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --exact-until 200000 \
  --max-chunk 25000000 \
  --max-sum-eta 100.0 \
  --seed 12345

echo "Done. Outputs:"
ls -lh results_pg_delta2_invsqrt
