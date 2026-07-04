#!/bin/bash
#SBATCH --account=def-szepesva
#SBATCH --job-name=pg_stepsizes_k40
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results_pg_stepsizes_k40

echo "Running in:"
pwd

if [ ! -f plot_pg_stepsizes_k40.py ]; then
    echo "ERROR: plot_pg_stepsizes_k40.py is missing."
    echo "Put plot_pg_stepsizes_k40.py in this directory:"
    pwd
    exit 1
fi

module purge
module load python/3.11 scipy-stack

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python plot_pg_stepsizes_k40.py \
  --outdir results_pg_stepsizes_k40 \
  --k 40 \
  --delta 0.002 \
  --horizon 1000000000 \
  --ntraj-display 40 \
  --ntraj-average 1000 \
  --n-grid 520 \
  --exact-until 20000 \
  --max-chunk 25000000 \
  --max-sum-eta 5.0 \
  --max-rel-step 0.05 \
  --seed 12345

echo "Done. Outputs:"
ls -lh results_pg_stepsizes_k40
