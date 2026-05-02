#!/bin/bash
#SBATCH --job-name=pg_val_combine
#SBATCH --account=def-szepesva
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Combine per-schedule validation CSVs into one folder.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs multi_schedule_results/validation_combined
module load python scipy-stack || true
export PYTHONUNBUFFERED=1

python - <<'PY'
from pathlib import Path
import csv

base = Path("multi_schedule_results/validation_by_schedule")
out = Path("multi_schedule_results/validation_combined")
out.mkdir(parents=True, exist_ok=True)

files = {
    "validation_final.csv": "validation_final_all_schedules.csv",
    "validation_history.csv": "validation_history_all_schedules.csv",
    "validation_comparison.csv": "validation_comparison_all_schedules.csv",
}

for input_name, output_name in files.items():
    rows = []
    for path in sorted(base.glob(f"*/validation/{input_name}")):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print(f"No rows found for {input_name}")
        continue
    output_path = out / output_name
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path} ({len(rows)} rows)")
PY
