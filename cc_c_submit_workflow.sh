#!/bin/bash
# Submit c-multiplier workflow with SLURM dependencies.
# Edit #SBATCH --account in each script before running this.

set -euo pipefail
mkdir -p logs c_multiplier_results

SWEEP_JOB=$(sbatch --parsable cc_c_sweep_array.sh)
echo "Submitted sweep array: ${SWEEP_JOB}"

COMBINE_JOB=$(sbatch --parsable --dependency=afterok:${SWEEP_JOB} cc_c_combine_plot.sh)
echo "Submitted combine/plot: ${COMBINE_JOB}"

HISTORY_JOB=$(sbatch --parsable --dependency=afterok:${COMBINE_JOB} cc_c_history_array.sh)
echo "Submitted history array: ${HISTORY_JOB}"

PLOT_HISTORY_JOB=$(sbatch --parsable --dependency=afterok:${HISTORY_JOB} cc_c_plot_history.sh)
echo "Submitted history plot: ${PLOT_HISTORY_JOB}"

VALIDATE_JOB=$(sbatch --parsable --dependency=afterok:${COMBINE_JOB} cc_c_validate_array.sh)
echo "Submitted exact-vs-approx validation: ${VALIDATE_JOB}"
