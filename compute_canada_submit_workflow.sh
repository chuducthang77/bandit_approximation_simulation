#!/bin/bash
# Submit the whole Compute Canada / Alliance workflow from a login node.
# Usage:
#   bash compute_canada_submit_workflow.sh
#
# This submits jobs with dependencies:
#   sweep array -> combine sweep -> history array -> plot history
# and also:
#   combine sweep -> validation array -> combine validation

set -euo pipefail

sweep_job=$(sbatch --parsable compute_canada_sweep_array.sh)
echo "Submitted sweep array: ${sweep_job}"

combine_job=$(sbatch --parsable --dependency=afterok:${sweep_job} compute_canada_combine_sweep.sh)
echo "Submitted combine sweep: ${combine_job}"

history_job=$(sbatch --parsable --dependency=afterok:${combine_job} compute_canada_history_array.sh)
echo "Submitted history array: ${history_job}"

plot_job=$(sbatch --parsable --dependency=afterok:${history_job} compute_canada_plot_history.sh)
echo "Submitted history plot: ${plot_job}"

validate_job=$(sbatch --parsable --dependency=afterok:${combine_job} compute_canada_validate_array.sh)
echo "Submitted validation array: ${validate_job}"

validate_combine_job=$(sbatch --parsable --dependency=afterok:${validate_job} compute_canada_combine_validation.sh)
echo "Submitted validation combine: ${validate_combine_job}"
