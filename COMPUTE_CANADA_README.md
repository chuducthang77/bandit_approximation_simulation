# Compute Canada / Alliance scripts for the policy-gradient bandit experiment

Files expected in the same directory:

```text
pg_bandit_core.py
run_pg_bandit_experiments.py
compute_canada_sweep_array.sh
compute_canada_combine_sweep.sh
compute_canada_history_array.sh
compute_canada_plot_history.sh
compute_canada_validate_array.sh
compute_canada_combine_validation.sh
compute_canada_submit_workflow.sh
```

Before submitting jobs, edit every line like this:

```bash
#SBATCH --account=YOUR_ACCOUNT_HERE
```

Example:

```bash
#SBATCH --account=def-yourpi
```

Then run either the full dependency workflow:

```bash
mkdir -p logs multi_schedule_results
bash compute_canada_submit_workflow.sh
```

or run step by step:

```bash
sbatch compute_canada_sweep_array.sh
# after the sweep finishes:
sbatch compute_canada_combine_sweep.sh
# after combine finishes:
sbatch compute_canada_history_array.sh
# after histories finish:
sbatch compute_canada_plot_history.sh
```

Validation at T=100000:

```bash
sbatch compute_canada_validate_array.sh
# after validation finishes:
sbatch compute_canada_combine_validation.sh
```

Main outputs:

```text
multi_schedule_results/combined_gap_sweep_all_schedules.csv
multi_schedule_results/hardest_gaps_by_schedule.csv
multi_schedule_results/history/history_*.csv
multi_schedule_results/hardest_gap_regret_histories_linear_scale.png
multi_schedule_results/hardest_gap_regret_histories_logx.png
multi_schedule_results/hardest_gap_regret_histories_loglog.png
multi_schedule_results/validation_combined/validation_comparison_all_schedules.csv
```

The sweep array is hard-coded for 6 schedules x 101 gaps = 606 tasks:

```bash
#SBATCH --array=0-605%100
```

If you change `NUM_GAPS`, update the array upper bound to `6 * NUM_GAPS - 1`.
