# c-multiplier stepsize sweep

This experiment tests schedules of the form

```text
eta_t = c * base_eta_t
```

where `base_eta_t` is one of

```text
1/t
1/t^(2/3)
log(t)/t
1/sqrt(t)
sqrt(log(t)/t)
1/log(t)
```

All log schedules use `log(max(t, 2))`.

## Main outputs

After the sweep and combine:

```text
c_multiplier_results/combined_c_gap_sweep.csv
c_multiplier_results/hardest_gap_by_schedule_and_c.csv
c_multiplier_results/c_sweep_regret_per_round_vs_c.png
c_multiplier_results/c_sweep_hardest_gap_vs_c.png
```

After histories:

```text
c_multiplier_results/c_sweep_histories_cumulative_linear.png
c_multiplier_results/c_sweep_histories_cumulative_logx.png
c_multiplier_results/c_sweep_histories_cumulative_loglog.png
c_multiplier_results/c_sweep_histories_regret_per_round.png
```

The key diagnostic is `c_sweep_regret_per_round_vs_c.png`.  If increasing `c` makes `R_T/T` larger and keeps it away from zero, that supports the finite-horizon almost-linear-regret hypothesis.

## Compute Canada / Alliance workflow

Edit every `#SBATCH --account=YOUR_ACCOUNT_HERE` line. Then run:

```bash
bash cc_c_submit_workflow.sh
```

The default sweep uses:

```text
T = 1e9
K = 40
50,000 trajectories
101 gaps in [0, 1]
8 c values: 0.125, 0.25, 0.5, 1, 2, 4, 8, 16
6 base schedules
```

The array size is `6 * 8 * 101 = 4848`, so the sweep script uses:

```text
#SBATCH --array=0-4847%200
```

If you change the number of c values or gaps, update the array range.

## Short exact-vs-approx validation

The validation script compares exact round-by-round updates with the blocked Gaussian approximation at `T=100000` on the hardest gaps found by the sweep:

```bash
sbatch cc_c_validate_array.sh
```

or directly:

```bash
python pg_c_multiplier_sweep.py validate \
  --from-hardest \
  --schedule-slugs inv_sqrt_t \
  --horizon 100000 \
  --trajectories 1000 \
  --workers 8 \
  --plot
```
