#!/usr/bin/env python3
"""
Experiment runner, plotting, and Compute Canada / SLURM helpers.

The algorithmic code is in pg_bandit_core.py. This file only handles:
  - command-line arguments,
  - multiprocessing and CSV files,
  - hardest-gap selection,
  - horizon-wise hardest-gap envelope selection,
  - plotting the attached regret-history figure in three scales,
  - optional exact-vs-approx validation at T=100k,
  - optional SLURM template generation.

Common workflow:

  # 1) Approximate sweep for all schedules and gaps.
  python run_pg_bandit_experiments.py sweep --run-all --num-gaps 101

  # 2) Combine sweep outputs and choose the hardest gap per schedule.
  python run_pg_bandit_experiments.py combine-sweep

  # 3) Run regret histories at each schedule's hardest gap.
  python run_pg_bandit_experiments.py history --num-checkpoints 250

  # 4) Plot the attached all-schedule regret-history figure in three scales.
  python run_pg_bandit_experiments.py plot-history

  # 5) Validate exact vs approximate updates at T=100k.
  python run_pg_bandit_experiments.py validate --horizon 100000 --trajectories 1000

  # Alternative: hardest possible regret at every intermediate horizon.
  # This computes max_Delta R_t(Delta) separately at each checkpoint t.
  python run_pg_bandit_experiments.py envelope-sweep --run-all
  python run_pg_bandit_experiments.py combine-envelope --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pg_bandit_core import (
    BlockConfig,
    SimulationOutput,
    all_schedules,
    log_spaced_checkpoints,
    schedule_by_slug,
    select_schedules,
    simulate_policy_gradient,
    standard_errors,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------------------------------------------------------
# Small IO and multiprocessing helpers
# ---------------------------------------------------------------------------


def default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def split_trajectories(total: int, workers: int, chunk_trajectories: Optional[int]) -> List[int]:
    if total <= 0:
        raise ValueError("--trajectories must be positive")

    if chunk_trajectories is not None and chunk_trajectories > 0:
        chunks: List[int] = []
        remaining = total
        while remaining > 0:
            n = min(chunk_trajectories, remaining)
            chunks.append(n)
            remaining -= n
        return chunks

    workers = max(1, min(workers, total))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def gap_grid(gap_start: float, gap_stop: float, num_gaps: int) -> np.ndarray:
    if num_gaps < 2:
        raise ValueError("--num-gaps must be at least 2")
    return np.linspace(gap_start, gap_stop, num_gaps, dtype=np.float64)


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: pathlib.Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path} ({len(rows)} rows)", flush=True)


def read_csv(path: pathlib.Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_one_row(path: pathlib.Path) -> dict:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {path}, found {len(rows)}")
    return rows[0]


def block_config_from_args(args: argparse.Namespace) -> BlockConfig:
    return BlockConfig(
        max_mean_change=float(args.max_mean_change),
        max_noise_change=float(args.max_noise_change),
        max_block_size=int(args.max_block_size),
        block_quantile=float(args.block_quantile),
        exact_eta_sum_threshold=int(args.exact_eta_sum_threshold),
        exact_small_block_threshold=int(args.exact_small_block_threshold),
    )


def _simulate_chunk_worker(kwargs: dict) -> SimulationOutput:
    return simulate_policy_gradient(**kwargs)


def aggregate_outputs(outputs: Sequence[SimulationOutput]) -> SimulationOutput:
    if not outputs:
        raise ValueError("No outputs to aggregate")

    first = outputs[0]
    checkpoint_times = first.checkpoint_times
    total_n = sum(o.num_trajectories for o in outputs)

    sum_regret = np.zeros_like(first.sum_regret)
    sumsq_regret = np.zeros_like(first.sumsq_regret)
    sum_pi1 = np.zeros_like(first.sum_pi1)
    sumsq_pi1 = np.zeros_like(first.sumsq_pi1)
    total_blocks = 0

    final_regret_samples = []
    final_pi1_samples = []

    for o in outputs:
        if not np.array_equal(o.checkpoint_times, checkpoint_times):
            raise ValueError("Cannot aggregate outputs with different checkpoints")
        sum_regret += o.sum_regret
        sumsq_regret += o.sumsq_regret
        sum_pi1 += o.sum_pi1
        sumsq_pi1 += o.sumsq_pi1
        total_blocks += o.num_blocks
        if o.final_regret_samples is not None:
            final_regret_samples.append(o.final_regret_samples)
        if o.final_pi1_samples is not None:
            final_pi1_samples.append(o.final_pi1_samples)

    return SimulationOutput(
        schedule_slug=first.schedule_slug,
        method=first.method,
        gap_arm2=first.gap_arm2,
        horizon_steps=first.horizon_steps,
        num_arms=first.num_arms,
        num_trajectories=total_n,
        checkpoint_times=checkpoint_times.copy(),
        sum_regret=sum_regret,
        sumsq_regret=sumsq_regret,
        sum_pi1=sum_pi1,
        sumsq_pi1=sumsq_pi1,
        q10_regret=None,
        q90_regret=None,
        final_regret_samples=np.concatenate(final_regret_samples) if final_regret_samples else None,
        final_pi1_samples=np.concatenate(final_pi1_samples) if final_pi1_samples else None,
        num_blocks=total_blocks,
    )


def simulate_parallel(
    *,
    schedule_slug: str,
    gap_arm2: float,
    method: str,
    horizon_steps: int,
    num_arms: int,
    num_trajectories: int,
    random_seed: int,
    workers: int,
    chunk_trajectories: Optional[int],
    checkpoints: Sequence[int],
    block_config: BlockConfig,
    return_trajectory_samples: bool = False,
) -> SimulationOutput:
    chunks = split_trajectories(num_trajectories, workers, chunk_trajectories)
    workers = max(1, min(workers, len(chunks)))

    method_offset = 0 if method == "exact" else 50_000_000
    jobs = []
    for chunk_id, chunk_n in enumerate(chunks):
        jobs.append({
            "schedule_slug": schedule_slug,
            "gap_arm2": gap_arm2,
            "horizon_steps": horizon_steps,
            "num_arms": num_arms,
            "num_trajectories": int(chunk_n),
            "random_seed": int(random_seed + method_offset + 9_176 * chunk_id),
            "method": method,
            "checkpoints": checkpoints,
            "block_config": block_config,
            "track_quantiles": False,
            "return_trajectory_samples": return_trajectory_samples,
        })

    print(
        f"[simulate] method={method:6s} schedule={schedule_slug:20s} "
        f"gap={gap_arm2:.8g} horizon={horizon_steps} trajectories={num_trajectories} "
        f"chunks={chunks} workers={workers}",
        flush=True,
    )

    t0 = time.time()
    outputs: List[SimulationOutput] = []
    if workers == 1:
        for job in jobs:
            outputs.append(_simulate_chunk_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_simulate_chunk_worker, job) for job in jobs]
            for future in as_completed(futures):
                outputs.append(future.result())

    combined = aggregate_outputs(outputs)
    print(
        f"[done] method={method:6s} schedule={schedule_slug:20s} "
        f"gap={gap_arm2:.8g} final_mean={combined.mean_regret[-1]:.6e} "
        f"blocks_total={combined.num_blocks} seconds={time.time() - t0:.1f}",
        flush=True,
    )
    return combined


# ---------------------------------------------------------------------------
# Sweep and hardest-gap selection
# ---------------------------------------------------------------------------


def sweep_task_indices(args: argparse.Namespace) -> List[Tuple[int, object, int, float]]:
    schedules = select_schedules(args.schedule_slugs)
    gaps = gap_grid(args.gap_start, args.gap_stop, args.num_gaps)

    if args.task_index is not None:
        total_tasks = len(schedules) * len(gaps)
        if args.task_index < 0 or args.task_index >= total_tasks:
            raise IndexError(f"--task-index must be in [0, {total_tasks - 1}]")
        local_schedule_index = args.task_index // len(gaps)
        gap_index = args.task_index % len(gaps)
        schedule = schedules[local_schedule_index]
        global_schedule_index = all_schedules().index(schedule)
        return [(global_schedule_index, schedule, gap_index, float(gaps[gap_index]))]

    if args.schedule_index is not None and args.gap_index is not None:
        if args.schedule_index < 0 or args.schedule_index >= len(schedules):
            raise IndexError(f"--schedule-index must be in [0, {len(schedules) - 1}]")
        if args.gap_index < 0 or args.gap_index >= len(gaps):
            raise IndexError(f"--gap-index must be in [0, {len(gaps) - 1}]")
        schedule = schedules[args.schedule_index]
        global_schedule_index = all_schedules().index(schedule)
        return [(global_schedule_index, schedule, args.gap_index, float(gaps[args.gap_index]))]

    if args.run_all:
        tasks = []
        for schedule in schedules:
            global_schedule_index = all_schedules().index(schedule)
            for gap_index, gap in enumerate(gaps):
                tasks.append((global_schedule_index, schedule, gap_index, float(gap)))
        return tasks

    raise ValueError("Use --task-index, --schedule-index with --gap-index, or --run-all.")


def run_sweep(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    workers = args.workers if args.workers is not None else default_workers()
    block_config = block_config_from_args(args)
    checkpoints = [int(args.horizon)]

    for schedule_index, schedule, gap_index, gap in sweep_task_indices(args):
        sweep_dir = outdir / "sweep" / schedule.slug
        output_path = sweep_dir / f"gap_{gap_index:04d}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists. Use --overwrite to recompute.", flush=True)
            continue

        seed = int(args.seed + 10_000_019 * schedule_index + 1_000_003 * gap_index)
        result = simulate_parallel(
            schedule_slug=schedule.slug,
            gap_arm2=gap,
            method="approx",
            horizon_steps=int(args.horizon),
            num_arms=int(args.num_arms),
            num_trajectories=int(args.trajectories),
            random_seed=seed,
            workers=workers,
            chunk_trajectories=args.chunk_trajectories,
            checkpoints=checkpoints,
            block_config=block_config,
            return_trajectory_samples=False,
        )

        row = {
            "schedule_index": schedule_index,
            "schedule_slug": schedule.slug,
            "schedule_label": schedule.label,
            "gap_index": gap_index,
            "gap_arm2": gap,
            "mean_regret": float(result.mean_regret[-1]),
            "standard_error": float(result.se_regret[-1]),
            "mean_final_pi1": float(result.mean_pi1[-1]),
            "se_final_pi1": float(result.se_pi1[-1]),
            "num_trajectories": result.num_trajectories,
            "horizon_steps": result.horizon_steps,
            "num_arms": result.num_arms,
            "method": "approx",
            "exact_small_block_threshold": block_config.exact_small_block_threshold,
            "max_mean_change": block_config.max_mean_change,
            "max_noise_change": block_config.max_noise_change,
            "block_quantile": block_config.block_quantile,
            "num_blocks_total": result.num_blocks,
        }
        write_csv(output_path, [row])


def combine_sweep(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    paths = sorted((outdir / "sweep").glob("*/gap_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No sweep CSV files found under {outdir / 'sweep'}")

    rows = [read_one_row(p) for p in paths]
    rows.sort(key=lambda r: (int(r["schedule_index"]), int(r["gap_index"])))

    combined_path = outdir / "combined_gap_sweep_all_schedules.csv"
    write_csv(combined_path, rows)

    hardest_rows = []
    by_schedule: Dict[str, List[dict]] = {}
    for row in rows:
        by_schedule.setdefault(row["schedule_slug"], []).append(row)

    for schedule_slug, schedule_rows in by_schedule.items():
        hardest_rows.append(max(schedule_rows, key=lambda r: float(r["mean_regret"])))

    hardest_rows.sort(key=lambda r: int(r["schedule_index"]))
    hardest_path = outdir / "hardest_gaps_by_schedule.csv"
    write_csv(hardest_path, hardest_rows)

    print("\nHardest gaps:", flush=True)
    for row in hardest_rows:
        print(
            f"  {row['schedule_slug']:20s} Delta={float(row['gap_arm2']):.6g} "
            f"mean_regret={float(row['mean_regret']):.6e}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Histories at hardest gaps and the attached plot
# ---------------------------------------------------------------------------


def run_history(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    hardest_path = pathlib.Path(args.hardest_csv) if args.hardest_csv else outdir / "hardest_gaps_by_schedule.csv"
    if not hardest_path.exists():
        raise FileNotFoundError(f"Hardest-gap CSV not found: {hardest_path}. Run combine-sweep first.")

    hardest_rows = read_csv(hardest_path)
    hardest_rows.sort(key=lambda r: int(r["schedule_index"]))

    if args.history_schedule_index is not None:
        hardest_rows = [r for r in hardest_rows if int(r["schedule_index"]) == int(args.history_schedule_index)]
        if not hardest_rows:
            raise ValueError(f"No schedule_index={args.history_schedule_index} found in {hardest_path}")

    if args.schedule_slugs and args.schedule_slugs.strip().lower() not in {"", "all"}:
        wanted = {s.strip() for s in args.schedule_slugs.split(",") if s.strip()}
        hardest_rows = [r for r in hardest_rows if r["schedule_slug"] in wanted]

    workers = args.workers if args.workers is not None else default_workers()
    block_config = block_config_from_args(args)
    checkpoints = log_spaced_checkpoints(
        int(args.horizon),
        int(args.num_checkpoints),
        include_early=bool(args.linear_early_checkpoints),
    )

    history_dir = outdir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    for row in hardest_rows:
        schedule = schedule_by_slug(row["schedule_slug"])
        schedule_index = int(row["schedule_index"])
        gap_index = int(row["gap_index"])
        gap = float(row["gap_arm2"])
        output_path = history_dir / f"history_{schedule.slug}.csv"

        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists. Use --overwrite to recompute.", flush=True)
            continue

        seed = int(args.seed + 50_000_021 * schedule_index + 1_000_003 * gap_index)
        result = simulate_parallel(
            schedule_slug=schedule.slug,
            gap_arm2=gap,
            method="approx",
            horizon_steps=int(args.horizon),
            num_arms=int(args.num_arms),
            num_trajectories=int(args.trajectories),
            random_seed=seed,
            workers=workers,
            chunk_trajectories=args.chunk_trajectories,
            checkpoints=checkpoints,
            block_config=block_config,
            return_trajectory_samples=False,
        )

        rows = []
        for i, t in enumerate(result.checkpoint_times):
            rows.append({
                "schedule_index": schedule_index,
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "hardest_gap_index": gap_index,
                "hardest_gap_arm2": gap,
                "time": int(t),
                "mean_regret": float(result.mean_regret[i]),
                "standard_error": float(result.se_regret[i]),
                "mean_pi1": float(result.mean_pi1[i]),
                "se_pi1": float(result.se_pi1[i]),
                "num_trajectories": result.num_trajectories,
                "horizon_steps": result.horizon_steps,
                "num_arms": result.num_arms,
                "method": "approx",
                "exact_small_block_threshold": block_config.exact_small_block_threshold,
            })
        write_csv(output_path, rows)

    if args.plot:
        plot_history(args)


def load_histories(outdir: pathlib.Path) -> List[Tuple[object, np.ndarray, np.ndarray, np.ndarray, float]]:
    history_dir = outdir / "history"
    paths = sorted(history_dir.glob("history_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No history_*.csv files found under {history_dir}")

    histories = []
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        schedule = schedule_by_slug(rows[0]["schedule_slug"])
        t = np.asarray([int(r["time"]) for r in rows], dtype=np.float64)
        mean = np.asarray([float(r["mean_regret"]) for r in rows], dtype=np.float64)
        se = np.asarray([float(r["standard_error"]) for r in rows], dtype=np.float64)
        gap = float(rows[0]["hardest_gap_arm2"])
        order = np.argsort(t)
        histories.append((schedule, t[order], mean[order], se[order], gap))
    return histories


def make_one_history_plot(
    histories: Sequence[Tuple[object, np.ndarray, np.ndarray, np.ndarray, float]],
    outpath: pathlib.Path,
    regret_scale: float,
    xscale: str,
    yscale: str,
    show_bands: bool,
    include_baselines: bool,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=180)

    max_time = 0.0
    for schedule, t, mean, se, gap in histories:
        max_time = max(max_time, float(np.max(t)))
        label = f"{schedule.label}, hardest $\\Delta={gap:.4g}$"
        y = mean / regret_scale
        ax.plot(t, y, linewidth=2.2, label=label)
        if show_bands:
            ax.fill_between(t, (mean - 2.0 * se) / regret_scale, (mean + 2.0 * se) / regret_scale, alpha=0.08, linewidth=0)

    if include_baselines:
        # Raw baselines in the same y-units as regret. This keeps T above T^{2/3}
        # above sqrt(T), unlike matched-final guide curves.
        t_ref = histories[0][1]
        baselines = [
            (0.5, r"raw $\sqrt{T}$"),
            (2.0 / 3.0, r"raw $T^{2/3}$"),
            (1.0, r"raw $T$"),
        ]
        for alpha, label in baselines:
            ax.plot(t_ref, np.power(t_ref, alpha) / regret_scale, linestyle="--", linewidth=1.8, alpha=0.85, label=label)

    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative regret" if regret_scale == 1.0 else rf"Average cumulative regret / ${regret_scale:.0e}$")
    ax.set_title("Hardest-gap regret histories by stepsize schedule")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)


def plot_history(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    histories = load_histories(outdir)
    if not histories:
        raise RuntimeError("No histories loaded")

    scale = float(args.regret_scale)
    make_one_history_plot(
        histories,
        outdir / "hardest_gap_regret_histories_linear_scale.png",
        scale,
        xscale="linear",
        yscale="linear",
        show_bands=args.show_bands,
        include_baselines=not args.no_baselines,
    )
    make_one_history_plot(
        histories,
        outdir / "hardest_gap_regret_histories_logx.png",
        scale,
        xscale="log",
        yscale="linear",
        show_bands=args.show_bands,
        include_baselines=not args.no_baselines,
    )
    make_one_history_plot(
        histories,
        outdir / "hardest_gap_regret_histories_loglog.png",
        scale,
        xscale="log",
        yscale="log",
        show_bands=args.show_bands,
        include_baselines=not args.no_baselines,
    )



# ---------------------------------------------------------------------------
# Horizon-wise hardest-gap envelope
# ---------------------------------------------------------------------------


def horizon_grid_from_args(args: argparse.Namespace) -> np.ndarray:
    """Return the horizons at which to compute max_Delta R_t(Delta).

    This is the grid for the envelope experiment.  The default is intentionally
    much smaller than 1e9: a log-spaced grid up to 1e7 is usually enough to see
    the finite-horizon scaling without making the sweep too expensive.
    """
    explicit = parse_int_list(getattr(args, "horizon_values", None))
    if explicit is not None:
        pts = np.unique(np.asarray(explicit, dtype=np.int64))
        pts = pts[pts >= 1]
        if pts.size == 0:
            raise ValueError("--horizon-values must contain at least one positive integer")
        return pts

    return log_spaced_checkpoints(
        int(args.horizon),
        int(args.num_horizons),
        include_early=False,
    )


def run_envelope_sweep(args: argparse.Namespace) -> None:
    """Run one or more schedule-gap tasks and record all horizon checkpoints.

    A single task simulates one fixed gap to the largest horizon and records
    mean regret at all checkpoints.  Combining across gaps then gives
        R_star(t) = max_Delta R_t(Delta)
    without rerunning a separate sweep for every horizon t.
    """
    outdir = pathlib.Path(args.outdir)
    workers = args.workers if args.workers is not None else default_workers()
    block_config = block_config_from_args(args)
    checkpoints = horizon_grid_from_args(args)
    max_horizon = int(np.max(checkpoints))

    for schedule_index, schedule, gap_index, gap in sweep_task_indices(args):
        sweep_dir = outdir / "envelope_sweep" / schedule.slug
        output_path = sweep_dir / f"gap_{gap_index:04d}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists. Use --overwrite to recompute.", flush=True)
            continue

        seed = int(args.seed + 90_000_071 * schedule_index + 1_000_003 * gap_index)
        result = simulate_parallel(
            schedule_slug=schedule.slug,
            gap_arm2=gap,
            method="approx",
            horizon_steps=max_horizon,
            num_arms=int(args.num_arms),
            num_trajectories=int(args.trajectories),
            random_seed=seed,
            workers=workers,
            chunk_trajectories=args.chunk_trajectories,
            checkpoints=checkpoints,
            block_config=block_config,
            return_trajectory_samples=False,
        )

        rows = []
        for i, t in enumerate(result.checkpoint_times):
            rows.append({
                "schedule_index": schedule_index,
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "gap_index": gap_index,
                "gap_arm2": gap,
                "time": int(t),
                "mean_regret": float(result.mean_regret[i]),
                "standard_error": float(result.se_regret[i]),
                "mean_pi1": float(result.mean_pi1[i]),
                "se_pi1": float(result.se_pi1[i]),
                "num_trajectories": result.num_trajectories,
                "horizon_steps": max_horizon,
                "num_arms": result.num_arms,
                "method": "approx",
                "experiment": "horizon_wise_gap_envelope",
                "exact_small_block_threshold": block_config.exact_small_block_threshold,
                "max_mean_change": block_config.max_mean_change,
                "max_noise_change": block_config.max_noise_change,
                "block_quantile": block_config.block_quantile,
                "num_blocks_total": result.num_blocks,
            })
        write_csv(output_path, rows)


def combine_envelope(args: argparse.Namespace) -> None:
    """Combine envelope sweep files and select hardest gap separately at each time."""
    outdir = pathlib.Path(args.outdir)
    paths = sorted((outdir / "envelope_sweep").glob("*/gap_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No envelope sweep CSV files found under {outdir / 'envelope_sweep'}")

    rows: List[dict] = []
    for path in paths:
        rows.extend(read_csv(path))

    rows.sort(key=lambda r: (int(r["schedule_index"]), int(r["time"]), int(r["gap_index"])))
    combined_path = outdir / "combined_envelope_sweep.csv"
    write_csv(combined_path, rows)

    by_schedule_time: Dict[Tuple[str, int], List[dict]] = {}
    for row in rows:
        key = (row["schedule_slug"], int(row["time"]))
        by_schedule_time.setdefault(key, []).append(row)

    hardest_rows = []
    for key, group in by_schedule_time.items():
        best = max(group, key=lambda r: float(r["mean_regret"]))
        # Rename the selected gap columns for clarity while preserving the raw row.
        best = dict(best)
        best["hardest_gap_index"] = best["gap_index"]
        best["hardest_gap_arm2"] = best["gap_arm2"]
        hardest_rows.append(best)

    hardest_rows.sort(key=lambda r: (int(r["schedule_index"]), int(r["time"])))
    hardest_path = outdir / "hardest_gap_by_time.csv"
    write_csv(hardest_path, hardest_rows)

    print("\nHorizon-wise hardest gaps at final recorded time:", flush=True)
    final_by_schedule: Dict[str, dict] = {}
    for row in hardest_rows:
        slug = row["schedule_slug"]
        if slug not in final_by_schedule or int(row["time"]) > int(final_by_schedule[slug]["time"]):
            final_by_schedule[slug] = row
    for slug, row in sorted(final_by_schedule.items(), key=lambda x: int(x[1]["schedule_index"])):
        print(
            f"  {slug:20s} T={int(row['time']):>10d} "
            f"Delta_t={float(row['hardest_gap_arm2']):.6g} "
            f"mean_regret={float(row['mean_regret']):.6e}",
            flush=True,
        )

    if getattr(args, "plot", False):
        plot_envelope(args)


def load_envelope(outdir: pathlib.Path) -> List[Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    hardest_path = outdir / "hardest_gap_by_time.csv"
    if not hardest_path.exists():
        raise FileNotFoundError(f"Missing {hardest_path}. Run combine-envelope first.")

    rows = read_csv(hardest_path)
    by_schedule: Dict[str, List[dict]] = {}
    for row in rows:
        by_schedule.setdefault(row["schedule_slug"], []).append(row)

    envelopes = []
    for schedule in all_schedules():
        group = by_schedule.get(schedule.slug)
        if not group:
            continue
        group.sort(key=lambda r: int(r["time"]))
        t = np.asarray([int(r["time"]) for r in group], dtype=np.float64)
        mean = np.asarray([float(r["mean_regret"]) for r in group], dtype=np.float64)
        se = np.asarray([float(r["standard_error"]) for r in group], dtype=np.float64)
        gap = np.asarray([float(r["hardest_gap_arm2"]) for r in group], dtype=np.float64)
        envelopes.append((schedule, t, mean, se, gap))
    return envelopes


def make_one_envelope_plot(
    envelopes: Sequence[Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    outpath: pathlib.Path,
    regret_scale: float,
    xscale: str,
    yscale: str,
    show_bands: bool,
    include_baselines: bool,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=180)

    for schedule, t, mean, se, gap in envelopes:
        label = f"{schedule.label}, horizon-wise hardest $\\Delta_t$"
        y = mean / regret_scale
        ax.plot(t, y, linewidth=2.2, label=label)
        if show_bands:
            ax.fill_between(
                t,
                (mean - 2.0 * se) / regret_scale,
                (mean + 2.0 * se) / regret_scale,
                alpha=0.08,
                linewidth=0,
            )

    if include_baselines and envelopes:
        t_ref = envelopes[0][1]
        baselines = [
            (0.5, r"raw $\sqrt{T}$"),
            (2.0 / 3.0, r"raw $T^{2/3}$"),
            (1.0, r"raw $T$"),
        ]
        for alpha, label in baselines:
            ax.plot(
                t_ref,
                np.power(t_ref, alpha) / regret_scale,
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
                label=label,
            )

    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel("Horizon / time")
    ax.set_ylabel("Worst-gap average cumulative regret" if regret_scale == 1.0 else rf"Worst-gap average cumulative regret / ${regret_scale:.0e}$")
    ax.set_title(r"Horizon-wise hardest regret envelope, $\max_{\Delta} R_t(\Delta)$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)


def plot_envelope(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    envelopes = load_envelope(outdir)
    if not envelopes:
        raise RuntimeError("No envelope rows loaded")

    scale = float(args.regret_scale)
    include_baselines = not getattr(args, "no_baselines", False)
    show_bands = bool(getattr(args, "show_bands", False))

    make_one_envelope_plot(
        envelopes,
        outdir / "envelope_hardest_regret_linear_scale.png",
        scale,
        xscale="linear",
        yscale="linear",
        show_bands=show_bands,
        include_baselines=include_baselines,
    )
    make_one_envelope_plot(
        envelopes,
        outdir / "envelope_hardest_regret_logx.png",
        scale,
        xscale="log",
        yscale="linear",
        show_bands=show_bands,
        include_baselines=include_baselines,
    )
    make_one_envelope_plot(
        envelopes,
        outdir / "envelope_hardest_regret_loglog.png",
        scale,
        xscale="log",
        yscale="log",
        show_bands=show_bands,
        include_baselines=include_baselines,
    )

    if plt is None:
        return

    # Plot the selected hardest gap as a function of horizon.
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    for schedule, t, mean, se, gap in envelopes:
        ax.plot(t, gap, linewidth=2.2, marker="o", markersize=3, label=schedule.label)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon / time")
    ax.set_ylabel(r"Selected hardest gap $\Delta_t$")
    ax.set_title(r"Gap selected by $\arg\max_{\Delta} R_t(\Delta)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    outpath = outdir / "envelope_hardest_gap_vs_time.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)

    # Plot R_t/t, which is the most direct diagnostic for almost-linear regret.
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    for schedule, t, mean, se, gap in envelopes:
        ax.plot(t, mean / t, linewidth=2.2, label=schedule.label)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon / time")
    ax.set_ylabel(r"Worst-gap average regret per round, $R_t/t$")
    ax.set_title(r"Diagnostic for linear worst-gap regret")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    outpath = outdir / "envelope_regret_per_round.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)

# ---------------------------------------------------------------------------
# Exact-vs-approx validation at T=100k
# ---------------------------------------------------------------------------


def validation_pairs(args: argparse.Namespace) -> List[Tuple[int, object, int, float]]:
    schedules = select_schedules(args.schedule_slugs)
    gaps = parse_float_list(args.gap_values)

    if gaps is not None:
        pairs = []
        for schedule in schedules:
            schedule_index = all_schedules().index(schedule)
            for gap_index, gap in enumerate(gaps):
                pairs.append((schedule_index, schedule, gap_index, float(gap)))
        return pairs

    hardest_path = pathlib.Path(args.hardest_csv) if args.hardest_csv else pathlib.Path(args.outdir) / "hardest_gaps_by_schedule.csv"
    if hardest_path.exists():
        rows = read_csv(hardest_path)
        wanted_slugs = {s.slug for s in schedules}
        pairs = []
        for row in rows:
            if row["schedule_slug"] not in wanted_slugs:
                continue
            schedule = schedule_by_slug(row["schedule_slug"])
            pairs.append((int(row["schedule_index"]), schedule, int(row["gap_index"]), float(row["gap_arm2"])))
        if pairs:
            return pairs

    raise FileNotFoundError(
        "Validation needs either --gap-values or an existing hardest_gaps_by_schedule.csv. "
        "Run combine-sweep first, or pass --gap-values such as --gap-values 0.2,0.25,0.35."
    )


def run_validation(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir) / "validation"
    outdir.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else default_workers()
    block_config = block_config_from_args(args)
    checkpoints = log_spaced_checkpoints(int(args.horizon), int(args.num_checkpoints), include_early=False)

    final_rows = []
    history_rows = []

    for schedule_index, schedule, gap_index, gap in validation_pairs(args):
        for method in ["exact", "approx"]:
            seed = int(args.seed + 80_000_033 * schedule_index + 1_000_003 * gap_index)
            result = simulate_parallel(
                schedule_slug=schedule.slug,
                gap_arm2=gap,
                method=method,
                horizon_steps=int(args.horizon),
                num_arms=int(args.num_arms),
                num_trajectories=int(args.trajectories),
                random_seed=seed,
                workers=workers,
                chunk_trajectories=args.chunk_trajectories,
                checkpoints=checkpoints,
                block_config=block_config,
                return_trajectory_samples=False,
            )

            final_rows.append({
                "schedule_index": schedule_index,
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "gap_index": gap_index,
                "gap_arm2": gap,
                "method": method,
                "horizon_steps": result.horizon_steps,
                "num_arms": result.num_arms,
                "num_trajectories": result.num_trajectories,
                "mean_regret": float(result.mean_regret[-1]),
                "standard_error": float(result.se_regret[-1]),
                "mean_final_pi1": float(result.mean_pi1[-1]),
                "se_final_pi1": float(result.se_pi1[-1]),
                "num_blocks_total": result.num_blocks,
                "exact_small_block_threshold": block_config.exact_small_block_threshold,
                "max_mean_change": block_config.max_mean_change,
                "max_noise_change": block_config.max_noise_change,
                "block_quantile": block_config.block_quantile,
            })

            for i, t in enumerate(result.checkpoint_times):
                history_rows.append({
                    "schedule_index": schedule_index,
                    "schedule_slug": schedule.slug,
                    "schedule_label": schedule.label,
                    "gap_index": gap_index,
                    "gap_arm2": gap,
                    "method": method,
                    "time": int(t),
                    "mean_regret": float(result.mean_regret[i]),
                    "standard_error": float(result.se_regret[i]),
                    "mean_pi1": float(result.mean_pi1[i]),
                    "se_pi1": float(result.se_pi1[i]),
                    "num_trajectories": result.num_trajectories,
                })

    write_csv(outdir / "validation_final.csv", final_rows)
    write_csv(outdir / "validation_history.csv", history_rows)

    comparison = make_validation_comparison_rows(final_rows)
    write_csv(outdir / "validation_comparison.csv", comparison)

    if args.plot:
        plot_validation(outdir, float(args.regret_scale))


def make_validation_comparison_rows(final_rows: Sequence[dict]) -> List[dict]:
    by_key: Dict[Tuple[str, int], Dict[str, dict]] = {}
    for row in final_rows:
        key = (row["schedule_slug"], int(row["gap_index"]))
        by_key.setdefault(key, {})[row["method"]] = row

    comparison = []
    for key in sorted(by_key):
        group = by_key[key]
        if "exact" not in group or "approx" not in group:
            continue
        exact = group["exact"]
        approx = group["approx"]
        exact_mean = float(exact["mean_regret"])
        approx_mean = float(approx["mean_regret"])
        exact_se = float(exact["standard_error"])
        approx_se = float(approx["standard_error"])
        diff = approx_mean - exact_mean
        denom = math.sqrt(exact_se * exact_se + approx_se * approx_se)
        comparison.append({
            "schedule_index": exact["schedule_index"],
            "schedule_slug": exact["schedule_slug"],
            "schedule_label": exact["schedule_label"],
            "gap_index": exact["gap_index"],
            "gap_arm2": exact["gap_arm2"],
            "exact_mean_regret": exact_mean,
            "approx_mean_regret": approx_mean,
            "difference_approx_minus_exact": diff,
            "relative_difference": diff / exact_mean if exact_mean != 0.0 else float("nan"),
            "z_score": diff / denom if denom > 0.0 else float("nan"),
            "exact_mean_final_pi1": exact["mean_final_pi1"],
            "approx_mean_final_pi1": approx["mean_final_pi1"],
        })
    return comparison


def plot_validation(validation_dir: pathlib.Path, regret_scale: float) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    rows = read_csv(validation_dir / "validation_history.csv")
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=180)
    grouped: Dict[Tuple[str, str, float], List[dict]] = {}
    for row in rows:
        key = (row["schedule_slug"], row["method"], float(row["gap_arm2"]))
        grouped.setdefault(key, []).append(row)

    for (slug, method, gap), group in sorted(grouped.items()):
        schedule = schedule_by_slug(slug)
        group.sort(key=lambda r: int(r["time"]))
        t = np.asarray([int(r["time"]) for r in group], dtype=np.float64)
        mean = np.asarray([float(r["mean_regret"]) for r in group], dtype=np.float64) / regret_scale
        linestyle = "-" if method == "exact" else "--"
        label = f"{schedule.label}, {method}, $\\Delta={gap:.4g}$"
        ax.plot(t, mean, linestyle=linestyle, linewidth=1.9, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative regret" if regret_scale == 1.0 else rf"Average cumulative regret / ${regret_scale:.0e}$")
    ax.set_title("Exact vs approximate update validation")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=7)
    fig.tight_layout()
    outpath = validation_dir / "validation_exact_vs_approx_history.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Compute Canada / SLURM template writer
# ---------------------------------------------------------------------------


def write_slurm_templates(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num_schedules = len(select_schedules(args.schedule_slugs))
    total_tasks = num_schedules * int(args.num_gaps)
    array_max = total_tasks - 1

    sweep_script = f"""#!/bin/bash
#SBATCH --job-name=pg_sweep
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time={args.time}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --array=0-{array_max}%{args.array_parallelism}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_pg_bandit_experiments.py sweep \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --outdir {args.outdir} \\
  --schedule-slugs {args.schedule_slugs} \\
  --num-gaps {args.num_gaps} \\
  --gap-start {args.gap_start} \\
  --gap-stop {args.gap_stop} \\
  --horizon {args.horizon} \\
  --num-arms {args.num_arms} \\
  --trajectories {args.trajectories} \\
  --workers "${{SLURM_CPUS_PER_TASK}}" \\
  --chunk-trajectories {args.chunk_trajectories or 0} \\
  --max-mean-change {args.max_mean_change} \\
  --max-noise-change {args.max_noise_change} \\
  --block-quantile {args.block_quantile} \\
  --exact-small-block-threshold {args.exact_small_block_threshold}
"""

    history_script = f"""#!/bin/bash
#SBATCH --job-name=pg_history
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time={args.time}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_pg_bandit_experiments.py history \\
  --outdir {args.outdir} \\
  --horizon {args.horizon} \\
  --num-arms {args.num_arms} \\
  --trajectories {args.trajectories} \\
  --workers "${{SLURM_CPUS_PER_TASK}}" \\
  --chunk-trajectories {args.chunk_trajectories or 0} \\
  --num-checkpoints {args.num_checkpoints} \\
  --max-mean-change {args.max_mean_change} \\
  --max-noise-change {args.max_noise_change} \\
  --block-quantile {args.block_quantile} \\
  --exact-small-block-threshold {args.exact_small_block_threshold}

python run_pg_bandit_experiments.py plot-history --outdir {args.outdir}
"""

    (outdir / "submit_sweep_array.sh").write_text(sweep_script)
    (outdir / "submit_history.sh").write_text(history_script)
    print(f"[write] {outdir / 'submit_sweep_array.sh'}", flush=True)
    print(f"[write] {outdir / 'submit_history.sh'}", flush=True)




def write_envelope_slurm_templates(args: argparse.Namespace) -> None:
    """Write SLURM scripts for the horizon-wise envelope workflow."""
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num_schedules = len(select_schedules(args.schedule_slugs))
    total_tasks = num_schedules * int(args.num_gaps)
    array_max = total_tasks - 1

    horizon_arg = f"--horizon-values {args.horizon_values}" if args.horizon_values else f"--num-horizons {args.num_horizons}"

    sweep_script = f"""#!/bin/bash
#SBATCH --job-name=pg_env_sweep
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time={args.time}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --array=0-{array_max}%{args.array_parallelism}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_pg_bandit_experiments.py envelope-sweep \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --outdir {args.outdir} \\
  --schedule-slugs {args.schedule_slugs} \\
  --num-gaps {args.num_gaps} \\
  --gap-start {args.gap_start} \\
  --gap-stop {args.gap_stop} \\
  --horizon {args.horizon} \\
  {horizon_arg} \\
  --num-arms {args.num_arms} \\
  --trajectories {args.trajectories} \\
  --workers "${{SLURM_CPUS_PER_TASK}}" \\
  --chunk-trajectories {args.chunk_trajectories or 0} \\
  --max-mean-change {args.max_mean_change} \\
  --max-noise-change {args.max_noise_change} \\
  --block-quantile {args.block_quantile} \\
  --exact-small-block-threshold {args.exact_small_block_threshold}
"""

    combine_script = f"""#!/bin/bash
#SBATCH --job-name=pg_env_plot
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true

python run_pg_bandit_experiments.py combine-envelope \\
  --outdir {args.outdir} \\
  --plot \\
  --regret-scale {args.regret_scale}
"""

    (outdir / "submit_envelope_sweep_array.sh").write_text(sweep_script)
    (outdir / "submit_envelope_combine_plot.sh").write_text(combine_script)
    print(f"[write] {outdir / 'submit_envelope_sweep_array.sh'}", flush=True)
    print(f"[write] {outdir / 'submit_envelope_combine_plot.sh'}", flush=True)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon", type=int, default=10**9)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--outdir", type=str, default="multi_schedule_results")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max-mean-change", type=float, default=0.20)
    parser.add_argument("--max-noise-change", type=float, default=0.80)
    parser.add_argument("--max-block-size", type=int, default=50_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=4096)
    parser.add_argument("--exact-small-block-threshold", type=int, default=64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and plot policy-gradient bandit experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Run approximate final-regret sweep tasks.")
    add_common_sim_args(sweep)
    sweep.add_argument("--task-index", type=int, default=None)
    sweep.add_argument("--schedule-index", type=int, default=None)
    sweep.add_argument("--gap-index", type=int, default=None)
    sweep.add_argument("--run-all", action="store_true")
    sweep.add_argument("--schedule-slugs", type=str, default="all")
    sweep.add_argument("--gap-start", type=float, default=0.0)
    sweep.add_argument("--gap-stop", type=float, default=1.0)
    sweep.add_argument("--num-gaps", type=int, default=101)

    combine = sub.add_parser("combine-sweep", help="Combine sweep CSVs and find hardest gap per schedule.")
    combine.add_argument("--outdir", type=str, default="multi_schedule_results")

    history = sub.add_parser("history", help="Run approximate regret histories at hardest gaps.")
    add_common_sim_args(history)
    history.add_argument("--hardest-csv", type=str, default=None)
    history.add_argument("--history-schedule-index", type=int, default=None)
    history.add_argument("--schedule-slugs", type=str, default="all")
    history.add_argument("--num-checkpoints", type=int, default=250)
    history.add_argument("--linear-early-checkpoints", action="store_true")
    history.add_argument("--plot", action="store_true")
    history.add_argument("--regret-scale", type=float, default=1_000_000.0)
    history.add_argument("--show-bands", action="store_true")
    history.add_argument("--no-baselines", action="store_true")

    plot_history_parser = sub.add_parser("plot-history", help="Plot existing history CSV files in three scales.")
    plot_history_parser.add_argument("--outdir", type=str, default="multi_schedule_results")
    plot_history_parser.add_argument("--regret-scale", type=float, default=1_000_000.0)
    plot_history_parser.add_argument("--show-bands", action="store_true")
    plot_history_parser.add_argument("--no-baselines", action="store_true")

    envelope_sweep = sub.add_parser(
        "envelope-sweep",
        help="Run approximate sweep that records max-Delta candidates at many horizon checkpoints.",
    )
    add_common_sim_args(envelope_sweep)
    envelope_sweep.set_defaults(horizon=10_000_000, trajectories=10_000, outdir="envelope_results")
    envelope_sweep.add_argument("--task-index", type=int, default=None)
    envelope_sweep.add_argument("--schedule-index", type=int, default=None)
    envelope_sweep.add_argument("--gap-index", type=int, default=None)
    envelope_sweep.add_argument("--run-all", action="store_true")
    envelope_sweep.add_argument("--schedule-slugs", type=str, default="all")
    envelope_sweep.add_argument("--gap-start", type=float, default=0.0)
    envelope_sweep.add_argument("--gap-stop", type=float, default=1.0)
    envelope_sweep.add_argument("--num-gaps", type=int, default=51)
    envelope_sweep.add_argument("--num-horizons", type=int, default=21)
    envelope_sweep.add_argument(
        "--horizon-values",
        type=str,
        default=None,
        help="Optional comma-separated horizons, e.g. 1000,3000,10000,30000,100000.",
    )

    combine_envelope_parser = sub.add_parser(
        "combine-envelope",
        help="Combine envelope sweep outputs and choose hardest gap separately at each time.",
    )
    combine_envelope_parser.add_argument("--outdir", type=str, default="envelope_results")
    combine_envelope_parser.add_argument("--plot", action="store_true")
    combine_envelope_parser.add_argument("--regret-scale", type=float, default=1_000_000.0)
    combine_envelope_parser.add_argument("--show-bands", action="store_true")
    combine_envelope_parser.add_argument("--no-baselines", action="store_true")

    plot_envelope_parser = sub.add_parser(
        "plot-envelope",
        help="Plot existing horizon-wise hardest-gap envelope outputs.",
    )
    plot_envelope_parser.add_argument("--outdir", type=str, default="envelope_results")
    plot_envelope_parser.add_argument("--regret-scale", type=float, default=1_000_000.0)
    plot_envelope_parser.add_argument("--show-bands", action="store_true")
    plot_envelope_parser.add_argument("--no-baselines", action="store_true")

    validate = sub.add_parser("validate", help="Compare exact and approximate updates at a shorter horizon.")
    add_common_sim_args(validate)
    validate.set_defaults(horizon=100_000, trajectories=1000)
    validate.add_argument("--schedule-slugs", type=str, default="all")
    validate.add_argument("--hardest-csv", type=str, default=None)
    validate.add_argument("--gap-values", type=str, default=None,
                          help="Comma-separated gaps. If omitted, uses hardest_gaps_by_schedule.csv.")
    validate.add_argument("--num-checkpoints", type=int, default=80)
    validate.add_argument("--plot", action="store_true")
    validate.add_argument("--regret-scale", type=float, default=1_000_000.0)
    # Stricter validation defaults than the large 1e9 sweep.
    validate.set_defaults(max_mean_change=0.02, max_noise_change=0.10, max_block_size=1_000_000, block_quantile=1.0)

    slurm = sub.add_parser("write-slurm", help="Write simple SLURM templates into --outdir.")
    add_common_sim_args(slurm)
    slurm.add_argument("--schedule-slugs", type=str, default="all")
    slurm.add_argument("--gap-start", type=float, default=0.0)
    slurm.add_argument("--gap-stop", type=float, default=1.0)
    slurm.add_argument("--num-gaps", type=int, default=101)
    slurm.add_argument("--num-checkpoints", type=int, default=250)
    slurm.add_argument("--time", type=str, default="12:00:00")
    slurm.add_argument("--cpus-per-task", type=int, default=8)
    slurm.add_argument("--mem", type=str, default="16G")
    slurm.add_argument("--array-parallelism", type=int, default=100)

    envelope_slurm = sub.add_parser("write-envelope-slurm", help="Write SLURM templates for the horizon-wise envelope workflow.")
    add_common_sim_args(envelope_slurm)
    envelope_slurm.set_defaults(horizon=10_000_000, trajectories=10_000, outdir="envelope_results")
    envelope_slurm.add_argument("--schedule-slugs", type=str, default="all")
    envelope_slurm.add_argument("--gap-start", type=float, default=0.0)
    envelope_slurm.add_argument("--gap-stop", type=float, default=1.0)
    envelope_slurm.add_argument("--num-gaps", type=int, default=51)
    envelope_slurm.add_argument("--num-horizons", type=int, default=21)
    envelope_slurm.add_argument("--horizon-values", type=str, default=None)
    envelope_slurm.add_argument("--regret-scale", type=float, default=1_000_000.0)
    envelope_slurm.add_argument("--time", type=str, default="08:00:00")
    envelope_slurm.add_argument("--cpus-per-task", type=int, default=8)
    envelope_slurm.add_argument("--mem", type=str, default="16G")
    envelope_slurm.add_argument("--array-parallelism", type=int, default=100)

    list_schedules = sub.add_parser("list-schedules", help="Print schedule slugs.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "list-schedules":
        for i, schedule in enumerate(all_schedules()):
            print(f"{i}: {schedule.slug:20s} {schedule.label}")
        return

    if args.command == "sweep":
        run_sweep(args)
    elif args.command == "combine-sweep":
        combine_sweep(args)
    elif args.command == "history":
        run_history(args)
    elif args.command == "plot-history":
        plot_history(args)
    elif args.command == "envelope-sweep":
        run_envelope_sweep(args)
    elif args.command == "combine-envelope":
        combine_envelope(args)
    elif args.command == "plot-envelope":
        plot_envelope(args)
    elif args.command == "validate":
        run_validation(args)
    elif args.command == "write-slurm":
        write_slurm_templates(args)
    elif args.command == "write-envelope-slurm":
        write_envelope_slurm_templates(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
