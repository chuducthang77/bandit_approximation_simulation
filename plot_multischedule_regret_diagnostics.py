#!/usr/bin/env python3
"""
Post-processing plots for the multi-schedule hardest-gap regret histories.

This script does not rerun the bandit simulations. It reads files of the form

    <outdir>/history/history_*.csv

produced by pg_multischedule_hardest_gap.py history, and writes diagnostic plots
for deciding whether the observed finite-horizon behavior is close to linear:

    1. cumulative regret vs time on log-log axes, with growth-rate guides
    2. average regret per round, R_t / t, vs time
    3. local log-log slope, d log R_t / d log t, vs time

The cumulative plot can show raw guides sqrt(T), T^(2/3), T, matched-final
scaled guides, or both. Raw guides preserve the ordering

    T > T^(2/3) > sqrt(T)       for T > 1.

Matched-final guides are useful for comparing slopes because all guides end at
one common reference value, but their vertical ordering before the final horizon
is reversed because (t/n)^0.5 > (t/n)^(2/3) > (t/n)^1 for t < n.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SCHEDULE_LABELS = {
    "inv_t": r"$\eta_t = 1/t$",
    "inv_t_two_thirds": r"$\eta_t = 1/t^{2/3}$",
    "log_over_t": r"$\eta_t = \log(t)/t$",
    "inv_sqrt_t": r"$\eta_t = 1/\sqrt{t}$",
    "sqrt_log_over_t": r"$\eta_t = \sqrt{\log(t)/t}$",
    "inv_log_t": r"$\eta_t = 1/\log(t)$",
}


@dataclass
class History:
    path: pathlib.Path
    schedule_slug: str
    schedule_label: str
    hardest_gap: float
    time: np.ndarray
    mean_regret: np.ndarray
    standard_error: np.ndarray
    num_trajectories: Optional[int] = None

    @property
    def final_time(self) -> float:
        return float(self.time[-1])

    @property
    def final_mean_regret(self) -> float:
        return float(self.mean_regret[-1])

    @property
    def final_regret_per_round(self) -> float:
        return self.final_mean_regret / self.final_time


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------


def first_existing_column(row: dict, candidates: Sequence[str], *, path: pathlib.Path) -> str:
    for name in candidates:
        if name in row:
            return name
    raise ValueError(f"None of these columns were found in {path}: {candidates}")


def read_csv_rows(path: pathlib.Path) -> List[dict]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV file: {path}")
    return rows


def load_history(path: pathlib.Path) -> History:
    rows = read_csv_rows(path)
    first = rows[0]

    time_col = first_existing_column(
        first,
        ["time", "timestep", "checkpoint_time"],
        path=path,
    )
    mean_col = first_existing_column(
        first,
        ["mean_regret", "average_regret", "average_cumulative_regret"],
        path=path,
    )

    if "standard_error" in first:
        se_col = "standard_error"
    elif "std_error" in first:
        se_col = "std_error"
    else:
        se_col = None

    schedule_slug = first.get("schedule_slug", path.stem.replace("history_", ""))
    schedule_label = first.get("schedule_label", DEFAULT_SCHEDULE_LABELS.get(schedule_slug, schedule_slug))
    # Some CSV writers or manual edits store LaTeX labels with doubled backslashes.
    # Matplotlib mathtext expects single backslashes inside $...$.
    schedule_label = schedule_label.replace("\\\\", "\\")

    if "hardest_gap_arm2" in first:
        hardest_gap = float(first["hardest_gap_arm2"])
    elif "gap_arm2" in first:
        hardest_gap = float(first["gap_arm2"])
    else:
        hardest_gap = float("nan")

    time = np.array([float(r[time_col]) for r in rows], dtype=np.float64)
    mean_regret = np.array([float(r[mean_col]) for r in rows], dtype=np.float64)
    if se_col is None:
        standard_error = np.zeros_like(mean_regret)
    else:
        standard_error = np.array([float(r[se_col]) for r in rows], dtype=np.float64)

    order = np.argsort(time)
    time = time[order]
    mean_regret = mean_regret[order]
    standard_error = standard_error[order]

    # Merge duplicate checkpoints, keeping the last occurrence.
    unique_times, unique_indices = np.unique(time, return_index=True)
    if len(unique_times) != len(time):
        last_indices = []
        for t in unique_times:
            matching = np.flatnonzero(time == t)
            last_indices.append(matching[-1])
        last_indices = np.array(last_indices, dtype=int)
        time = time[last_indices]
        mean_regret = mean_regret[last_indices]
        standard_error = standard_error[last_indices]

    num_trajectories = None
    if "num_trajectories" in first:
        try:
            num_trajectories = int(float(first["num_trajectories"]))
        except ValueError:
            num_trajectories = None

    return History(
        path=path,
        schedule_slug=schedule_slug,
        schedule_label=schedule_label,
        hardest_gap=hardest_gap,
        time=time,
        mean_regret=mean_regret,
        standard_error=standard_error,
        num_trajectories=num_trajectories,
    )


def load_histories(history_dir: pathlib.Path, schedule_slugs: Optional[str]) -> List[History]:
    if not history_dir.exists():
        raise FileNotFoundError(f"History directory does not exist: {history_dir}")

    requested = None
    if schedule_slugs and schedule_slugs.strip().lower() not in {"", "all"}:
        requested = {s.strip() for s in schedule_slugs.split(",") if s.strip()}

    paths = sorted(history_dir.glob("history_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No history_*.csv files found in {history_dir}")

    histories = []
    for path in paths:
        history = load_history(path)
        if requested is not None and history.schedule_slug not in requested:
            continue
        histories.append(history)

    if not histories:
        raise ValueError(f"No histories matched --schedule-slugs={schedule_slugs!r}")

    # Stable plotting order used in the simulation script.
    preferred_order = [
        "inv_t",
        "inv_t_two_thirds",
        "log_over_t",
        "inv_sqrt_t",
        "sqrt_log_over_t",
        "inv_log_t",
    ]
    order = {slug: i for i, slug in enumerate(preferred_order)}
    histories.sort(key=lambda h: (order.get(h.schedule_slug, 10_000), h.schedule_slug))
    return histories


# -----------------------------------------------------------------------------
# Numerical diagnostics
# -----------------------------------------------------------------------------


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, values.size)
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def local_loglog_slope(
    time: np.ndarray,
    regret: np.ndarray,
    *,
    smooth_window: int,
    min_regret: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (time > 1.0) & (regret > min_regret) & np.isfinite(time) & np.isfinite(regret)
    t = time[mask]
    r = regret[mask]
    if t.size < 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    log_t = np.log(t)
    log_r = np.log(r)
    log_r_smooth = smooth_series(log_r, smooth_window)
    slope = np.gradient(log_r_smooth, log_t)
    return t, slope


def write_summary_csv(histories: Sequence[History], outdir: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    path = outdir / f"{args.prefix}_summary.csv"
    rows = []

    for history in histories:
        slope_t, slope = local_loglog_slope(
            history.time,
            history.mean_regret,
            smooth_window=args.slope_smooth_window,
            min_regret=args.slope_min_regret,
        )
        last_slope = float(slope[-1]) if slope.size else float("nan")
        median_tail_slope = float(np.median(slope[-min(10, slope.size):])) if slope.size else float("nan")

        rows.append(
            {
                "schedule_slug": history.schedule_slug,
                "schedule_label": history.schedule_label,
                "hardest_gap_arm2": history.hardest_gap,
                "final_time": int(history.final_time),
                "final_mean_regret": history.final_mean_regret,
                "final_regret_per_round": history.final_regret_per_round,
                "last_local_loglog_slope": last_slope,
                "median_last_10_local_loglog_slopes": median_tail_slope,
                "num_trajectories": history.num_trajectories if history.num_trajectories is not None else "",
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def display_label(history: History) -> str:
    if math.isfinite(history.hardest_gap):
        return f"{history.schedule_label}, hardest $\\Delta={history.hardest_gap:.4g}$"
    return history.schedule_label


def positive_curve(time: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (time > 0.0) & (y > 0.0) & np.isfinite(time) & np.isfinite(y)
    return time[mask], y[mask]


def guide_time_grid(histories: Sequence[History]) -> np.ndarray:
    # Use the densest time grid among the histories as the guide grid.
    return max((h.time for h in histories), key=lambda x: x.size)


def plot_guides(
    ax,
    t_ref: np.ndarray,
    *,
    mode: str,
    max_final_regret: float,
    horizon: float,
    scale: float,
) -> None:
    baseline_specs = [
        (0.5, r"$\sqrt{T}$"),
        (2.0 / 3.0, r"$T^{2/3}$"),
        (1.0, r"$T$"),
    ]

    for alpha, label in baseline_specs:
        if mode == "raw":
            guide = np.power(t_ref, alpha)
            guide_label = label + " raw guide"
        elif mode == "matched-final":
            guide = max_final_regret * np.power(t_ref / horizon, alpha)
            guide_label = label + " matched-final guide"
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

        t, y = positive_curve(t_ref, guide / scale)
        ax.plot(t, y, linestyle="--", linewidth=1.8, alpha=0.85, label=guide_label)


def save_figure(fig, path: pathlib.Path, also_pdf: bool) -> None:
    fig.savefig(path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}", flush=True)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_cumulative_loglog(
    histories: Sequence[History],
    outdir: pathlib.Path,
    args: argparse.Namespace,
    *,
    baseline_mode: str,
) -> pathlib.Path:
    scale = float(args.regret_scale)
    max_final_regret = max(h.final_mean_regret for h in histories)
    horizon = max(h.final_time for h in histories)
    t_ref = guide_time_grid(histories)

    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=args.dpi)

    for history in histories:
        t, y = positive_curve(history.time, history.mean_regret / scale)
        ax.plot(t, y, linewidth=2.2, label=display_label(history))

        if args.show_bands:
            lower = history.mean_regret - 2.0 * history.standard_error
            upper = history.mean_regret + 2.0 * history.standard_error
            band_mask = (history.time > 0.0) & (upper > 0.0) & np.isfinite(upper)
            lower = np.maximum(lower, np.nanmin(history.mean_regret[history.mean_regret > 0.0]) * 1e-3)
            ax.fill_between(
                history.time[band_mask],
                lower[band_mask] / scale,
                upper[band_mask] / scale,
                alpha=0.08,
                linewidth=0,
            )

    plot_guides(
        ax,
        t_ref,
        mode=baseline_mode,
        max_final_regret=max_final_regret,
        horizon=horizon,
        scale=scale,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative regret" if scale == 1.0 else rf"Average cumulative regret / ${scale:.0e}$")

    if baseline_mode == "raw":
        title_suffix = "raw growth-rate guides"
    else:
        title_suffix = "matched-final growth-rate guides"
    ax.set_title(f"Hardest-gap regret histories ({title_suffix})")

    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True, fontsize=args.legend_fontsize)
    fig.tight_layout()

    path = outdir / f"{args.prefix}_cumulative_loglog_{baseline_mode.replace('-', '_')}.png"
    save_figure(fig, path, args.pdf)
    return path


def plot_regret_per_round(histories: Sequence[History], outdir: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=args.dpi)

    for history in histories:
        y = history.mean_regret / history.time
        t, y = positive_curve(history.time, y)
        ax.plot(t, y, linewidth=2.2, label=display_label(history))

    ax.set_xscale("log")
    if args.per_round_log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Average regret per round, $R_t/t$")
    ax.set_title(r"Diagnostic for linear regret: $R_t/t$")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True, fontsize=args.legend_fontsize)
    fig.tight_layout()

    suffix = "logy" if args.per_round_log_y else "linear_y"
    path = outdir / f"{args.prefix}_regret_per_round_{suffix}.png"
    save_figure(fig, path, args.pdf)
    return path


def plot_local_slope(histories: Sequence[History], outdir: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=args.dpi)

    for history in histories:
        t, slope = local_loglog_slope(
            history.time,
            history.mean_regret,
            smooth_window=args.slope_smooth_window,
            min_regret=args.slope_min_regret,
        )
        if t.size == 0:
            continue
        ax.plot(t, slope, linewidth=2.1, label=display_label(history))

    ax.axhline(1.0, linestyle="--", linewidth=1.5, alpha=0.85, label=r"linear slope $1$")
    ax.axhline(2.0 / 3.0, linestyle="--", linewidth=1.5, alpha=0.85, label=r"$2/3$ slope")
    ax.axhline(0.5, linestyle="--", linewidth=1.5, alpha=0.85, label=r"$1/2$ slope")

    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Local slope $d\log R_t / d\log t$")
    ax.set_title("Local power-law exponent diagnostic")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(frameon=True, fontsize=args.legend_fontsize)
    fig.tight_layout()

    path = outdir / f"{args.prefix}_local_loglog_slope.png"
    save_figure(fig, path, args.pdf)
    return path


def plot_all(histories: Sequence[History], outdir: pathlib.Path, args: argparse.Namespace) -> List[pathlib.Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[pathlib.Path] = []

    baseline_modes: List[str]
    if args.baseline_mode == "both":
        baseline_modes = ["raw", "matched-final"]
    else:
        baseline_modes = [args.baseline_mode]

    for mode in baseline_modes:
        paths.append(plot_cumulative_loglog(histories, outdir, args, baseline_mode=mode))

    paths.append(plot_regret_per_round(histories, outdir, args))

    # Also write a log-y version of R_t/t when the main version uses linear y.
    if not args.per_round_log_y and args.also_per_round_log_y:
        old_value = args.per_round_log_y
        args.per_round_log_y = True
        paths.append(plot_regret_per_round(histories, outdir, args))
        args.per_round_log_y = old_value

    paths.append(plot_local_slope(histories, outdir, args))
    summary_path = write_summary_csv(histories, outdir, args)
    print(f"[summary] wrote {summary_path}", flush=True)
    return paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regret-history diagnostics for the multi-schedule hardest-gap experiment."
    )

    parser.add_argument("--outdir", type=str, default="multi_schedule_results",
                        help="Directory containing the history/ subdirectory and where plots are written.")
    parser.add_argument("--history-dir", type=str, default=None,
                        help="Override input history directory. Default: <outdir>/history.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Override output plot directory. Default: <outdir>.")
    parser.add_argument("--schedule-slugs", type=str, default="all",
                        help="Comma-separated schedules to plot, or 'all'.")

    parser.add_argument("--regret-scale", type=float, default=1_000_000.0,
                        help="Divide cumulative regret by this number in cumulative plots.")
    parser.add_argument("--baseline-mode", choices=["raw", "matched-final", "both"], default="both",
                        help="Growth-rate guides to include in the cumulative log-log plot.")
    parser.add_argument("--show-bands", action="store_true",
                        help="Show +/- 2 standard-error bands when available.")

    parser.add_argument("--per-round-log-y", action="store_true",
                        help="Use a log y-axis for the R_t/t plot.")
    parser.add_argument("--also-per-round-log-y", action="store_true", default=True,
                        help="Also save a log-y version of the R_t/t plot when the main one uses linear y.")

    parser.add_argument("--slope-smooth-window", type=int, default=7,
                        help="Moving-average window applied to log regret before computing local slope.")
    parser.add_argument("--slope-min-regret", type=float, default=1.0,
                        help="Ignore points with regret below this value in the local-slope plot.")

    parser.add_argument("--prefix", type=str, default="diagnostic",
                        help="Output filename prefix.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pdf", action="store_true",
                        help="Also save each figure as PDF.")
    parser.add_argument("--legend-fontsize", type=float, default=8.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    history_dir = pathlib.Path(args.history_dir) if args.history_dir else outdir / "history"
    plot_dir = pathlib.Path(args.plot_dir) if args.plot_dir else outdir

    histories = load_histories(history_dir, args.schedule_slugs)
    print(f"[load] loaded {len(histories)} history files from {history_dir}", flush=True)
    for history in histories:
        print(
            f"  {history.schedule_slug:22s} gap={history.hardest_gap:.6g} "
            f"final_R={history.final_mean_regret:.6e} R/T={history.final_regret_per_round:.6e}",
            flush=True,
        )

    plot_all(histories, plot_dir, args)


if __name__ == "__main__":
    main()
