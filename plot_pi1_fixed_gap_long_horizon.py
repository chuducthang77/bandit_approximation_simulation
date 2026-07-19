#!/usr/bin/env python3
"""
Fixed-gap pi_1 sample-path experiment.

Setting:
    Delta = 0.002
    number of trajectories = 40
    horizon = 1e11

Arm means:
    mu = (1, 1-Delta, 0, ..., 0)

Stepsizes:
    1) eta_t = 1/sqrt(t+1)
    2) eta   = Delta^2

Arm counts:
    K = 3
    K = 1000

Outputs:
    For each (K, schedule), create:
        - logit-scale sample-path plot
        - linear-scale sample-path plot
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_log_checkpoints(horizon: int, num_checkpoints: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if num_checkpoints <= 1:
        return np.array([horizon], dtype=np.int64)

    pts = np.unique(
        np.round(np.logspace(0, math.log10(horizon), int(num_checkpoints))).astype(np.int64)
    )
    pts = pts[(pts >= 1) & (pts <= horizon)]
    if pts.size == 0 or pts[-1] != horizon:
        pts = np.unique(np.concatenate([pts, np.array([horizon], dtype=np.int64)]))
    return pts


def style_axis(ax, *, log_grid: bool = False) -> None:
    # Less dense grid: major only
    if log_grid:
        ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
        ax.grid(False, which="minor")
    else:
        ax.grid(True, which="major", alpha=0.28, linewidth=0.8)


def stable_policy_probabilities(
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta_other: np.ndarray,
    num_other_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_score = np.maximum(np.maximum(theta1, theta2), theta_other)

    w1 = np.exp(theta1 - max_score)
    w2 = np.exp(theta2 - max_score)
    wo = np.exp(theta_other - max_score)

    denom = w1 + w2 + num_other_arms * wo

    p1 = w1 / denom
    p2 = w2 / denom
    po = wo / denom
    return p1, p2, po


def eta_value(schedule: str, step_index_zero_based, delta: float) -> np.ndarray:
    t = np.asarray(step_index_zero_based, dtype=np.float64)

    if schedule == "inv_sqrt_tplus1":
        return 1.0 / np.sqrt(t + 1.0)

    if schedule == "constant_delta_squared":
        return np.full_like(t, float(delta) ** 2, dtype=np.float64)

    raise ValueError(f"Unknown schedule: {schedule}")


def harmonic_number(n: int) -> float:
    n = int(n)
    if n <= 0:
        return 0.0

    if n < 200_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))

    gamma = 0.5772156649015328606
    x = float(n)
    return math.log(x) + gamma + 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)


def eta_sums_for_block(
    schedule: str,
    start_step: int,
    block_size: int,
    delta: float,
) -> Tuple[float, float]:
    if block_size <= 0:
        return 0.0, 0.0

    if schedule == "constant_delta_squared":
        eta = float(delta) ** 2
        return block_size * eta, block_size * eta * eta

    if schedule != "inv_sqrt_tplus1":
        raise ValueError(f"Unknown schedule: {schedule}")

    if block_size <= 200_000:
        t = np.arange(start_step, start_step + block_size, dtype=np.float64)
        eta = 1.0 / np.sqrt(t + 1.0)
        return float(np.sum(eta)), float(np.sum(eta * eta))

    sum_eta = 2.0 * (
        math.sqrt(start_step + block_size + 1.0) - math.sqrt(start_step + 1.0)
    )
    sum_eta_sq = harmonic_number(start_step + block_size) - harmonic_number(start_step)
    return float(sum_eta), float(sum_eta_sq)


def choose_block_size(
    *,
    schedule: str,
    current_step: int,
    remaining_steps: int,
    delta: float,
    p1: np.ndarray,
    p2: np.ndarray,
    po: np.ndarray,
    num_other_arms: int,
    max_mean_change: float,
    max_noise_change: float,
    max_block_size: int,
    block_quantile: float,
) -> int:
    eta_now = float(eta_value(schedule, current_step, delta))
    if eta_now <= 0.0:
        return int(min(remaining_steps, max_block_size))

    inst_regret = num_other_arms * po + float(delta) * p2

    grad1 = p1 * inst_regret
    grad2 = p2 * (inst_regret - float(delta))
    grad_other = -(grad1 + grad2) / float(num_other_arms)

    grad_abs = np.maximum.reduce([np.abs(grad1), np.abs(grad2), np.abs(grad_other)])
    mean_grad_scale = float(np.quantile(grad_abs, float(block_quantile)))

    mu2 = 1.0 - float(delta)
    ey1_sq = 2.0
    ey2_sq = mu2 * mu2 + 1.0

    var1 = p1 * (1.0 - p1) ** 2 * ey1_sq + p2 * p1 ** 2 * ey2_sq
    var2 = p1 * p2 ** 2 * ey1_sq + p2 * (1.0 - p2) ** 2 * ey2_sq
    varo = p1 * po ** 2 * ey1_sq + p2 * po ** 2 * ey2_sq

    variance_scale = float(np.quantile(np.maximum.reduce([var1, var2, varo]), float(block_quantile)))

    limit_by_mean = math.inf
    if mean_grad_scale > 1e-300:
        limit_by_mean = float(max_mean_change) / (eta_now * mean_grad_scale)

    limit_by_noise = math.inf
    if variance_scale > 1e-300:
        limit_by_noise = (float(max_noise_change) / (eta_now * math.sqrt(variance_scale))) ** 2

    block_size = int(
        max(
            1,
            math.floor(
                min(
                    int(remaining_steps),
                    int(max_block_size),
                    limit_by_mean,
                    limit_by_noise,
                )
            ),
        )
    )
    return block_size


def run_exact_steps(
    *,
    schedule: str,
    num_steps: int,
    current_step: int,
    rng: np.random.Generator,
    delta: float,
    num_other_arms: int,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta_other: np.ndarray,
) -> None:
    n = theta1.size
    mu2 = 1.0 - float(delta)

    for local_step in range(int(num_steps)):
        t = current_step + local_step
        eta = float(eta_value(schedule, t, delta))

        p1, p2, po = stable_policy_probabilities(theta1, theta2, theta_other, num_other_arms)

        if eta <= 0.0:
            continue

        u = rng.random(n)
        choose1 = u < p1
        choose2 = (u >= p1) & (u < p1 + p2)

        reward = np.zeros(n, dtype=np.float64)

        c1 = int(np.sum(choose1))
        c2 = int(np.sum(choose2))

        if c1 > 0:
            reward[choose1] = 1.0 + rng.normal(size=c1)

        if c2 > 0:
            reward[choose2] = mu2 + rng.normal(size=c2)

        theta1 += eta * reward * (choose1.astype(np.float64) - p1)
        theta2 += eta * reward * (choose2.astype(np.float64) - p2)
        theta_other += eta * reward * (-po)


def run_gaussian_approx_block(
    *,
    schedule: str,
    block_size: int,
    current_step: int,
    rng: np.random.Generator,
    delta: float,
    num_other_arms: int,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta_other: np.ndarray,
) -> None:
    p1, p2, po = stable_policy_probabilities(theta1, theta2, theta_other, num_other_arms)

    sum_eta, sum_eta_sq = eta_sums_for_block(schedule, current_step, block_size, delta)
    if sum_eta == 0.0 and sum_eta_sq == 0.0:
        return

    mu1 = 1.0
    mu2 = 1.0 - float(delta)

    ey1_sq = 2.0
    ey2_sq = mu2 * mu2 + 1.0

    mean_s1 = p1 * mu1 * sum_eta
    mean_s2 = p2 * mu2 * sum_eta

    var_s1 = (p1 * ey1_sq - (p1 * mu1) ** 2) * sum_eta_sq
    var_s2 = (p2 * ey2_sq - (p2 * mu2) ** 2) * sum_eta_sq
    cov_s1_s2 = -(p1 * mu1) * (p2 * mu2) * sum_eta_sq

    var_s1 = np.maximum(var_s1, 0.0)
    var_s2 = np.maximum(var_s2, 0.0)

    z1 = rng.normal(size=p1.size)
    z2 = rng.normal(size=p1.size)

    sqrt_var_s1 = np.sqrt(var_s1)
    wr1 = mean_s1 + sqrt_var_s1 * z1

    coeff = np.zeros_like(var_s1)
    safe = var_s1 > 1e-300
    coeff[safe] = cov_s1_s2[safe] / sqrt_var_s1[safe]

    cond_var_s2 = np.maximum(var_s2 - coeff ** 2, 0.0)
    wr2 = mean_s2 + coeff * z1 + np.sqrt(cond_var_s2) * z2

    theta1 += (1.0 - p1) * wr1 - p1 * wr2
    theta2 += -p2 * wr1 + (1.0 - p2) * wr2
    theta_other += -po * (wr1 + wr2)


def simulate_pi1_paths(
    *,
    num_arms: int,
    delta: float,
    horizon: int,
    checkpoints: Sequence[int],
    num_paths: int,
    seed: int,
    schedule: str,
    max_mean_change: float,
    max_noise_change: float,
    max_block_size: int,
    block_quantile: float,
    exact_small_block_threshold: int,
) -> np.ndarray:
    num_other_arms = int(num_arms) - 2
    if num_other_arms < 1:
        raise ValueError("num_arms must be at least 3")

    checkpoints = np.array(sorted(set(int(x) for x in checkpoints)), dtype=np.int64)
    if checkpoints[-1] != int(horizon):
        checkpoints = np.unique(np.concatenate([checkpoints, np.array([int(horizon)], dtype=np.int64)]))

    rng = np.random.default_rng(int(seed))
    n = int(num_paths)

    theta1 = np.zeros(n, dtype=np.float64)
    theta2 = np.zeros(n, dtype=np.float64)
    theta_other = np.zeros(n, dtype=np.float64)

    pi1_paths = np.zeros((len(checkpoints), n), dtype=np.float64)

    current_step = 0
    record_index = 0
    horizon = int(horizon)

    while current_step < horizon:
        next_checkpoint = int(checkpoints[record_index])
        remaining_to_horizon = horizon - current_step
        remaining_to_checkpoint = next_checkpoint - current_step

        if remaining_to_checkpoint <= 0:
            p1, _, _ = stable_policy_probabilities(theta1, theta2, theta_other, num_other_arms)
            pi1_paths[record_index, :] = p1
            record_index += 1
            if record_index >= len(checkpoints):
                break
            continue

        p1, p2, po = stable_policy_probabilities(theta1, theta2, theta_other, num_other_arms)

        block_size = choose_block_size(
            schedule=schedule,
            current_step=current_step,
            remaining_steps=remaining_to_horizon,
            delta=delta,
            p1=p1,
            p2=p2,
            po=po,
            num_other_arms=num_other_arms,
            max_mean_change=max_mean_change,
            max_noise_change=max_noise_change,
            max_block_size=max_block_size,
            block_quantile=block_quantile,
        )
        block_size = min(block_size, remaining_to_checkpoint)

        if block_size <= int(exact_small_block_threshold):
            run_exact_steps(
                schedule=schedule,
                num_steps=block_size,
                current_step=current_step,
                rng=rng,
                delta=delta,
                num_other_arms=num_other_arms,
                theta1=theta1,
                theta2=theta2,
                theta_other=theta_other,
            )
        else:
            run_gaussian_approx_block(
                schedule=schedule,
                block_size=block_size,
                current_step=current_step,
                rng=rng,
                delta=delta,
                num_other_arms=num_other_arms,
                theta1=theta1,
                theta2=theta2,
                theta_other=theta_other,
            )

        current_step += int(block_size)

        while record_index < len(checkpoints) and current_step >= int(checkpoints[record_index]):
            p1, _, _ = stable_policy_probabilities(theta1, theta2, theta_other, num_other_arms)
            pi1_paths[record_index, :] = p1
            record_index += 1
            if record_index >= len(checkpoints):
                break

    return pi1_paths



# -----------------------------------------------------------------------------
# Four-gap 1x4 plotting code
# -----------------------------------------------------------------------------

FOUR_PANEL_CASES = [
    {"delta": 0.2,   "horizon": 100_000_000_000},          # 10^11
    {"delta": 0.05,  "horizon": 100_000_000_000_000},          # 10^11
    {"delta": 0.02,  "horizon": 100_000_000_000_000},          # 10^11
    {"delta": 0.002, "horizon": 1_000_000_000_000_000_000_000},    # 10^15
]

DEFAULT_NUM_ARMS = 3
DEFAULT_NUM_PATHS = 40
DEFAULT_NUM_CHECKPOINTS = 500
DEFAULT_SCHEDULE = "inv_sqrt_tplus1"


def case_data_path(outdir: Path, case_index: int) -> Path:
    case = FOUR_PANEL_CASES[int(case_index)]
    delta = case["delta"]
    return outdir / f"case_{case_index}_delta_{delta:g}.npz"


def simulate_one_four_panel_case(args, case_index: int) -> Path:
    if case_index < 0 or case_index >= len(FOUR_PANEL_CASES):
        raise ValueError("case_index must be 0, 1, 2, or 3")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    case = FOUR_PANEL_CASES[int(case_index)]
    delta = float(case["delta"])
    horizon = int(case["horizon"])

    checkpoints = make_log_checkpoints(horizon, int(args.num_checkpoints))

    seed = int(args.seed) + 100_003 * int(case_index)

    print(
        f"Running case {case_index}: "
        f"K={args.num_arms}, Delta={delta:g}, horizon={horizon}, "
        f"paths={args.num_paths}, schedule={DEFAULT_SCHEDULE}"
    )

    pi1_paths = simulate_pi1_paths(
        num_arms=int(args.num_arms),
        delta=delta,
        horizon=horizon,
        checkpoints=checkpoints,
        num_paths=int(args.num_paths),
        seed=seed,
        schedule=DEFAULT_SCHEDULE,
        max_mean_change=float(args.max_mean_change),
        max_noise_change=float(args.max_noise_change),
        max_block_size=int(args.max_block_size),
        block_quantile=float(args.block_quantile),
        exact_small_block_threshold=int(args.exact_small_block_threshold),
    )

    output_path = case_data_path(outdir, case_index)

    np.savez_compressed(
        output_path,
        checkpoints=checkpoints,
        pi1_paths=pi1_paths,
        delta=delta,
        horizon=horizon,
        num_arms=int(args.num_arms),
        num_paths=int(args.num_paths),
        schedule=DEFAULT_SCHEDULE,
    )

    print(f"Wrote {output_path}")
    return output_path


def plot_four_panel_pi1(args) -> Path:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22.0, 5.0),
        dpi=180,
        sharey=True,
    )

    for case_index, ax in enumerate(axes):
        data_path = case_data_path(outdir, case_index)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing {data_path}. "
                f"Run case {case_index} first, or run without --plot-only."
            )

        data = np.load(data_path)

        checkpoints = data["checkpoints"].astype(np.float64)
        pi1_paths = np.clip(data["pi1_paths"], 0.0, 1.0)
        delta = float(data["delta"])
        horizon = int(data["horizon"])

        average_path = np.mean(pi1_paths, axis=1)

        for path_index in range(pi1_paths.shape[1]):
            ax.plot(
                checkpoints,
                pi1_paths[:, path_index],
                color="0.72",
                alpha=0.45,
                linewidth=0.9,
            )

        ax.plot(
            checkpoints,
            average_path,
            color="red",
            linewidth=2.8,
            label="average",
        )

        ax.set_xscale("log")
        ax.set_xlim(1, horizon)
        ax.set_ylim(0.0, 1.0)

        ax.set_title(rf"$\Delta={delta:g}$", fontsize=16)

        # Major grid only, less dense.
        ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
        ax.grid(False, which="minor")

        # No repeated per-panel x-labels.
        ax.set_xlabel("")

        if case_index == 0:
            ax.set_ylabel(r"$\pi_t(1)$", fontsize=15)

        if case_index == len(axes) - 1:
            ax.legend(loc="lower right", frameon=True)

    # One shared x-axis label. No global title.
    fig.supxlabel("Time", fontsize=16)

    fig.tight_layout(rect=(0.02, 0.04, 1.0, 1.0))

    output_path = outdir / "pi1_four_gaps_1x4_linear.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1x4 sample-path plot for pi_t(1), K=3, eta_t=1/sqrt(t+1)."
    )

    parser.add_argument("--outdir", default="pi1_four_gaps_results")
    parser.add_argument("--num-arms", type=int, default=DEFAULT_NUM_ARMS)
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--num-checkpoints", type=int, default=DEFAULT_NUM_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only combine existing case_*.npz files into the 1x4 plot.",
    )

    parser.add_argument("--max-mean-change", type=float, default=0.08)
    parser.add_argument("--max-noise-change", type=float, default=0.35)
    parser.add_argument("--max-block-size", type=int, default=10_000_000_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-small-block-threshold", type=int, default=64)

    args = parser.parse_args()

    if args.plot_only:
        plot_four_panel_pi1(args)
        return

    if args.case_index is not None:
        simulate_one_four_panel_case(args, int(args.case_index))
        return

    for case_index in range(len(FOUR_PANEL_CASES)):
        simulate_one_four_panel_case(args, case_index)

    plot_four_panel_pi1(args)


if __name__ == "__main__":
    main()
