#!/usr/bin/env python3
"""
High-throughput gap sweep for softmax policy gradient on the lower-bound bandit
instance from Lattimore (2026), Appendix F / Theorem 10.

Default experiment:
    k = 40 arms
    horizon n = 1e9
    mu = (1, 1-Delta, 0, ..., 0)
    sigma = (1, 1, 0, ..., 0)
    eta_t = 1 / sqrt(t)
    50,000 Monte Carlo trajectories per gap

The script is designed for SLURM job arrays. Each array task can compute one gap
value, using multiprocessing to split trajectories across CPU cores.

Important: the horizon is huge, so this uses a blocked Gaussian aggregate
approximation of Algorithm 1. Within each block the policy is frozen; the sums
of eta_t-weighted rewards for arms 1 and 2 are sampled from their matching
2-dimensional Gaussian approximation. This is much faster than literal
round-by-round simulation and is appropriate for large-scale sweeps.

Examples:
    # One gap value, e.g. SLURM_ARRAY_TASK_ID=17
    python pg_gap_sweep_hpc.py --gap-index 17 --num-gaps 101 --workers 8

    # All gaps on one node/process pool; useful for testing only
    python pg_gap_sweep_hpc.py --run-all --num-gaps 11 --trajectories 1000 --workers 4

    # Combine array outputs and make plot
    python pg_gap_sweep_hpc.py --combine --outdir results --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Iterable, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional on compute nodes
    plt = None


EULER_GAMMA = 0.5772156649015328606


@dataclass(frozen=True)
class SimulationConfig:
    horizon_steps: int = 10**9
    num_arms: int = 40
    num_trajectories: int = 50_000
    random_seed: int = 20260310

    # Blocked approximation controls. Smaller values are more faithful but slower.
    max_mean_score_change_per_block: float = 0.20
    max_noise_score_change_per_block: float = 0.80
    max_block_size: int = 50_000_000

    # Use a high quantile rather than the absolute worst trajectory to choose
    # block sizes. This avoids one extreme trajectory forcing tiny blocks for all
    # other trajectories. Set to 1.0 for the conservative max rule.
    block_quantile: float = 0.995

    # Exact summation threshold for eta_t sums inside a block.
    # Larger values are slightly more accurate and slightly slower.
    exact_eta_sum_threshold: int = 4096

    # Stop if the current gap's output file already exists.
    skip_existing: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast gap sweep for policy-gradient bandit with eta_t=1/sqrt(t)."
    )

    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--gap-index", type=int, help="Index of gap to run, 0-based.")
    run_group.add_argument("--run-all", action="store_true", help="Run all gap values in this process.")
    run_group.add_argument("--combine", action="store_true", help="Combine per-gap CSV files.")

    parser.add_argument("--gap-start", type=float, default=0.0)
    parser.add_argument("--gap-stop", type=float, default=1.0)
    parser.add_argument("--num-gaps", type=int, default=101)

    parser.add_argument("--horizon", type=int, default=10**9)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-file", type=str, default="gap_sweep_eta_inv_sqrt.png")
    parser.add_argument("--regret-scale", type=float, default=1_000_000.0)

    parser.add_argument("--max-mean-change", type=float, default=0.20)
    parser.add_argument("--max-noise-change", type=float, default=0.80)
    parser.add_argument("--max-block-size", type=int, default=50_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995,
                        help="High quantile used for adaptive block-size scales; use 1.0 for max.")
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def gap_grid(gap_start: float, gap_stop: float, num_gaps: int) -> np.ndarray:
    if num_gaps < 2:
        raise ValueError("--num-gaps must be at least 2")
    return np.linspace(gap_start, gap_stop, num_gaps, dtype=np.float64)


def harmonic_approx(n: int) -> float:
    """Fast approximation to H_n = sum_{t=1}^n 1/t."""
    if n <= 0:
        return 0.0
    # Exact for small n avoids approximation artifacts at the beginning.
    if n <= 10_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    x = float(n)
    x2 = x * x
    return math.log(x) + EULER_GAMMA + 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)


def euler_maclaurin_sum_inv_sqrt(start_time: int, block_size: int) -> float:
    """Approximate sum_{t=start}^{start+B-1} t^{-1/2}."""
    a = float(start_time)
    b = float(start_time + block_size - 1)

    # Euler-Maclaurin for f(x)=x^{-1/2} over integer points a..b.
    integral = 2.0 * (math.sqrt(b) - math.sqrt(a))
    endpoints = 0.5 * (a ** -0.5 + b ** -0.5)
    derivative_correction = (1.0 / 12.0) * (-0.5 * b ** -1.5 + 0.5 * a ** -1.5)
    return integral + endpoints + derivative_correction


def learning_rate_sums(start_time: int, block_size: int, exact_threshold: int) -> Tuple[float, float]:
    """Return sum eta_t and sum eta_t^2 for eta_t=1/sqrt(t) over one block."""
    if block_size <= exact_threshold:
        t = np.arange(start_time, start_time + block_size, dtype=np.float64)
        inv_sqrt = 1.0 / np.sqrt(t)
        return float(np.sum(inv_sqrt)), float(np.sum(inv_sqrt * inv_sqrt))

    end_time = start_time + block_size - 1
    sum_eta = euler_maclaurin_sum_inv_sqrt(start_time, block_size)
    sum_eta_squared = harmonic_approx(end_time) - harmonic_approx(start_time - 1)
    return sum_eta, sum_eta_squared


def stable_action_probabilities(
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    num_other_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Softmax probabilities for arm 1, arm 2, and each identical arm 3..k."""
    other_arm_score = -(optimal_arm_score + second_arm_score) / num_other_arms

    max_score = np.maximum(np.maximum(optimal_arm_score, second_arm_score), other_arm_score)
    optimal_weight = np.exp(optimal_arm_score - max_score)
    second_weight = np.exp(second_arm_score - max_score)
    other_weight = np.exp(other_arm_score - max_score)

    total_weight = optimal_weight + second_weight + num_other_arms * other_weight
    optimal_prob = optimal_weight / total_weight
    second_prob = second_weight / total_weight
    other_prob_per_arm = other_weight / total_weight

    return optimal_prob, second_prob, other_prob_per_arm, other_arm_score


def high_quantile_scale(values: np.ndarray, quantile: float) -> float:
    """Return a high quantile of nonnegative scale values.

    quantile=1.0 returns the exact maximum. For large Monte Carlo batches, a
    quantile such as 0.995 prevents one extreme trajectory from making all
    blocks tiny.
    """
    if values.size == 0:
        return 0.0
    if quantile >= 1.0:
        return float(np.max(values))
    if quantile <= 0.0:
        return float(np.min(values))
    kth = int(math.ceil(quantile * (values.size - 1)))
    return float(np.partition(values, kth)[kth])


def choose_block_size(
    *,
    current_time: int,
    remaining_steps: int,
    gap_arm2: float,
    optimal_prob: np.ndarray,
    second_prob: np.ndarray,
    other_prob_per_arm: np.ndarray,
    num_other_arms: int,
    config: SimulationConfig,
) -> int:
    """Conservative adaptive block size for the frozen-policy approximation."""
    eta_now = 1.0 / math.sqrt(float(current_time))
    second_arm_mean = 1.0 - gap_arm2
    other_total_prob = num_other_arms * other_prob_per_arm

    instantaneous_regret = other_total_prob + gap_arm2 * second_prob

    mean_gradient_optimal = optimal_prob * instantaneous_regret
    mean_gradient_second = second_prob * (instantaneous_regret - gap_arm2)
    mean_gradient_other = -(mean_gradient_optimal + mean_gradient_second) / num_other_arms

    block_quantile = config.block_quantile
    mean_gradient_scale = max(
        high_quantile_scale(np.abs(mean_gradient_optimal), block_quantile),
        high_quantile_scale(np.abs(mean_gradient_second), block_quantile),
        high_quantile_scale(np.abs(mean_gradient_other), block_quantile),
    )

    # Upper bound using second moments of per-round gradient coordinates.
    # Arm 1 reward: N(1, 1), so E[Y^2] = 2.
    # Arm 2 reward: N(1-Delta, 1), so E[Y^2] = (1-Delta)^2 + 1.
    second_moment_arm1_reward = 2.0
    second_moment_arm2_reward = second_arm_mean * second_arm_mean + 1.0

    second_moment_gradient_optimal = (
        optimal_prob * (1.0 - optimal_prob) ** 2 * second_moment_arm1_reward
        + second_prob * optimal_prob**2 * second_moment_arm2_reward
    )
    second_moment_gradient_second = (
        optimal_prob * second_prob**2 * second_moment_arm1_reward
        + second_prob * (1.0 - second_prob) ** 2 * second_moment_arm2_reward
    )
    second_moment_gradient_other = (
        optimal_prob * other_prob_per_arm**2 * second_moment_arm1_reward
        + second_prob * other_prob_per_arm**2 * second_moment_arm2_reward
    )

    noise_gradient_scale = max(
        high_quantile_scale(second_moment_gradient_optimal, block_quantile),
        high_quantile_scale(second_moment_gradient_second, block_quantile),
        high_quantile_scale(second_moment_gradient_other, block_quantile),
    )

    limit_by_mean_change = math.inf
    if mean_gradient_scale > 1e-300:
        limit_by_mean_change = config.max_mean_score_change_per_block / (eta_now * mean_gradient_scale)

    limit_by_noise_change = math.inf
    if noise_gradient_scale > 1e-300:
        limit_by_noise_change = (
            config.max_noise_score_change_per_block / (eta_now * math.sqrt(noise_gradient_scale))
        ) ** 2

    return int(
        max(
            1,
            math.floor(
                min(
                    remaining_steps,
                    config.max_block_size,
                    limit_by_mean_change,
                    limit_by_noise_change,
                )
            ),
        )
    )


def sample_weighted_reward_sums_gaussian(
    *,
    random_generator: np.random.Generator,
    optimal_prob: np.ndarray,
    second_prob: np.ndarray,
    gap_arm2: float,
    sum_eta: float,
    sum_eta_squared: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample aggregate eta-weighted reward sums for arms 1 and 2.

    For each block with frozen policy, approximate
        S_j = sum_t eta_t 1{A_t=j} Y_t
    by its matching bivariate normal distribution.
    """
    mu1 = 1.0
    mu2 = 1.0 - gap_arm2
    second_moment_reward1 = 2.0               # sigma^2 + mu^2 = 1 + 1^2
    second_moment_reward2 = 1.0 + mu2 * mu2   # sigma^2 + mu^2

    mean1 = optimal_prob * mu1 * sum_eta
    mean2 = second_prob * mu2 * sum_eta

    var1 = (optimal_prob * second_moment_reward1 - (optimal_prob * mu1) ** 2) * sum_eta_squared
    var2 = (second_prob * second_moment_reward2 - (second_prob * mu2) ** 2) * sum_eta_squared
    cov12 = -(optimal_prob * mu1) * (second_prob * mu2) * sum_eta_squared

    # Numerical safety.
    var1 = np.maximum(var1, 0.0)
    var2 = np.maximum(var2, 0.0)

    z1 = random_generator.standard_normal(size=optimal_prob.shape[0])
    z2 = random_generator.standard_normal(size=optimal_prob.shape[0])

    sd1 = np.sqrt(var1)
    sd2 = np.sqrt(var2)

    denom = sd1 * sd2
    correlation = np.divide(cov12, denom, out=np.zeros_like(cov12), where=denom > 1e-300)
    np.clip(correlation, -0.999999999, 0.999999999, out=correlation)

    weighted_reward_sum1 = mean1 + sd1 * z1
    weighted_reward_sum2 = mean2 + sd2 * (correlation * z1 + np.sqrt(1.0 - correlation * correlation) * z2)

    return weighted_reward_sum1, weighted_reward_sum2


def simulate_chunk(
    gap_arm2: float,
    chunk_trajectories: int,
    seed: int,
    config: SimulationConfig,
) -> Tuple[int, float, float, int, float]:
    """Simulate a chunk of trajectories for one gap.

    Returns:
        n, sum_regret, sum_regret_squared, num_blocks, seconds
    """
    start_wall_time = time.time()
    random_generator = np.random.default_rng(seed)
    num_other_arms = config.num_arms - 2

    optimal_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    second_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    regret_by_trajectory = np.zeros(chunk_trajectories, dtype=np.float64)

    current_time = 1
    num_blocks = 0

    while current_time <= config.horizon_steps:
        optimal_prob, second_prob, other_prob_per_arm, _ = stable_action_probabilities(
            optimal_arm_score,
            second_arm_score,
            num_other_arms,
        )

        remaining_steps = config.horizon_steps - current_time + 1
        block_size = choose_block_size(
            current_time=current_time,
            remaining_steps=remaining_steps,
            gap_arm2=gap_arm2,
            optimal_prob=optimal_prob,
            second_prob=second_prob,
            other_prob_per_arm=other_prob_per_arm,
            num_other_arms=num_other_arms,
            config=config,
        )

        other_total_prob = 1.0 - optimal_prob - second_prob
        instantaneous_regret = other_total_prob + gap_arm2 * second_prob
        regret_by_trajectory += block_size * instantaneous_regret

        sum_eta, sum_eta_squared = learning_rate_sums(
            current_time,
            block_size,
            config.exact_eta_sum_threshold,
        )

        weighted_reward_sum1, weighted_reward_sum2 = sample_weighted_reward_sums_gaussian(
            random_generator=random_generator,
            optimal_prob=optimal_prob,
            second_prob=second_prob,
            gap_arm2=gap_arm2,
            sum_eta=sum_eta,
            sum_eta_squared=sum_eta_squared,
        )

        # Algorithm 1 update aggregated over the block.
        optimal_arm_score += (
            (1.0 - optimal_prob) * weighted_reward_sum1
            - optimal_prob * weighted_reward_sum2
        )
        second_arm_score += (
            -second_prob * weighted_reward_sum1
            + (1.0 - second_prob) * weighted_reward_sum2
        )

        current_time += block_size
        num_blocks += 1

    elapsed_seconds = time.time() - start_wall_time
    sum_regret = float(np.sum(regret_by_trajectory))
    sum_regret_squared = float(np.sum(regret_by_trajectory * regret_by_trajectory))
    return chunk_trajectories, sum_regret, sum_regret_squared, num_blocks, elapsed_seconds


def split_trajectories(total: int, workers: int, chunk_trajectories: int | None) -> List[int]:
    if chunk_trajectories is not None and chunk_trajectories > 0:
        chunks = []
        remaining = total
        while remaining > 0:
            n = min(chunk_trajectories, remaining)
            chunks.append(n)
            remaining -= n
        return chunks

    workers = max(1, min(workers, total))
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


def default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def run_one_gap(
    *,
    gap_index: int,
    gap_arm2: float,
    args: argparse.Namespace,
    config: SimulationConfig,
) -> pathlib.Path:
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"gap_{gap_index:04d}.csv"
    metadata_path = outdir / f"gap_{gap_index:04d}.json"

    if output_path.exists() and config.skip_existing:
        print(f"[skip] {output_path} already exists", flush=True)
        return output_path

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)
    chunks = split_trajectories(config.num_trajectories, workers, args.chunk_trajectories)

    print(
        f"[run] gap_index={gap_index} gap={gap_arm2:.10g} "
        f"trajectories={config.num_trajectories} chunks={chunks} workers={workers}",
        flush=True,
    )

    total_n = 0
    total_sum = 0.0
    total_sum_squared = 0.0
    total_blocks = 0
    max_chunk_seconds = 0.0
    start_wall_time = time.time()

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_id, chunk_n in enumerate(chunks):
            chunk_seed = int(config.random_seed + 1_000_003 * gap_index + 9_176 * chunk_id)
            futures.append(
                executor.submit(
                    simulate_chunk,
                    float(gap_arm2),
                    int(chunk_n),
                    chunk_seed,
                    config,
                )
            )

        for future in as_completed(futures):
            chunk_n, sum_regret, sum_regret_squared, num_blocks, elapsed_seconds = future.result()
            total_n += chunk_n
            total_sum += sum_regret
            total_sum_squared += sum_regret_squared
            total_blocks += num_blocks
            max_chunk_seconds = max(max_chunk_seconds, elapsed_seconds)
            print(
                f"  finished chunk: n={chunk_n} mean_regret={sum_regret/chunk_n:.6e} "
                f"blocks={num_blocks} seconds={elapsed_seconds:.1f}",
                flush=True,
            )

    mean_regret = total_sum / total_n
    if total_n > 1:
        # Unbiased sample variance from aggregate sums.
        variance = (total_sum_squared - total_sum * total_sum / total_n) / (total_n - 1)
        variance = max(0.0, variance)
        standard_error = math.sqrt(variance / total_n)
    else:
        standard_error = float("nan")

    elapsed_total = time.time() - start_wall_time
    row = {
        "gap_index": gap_index,
        "gap_arm2": float(gap_arm2),
        "mean_regret": mean_regret,
        "standard_error": standard_error,
        "num_trajectories": total_n,
        "num_chunks": len(chunks),
        "workers": workers,
        "mean_blocks_per_chunk": total_blocks / len(chunks),
        "max_chunk_seconds": max_chunk_seconds,
        "wall_seconds": elapsed_total,
        "horizon_steps": config.horizon_steps,
        "num_arms": config.num_arms,
        "eta_schedule": "eta_t = 1/sqrt(t)",
        "simulation": "blocked_gaussian_aggregate",
    }

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with metadata_path.open("w") as f:
        json.dump({"config": asdict(config), "row": row}, f, indent=2)

    print(f"[done] wrote {output_path} mean_regret={mean_regret:.6e} se={standard_error:.6e}", flush=True)
    return output_path


def read_single_row_csv(path: pathlib.Path) -> dict:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {path}, found {len(rows)}")
    return rows[0]


def combine_results(args: argparse.Namespace) -> pathlib.Path:
    outdir = pathlib.Path(args.outdir)
    paths = sorted(outdir.glob("gap_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No gap_*.csv files found in {outdir}")

    rows = [read_single_row_csv(path) for path in paths]
    rows.sort(key=lambda r: int(r["gap_index"]))

    combined_path = outdir / "combined_gap_sweep.csv"
    fieldnames = list(rows[0].keys())
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[combine] wrote {combined_path} with {len(rows)} rows", flush=True)

    if args.plot:
        if plt is None:
            print("[plot] matplotlib unavailable; skipping plot", file=sys.stderr)
        else:
            plot_combined_results(combined_path, args)

    return combined_path


def plot_combined_results(combined_path: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    data = np.genfromtxt(combined_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    gap = np.asarray(data["gap_arm2"], dtype=np.float64)
    mean_regret = np.asarray(data["mean_regret"], dtype=np.float64)
    standard_error = np.asarray(data["standard_error"], dtype=np.float64)

    order = np.argsort(gap)
    gap = gap[order]
    mean_regret = mean_regret[order]
    standard_error = standard_error[order]

    scale = float(args.regret_scale)
    y = mean_regret / scale
    se = standard_error / scale

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    ax.plot(gap, y, linewidth=2.4, label=r"$\eta_t = 1/\sqrt{t}$")
    ax.fill_between(gap, y - 2.0 * se, y + 2.0 * se, alpha=0.18, linewidth=0)

    ax.set_xlim(float(np.min(gap)), float(np.max(gap)))
    ax.set_xlabel(r"$\Delta$")
    if scale == 1.0:
        ax.set_ylabel("Expected regret")
    else:
        ax.set_ylabel(rf"Expected regret / ${scale:.0e}$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()

    plot_path = pathlib.Path(args.outdir) / args.plot_file
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[plot] wrote {plot_path}", flush=True)
    return plot_path


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        horizon_steps=args.horizon,
        num_arms=args.num_arms,
        num_trajectories=args.trajectories,
        random_seed=args.seed,
        max_mean_score_change_per_block=args.max_mean_change,
        max_noise_score_change_per_block=args.max_noise_change,
        max_block_size=args.max_block_size,
        exact_eta_sum_threshold=args.exact_eta_sum_threshold,
        block_quantile=args.block_quantile,
        skip_existing=not args.overwrite,
    )

    if args.combine:
        combine_results(args)
        return

    gaps = gap_grid(args.gap_start, args.gap_stop, args.num_gaps)

    if args.gap_index is not None:
        if args.gap_index < 0 or args.gap_index >= len(gaps):
            raise IndexError(f"gap-index must be between 0 and {len(gaps)-1}")
        run_one_gap(
            gap_index=args.gap_index,
            gap_arm2=float(gaps[args.gap_index]),
            args=args,
            config=config,
        )
        return

    if args.run_all:
        for gap_index, gap_arm2 in enumerate(gaps):
            run_one_gap(
                gap_index=gap_index,
                gap_arm2=float(gap_arm2),
                args=args,
                config=config,
            )
        return


if __name__ == "__main__":
    main()
