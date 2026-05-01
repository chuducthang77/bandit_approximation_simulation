#!/usr/bin/env python3
"""
Find the hardest gap from a previous gap sweep, then rerun the lower-bound
bandit instance at that gap and record the average cumulative regret over time.

Default setup:
    k = 40 arms
    horizon n = 1e9
    mu = (1, 1-Delta, 0, ..., 0)
    sigma = (1, 1, 0, ..., 0)
    eta_t = 1 / sqrt(t)
    50,000 Monte Carlo trajectories

The script expects a completed sweep from pg_gap_sweep_hpc.py, typically:
    results/combined_gap_sweep.csv

It defines "hardest gap" as the gap with the largest final mean_regret in the
sweep CSV. Then it simulates that single gap again and records E[Reg_t] at
log-spaced checkpoints.

Because n can be 1e9, this uses the same blocked Gaussian aggregate
approximation as the gap-sweep code:
    - within a block, the policy is frozen;
    - eta_t-weighted reward sums for arms 1 and 2 are sampled from their
      matching bivariate Gaussian approximation;
    - regret is accumulated exactly under the frozen policy inside the block.

Quick local test:
    python pg_hardest_gap_regret_history.py \
        --sweep-csv results/combined_gap_sweep.csv \
        --horizon 100000 \
        --trajectories 2000 \
        --workers 4 \
        --outdir hardest_gap_results

Production-style run:
    python pg_hardest_gap_regret_history.py \
        --sweep-csv results/combined_gap_sweep.csv \
        --horizon 1000000000 \
        --num-arms 40 \
        --trajectories 50000 \
        --workers ${SLURM_CPUS_PER_TASK:-16} \
        --chunk-trajectories 2500 \
        --num-checkpoints 250 \
        --outdir hardest_gap_results
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
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
    block_quantile: float = 0.995
    exact_eta_sum_threshold: int = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the hardest gap from a sweep, rerun eta_t=1/sqrt(t), "
            "and plot average cumulative regret over time."
        )
    )

    parser.add_argument(
        "--sweep-csv",
        type=str,
        default="results/combined_gap_sweep.csv",
        help="CSV from the gap sweep. Must contain gap_arm2 and mean_regret columns.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="results",
        help="Directory containing gap_*.csv files. Used if --sweep-csv does not exist.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="Override hardest-gap detection and run this gap directly.",
    )

    parser.add_argument("--horizon", type=int, default=10**9)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--outdir", type=str, default="hardest_gap_results")
    parser.add_argument("--history-csv", type=str, default="hardest_gap_regret_history.csv")
    parser.add_argument("--summary-json", type=str, default="hardest_gap_summary.json")
    parser.add_argument("--plot-file", type=str, default="hardest_gap_regret_history.png")
    parser.add_argument("--regret-scale", type=float, default=1_000_000.0)

    parser.add_argument("--num-checkpoints", type=int, default=250)
    parser.add_argument(
        "--checkpoint-mode",
        choices=["log", "linear"],
        default="log",
        help="Checkpoints at which to record average regret.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Optional CSV/text file containing explicit integer checkpoint times.",
    )

    parser.add_argument("--max-mean-change", type=float, default=0.20)
    parser.add_argument("--max-noise-change", type=float, default=0.80)
    parser.add_argument("--max-block-size", type=int, default=50_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=4096)

    parser.add_argument(
        "--no-error-band",
        action="store_true",
        help="Do not draw a +/- 2 standard error band in the plot.",
    )
    parser.add_argument(
        "--plot-average-per-round",
        action="store_true",
        help="Also create a second plot of E[Reg_t] / t.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


def default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def read_csv_rows(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_sweep_rows(sweep_csv: pathlib.Path, sweep_dir: pathlib.Path) -> List[Dict[str, str]]:
    """Read combined_gap_sweep.csv or fall back to per-gap gap_*.csv files."""
    if sweep_csv.exists():
        rows = read_csv_rows(sweep_csv)
        if rows:
            print(f"[sweep] read {len(rows)} rows from {sweep_csv}", flush=True)
            return rows
        raise ValueError(f"Sweep CSV exists but is empty: {sweep_csv}")

    paths = sorted(sweep_dir.glob("gap_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"Could not find {sweep_csv} and no gap_*.csv files were found in {sweep_dir}."
        )

    rows: List[Dict[str, str]] = []
    for path in paths:
        path_rows = read_csv_rows(path)
        if len(path_rows) != 1:
            raise ValueError(f"Expected one row in {path}, found {len(path_rows)}")
        rows.append(path_rows[0])

    rows.sort(key=lambda row: int(row.get("gap_index", 0)))
    print(f"[sweep] read {len(rows)} per-gap rows from {sweep_dir}", flush=True)
    return rows


def find_hardest_gap_from_rows(rows: List[Dict[str, str]]) -> Tuple[float, Dict[str, str]]:
    """Pick the gap with largest final mean_regret in the sweep."""
    if not rows:
        raise ValueError("No sweep rows were supplied.")

    required = {"gap_arm2", "mean_regret"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Sweep rows are missing required columns: {sorted(missing)}")

    def regret_value(row: Dict[str, str]) -> float:
        return float(row["mean_regret"])

    hardest_row = max(rows, key=regret_value)
    hardest_gap = float(hardest_row["gap_arm2"])
    return hardest_gap, hardest_row


def make_checkpoints(
    horizon_steps: int,
    num_checkpoints: int,
    mode: str,
    checkpoint_file: Optional[str],
) -> np.ndarray:
    if checkpoint_file is not None:
        values: List[int] = []
        path = pathlib.Path(checkpoint_file)
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Accept either one integer per line or CSV-style rows.
                for part in line.split(","):
                    part = part.strip()
                    if part:
                        values.append(int(float(part)))
        checkpoints = np.array(values, dtype=np.int64)
    elif mode == "linear":
        checkpoints = np.rint(
            np.linspace(1, horizon_steps, num_checkpoints, dtype=np.float64)
        ).astype(np.int64)
    else:
        checkpoints = np.rint(
            np.geomspace(1, horizon_steps, num_checkpoints, dtype=np.float64)
        ).astype(np.int64)

    checkpoints = np.unique(checkpoints)
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= horizon_steps)]

    if checkpoints.size == 0 or checkpoints[0] != 1:
        checkpoints = np.insert(checkpoints, 0, 1)
    if checkpoints[-1] != horizon_steps:
        checkpoints = np.append(checkpoints, horizon_steps)

    return checkpoints.astype(np.int64)


def harmonic_approx(n: int) -> float:
    """Fast approximation to H_n = sum_{t=1}^n 1/t."""
    if n <= 0:
        return 0.0
    if n <= 10_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    x = float(n)
    x2 = x * x
    return math.log(x) + EULER_GAMMA + 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)


def euler_maclaurin_sum_inv_sqrt(start_time: int, block_size: int) -> float:
    """Approximate sum_{t=start}^{start+B-1} t^{-1/2}."""
    a = float(start_time)
    b = float(start_time + block_size - 1)

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
    """Adaptive block size for the frozen-policy approximation."""
    eta_now = 1.0 / math.sqrt(float(current_time))
    second_arm_mean = 1.0 - gap_arm2
    other_total_prob = num_other_arms * other_prob_per_arm

    instantaneous_regret = other_total_prob + gap_arm2 * second_prob

    mean_gradient_optimal = optimal_prob * instantaneous_regret
    mean_gradient_second = second_prob * (instantaneous_regret - gap_arm2)
    mean_gradient_other = -(mean_gradient_optimal + mean_gradient_second) / num_other_arms

    q = config.block_quantile
    mean_gradient_scale = max(
        high_quantile_scale(np.abs(mean_gradient_optimal), q),
        high_quantile_scale(np.abs(mean_gradient_second), q),
        high_quantile_scale(np.abs(mean_gradient_other), q),
    )

    # Second-moment scale for one-step stochastic gradient coordinates.
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
        high_quantile_scale(second_moment_gradient_optimal, q),
        high_quantile_scale(second_moment_gradient_second, q),
        high_quantile_scale(second_moment_gradient_other, q),
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
    """Sample eta-weighted aggregate reward sums for arms 1 and 2."""
    mu1 = 1.0
    mu2 = 1.0 - gap_arm2
    second_moment_reward1 = 2.0
    second_moment_reward2 = 1.0 + mu2 * mu2

    mean1 = optimal_prob * mu1 * sum_eta
    mean2 = second_prob * mu2 * sum_eta

    var1 = (optimal_prob * second_moment_reward1 - (optimal_prob * mu1) ** 2) * sum_eta_squared
    var2 = (second_prob * second_moment_reward2 - (second_prob * mu2) ** 2) * sum_eta_squared
    cov12 = -(optimal_prob * mu1) * (second_prob * mu2) * sum_eta_squared

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
    weighted_reward_sum2 = mean2 + sd2 * (
        correlation * z1 + np.sqrt(1.0 - correlation * correlation) * z2
    )

    return weighted_reward_sum1, weighted_reward_sum2


def simulate_history_chunk(
    gap_arm2: float,
    chunk_trajectories: int,
    seed: int,
    config: SimulationConfig,
    checkpoints: np.ndarray,
) -> Dict[str, object]:
    """Simulate one chunk and return aggregate regret history statistics."""
    start_wall_time = time.time()
    random_generator = np.random.default_rng(seed)
    num_other_arms = config.num_arms - 2
    num_checkpoints = int(checkpoints.size)

    optimal_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    second_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    cumulative_regret = np.zeros(chunk_trajectories, dtype=np.float64)

    regret_sum_at_checkpoints = np.zeros(num_checkpoints, dtype=np.float64)
    regret_sq_sum_at_checkpoints = np.zeros(num_checkpoints, dtype=np.float64)
    pi1_sum_at_checkpoints = np.zeros(num_checkpoints, dtype=np.float64)
    pi2_sum_at_checkpoints = np.zeros(num_checkpoints, dtype=np.float64)

    current_time = 1
    checkpoint_index = 0
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
        block_end_time = current_time + block_size - 1

        other_total_prob = 1.0 - optimal_prob - second_prob
        instantaneous_regret = other_total_prob + gap_arm2 * second_prob

        # Record checkpoints inside this frozen-policy block.
        while checkpoint_index < num_checkpoints and checkpoints[checkpoint_index] <= block_end_time:
            checkpoint_time = int(checkpoints[checkpoint_index])
            offset = checkpoint_time - current_time + 1
            regret_at_checkpoint = cumulative_regret + offset * instantaneous_regret

            regret_sum_at_checkpoints[checkpoint_index] = float(np.sum(regret_at_checkpoint))
            regret_sq_sum_at_checkpoints[checkpoint_index] = float(
                np.sum(regret_at_checkpoint * regret_at_checkpoint)
            )
            pi1_sum_at_checkpoints[checkpoint_index] = float(np.sum(optimal_prob))
            pi2_sum_at_checkpoints[checkpoint_index] = float(np.sum(second_prob))
            checkpoint_index += 1

        cumulative_regret += block_size * instantaneous_regret

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

    # Safety: all checkpoints should have been filled because the final checkpoint is horizon.
    if checkpoint_index != num_checkpoints:
        raise RuntimeError(
            f"Only filled {checkpoint_index}/{num_checkpoints} checkpoints. "
            f"Last time={current_time}, horizon={config.horizon_steps}."
        )

    return {
        "num_trajectories": chunk_trajectories,
        "regret_sum": regret_sum_at_checkpoints,
        "regret_sq_sum": regret_sq_sum_at_checkpoints,
        "pi1_sum": pi1_sum_at_checkpoints,
        "pi2_sum": pi2_sum_at_checkpoints,
        "num_blocks": num_blocks,
        "elapsed_seconds": elapsed_seconds,
    }


def split_trajectories(total: int, workers: int, chunk_trajectories: Optional[int]) -> List[int]:
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


def simulate_hardest_gap_history(
    gap_arm2: float,
    config: SimulationConfig,
    checkpoints: np.ndarray,
    workers: int,
    chunk_trajectories: Optional[int],
) -> Dict[str, object]:
    chunks = split_trajectories(config.num_trajectories, workers, chunk_trajectories)

    print(
        f"[run] hardest_gap={gap_arm2:.12g} trajectories={config.num_trajectories} "
        f"chunks={chunks} workers={workers} checkpoints={len(checkpoints)}",
        flush=True,
    )

    total_n = 0
    total_regret_sum = np.zeros(checkpoints.size, dtype=np.float64)
    total_regret_sq_sum = np.zeros(checkpoints.size, dtype=np.float64)
    total_pi1_sum = np.zeros(checkpoints.size, dtype=np.float64)
    total_pi2_sum = np.zeros(checkpoints.size, dtype=np.float64)
    total_blocks = 0
    max_chunk_seconds = 0.0
    start_wall_time = time.time()

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_id, chunk_n in enumerate(chunks):
            chunk_seed = int(config.random_seed + 1_000_003 * chunk_id)
            futures.append(
                executor.submit(
                    simulate_history_chunk,
                    float(gap_arm2),
                    int(chunk_n),
                    chunk_seed,
                    config,
                    checkpoints,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            chunk_n = int(result["num_trajectories"])
            total_n += chunk_n
            total_regret_sum += np.asarray(result["regret_sum"], dtype=np.float64)
            total_regret_sq_sum += np.asarray(result["regret_sq_sum"], dtype=np.float64)
            total_pi1_sum += np.asarray(result["pi1_sum"], dtype=np.float64)
            total_pi2_sum += np.asarray(result["pi2_sum"], dtype=np.float64)
            total_blocks += int(result["num_blocks"])
            max_chunk_seconds = max(max_chunk_seconds, float(result["elapsed_seconds"]))
            final_mean = np.asarray(result["regret_sum"], dtype=np.float64)[-1] / chunk_n
            print(
                f"  finished chunk: n={chunk_n} final_mean_regret={final_mean:.6e} "
                f"blocks={result['num_blocks']} seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )

    mean_regret = total_regret_sum / total_n
    if total_n > 1:
        variance = (total_regret_sq_sum - total_regret_sum * total_regret_sum / total_n) / (total_n - 1)
        variance = np.maximum(variance, 0.0)
        standard_error = np.sqrt(variance / total_n)
    else:
        standard_error = np.full(checkpoints.size, np.nan)

    mean_pi1 = total_pi1_sum / total_n
    mean_pi2 = total_pi2_sum / total_n
    wall_seconds = time.time() - start_wall_time

    return {
        "time_step": checkpoints,
        "mean_regret": mean_regret,
        "standard_error": standard_error,
        "mean_pi1": mean_pi1,
        "mean_pi2": mean_pi2,
        "num_trajectories": total_n,
        "total_blocks": total_blocks,
        "mean_blocks_per_chunk": total_blocks / max(1, len(chunks)),
        "max_chunk_seconds": max_chunk_seconds,
        "wall_seconds": wall_seconds,
        "num_chunks": len(chunks),
    }


def write_history_csv(path: pathlib.Path, history: Dict[str, object]) -> None:
    time_step = np.asarray(history["time_step"], dtype=np.int64)
    mean_regret = np.asarray(history["mean_regret"], dtype=np.float64)
    standard_error = np.asarray(history["standard_error"], dtype=np.float64)
    mean_pi1 = np.asarray(history["mean_pi1"], dtype=np.float64)
    mean_pi2 = np.asarray(history["mean_pi2"], dtype=np.float64)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_step",
            "mean_regret",
            "standard_error",
            "lower_2se",
            "upper_2se",
            "mean_regret_per_round",
            "mean_pi1",
            "mean_pi2",
        ])
        for t, m, se, p1, p2 in zip(time_step, mean_regret, standard_error, mean_pi1, mean_pi2):
            writer.writerow([
                int(t),
                float(m),
                float(se),
                float(m - 2.0 * se),
                float(m + 2.0 * se),
                float(m / t),
                float(p1),
                float(p2),
            ])


def plot_regret_history(
    history: Dict[str, object],
    gap_arm2: float,
    plot_path: pathlib.Path,
    regret_scale: float,
    draw_error_band: bool,
) -> None:
    if plt is None:
        print("[plot] matplotlib unavailable; skipping plot", file=sys.stderr)
        return

    t = np.asarray(history["time_step"], dtype=np.float64)
    mean_regret = np.asarray(history["mean_regret"], dtype=np.float64)
    standard_error = np.asarray(history["standard_error"], dtype=np.float64)

    scale = float(regret_scale)
    y = mean_regret / scale
    se = standard_error / scale

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    ax.plot(t, y, linewidth=2.4, label=rf"hardest $\Delta={gap_arm2:.5g}$")
    if draw_error_band:
        ax.fill_between(t, y - 2.0 * se, y + 2.0 * se, alpha=0.18, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Time")
    if scale == 1.0:
        ax.set_ylabel("Average cumulative regret")
    else:
        ax.set_ylabel(rf"Average cumulative regret / ${scale:.0e}$")
    ax.set_title(r"$\eta_t = 1/\sqrt{t}$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {plot_path}", flush=True)


def plot_average_regret_per_round(
    history: Dict[str, object],
    gap_arm2: float,
    plot_path: pathlib.Path,
    draw_error_band: bool,
) -> None:
    if plt is None:
        print("[plot] matplotlib unavailable; skipping per-round plot", file=sys.stderr)
        return

    t = np.asarray(history["time_step"], dtype=np.float64)
    mean_regret = np.asarray(history["mean_regret"], dtype=np.float64) / t
    standard_error = np.asarray(history["standard_error"], dtype=np.float64) / t

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    ax.plot(t, mean_regret, linewidth=2.4, label=rf"hardest $\Delta={gap_arm2:.5g}$")
    if draw_error_band:
        ax.fill_between(
            t,
            mean_regret - 2.0 * standard_error,
            mean_regret + 2.0 * standard_error,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Average regret per round, $E[Reg_t]/t$")
    ax.set_title(r"$\eta_t = 1/\sqrt{t}$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {plot_path}", flush=True)


def main() -> None:
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    history_path = outdir / args.history_csv
    summary_path = outdir / args.summary_json
    plot_path = outdir / args.plot_file

    if not args.overwrite:
        existing = [path for path in [history_path, summary_path, plot_path] if path.exists()]
        if existing:
            raise FileExistsError(
                "Output file(s) already exist. Use --overwrite to replace them: "
                + ", ".join(str(path) for path in existing)
            )

    hardest_row: Optional[Dict[str, str]] = None
    if args.gap is not None:
        hardest_gap = float(args.gap)
        print(f"[hardest-gap] using user-specified gap={hardest_gap:.12g}", flush=True)
    else:
        rows = read_sweep_rows(pathlib.Path(args.sweep_csv), pathlib.Path(args.sweep_dir))
        hardest_gap, hardest_row = find_hardest_gap_from_rows(rows)
        print(
            f"[hardest-gap] selected gap={hardest_gap:.12g} "
            f"from sweep final mean_regret={float(hardest_row['mean_regret']):.6e}",
            flush=True,
        )

    config = SimulationConfig(
        horizon_steps=args.horizon,
        num_arms=args.num_arms,
        num_trajectories=args.trajectories,
        random_seed=args.seed,
        max_mean_score_change_per_block=args.max_mean_change,
        max_noise_score_change_per_block=args.max_noise_change,
        max_block_size=args.max_block_size,
        block_quantile=args.block_quantile,
        exact_eta_sum_threshold=args.exact_eta_sum_threshold,
    )

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)

    checkpoints = make_checkpoints(
        horizon_steps=config.horizon_steps,
        num_checkpoints=args.num_checkpoints,
        mode=args.checkpoint_mode,
        checkpoint_file=args.checkpoint_file,
    )

    history = simulate_hardest_gap_history(
        gap_arm2=hardest_gap,
        config=config,
        checkpoints=checkpoints,
        workers=workers,
        chunk_trajectories=args.chunk_trajectories,
    )

    write_history_csv(history_path, history)
    print(f"[write] wrote {history_path}", flush=True)

    summary = {
        "hardest_gap": hardest_gap,
        "hardest_gap_selection": (
            "user-specified --gap" if args.gap is not None else "max mean_regret in sweep"
        ),
        "hardest_sweep_row": hardest_row,
        "config": asdict(config),
        "workers": workers,
        "chunk_trajectories": args.chunk_trajectories,
        "num_checkpoints": int(checkpoints.size),
        "eta_schedule": "eta_t = 1/sqrt(t)",
        "simulation": "blocked_gaussian_aggregate",
        "final_mean_regret": float(np.asarray(history["mean_regret"])[-1]),
        "final_standard_error": float(np.asarray(history["standard_error"])[-1]),
        "final_mean_regret_per_round": float(
            np.asarray(history["mean_regret"])[-1] / config.horizon_steps
        ),
        "final_mean_pi1": float(np.asarray(history["mean_pi1"])[-1]),
        "final_mean_pi2": float(np.asarray(history["mean_pi2"])[-1]),
        "num_trajectories_completed": int(history["num_trajectories"]),
        "total_blocks": int(history["total_blocks"]),
        "mean_blocks_per_chunk": float(history["mean_blocks_per_chunk"]),
        "max_chunk_seconds": float(history["max_chunk_seconds"]),
        "wall_seconds": float(history["wall_seconds"]),
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] wrote {summary_path}", flush=True)

    plot_regret_history(
        history,
        hardest_gap,
        plot_path,
        regret_scale=args.regret_scale,
        draw_error_band=not args.no_error_band,
    )

    if args.plot_average_per_round:
        per_round_plot_path = outdir / "hardest_gap_average_regret_per_round.png"
        plot_average_regret_per_round(
            history,
            hardest_gap,
            per_round_plot_path,
            draw_error_band=not args.no_error_band,
        )

    print(
        "[done] "
        f"gap={hardest_gap:.12g} final_mean_regret={summary['final_mean_regret']:.6e} "
        f"se={summary['final_standard_error']:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
