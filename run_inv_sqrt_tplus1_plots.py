#!/usr/bin/env python3
"""
Experiments for the lower-bound policy-gradient bandit with

    eta_t = 1 / sqrt(t + 1).

Bandit instance for a given gap Delta:

    mu    = (1, 1 - Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)

The number of zero-mean arms is

    m = K - 2.

This version is hard-coded by default for:

    main K:       1000
    comparison K: 10
    gap grid:     101 values in [0, 1]

Commands:

    sweep-one
        Run one gap value for one arm count. Use this in the SLURM array.

    sweep-all
        Run all gaps locally. Useful for small tests.

    combine-plot
        Combine all sweep outputs and create:
            worst_gap_vs_predicted_gap.png
            sample_path_pi1_arms_1000.png
            sample_path_pi1_arms_10.png
            worst_case_regret_original_scale.png
            worst_case_regret_loglog.png
            few_arms_comparison_original_scale.png
            few_arms_comparison_loglog.png

    sample-path
        Create one sample-path plot for a given K and gap.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Defaults: these are the important values.
# =============================================================================

DEFAULT_OUTDIR = "inv_sqrt_tplus1_results"

# Main experiment: K = 1000, so m = 998 zero-mean arms.
DEFAULT_MAIN_NUM_ARMS = 1000

# We also run K = 10, so m = 8 zero-mean arms.
DEFAULT_NUM_ARMS_LIST = "1000,10"

# 101 gaps means Delta = 0.00, 0.01, ..., 1.00.
DEFAULT_NUM_GAPS = 101

DEFAULT_HORIZON = 10_000_000
DEFAULT_NUM_HORIZONS = 21
DEFAULT_TRAJECTORIES = 10_000
DEFAULT_WORKERS = 1
DEFAULT_CHUNK_TRAJECTORIES = 2_500
DEFAULT_SEED = 20260310


# =============================================================================
# Utility functions
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_gap_grid(num_gaps: int, gap_start: float, gap_stop: float) -> np.ndarray:
    if num_gaps <= 1:
        return np.array([float(gap_start)], dtype=np.float64)
    return np.linspace(float(gap_start), float(gap_stop), int(num_gaps), dtype=np.float64)


def make_log_checkpoints(horizon: int, num_horizons: int) -> np.ndarray:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if num_horizons <= 1:
        return np.array([horizon], dtype=np.int64)

    raw = np.unique(
        np.round(np.logspace(0, math.log10(horizon), int(num_horizons))).astype(np.int64)
    )
    raw = raw[(raw >= 1) & (raw <= horizon)]

    if raw.size == 0 or raw[-1] != horizon:
        raw = np.unique(np.concatenate([raw, np.array([horizon], dtype=np.int64)]))

    return raw.astype(np.int64)


def eta_inv_sqrt_tplus1(step_index_zero_based: int | np.ndarray) -> float | np.ndarray:
    """
    eta_t = 1 / sqrt(t + 1), with t zero-based.

    So at the first update, t = 0 and eta_0 = 1.
    """
    return 1.0 / np.sqrt(np.asarray(step_index_zero_based, dtype=np.float64) + 1.0)


def harmonic_number(n: int) -> float:
    """
    Fast approximation to H_n = sum_{j=1}^n 1/j.

    Used for sum eta_t^2 because eta_t^2 = 1/(t+1).
    """
    n = int(n)
    if n <= 0:
        return 0.0

    if n < 200_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))

    x = float(n)
    gamma = 0.5772156649015328606
    return math.log(x) + gamma + 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)


def eta_sums_for_block(start_step: int, block_size: int) -> Tuple[float, float]:
    """
    Return

        sum eta_t
        sum eta_t^2

    over t = start_step, ..., start_step + block_size - 1.
    """
    start_step = int(start_step)
    block_size = int(block_size)

    if block_size <= 0:
        return 0.0, 0.0

    if block_size <= 200_000:
        t = np.arange(start_step, start_step + block_size, dtype=np.float64)
        eta = 1.0 / np.sqrt(t + 1.0)
        return float(np.sum(eta)), float(np.sum(eta * eta))

    # Integral approximation for sum 1/sqrt(t+1).
    sum_eta = 2.0 * (
        math.sqrt(start_step + block_size + 1.0) - math.sqrt(start_step + 1.0)
    )

    # Exact identity:
    #   sum_{t=a}^{a+B-1} 1/(t+1) = H_{a+B} - H_a.
    sum_eta_squared = harmonic_number(start_step + block_size) - harmonic_number(start_step)

    return float(sum_eta), float(sum_eta_squared)


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)

    if not rows:
        raise ValueError(f"No rows to write to {path}")

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Policy and update rule
# =============================================================================

def stable_policy_probabilities(
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    other_arm_score: np.ndarray,
    num_other_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Softmax probabilities for:

        arm 1
        arm 2
        each zero-mean arm
    """
    max_score = np.maximum(np.maximum(optimal_arm_score, second_arm_score), other_arm_score)

    weight_optimal = np.exp(optimal_arm_score - max_score)
    weight_second = np.exp(second_arm_score - max_score)
    weight_other = np.exp(other_arm_score - max_score)

    total_weight = weight_optimal + weight_second + num_other_arms * weight_other

    optimal_prob = weight_optimal / total_weight
    second_prob = weight_second / total_weight
    other_prob_per_arm = weight_other / total_weight

    return optimal_prob, second_prob, other_prob_per_arm


def instantaneous_expected_regret(
    optimal_prob: np.ndarray,
    second_prob: np.ndarray,
    other_prob_per_arm: np.ndarray,
    gap_arm2: float,
    num_other_arms: int,
) -> np.ndarray:
    """
    Conditional expected regret under the current policy.

    Arm 1 gap: 0
    Arm 2 gap: Delta
    Arms 3,...,K gap: 1
    """
    other_total_prob = num_other_arms * other_prob_per_arm
    return other_total_prob + float(gap_arm2) * second_prob


def choose_block_size(
    *,
    current_step: int,
    remaining_steps: int,
    gap_arm2: float,
    optimal_prob: np.ndarray,
    second_prob: np.ndarray,
    other_prob_per_arm: np.ndarray,
    num_other_arms: int,
    max_mean_change: float,
    max_noise_change: float,
    max_block_size: int,
    block_quantile: float,
) -> int:
    """
    Adaptive block size for the frozen-policy approximation.

    Smaller max_mean_change and max_noise_change make the approximation stricter
    but slower.
    """
    eta_now = float(eta_inv_sqrt_tplus1(current_step))
    gap = float(gap_arm2)

    inst_regret = instantaneous_expected_regret(
        optimal_prob, second_prob, other_prob_per_arm, gap, num_other_arms
    )

    grad_optimal = optimal_prob * inst_regret
    grad_second = second_prob * (inst_regret - gap)
    grad_other = -(grad_optimal + grad_second) / float(num_other_arms)

    grad_abs = np.maximum.reduce(
        [np.abs(grad_optimal), np.abs(grad_second), np.abs(grad_other)]
    )
    mean_gradient_scale = float(np.quantile(grad_abs, float(block_quantile)))

    second_mean = 1.0 - gap
    second_moment_arm1 = 2.0
    second_moment_arm2 = second_mean * second_mean + 1.0

    var_optimal = (
        optimal_prob * (1.0 - optimal_prob) ** 2 * second_moment_arm1
        + second_prob * optimal_prob ** 2 * second_moment_arm2
    )
    var_second = (
        optimal_prob * second_prob ** 2 * second_moment_arm1
        + second_prob * (1.0 - second_prob) ** 2 * second_moment_arm2
    )
    var_other = (
        optimal_prob * other_prob_per_arm ** 2 * second_moment_arm1
        + second_prob * other_prob_per_arm ** 2 * second_moment_arm2
    )

    variance_scale_array = np.maximum.reduce([var_optimal, var_second, var_other])
    variance_scale = float(np.quantile(variance_scale_array, float(block_quantile)))

    limit_by_mean = math.inf
    if mean_gradient_scale > 1e-300:
        limit_by_mean = float(max_mean_change) / (eta_now * mean_gradient_scale)

    limit_by_noise = math.inf
    if variance_scale > 1e-300:
        limit_by_noise = (
            float(max_noise_change) / (eta_now * math.sqrt(variance_scale))
        ) ** 2

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


@dataclass
class SimulationResult:
    checkpoints: np.ndarray
    regret_sum: np.ndarray
    regret_sumsq: np.ndarray
    pi1_sum: np.ndarray
    pi2_sum: np.ndarray
    other_mass_sum: np.ndarray
    num_trajectories: int
    pi1_paths: Optional[np.ndarray] = None


def record_checkpoint(
    *,
    record_index: int,
    regret_by_trajectory: np.ndarray,
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    other_arm_score: np.ndarray,
    num_other_arms: int,
    regret_sum: np.ndarray,
    regret_sumsq: np.ndarray,
    pi1_sum: np.ndarray,
    pi2_sum: np.ndarray,
    other_mass_sum: np.ndarray,
    pi1_paths: Optional[np.ndarray],
) -> None:
    p1, p2, po = stable_policy_probabilities(
        optimal_arm_score, second_arm_score, other_arm_score, num_other_arms
    )

    other_mass = num_other_arms * po

    regret_sum[record_index] = float(np.sum(regret_by_trajectory))
    regret_sumsq[record_index] = float(np.sum(regret_by_trajectory ** 2))
    pi1_sum[record_index] = float(np.sum(p1))
    pi2_sum[record_index] = float(np.sum(p2))
    other_mass_sum[record_index] = float(np.sum(other_mass))

    if pi1_paths is not None:
        pi1_paths[record_index, :] = p1


def run_exact_steps(
    *,
    num_steps: int,
    current_step: int,
    random_generator: np.random.Generator,
    gap_arm2: float,
    num_other_arms: int,
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    other_arm_score: np.ndarray,
    regret_by_trajectory: np.ndarray,
) -> None:
    """
    Literal round-by-round Algorithm-1 update for a small number of steps.
    """
    n = optimal_arm_score.size
    gap = float(gap_arm2)
    second_mean = 1.0 - gap

    for local_step in range(int(num_steps)):
        step = current_step + local_step
        eta = float(eta_inv_sqrt_tplus1(step))

        p1, p2, po = stable_policy_probabilities(
            optimal_arm_score, second_arm_score, other_arm_score, num_other_arms
        )

        other_total = num_other_arms * po
        regret_by_trajectory += other_total + gap * p2

        u = random_generator.random(n)

        choose_arm1 = u < p1
        choose_arm2 = (u >= p1) & (u < p1 + p2)

        reward = np.zeros(n, dtype=np.float64)

        count_arm1 = int(np.sum(choose_arm1))
        count_arm2 = int(np.sum(choose_arm2))

        if count_arm1:
            reward[choose_arm1] = 1.0 + random_generator.normal(size=count_arm1)

        if count_arm2:
            reward[choose_arm2] = second_mean + random_generator.normal(size=count_arm2)

        optimal_arm_score += eta * reward * (choose_arm1.astype(np.float64) - p1)
        second_arm_score += eta * reward * (choose_arm2.astype(np.float64) - p2)

        # Each zero-mean arm has the same score, so we store one representative.
        other_arm_score += eta * reward * (-po)


def run_gaussian_approx_block(
    *,
    block_size: int,
    current_step: int,
    random_generator: np.random.Generator,
    gap_arm2: float,
    num_other_arms: int,
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    other_arm_score: np.ndarray,
    regret_by_trajectory: np.ndarray,
) -> None:
    """
    Blocked Gaussian approximation with frozen policy inside the block.

    It samples the weighted reward sums

        S1 = sum eta_t Y_t 1{A_t = 1}
        S2 = sum eta_t Y_t 1{A_t = 2}

    by matching their Gaussian mean and covariance.
    """
    gap = float(gap_arm2)
    second_mean = 1.0 - gap

    p1, p2, po = stable_policy_probabilities(
        optimal_arm_score, second_arm_score, other_arm_score, num_other_arms
    )

    other_total = num_other_arms * po
    inst_regret = other_total + gap * p2
    regret_by_trajectory += int(block_size) * inst_regret

    sum_eta, sum_eta_squared = eta_sums_for_block(current_step, block_size)

    mu1 = 1.0
    mu2 = second_mean

    ey1_sq = 2.0
    ey2_sq = mu2 * mu2 + 1.0

    mean_s1 = p1 * mu1 * sum_eta
    mean_s2 = p2 * mu2 * sum_eta

    var_s1 = (p1 * ey1_sq - (p1 * mu1) ** 2) * sum_eta_squared
    var_s2 = (p2 * ey2_sq - (p2 * mu2) ** 2) * sum_eta_squared
    cov_s1_s2 = -(p1 * mu1) * (p2 * mu2) * sum_eta_squared

    var_s1 = np.maximum(var_s1, 0.0)
    var_s2 = np.maximum(var_s2, 0.0)

    z1 = random_generator.normal(size=p1.size)
    z2 = random_generator.normal(size=p1.size)

    sqrt_var_s1 = np.sqrt(var_s1)
    weighted_reward_arm1 = mean_s1 + sqrt_var_s1 * z1

    safe = var_s1 > 1e-300

    coeff = np.zeros_like(var_s1)
    coeff[safe] = cov_s1_s2[safe] / sqrt_var_s1[safe]

    conditional_var_s2 = var_s2 - coeff ** 2
    conditional_var_s2 = np.maximum(conditional_var_s2, 0.0)

    weighted_reward_arm2 = mean_s2 + coeff * z1 + np.sqrt(conditional_var_s2) * z2

    optimal_arm_score += (1.0 - p1) * weighted_reward_arm1 - p1 * weighted_reward_arm2
    second_arm_score += -p2 * weighted_reward_arm1 + (1.0 - p2) * weighted_reward_arm2
    other_arm_score += -po * (weighted_reward_arm1 + weighted_reward_arm2)


def simulate_bandit_history(
    *,
    gap_arm2: float,
    num_arms: int,
    horizon: int,
    checkpoints: Sequence[int],
    num_trajectories: int,
    seed: int,
    method: str = "approx",
    max_mean_change: float = 0.08,
    max_noise_change: float = 0.35,
    max_block_size: int = 10_000_000,
    block_quantile: float = 0.995,
    exact_small_block_threshold: int = 64,
    return_pi1_paths: bool = False,
) -> SimulationResult:
    if num_arms < 3:
        raise ValueError("num_arms must be at least 3")

    if method not in {"exact", "approx"}:
        raise ValueError("method must be either 'exact' or 'approx'")

    checkpoints_array = np.array(
        sorted(set(int(x) for x in checkpoints if 1 <= int(x) <= int(horizon))),
        dtype=np.int64,
    )

    if checkpoints_array.size == 0 or checkpoints_array[-1] != int(horizon):
        checkpoints_array = np.unique(
            np.concatenate([checkpoints_array, np.array([int(horizon)], dtype=np.int64)])
        )

    random_generator = np.random.default_rng(int(seed))

    num_other_arms = int(num_arms) - 2
    n = int(num_trajectories)

    optimal_arm_score = np.zeros(n, dtype=np.float64)
    second_arm_score = np.zeros(n, dtype=np.float64)
    other_arm_score = np.zeros(n, dtype=np.float64)

    regret_by_trajectory = np.zeros(n, dtype=np.float64)

    num_records = int(checkpoints_array.size)

    regret_sum = np.zeros(num_records, dtype=np.float64)
    regret_sumsq = np.zeros(num_records, dtype=np.float64)
    pi1_sum = np.zeros(num_records, dtype=np.float64)
    pi2_sum = np.zeros(num_records, dtype=np.float64)
    other_mass_sum = np.zeros(num_records, dtype=np.float64)

    pi1_paths = np.zeros((num_records, n), dtype=np.float64) if return_pi1_paths else None

    current_step = 0
    record_index = 0
    horizon = int(horizon)

    while current_step < horizon:
        next_checkpoint = int(checkpoints_array[record_index])

        remaining_to_horizon = horizon - current_step
        remaining_to_checkpoint = next_checkpoint - current_step

        if remaining_to_checkpoint <= 0:
            record_checkpoint(
                record_index=record_index,
                regret_by_trajectory=regret_by_trajectory,
                optimal_arm_score=optimal_arm_score,
                second_arm_score=second_arm_score,
                other_arm_score=other_arm_score,
                num_other_arms=num_other_arms,
                regret_sum=regret_sum,
                regret_sumsq=regret_sumsq,
                pi1_sum=pi1_sum,
                pi2_sum=pi2_sum,
                other_mass_sum=other_mass_sum,
                pi1_paths=pi1_paths,
            )

            record_index += 1

            if record_index >= num_records:
                break

            continue

        if method == "exact":
            block_size = min(remaining_to_horizon, remaining_to_checkpoint, 1000)

            run_exact_steps(
                num_steps=block_size,
                current_step=current_step,
                random_generator=random_generator,
                gap_arm2=gap_arm2,
                num_other_arms=num_other_arms,
                optimal_arm_score=optimal_arm_score,
                second_arm_score=second_arm_score,
                other_arm_score=other_arm_score,
                regret_by_trajectory=regret_by_trajectory,
            )

        else:
            p1, p2, po = stable_policy_probabilities(
                optimal_arm_score,
                second_arm_score,
                other_arm_score,
                num_other_arms,
            )

            block_size = choose_block_size(
                current_step=current_step,
                remaining_steps=remaining_to_horizon,
                gap_arm2=gap_arm2,
                optimal_prob=p1,
                second_prob=p2,
                other_prob_per_arm=po,
                num_other_arms=num_other_arms,
                max_mean_change=max_mean_change,
                max_noise_change=max_noise_change,
                max_block_size=max_block_size,
                block_quantile=block_quantile,
            )

            block_size = min(block_size, remaining_to_checkpoint)

            if block_size <= int(exact_small_block_threshold):
                run_exact_steps(
                    num_steps=block_size,
                    current_step=current_step,
                    random_generator=random_generator,
                    gap_arm2=gap_arm2,
                    num_other_arms=num_other_arms,
                    optimal_arm_score=optimal_arm_score,
                    second_arm_score=second_arm_score,
                    other_arm_score=other_arm_score,
                    regret_by_trajectory=regret_by_trajectory,
                )
            else:
                run_gaussian_approx_block(
                    block_size=block_size,
                    current_step=current_step,
                    random_generator=random_generator,
                    gap_arm2=gap_arm2,
                    num_other_arms=num_other_arms,
                    optimal_arm_score=optimal_arm_score,
                    second_arm_score=second_arm_score,
                    other_arm_score=other_arm_score,
                    regret_by_trajectory=regret_by_trajectory,
                )

        current_step += int(block_size)

        while record_index < num_records and current_step >= int(checkpoints_array[record_index]):
            record_checkpoint(
                record_index=record_index,
                regret_by_trajectory=regret_by_trajectory,
                optimal_arm_score=optimal_arm_score,
                second_arm_score=second_arm_score,
                other_arm_score=other_arm_score,
                num_other_arms=num_other_arms,
                regret_sum=regret_sum,
                regret_sumsq=regret_sumsq,
                pi1_sum=pi1_sum,
                pi2_sum=pi2_sum,
                other_mass_sum=other_mass_sum,
                pi1_paths=pi1_paths,
            )

            record_index += 1

            if record_index >= num_records:
                break

    return SimulationResult(
        checkpoints=checkpoints_array,
        regret_sum=regret_sum,
        regret_sumsq=regret_sumsq,
        pi1_sum=pi1_sum,
        pi2_sum=pi2_sum,
        other_mass_sum=other_mass_sum,
        num_trajectories=n,
        pi1_paths=pi1_paths,
    )


# =============================================================================
# Parallel simulation
# =============================================================================

def _simulate_chunk_worker(kwargs: Dict[str, object]) -> SimulationResult:
    return simulate_bandit_history(**kwargs)


def simulate_parallel(
    *,
    gap_arm2: float,
    num_arms: int,
    horizon: int,
    checkpoints: Sequence[int],
    trajectories: int,
    workers: int,
    chunk_trajectories: int,
    seed: int,
    method: str,
    max_mean_change: float,
    max_noise_change: float,
    max_block_size: int,
    block_quantile: float,
    exact_small_block_threshold: int,
) -> SimulationResult:
    trajectories = int(trajectories)
    workers = max(1, int(workers))
    chunk_trajectories = max(1, int(chunk_trajectories))

    chunks: List[int] = []
    remaining = trajectories

    while remaining > 0:
        size = min(chunk_trajectories, remaining)
        chunks.append(size)
        remaining -= size

    common = dict(
        gap_arm2=float(gap_arm2),
        num_arms=int(num_arms),
        horizon=int(horizon),
        checkpoints=list(int(x) for x in checkpoints),
        method=str(method),
        max_mean_change=float(max_mean_change),
        max_noise_change=float(max_noise_change),
        max_block_size=int(max_block_size),
        block_quantile=float(block_quantile),
        exact_small_block_threshold=int(exact_small_block_threshold),
        return_pi1_paths=False,
    )

    if workers == 1 or len(chunks) == 1:
        results = []

        for chunk_index, chunk_size in enumerate(chunks):
            kwargs = dict(common)
            kwargs.update(
                num_trajectories=chunk_size,
                seed=int(seed) + 10_003 * chunk_index,
            )

            results.append(simulate_bandit_history(**kwargs))

    else:
        results = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []

            for chunk_index, chunk_size in enumerate(chunks):
                kwargs = dict(common)
                kwargs.update(
                    num_trajectories=chunk_size,
                    seed=int(seed) + 10_003 * chunk_index,
                )

                futures.append(executor.submit(_simulate_chunk_worker, kwargs))

            for future in as_completed(futures):
                results.append(future.result())

    checkpoints_array = results[0].checkpoints

    regret_sum = np.sum([r.regret_sum for r in results], axis=0)
    regret_sumsq = np.sum([r.regret_sumsq for r in results], axis=0)
    pi1_sum = np.sum([r.pi1_sum for r in results], axis=0)
    pi2_sum = np.sum([r.pi2_sum for r in results], axis=0)
    other_mass_sum = np.sum([r.other_mass_sum for r in results], axis=0)

    total_trajectories = int(sum(r.num_trajectories for r in results))

    return SimulationResult(
        checkpoints=checkpoints_array,
        regret_sum=regret_sum,
        regret_sumsq=regret_sumsq,
        pi1_sum=pi1_sum,
        pi2_sum=pi2_sum,
        other_mass_sum=other_mass_sum,
        num_trajectories=total_trajectories,
        pi1_paths=None,
    )


def result_to_rows(
    *,
    result: SimulationResult,
    gap_index: int,
    gap_arm2: float,
    num_arms: int,
    method: str,
) -> List[Dict[str, object]]:
    n = result.num_trajectories
    rows: List[Dict[str, object]] = []

    for i, time_value in enumerate(result.checkpoints):
        mean_regret = result.regret_sum[i] / n

        if n > 1:
            variance = max(result.regret_sumsq[i] / n - mean_regret ** 2, 0.0)
            standard_error = math.sqrt(variance / n)
        else:
            standard_error = 0.0

        rows.append(
            {
                "num_arms": int(num_arms),
                "zero_mean_arms_m": int(num_arms) - 2,
                "gap_index": int(gap_index),
                "gap_arm2": float(gap_arm2),
                "time": int(time_value),
                "mean_regret": float(mean_regret),
                "standard_error": float(standard_error),
                "mean_pi1": float(result.pi1_sum[i] / n),
                "mean_pi2": float(result.pi2_sum[i] / n),
                "mean_other_mass": float(result.other_mass_sum[i] / n),
                "num_trajectories": int(n),
                "method": str(method),
                "stepsize": "eta_t=1/sqrt(t+1)",
            }
        )

    return rows


def sweep_output_path(outdir: Path, num_arms: int, gap_index: int) -> Path:
    return outdir / "sweep" / f"arms_{int(num_arms)}" / f"gap_{int(gap_index):04d}.csv"


def run_sweep_one(args: argparse.Namespace) -> Path:
    outdir = Path(args.outdir)

    gaps = make_gap_grid(args.num_gaps, args.gap_start, args.gap_stop)

    if args.gap_index < 0 or args.gap_index >= len(gaps):
        raise ValueError(f"gap-index {args.gap_index} is outside 0,...,{len(gaps)-1}")

    gap = float(gaps[int(args.gap_index)])
    checkpoints = make_log_checkpoints(args.horizon, args.num_horizons)

    seed = int(args.seed) + 1_000_003 * int(args.num_arms) + 101 * int(args.gap_index)

    result = simulate_parallel(
        gap_arm2=gap,
        num_arms=args.num_arms,
        horizon=args.horizon,
        checkpoints=checkpoints,
        trajectories=args.trajectories,
        workers=args.workers,
        chunk_trajectories=args.chunk_trajectories,
        seed=seed,
        method=args.method,
        max_mean_change=args.max_mean_change,
        max_noise_change=args.max_noise_change,
        max_block_size=args.max_block_size,
        block_quantile=args.block_quantile,
        exact_small_block_threshold=args.exact_small_block_threshold,
    )

    rows = result_to_rows(
        result=result,
        gap_index=args.gap_index,
        gap_arm2=gap,
        num_arms=args.num_arms,
        method=args.method,
    )

    path = sweep_output_path(outdir, args.num_arms, args.gap_index)
    write_rows_csv(path, rows)

    print(f"Wrote {path}")
    return path


def run_sweep_all(args: argparse.Namespace) -> None:
    arms_list = parse_int_list(args.num_arms_list)

    for num_arms in arms_list:
        for gap_index in range(int(args.num_gaps)):
            one_args = argparse.Namespace(**vars(args))
            one_args.num_arms = int(num_arms)
            one_args.gap_index = int(gap_index)

            run_sweep_one(one_args)


# =============================================================================
# Combining and plotting
# =============================================================================

def combine_envelope_for_num_arms(
    outdir: Path,
    num_arms: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    sweep_dir = outdir / "sweep" / f"arms_{int(num_arms)}"
    files = sorted(sweep_dir.glob("gap_*.csv"))

    if not files:
        raise FileNotFoundError(f"No sweep files found in {sweep_dir}")

    all_rows: List[Dict[str, object]] = []

    for path in files:
        for row in read_rows_csv(path):
            if int(float(row["num_arms"])) != int(num_arms):
                continue

            converted: Dict[str, object] = {
                "num_arms": int(float(row["num_arms"])),
                "zero_mean_arms_m": int(float(row["zero_mean_arms_m"])),
                "gap_index": int(float(row["gap_index"])),
                "gap_arm2": float(row["gap_arm2"]),
                "time": int(float(row["time"])),
                "mean_regret": float(row["mean_regret"]),
                "standard_error": float(row["standard_error"]),
                "mean_pi1": float(row["mean_pi1"]),
                "mean_pi2": float(row.get("mean_pi2", 0.0)),
                "mean_other_mass": float(row.get("mean_other_mass", 0.0)),
                "num_trajectories": int(float(row["num_trajectories"])),
                "method": row.get("method", "approx"),
                "stepsize": row.get("stepsize", "eta_t=1/sqrt(t+1)"),
            }

            all_rows.append(converted)

    if not all_rows:
        raise RuntimeError(f"No usable rows found for K={num_arms}")

    by_time: Dict[int, List[Dict[str, object]]] = {}

    for row in all_rows:
        by_time.setdefault(int(row["time"]), []).append(row)

    hardest_rows: List[Dict[str, object]] = []

    for time_value in sorted(by_time.keys()):
        candidates = by_time[time_value]
        best = max(candidates, key=lambda r: float(r["mean_regret"]))

        hardest_rows.append(
            {
                "num_arms": int(num_arms),
                "zero_mean_arms_m": int(num_arms) - 2,
                "time": int(time_value),
                "hardest_gap_index": int(best["gap_index"]),
                "hardest_gap_arm2": float(best["gap_arm2"]),
                "mean_regret": float(best["mean_regret"]),
                "standard_error": float(best["standard_error"]),
                "mean_pi1": float(best["mean_pi1"]),
                "mean_pi2": float(best["mean_pi2"]),
                "mean_other_mass": float(best["mean_other_mass"]),
                "num_trajectories": int(best["num_trajectories"]),
                "method": best["method"],
                "stepsize": best["stepsize"],
            }
        )

    return all_rows, hardest_rows


def rows_to_arrays(
    rows: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.array([float(r["time"]) for r in rows], dtype=np.float64)
    regret = np.array([float(r["mean_regret"]) for r in rows], dtype=np.float64)
    se = np.array([float(r["standard_error"]) for r in rows], dtype=np.float64)
    gap = np.array([float(r["hardest_gap_arm2"]) for r in rows], dtype=np.float64)

    order = np.argsort(time)

    return time[order], regret[order], se[order], gap[order]


def plot_worst_case_regret_main(
    hardest_rows: Sequence[Dict[str, object]],
    outdir: Path,
    regret_scale: float,
) -> List[Path]:
    time, regret, se, _gap = rows_to_arrays(hardest_rows)

    num_arms = int(hardest_rows[0]["num_arms"])
    scale = float(regret_scale)

    ylabel = (
        "Worst-case average regret"
        if scale == 1.0
        else rf"Worst-case average regret / ${scale:.0e}$"
    )

    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(time, regret / scale, linewidth=2.5, label=rf"$K={num_arms}$")
    ax.fill_between(
        time,
        (regret - 2.0 * se) / scale,
        (regret + 2.0 * se) / scale,
        alpha=0.18,
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(r"Horizon-wise worst-case regret, $\eta_t = 1/\sqrt{t+1}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / "worst_case_regret_original_scale.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    positive = (time > 0) & (regret > 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(time[positive], regret[positive] / scale, linewidth=2.5, label=rf"$K={num_arms}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title("Horizon-wise worst-case regret, log-log scale")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / "worst_case_regret_loglog.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    return paths


def plot_worst_gap_vs_prediction(
    hardest_rows: Sequence[Dict[str, object]],
    outdir: Path,
    c_values: Sequence[float],
) -> Path:
    time, _regret, _se, gap = rows_to_arrays(hardest_rows)

    num_arms = int(hardest_rows[0]["num_arms"])
    m = int(num_arms) - 2

    if m <= 0:
        raise ValueError("num_arms must be at least 3")

    m_term = math.sqrt(math.log(max(m, 2)) / float(m))
    n_term = 1.0 / np.log(np.maximum(time, 3.0))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    ax.plot(time, gap, marker="o", linewidth=2.5, label="empirical hardest gap")

    for c in c_values:
        predicted = float(c) * np.minimum(m_term, n_term)
        predicted = np.minimum(predicted, 1.0)

        ax.plot(
            time,
            predicted,
            linestyle="--",
            linewidth=1.8,
            label=rf"$c={c:g}$ prediction",
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Horizon $n$")
    ax.set_ylabel(r"Gap $\Delta$")
    ax.set_title(
        rf"Hardest gap vs. $c\min\{{\sqrt{{\log m/m}},1/\log n\}}$, "
        rf"$K={num_arms}$, $m={m}$"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = outdir / "worst_gap_vs_predicted_gap.png"
    fig.savefig(path)
    plt.close(fig)

    return path


def plot_few_arms_comparison(
    hardest_by_arms: Dict[int, Sequence[Dict[str, object]]],
    outdir: Path,
    regret_scale: float,
) -> List[Path]:
    scale = float(regret_scale)

    ylabel = (
        "Worst-case average regret"
        if scale == 1.0
        else rf"Worst-case average regret / ${scale:.0e}$"
    )

    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    for num_arms, rows in sorted(hardest_by_arms.items(), reverse=True):
        time, regret, _se, _gap = rows_to_arrays(rows)
        ax.plot(
            time,
            regret / scale,
            linewidth=2.5,
            label=rf"$K={num_arms}$, $m={num_arms - 2}$",
        )

    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title("Few-arm comparison: large m is harder")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / "few_arms_comparison_original_scale.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    for num_arms, rows in sorted(hardest_by_arms.items(), reverse=True):
        time, regret, _se, _gap = rows_to_arrays(rows)
        positive = (time > 0) & (regret > 0)

        ax.plot(
            time[positive],
            regret[positive] / scale,
            linewidth=2.5,
            label=rf"$K={num_arms}$, $m={num_arms - 2}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title("Few-arm comparison, log-log scale")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / "few_arms_comparison_loglog.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    return paths


def plot_sample_paths(
    *,
    outdir: Path,
    gap_arm2: float,
    horizon: int,
    num_arms: int,
    num_paths: int,
    num_checkpoints: int,
    seed: int,
    method: str,
    max_mean_change: float,
    max_noise_change: float,
    max_block_size: int,
    block_quantile: float,
    exact_small_block_threshold: int,
) -> Path:
    checkpoints = make_log_checkpoints(horizon, num_checkpoints)

    result = simulate_bandit_history(
        gap_arm2=float(gap_arm2),
        num_arms=int(num_arms),
        horizon=int(horizon),
        checkpoints=checkpoints,
        num_trajectories=int(num_paths),
        seed=int(seed),
        method=str(method),
        max_mean_change=float(max_mean_change),
        max_noise_change=float(max_noise_change),
        max_block_size=int(max_block_size),
        block_quantile=float(block_quantile),
        exact_small_block_threshold=int(exact_small_block_threshold),
        return_pi1_paths=True,
    )

    assert result.pi1_paths is not None

    t = result.checkpoints.astype(np.float64)
    paths = np.clip(result.pi1_paths, 1e-12, 1.0 - 1e-12)
    mean_path = np.mean(paths, axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    for j in range(paths.shape[1]):
        ax.plot(t, paths[:, j], color="0.65", alpha=0.45, linewidth=0.9)

    ax.plot(t, mean_path, color="red", linewidth=2.8, label="average")

    ax.set_xscale("log")
    ax.set_yscale("logit")
    ax.set_ylim(1.0 / (10.0 * num_arms), 1.0 - 1e-4)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\pi_t(1)$")
    ax.set_title(
        rf"Sample paths of $\pi_t(1)$, $K={num_arms}$, hardest $\Delta={gap_arm2:.4g}$"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / f"sample_path_pi1_arms_{int(num_arms)}.png"
    fig.savefig(path)
    plt.close(fig)

    return path


def run_combine_plot(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    arms_list = parse_int_list(args.num_arms_list)
    main_num_arms = int(args.num_arms)

    if main_num_arms not in arms_list:
        arms_list = [main_num_arms] + arms_list

    hardest_by_arms: Dict[int, List[Dict[str, object]]] = {}
    all_rows_combined: List[Dict[str, object]] = []

    for num_arms in arms_list:
        all_rows, hardest_rows = combine_envelope_for_num_arms(outdir, num_arms)

        hardest_by_arms[num_arms] = hardest_rows
        all_rows_combined.extend(all_rows)

        write_rows_csv(outdir / f"combined_envelope_by_gap_arms_{num_arms}.csv", all_rows)
        write_rows_csv(outdir / f"hardest_gap_by_time_arms_{num_arms}.csv", hardest_rows)

        print(f"Combined K={num_arms}: {len(all_rows)} rows, {len(hardest_rows)} horizons")

    write_rows_csv(outdir / "combined_envelope_by_gap_all_arms.csv", all_rows_combined)

    paths: List[Path] = []

    main_hardest = hardest_by_arms[main_num_arms]

    paths.extend(plot_worst_case_regret_main(main_hardest, outdir, args.regret_scale))
    paths.append(plot_worst_gap_vs_prediction(main_hardest, outdir, parse_float_list(args.c_values)))

    if len(hardest_by_arms) >= 2:
        paths.extend(plot_few_arms_comparison(hardest_by_arms, outdir, args.regret_scale))

    if args.make_sample_path:
        for num_arms in arms_list:
            hardest_rows = hardest_by_arms[num_arms]

            final_hardest_gap = float(hardest_rows[-1]["hardest_gap_arm2"])

            if args.sample_gap is not None:
                gap_for_sample = float(args.sample_gap)
            else:
                gap_for_sample = final_hardest_gap

            path = plot_sample_paths(
                outdir=outdir,
                gap_arm2=gap_for_sample,
                horizon=args.horizon,
                num_arms=num_arms,
                num_paths=args.sample_paths,
                num_checkpoints=args.sample_checkpoints,
                seed=args.seed + 999_983 + 1009 * int(num_arms),
                method=args.method,
                max_mean_change=args.max_mean_change,
                max_noise_change=args.max_noise_change,
                max_block_size=args.max_block_size,
                block_quantile=args.block_quantile,
                exact_small_block_threshold=args.exact_small_block_threshold,
            )

            paths.append(path)

            print(f"Sample path for K={num_arms} used gap {gap_for_sample:.8g}")

    print("Generated plots/files:")

    for path in paths:
        print(f"  {path}")


def run_sample_path_command(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.sample_gap is not None:
        gap = float(args.sample_gap)
    else:
        hardest_path = outdir / f"hardest_gap_by_time_arms_{int(args.num_arms)}.csv"

        if not hardest_path.exists():
            raise FileNotFoundError(
                f"{hardest_path} does not exist. Run combine-plot first or pass --sample-gap."
            )

        rows = read_rows_csv(hardest_path)
        gap = float(rows[-1]["hardest_gap_arm2"])

    path = plot_sample_paths(
        outdir=outdir,
        gap_arm2=gap,
        horizon=args.horizon,
        num_arms=args.num_arms,
        num_paths=args.sample_paths,
        num_checkpoints=args.sample_checkpoints,
        seed=args.seed + 999_983 + 1009 * int(args.num_arms),
        method=args.method,
        max_mean_change=args.max_mean_change,
        max_noise_change=args.max_noise_change,
        max_block_size=args.max_block_size,
        block_quantile=args.block_quantile,
        exact_small_block_threshold=args.exact_small_block_threshold,
    )

    print(f"Wrote {path}")


# =============================================================================
# Command line interface
# =============================================================================

def add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    # IMPORTANT: default is 1000, not 40.
    parser.add_argument("--num-arms", type=int, default=DEFAULT_MAIN_NUM_ARMS)

    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--num-horizons", type=int, default=DEFAULT_NUM_HORIZONS)
    parser.add_argument("--trajectories", type=int, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-trajectories", type=int, default=DEFAULT_CHUNK_TRAJECTORIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--method", choices=["approx", "exact"], default="approx")

    parser.add_argument("--max-mean-change", type=float, default=0.08)
    parser.add_argument("--max-noise-change", type=float, default=0.35)
    parser.add_argument("--max-block-size", type=int, default=10_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-small-block-threshold", type=int, default=64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lower-bound bandit experiments for eta_t = 1/sqrt(t+1)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_one = subparsers.add_parser("sweep-one", help="Run one gap value for one K")
    add_common_sim_args(sweep_one)
    sweep_one.add_argument("--gap-index", type=int, required=True)
    sweep_one.add_argument("--num-gaps", type=int, default=DEFAULT_NUM_GAPS)
    sweep_one.add_argument("--gap-start", type=float, default=0.0)
    sweep_one.add_argument("--gap-stop", type=float, default=1.0)
    sweep_one.set_defaults(func=run_sweep_one)

    sweep_all = subparsers.add_parser("sweep-all", help="Run all gaps locally")
    add_common_sim_args(sweep_all)

    # IMPORTANT: default is 1000,10, not 40,10.
    sweep_all.add_argument("--num-arms-list", default=DEFAULT_NUM_ARMS_LIST)

    sweep_all.add_argument("--num-gaps", type=int, default=DEFAULT_NUM_GAPS)
    sweep_all.add_argument("--gap-start", type=float, default=0.0)
    sweep_all.add_argument("--gap-stop", type=float, default=1.0)
    sweep_all.set_defaults(func=run_sweep_all)

    combine_plot = subparsers.add_parser(
        "combine-plot",
        help="Combine sweep outputs and create plots",
    )
    add_common_sim_args(combine_plot)

    # IMPORTANT: default is 1000,10, not 40,10.
    combine_plot.add_argument("--num-arms-list", default=DEFAULT_NUM_ARMS_LIST)

    combine_plot.add_argument("--regret-scale", type=float, default=1_000_000.0)
    combine_plot.add_argument("--c-values", default="0.5,1,2,4")
    combine_plot.add_argument("--make-sample-path", action="store_true")
    combine_plot.add_argument("--sample-gap", type=float, default=None)
    combine_plot.add_argument("--sample-paths", type=int, default=40)
    combine_plot.add_argument("--sample-checkpoints", type=int, default=250)
    combine_plot.set_defaults(func=run_combine_plot)

    sample_path = subparsers.add_parser(
        "sample-path",
        help="Make one pi1 sample-path plot",
    )
    add_common_sim_args(sample_path)
    sample_path.add_argument("--sample-gap", type=float, default=None)
    sample_path.add_argument("--sample-paths", type=int, default=40)
    sample_path.add_argument("--sample-checkpoints", type=int, default=250)
    sample_path.set_defaults(func=run_sample_path_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
