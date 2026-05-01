#!/usr/bin/env python3
"""
Multi-schedule gap sweep and hardest-gap regret history for softmax policy
gradient on the lower-bound Gaussian bandit instance.

Setup:
    k arms, default k = 40
    horizon n, default n = 1e9
    mu    = (1, 1-Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)

Learning-rate schedules included by default:
    inv_t              eta_t = 1 / t
    inv_t_two_thirds   eta_t = 1 / t^(2/3)
    log_over_t         eta_t = log(t) / t
    inv_sqrt_t         eta_t = 1 / sqrt(t)
    sqrt_log_over_t    eta_t = sqrt(log(t) / t)
    inv_log_t          eta_t = 1 / log(t), regularized at t=1 by log(max(t, 2))

The horizon is large, so this script uses the same blocked Gaussian aggregate
approximation style as the previous HPC code: within a block the policy is
frozen; the eta_t-weighted reward sums for arms 1 and 2 are sampled from a
matching bivariate Gaussian approximation.

Typical SLURM workflow:

  # 1. Sweep all schedule-gap pairs with an array job. For 6 schedules and
  #    101 gaps, use array indices 0-605, and call
  #    python pg_multischedule_hardest_gap.py sweep --task-index $SLURM_ARRAY_TASK_ID ...

  # 2. Combine sweep outputs and find the hardest gap per schedule.
  python pg_multischedule_hardest_gap.py combine-sweep --outdir multi_results --plot

  # 3. Run histories for each hardest gap, either all at once or with a schedule array.
  python pg_multischedule_hardest_gap.py history --outdir multi_results --workers 16 --plot

  # 4. Plot histories and growth-rate guides again if needed.
  python pg_multischedule_hardest_gap.py plot-history --outdir multi_results
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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional on compute nodes
    plt = None


# Fixed Gauss-Legendre nodes for fast smooth-sum approximations.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


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
    # block sizes. Set to 1.0 for the conservative max rule.
    block_quantile: float = 0.995

    # Exact summation threshold for eta_t sums inside a block.
    exact_eta_sum_threshold: int = 4096

    skip_existing: bool = True


@dataclass(frozen=True)
class ScheduleSpec:
    slug: str
    label: str
    kind: str
    power: Optional[float] = None


def all_schedules() -> List[ScheduleSpec]:
    return [
        ScheduleSpec("inv_t", r"$\eta_t = 1/t$", "power", 1.0),
        ScheduleSpec("inv_t_two_thirds", r"$\eta_t = 1/t^{2/3}$", "power", 2.0 / 3.0),
        ScheduleSpec("log_over_t", r"$\eta_t = \log(t)/t$", "log_over_t", None),
        ScheduleSpec("inv_sqrt_t", r"$\eta_t = 1/\sqrt{t}$", "power", 0.5),
        ScheduleSpec("sqrt_log_over_t", r"$\eta_t = \sqrt{\log(t)/t}$", "sqrt_log_over_t", None),
        # 1/log(1) is undefined. This schedule uses log(max(t, 2)) so eta_1 is finite.
        ScheduleSpec("inv_log_t", r"$\eta_t = 1/\log(t)$", "inv_log_t", None),
    ]


def schedule_by_slug(slug: str) -> ScheduleSpec:
    for schedule in all_schedules():
        if schedule.slug == slug:
            return schedule
    valid = ", ".join(s.slug for s in all_schedules())
    raise KeyError(f"Unknown schedule '{slug}'. Valid schedules: {valid}")


def selected_schedules(schedule_slugs: Optional[str]) -> List[ScheduleSpec]:
    schedules = all_schedules()
    if schedule_slugs is None or schedule_slugs.strip().lower() in {"", "all"}:
        return schedules
    requested = [s.strip() for s in schedule_slugs.split(",") if s.strip()]
    return [schedule_by_slug(s) for s in requested]


def gap_grid(gap_start: float, gap_stop: float, num_gaps: int) -> np.ndarray:
    if num_gaps < 2:
        raise ValueError("--num-gaps must be at least 2")
    return np.linspace(gap_start, gap_stop, num_gaps, dtype=np.float64)


def safe_log_for_log_schedules(x: np.ndarray | float) -> np.ndarray | float:
    """log(t) with t clipped below at 1 for log(t)/t and sqrt(log(t)/t)."""
    return np.log(np.maximum(x, 1.0))


def eta_values(schedule: ScheduleSpec, t: np.ndarray | float) -> np.ndarray | float:
    """Evaluate eta_t. Works for scalar floats or numpy arrays."""
    x = np.asarray(t, dtype=np.float64)

    if schedule.kind == "power":
        values = np.power(np.maximum(x, 1.0), -float(schedule.power))
    elif schedule.kind == "log_over_t":
        x_safe = np.maximum(x, 1.0)
        values = np.log(x_safe) / x_safe
    elif schedule.kind == "sqrt_log_over_t":
        x_safe = np.maximum(x, 1.0)
        values = np.sqrt(np.maximum(np.log(x_safe), 0.0) / x_safe)
    elif schedule.kind == "inv_log_t":
        # eta_t = 1/log(t) is singular at t=1. Regularize at t=1 by using log(max(t, 2)).
        values = 1.0 / np.log(np.maximum(x, 2.0))
    else:
        raise ValueError(f"Unknown schedule kind: {schedule.kind}")

    if np.isscalar(t):
        return float(values)
    return values


def max_eta_over_interval(schedule: ScheduleSpec, start_time: int, end_time: int) -> float:
    """Conservative max eta_t over integer times in [start_time, end_time]."""
    s = float(start_time)
    e = float(end_time)

    if schedule.kind == "power":
        return float(eta_values(schedule, s))

    candidates = [s, e]

    if schedule.kind in {"log_over_t", "sqrt_log_over_t"}:
        # log(t)/t and sqrt(log(t)/t) peak at t=e.
        if s <= math.e <= e:
            candidates.append(math.e)
    elif schedule.kind == "inv_log_t":
        # With log(max(t,2)), the maximum is achieved at t <= 2.
        if s <= 2.0 <= e:
            candidates.append(2.0)
    else:
        raise ValueError(f"Unknown schedule kind: {schedule.kind}")

    return float(max(eta_values(schedule, c) for c in candidates))


def exact_eta_sums(schedule: ScheduleSpec, start_time: int, block_size: int) -> Tuple[float, float]:
    t = np.arange(start_time, start_time + block_size, dtype=np.float64)
    eta = eta_values(schedule, t)
    return float(np.sum(eta)), float(np.sum(eta * eta))


def power_sum_euler_maclaurin(start_time: int, block_size: int, p: float) -> float:
    """Approximate sum_{t=start}^{start+B-1} t^{-p}."""
    a = float(start_time)
    b = float(start_time + block_size - 1)

    if block_size <= 0:
        return 0.0
    if block_size == 1:
        return a ** (-p)

    if abs(p - 1.0) < 1e-14:
        integral = math.log(b / a)
    else:
        integral = (b ** (1.0 - p) - a ** (1.0 - p)) / (1.0 - p)

    f_a = a ** (-p)
    f_b = b ** (-p)

    # f'(x) = -p x^{-p-1}
    fp_a = -p * a ** (-p - 1.0)
    fp_b = -p * b ** (-p - 1.0)

    # f'''(x) = -p(-p-1)(-p-2)x^{-p-3}
    f3_a = -p * (-p - 1.0) * (-p - 2.0) * a ** (-p - 3.0)
    f3_b = -p * (-p - 1.0) * (-p - 2.0) * b ** (-p - 3.0)

    estimate = integral + 0.5 * (f_a + f_b) + (fp_b - fp_a) / 12.0 - (f3_b - f3_a) / 720.0
    return float(max(0.0, estimate))


def log_over_t_sum_euler_maclaurin(start_time: int, block_size: int, squared: bool) -> float:
    """Approximate sum log(t)/t or sum (log(t)/t)^2."""
    a = float(start_time)
    b = float(start_time + block_size - 1)

    if block_size <= 0:
        return 0.0
    if block_size == 1:
        val = math.log(max(a, 1.0)) / max(a, 1.0)
        return val * val if squared else val

    if not squared:
        # Integral of log(x)/x is 0.5 log(x)^2.
        la = math.log(max(a, 1.0))
        lb = math.log(max(b, 1.0))
        integral = 0.5 * (lb * lb - la * la)

        def f(x: float) -> float:
            return math.log(max(x, 1.0)) / max(x, 1.0)

        # derivative for x > 1: (1 - log x) / x^2.
        def fp(x: float) -> float:
            if x <= 1.0:
                return 1.0
            return (1.0 - math.log(x)) / (x * x)

        estimate = integral + 0.5 * (f(a) + f(b)) + (fp(b) - fp(a)) / 12.0
        return float(max(0.0, estimate))

    # Integral of log(x)^2 / x^2 is -(log(x)^2 + 2 log(x) + 2) / x.
    def F(x: float) -> float:
        x_safe = max(x, 1.0)
        lx = math.log(x_safe)
        return -(lx * lx + 2.0 * lx + 2.0) / x_safe

    def f2(x: float) -> float:
        x_safe = max(x, 1.0)
        lx = math.log(x_safe)
        return (lx / x_safe) ** 2

    def f2p(x: float) -> float:
        if x <= 1.0:
            return 0.0
        lx = math.log(x)
        return 2.0 * lx * (1.0 - lx) / (x ** 3)

    integral = F(b) - F(a)
    estimate = integral + 0.5 * (f2(a) + f2(b)) + (f2p(b) - f2p(a)) / 12.0
    return float(max(0.0, estimate))


def gauss_cell_sum(schedule: ScheduleSpec, start_time: int, block_size: int, squared: bool) -> float:
    """Approximate a discrete sum by integrating over unit cells with 64-point Gauss-Legendre."""
    if block_size <= 0:
        return 0.0

    # Sum f(t) is approximated by int_{start-1/2}^{end+1/2} f(x) dx.
    a = max(1.0, float(start_time) - 0.5)
    b = float(start_time + block_size - 1) + 0.5
    midpoint = 0.5 * (a + b)
    half_width = 0.5 * (b - a)
    x = midpoint + half_width * _GL_NODES
    values = eta_values(schedule, x)
    if squared:
        values = values * values
    return float(half_width * np.dot(_GL_WEIGHTS, values))


def learning_rate_sums(schedule: ScheduleSpec, start_time: int, block_size: int, exact_threshold: int) -> Tuple[float, float]:
    """Return sum eta_t and sum eta_t^2 over one block."""
    if block_size <= exact_threshold:
        return exact_eta_sums(schedule, start_time, block_size)

    if schedule.kind == "power":
        p = float(schedule.power)
        return (
            power_sum_euler_maclaurin(start_time, block_size, p),
            power_sum_euler_maclaurin(start_time, block_size, 2.0 * p),
        )

    if schedule.kind == "log_over_t":
        return (
            log_over_t_sum_euler_maclaurin(start_time, block_size, squared=False),
            log_over_t_sum_euler_maclaurin(start_time, block_size, squared=True),
        )

    if schedule.kind == "sqrt_log_over_t":
        # eta^2 = log(t)/t has an analytic EM approximation; eta itself uses GL.
        return (
            gauss_cell_sum(schedule, start_time, block_size, squared=False),
            log_over_t_sum_euler_maclaurin(start_time, block_size, squared=False),
        )

    if schedule.kind == "inv_log_t":
        return (
            gauss_cell_sum(schedule, start_time, block_size, squared=False),
            gauss_cell_sum(schedule, start_time, block_size, squared=True),
        )

    raise ValueError(f"Unknown schedule kind: {schedule.kind}")


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
    schedule: ScheduleSpec,
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
    end_time = current_time + remaining_steps - 1
    eta_scale = max_eta_over_interval(schedule, current_time, end_time)

    if eta_scale <= 0.0:
        # Only possible for degenerate one-step log schedules at t=1.
        return 1

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

    # Upper bound using second moments of per-round gradient coordinates.
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
        limit_by_mean_change = config.max_mean_score_change_per_block / (eta_scale * mean_gradient_scale)

    limit_by_noise_change = math.inf
    if noise_gradient_scale > 1e-300:
        limit_by_noise_change = (
            config.max_noise_score_change_per_block / (eta_scale * math.sqrt(noise_gradient_scale))
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
    """Sample aggregate eta-weighted reward sums for arms 1 and 2."""
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
    weighted_reward_sum2 = mean2 + sd2 * (correlation * z1 + np.sqrt(1.0 - correlation * correlation) * z2)

    return weighted_reward_sum1, weighted_reward_sum2


def apply_block_update(
    *,
    random_generator: np.random.Generator,
    schedule: ScheduleSpec,
    current_time: int,
    block_size: int,
    gap_arm2: float,
    optimal_arm_score: np.ndarray,
    second_arm_score: np.ndarray,
    optimal_prob: np.ndarray,
    second_prob: np.ndarray,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    sum_eta, sum_eta_squared = learning_rate_sums(
        schedule,
        current_time,
        block_size,
        config.exact_eta_sum_threshold,
    )

    if sum_eta == 0.0 and sum_eta_squared == 0.0:
        return optimal_arm_score, second_arm_score

    weighted_reward_sum1, weighted_reward_sum2 = sample_weighted_reward_sums_gaussian(
        random_generator=random_generator,
        optimal_prob=optimal_prob,
        second_prob=second_prob,
        gap_arm2=gap_arm2,
        sum_eta=sum_eta,
        sum_eta_squared=sum_eta_squared,
    )

    optimal_arm_score = optimal_arm_score + (
        (1.0 - optimal_prob) * weighted_reward_sum1
        - optimal_prob * weighted_reward_sum2
    )
    second_arm_score = second_arm_score + (
        -second_prob * weighted_reward_sum1
        + (1.0 - second_prob) * weighted_reward_sum2
    )

    return optimal_arm_score, second_arm_score


def simulate_sweep_chunk(
    schedule_slug: str,
    gap_arm2: float,
    chunk_trajectories: int,
    seed: int,
    config: SimulationConfig,
) -> Tuple[int, float, float, int, float]:
    """Simulate a chunk for final cumulative regret only."""
    schedule = schedule_by_slug(schedule_slug)
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
            schedule=schedule,
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

        optimal_arm_score, second_arm_score = apply_block_update(
            random_generator=random_generator,
            schedule=schedule,
            current_time=current_time,
            block_size=block_size,
            gap_arm2=gap_arm2,
            optimal_arm_score=optimal_arm_score,
            second_arm_score=second_arm_score,
            optimal_prob=optimal_prob,
            second_prob=second_prob,
            config=config,
        )

        current_time += block_size
        num_blocks += 1

    elapsed_seconds = time.time() - start_wall_time
    return (
        int(chunk_trajectories),
        float(np.sum(regret_by_trajectory)),
        float(np.sum(regret_by_trajectory * regret_by_trajectory)),
        int(num_blocks),
        float(elapsed_seconds),
    )


def simulate_history_chunk(
    schedule_slug: str,
    gap_arm2: float,
    checkpoints: np.ndarray,
    chunk_trajectories: int,
    seed: int,
    config: SimulationConfig,
) -> Tuple[int, np.ndarray, np.ndarray, int, float]:
    """Simulate a chunk and record cumulative regret at checkpoints."""
    schedule = schedule_by_slug(schedule_slug)
    start_wall_time = time.time()
    random_generator = np.random.default_rng(seed)
    num_other_arms = config.num_arms - 2

    optimal_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    second_arm_score = np.zeros(chunk_trajectories, dtype=np.float64)
    regret_by_trajectory = np.zeros(chunk_trajectories, dtype=np.float64)

    history_sum = np.zeros(checkpoints.shape[0], dtype=np.float64)
    history_sum_squared = np.zeros(checkpoints.shape[0], dtype=np.float64)
    checkpoint_index = 0

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
            schedule=schedule,
            current_time=current_time,
            remaining_steps=remaining_steps,
            gap_arm2=gap_arm2,
            optimal_prob=optimal_prob,
            second_prob=second_prob,
            other_prob_per_arm=other_prob_per_arm,
            num_other_arms=num_other_arms,
            config=config,
        )

        block_start_completed = current_time - 1
        block_end_completed = current_time + block_size - 1

        other_total_prob = 1.0 - optimal_prob - second_prob
        instantaneous_regret = other_total_prob + gap_arm2 * second_prob

        # Record all checkpoints crossed by this block.
        while checkpoint_index < checkpoints.shape[0] and checkpoints[checkpoint_index] <= block_end_completed:
            checkpoint = int(checkpoints[checkpoint_index])
            if checkpoint >= current_time:
                offset = checkpoint - block_start_completed
                regret_at_checkpoint = regret_by_trajectory + offset * instantaneous_regret
                history_sum[checkpoint_index] = float(np.sum(regret_at_checkpoint))
                history_sum_squared[checkpoint_index] = float(np.sum(regret_at_checkpoint * regret_at_checkpoint))
            checkpoint_index += 1

        regret_by_trajectory += block_size * instantaneous_regret

        optimal_arm_score, second_arm_score = apply_block_update(
            random_generator=random_generator,
            schedule=schedule,
            current_time=current_time,
            block_size=block_size,
            gap_arm2=gap_arm2,
            optimal_arm_score=optimal_arm_score,
            second_arm_score=second_arm_score,
            optimal_prob=optimal_prob,
            second_prob=second_prob,
            config=config,
        )

        current_time += block_size
        num_blocks += 1

    # Safety for checkpoints equal to horizon if not handled due to integer edge cases.
    while checkpoint_index < checkpoints.shape[0]:
        history_sum[checkpoint_index] = float(np.sum(regret_by_trajectory))
        history_sum_squared[checkpoint_index] = float(np.sum(regret_by_trajectory * regret_by_trajectory))
        checkpoint_index += 1

    elapsed_seconds = time.time() - start_wall_time
    return int(chunk_trajectories), history_sum, history_sum_squared, int(num_blocks), float(elapsed_seconds)


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


def default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def aggregate_mean_se(total_n: int, total_sum: float, total_sum_squared: float) -> Tuple[float, float]:
    mean = total_sum / total_n
    if total_n > 1:
        variance = (total_sum_squared - total_sum * total_sum / total_n) / (total_n - 1)
        variance = max(0.0, variance)
        se = math.sqrt(variance / total_n)
    else:
        se = float("nan")
    return mean, se


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        horizon_steps=int(args.horizon),
        num_arms=int(args.num_arms),
        num_trajectories=int(args.trajectories),
        random_seed=int(args.seed),
        max_mean_score_change_per_block=float(args.max_mean_change),
        max_noise_score_change_per_block=float(args.max_noise_change),
        max_block_size=int(args.max_block_size),
        block_quantile=float(args.block_quantile),
        exact_eta_sum_threshold=int(args.exact_eta_sum_threshold),
        skip_existing=not bool(args.overwrite),
    )


def run_one_sweep_task(
    *,
    schedule: ScheduleSpec,
    schedule_index: int,
    gap_index: int,
    gap_arm2: float,
    args: argparse.Namespace,
    config: SimulationConfig,
) -> pathlib.Path:
    outdir = pathlib.Path(args.outdir)
    sweep_dir = outdir / "sweep" / schedule.slug
    sweep_dir.mkdir(parents=True, exist_ok=True)
    output_path = sweep_dir / f"gap_{gap_index:04d}.csv"
    metadata_path = sweep_dir / f"gap_{gap_index:04d}.json"

    if output_path.exists() and config.skip_existing:
        print(f"[skip] {output_path} already exists", flush=True)
        return output_path

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)
    chunks = split_trajectories(config.num_trajectories, workers, args.chunk_trajectories)

    print(
        f"[sweep] schedule={schedule.slug} gap_index={gap_index} gap={gap_arm2:.10g} "
        f"trajectories={config.num_trajectories} chunks={chunks} workers={workers}",
        flush=True,
    )

    total_n = 0
    total_sum = 0.0
    total_sum_squared = 0.0
    total_blocks = 0
    max_chunk_seconds = 0.0
    start_wall_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk_id, chunk_n in enumerate(chunks):
            chunk_seed = int(config.random_seed + 10_000_019 * schedule_index + 1_000_003 * gap_index + 9_176 * chunk_id)
            futures.append(
                executor.submit(
                    simulate_sweep_chunk,
                    schedule.slug,
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

    mean_regret, standard_error = aggregate_mean_se(total_n, total_sum, total_sum_squared)
    elapsed_total = time.time() - start_wall_time

    row = {
        "schedule_index": schedule_index,
        "schedule_slug": schedule.slug,
        "schedule_label": schedule.label,
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


def combine_sweep_results(args: argparse.Namespace) -> Tuple[pathlib.Path, pathlib.Path]:
    outdir = pathlib.Path(args.outdir)
    paths = sorted((outdir / "sweep").glob("*/gap_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No sweep CSV files found under {outdir / 'sweep'}")

    rows = [read_single_row_csv(path) for path in paths]
    rows.sort(key=lambda r: (int(r["schedule_index"]), int(r["gap_index"])))

    combined_path = outdir / "combined_gap_sweep_all_schedules.csv"
    fieldnames = list(rows[0].keys())
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hardest_rows = []
    by_schedule: Dict[str, List[dict]] = {}
    for row in rows:
        by_schedule.setdefault(row["schedule_slug"], []).append(row)

    for schedule_slug, schedule_rows in by_schedule.items():
        best = max(schedule_rows, key=lambda r: float(r["mean_regret"]))
        hardest_rows.append(best)

    hardest_rows.sort(key=lambda r: int(r["schedule_index"]))
    hardest_path = outdir / "hardest_gaps_by_schedule.csv"
    with hardest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hardest_rows)

    print(f"[combine] wrote {combined_path} with {len(rows)} rows", flush=True)
    print(f"[combine] wrote {hardest_path} with {len(hardest_rows)} hardest gaps", flush=True)

    if args.plot:
        plot_gap_sweep(combined_path, args)

    return combined_path, hardest_path


def load_csv_dicts(path: pathlib.Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def plot_gap_sweep(combined_path: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    rows = load_csv_dicts(combined_path)
    by_schedule: Dict[str, List[dict]] = {}
    for row in rows:
        by_schedule.setdefault(row["schedule_slug"], []).append(row)

    scale = float(args.regret_scale)
    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=180)

    for schedule in all_schedules():
        schedule_rows = by_schedule.get(schedule.slug)
        if not schedule_rows:
            continue
        schedule_rows.sort(key=lambda r: float(r["gap_arm2"]))
        gap = np.array([float(r["gap_arm2"]) for r in schedule_rows])
        mean = np.array([float(r["mean_regret"]) for r in schedule_rows]) / scale
        se = np.array([float(r["standard_error"]) for r in schedule_rows]) / scale
        ax.plot(gap, mean, linewidth=2.0, label=schedule.label)
        ax.fill_between(gap, mean - 2.0 * se, mean + 2.0 * se, alpha=0.08, linewidth=0)

    ax.set_xlim(float(args.gap_start), float(args.gap_stop))
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("Expected regret" if scale == 1.0 else rf"Expected regret / ${scale:.0e}$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    plot_path = pathlib.Path(args.outdir) / "gap_sweep_all_schedules.png"
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[plot] wrote {plot_path}", flush=True)
    return plot_path


def make_checkpoints(horizon_steps: int, num_checkpoints: int, include_linear_early: bool = True) -> np.ndarray:
    if num_checkpoints < 2:
        raise ValueError("--num-checkpoints must be at least 2")

    log_points = np.unique(np.round(np.geomspace(1, horizon_steps, num_checkpoints)).astype(np.int64))
    if include_linear_early:
        early_limit = min(horizon_steps, 10_000)
        early = np.arange(1, early_limit + 1, dtype=np.int64)
        points = np.unique(np.concatenate([early, log_points]))
    else:
        points = log_points
    points = points[(points >= 1) & (points <= horizon_steps)]
    if points[-1] != horizon_steps:
        points = np.append(points, np.int64(horizon_steps))
    return points.astype(np.int64)


def run_one_history(
    *,
    schedule: ScheduleSpec,
    schedule_index: int,
    gap_arm2: float,
    gap_index: int,
    checkpoints: np.ndarray,
    args: argparse.Namespace,
    config: SimulationConfig,
) -> pathlib.Path:
    outdir = pathlib.Path(args.outdir)
    history_dir = outdir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    output_path = history_dir / f"history_{schedule.slug}.csv"
    metadata_path = history_dir / f"history_{schedule.slug}.json"

    if output_path.exists() and config.skip_existing:
        print(f"[skip] {output_path} already exists", flush=True)
        return output_path

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)
    chunks = split_trajectories(config.num_trajectories, workers, args.chunk_trajectories)

    print(
        f"[history] schedule={schedule.slug} hardest_gap={gap_arm2:.10g} gap_index={gap_index} "
        f"checkpoints={len(checkpoints)} trajectories={config.num_trajectories} chunks={chunks} workers={workers}",
        flush=True,
    )

    total_n = 0
    total_sum = np.zeros(checkpoints.shape[0], dtype=np.float64)
    total_sum_squared = np.zeros(checkpoints.shape[0], dtype=np.float64)
    total_blocks = 0
    max_chunk_seconds = 0.0
    start_wall_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk_id, chunk_n in enumerate(chunks):
            chunk_seed = int(config.random_seed + 50_000_021 * schedule_index + 1_000_003 * gap_index + 9_176 * chunk_id)
            futures.append(
                executor.submit(
                    simulate_history_chunk,
                    schedule.slug,
                    float(gap_arm2),
                    checkpoints,
                    int(chunk_n),
                    chunk_seed,
                    config,
                )
            )

        for future in as_completed(futures):
            chunk_n, history_sum, history_sum_squared, num_blocks, elapsed_seconds = future.result()
            total_n += chunk_n
            total_sum += history_sum
            total_sum_squared += history_sum_squared
            total_blocks += num_blocks
            max_chunk_seconds = max(max_chunk_seconds, elapsed_seconds)
            print(
                f"  finished history chunk: n={chunk_n} final_mean={history_sum[-1]/chunk_n:.6e} "
                f"blocks={num_blocks} seconds={elapsed_seconds:.1f}",
                flush=True,
            )

    mean = total_sum / total_n
    if total_n > 1:
        variance = (total_sum_squared - total_sum * total_sum / total_n) / (total_n - 1)
        variance = np.maximum(0.0, variance)
        se = np.sqrt(variance / total_n)
    else:
        se = np.full_like(mean, np.nan)

    rows = []
    for t, m, s in zip(checkpoints, mean, se):
        rows.append(
            {
                "schedule_index": schedule_index,
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "hardest_gap_index": gap_index,
                "hardest_gap_arm2": float(gap_arm2),
                "time": int(t),
                "mean_regret": float(m),
                "standard_error": float(s),
                "num_trajectories": total_n,
                "horizon_steps": config.horizon_steps,
                "num_arms": config.num_arms,
                "simulation": "blocked_gaussian_aggregate",
            }
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "config": asdict(config),
        "schedule": asdict(schedule),
        "hardest_gap_arm2": float(gap_arm2),
        "hardest_gap_index": int(gap_index),
        "num_checkpoints": int(checkpoints.shape[0]),
        "num_chunks": len(chunks),
        "workers": workers,
        "mean_blocks_per_chunk": total_blocks / len(chunks),
        "max_chunk_seconds": max_chunk_seconds,
        "wall_seconds": time.time() - start_wall_time,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[history done] wrote {output_path} final_mean={mean[-1]:.6e} final_se={se[-1]:.6e}", flush=True)
    return output_path


def run_histories(args: argparse.Namespace) -> List[pathlib.Path]:
    outdir = pathlib.Path(args.outdir)
    hardest_path = pathlib.Path(args.hardest_csv) if args.hardest_csv else outdir / "hardest_gaps_by_schedule.csv"
    if not hardest_path.exists():
        raise FileNotFoundError(f"Hardest-gap CSV not found: {hardest_path}. Run combine-sweep first.")

    hardest_rows = load_csv_dicts(hardest_path)
    hardest_rows.sort(key=lambda r: int(r["schedule_index"]))

    if args.history_schedule_index is not None:
        index = int(args.history_schedule_index)
        hardest_rows = [r for r in hardest_rows if int(r["schedule_index"]) == index]
        if not hardest_rows:
            raise ValueError(f"No schedule_index={index} found in {hardest_path}")

    if args.schedule_slugs and args.schedule_slugs.strip().lower() not in {"", "all"}:
        requested = {s.strip() for s in args.schedule_slugs.split(",") if s.strip()}
        hardest_rows = [r for r in hardest_rows if r["schedule_slug"] in requested]

    config = config_from_args(args)
    checkpoints = make_checkpoints(
        config.horizon_steps,
        int(args.num_checkpoints),
        include_linear_early=args.linear_early_checkpoints,
    )

    output_paths = []
    for row in hardest_rows:
        schedule = schedule_by_slug(row["schedule_slug"])
        output_paths.append(
            run_one_history(
                schedule=schedule,
                schedule_index=int(row["schedule_index"]),
                gap_arm2=float(row["gap_arm2"]),
                gap_index=int(row["gap_index"]),
                checkpoints=checkpoints,
                args=args,
                config=config,
            )
        )

    if args.plot:
        plot_histories(args)

    return output_paths


def load_history_file(path: pathlib.Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def plot_histories(args: argparse.Namespace) -> pathlib.Path:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")

    outdir = pathlib.Path(args.outdir)
    history_dir = outdir / "history"
    paths = sorted(history_dir.glob("history_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No history_*.csv files found in {history_dir}")

    histories: List[Tuple[ScheduleSpec, np.ndarray, np.ndarray, np.ndarray, float]] = []
    max_final_regret = 0.0
    horizon = 0

    for path in paths:
        rows = load_history_file(path)
        if not rows:
            continue
        schedule = schedule_by_slug(rows[0]["schedule_slug"])
        t = np.array([int(r["time"]) for r in rows], dtype=np.float64)
        mean = np.array([float(r["mean_regret"]) for r in rows], dtype=np.float64)
        se = np.array([float(r["standard_error"]) for r in rows], dtype=np.float64)
        gap = float(rows[0]["hardest_gap_arm2"])
        histories.append((schedule, t, mean, se, gap))
        max_final_regret = max(max_final_regret, float(mean[-1]))
        horizon = max(horizon, int(t[-1]))

    if not histories:
        raise RuntimeError("No nonempty history files found.")

    scale = float(args.regret_scale)
    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=180)

    for schedule, t, mean, se, gap in histories:
        label = f"{schedule.label}, hardest $\\Delta={gap:.4g}$"
        ax.plot(t, mean / scale, linewidth=2.2, label=label)
        if args.show_history_bands:
            ax.fill_between(t, (mean - 2.0 * se) / scale, (mean + 2.0 * se) / scale, alpha=0.08, linewidth=0)

    # Growth-rate baselines. Default: normalized guides that meet the largest final regret at horizon.
    t_ref = histories[0][1]
    if args.baseline_mode == "raw":
        baseline_scale = 1.0
        guide_label_suffix = ""
    else:
        baseline_scale = max_final_regret
        guide_label_suffix = " guide"

    baseline_specs = [
        (0.5, r"$\sqrt{T}$"),
        (2.0 / 3.0, r"$T^{2/3}$"),
        (1.0, r"$T$"),
    ]

    for alpha, label in baseline_specs:
        if args.baseline_mode == "raw":
            guide = np.power(t_ref, alpha)
        else:
            guide = baseline_scale * np.power(t_ref / float(horizon), alpha)
        ax.plot(t_ref, guide / scale, linestyle="--", linewidth=1.8, alpha=0.85, label=label + guide_label_suffix)

    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative regret" if scale == 1.0 else rf"Average cumulative regret / ${scale:.0e}$")
    ax.set_title("Hardest-gap regret histories by stepsize schedule")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    plot_path = outdir / "hardest_gap_regret_histories_all_schedules.png"
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[plot] wrote {plot_path}", flush=True)
    return plot_path


def add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon", type=int, default=10**9)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--outdir", type=str, default="multi_schedule_results")
    parser.add_argument("--max-mean-change", type=float, default=0.20)
    parser.add_argument("--max-noise-change", type=float, default=0.80)
    parser.add_argument("--max-block-size", type=int, default=50_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gap sweep, hardest-gap selection, and regret-history plotting for multiple eta_t schedules."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser("sweep", help="Run one or all schedule-gap sweep tasks.")
    add_common_sim_args(sweep)
    sweep.add_argument("--task-index", type=int, default=None,
                       help="Flat task index over selected schedules x gaps. Useful for SLURM arrays.")
    sweep.add_argument("--schedule-index", type=int, default=None,
                       help="Schedule index within selected schedules.")
    sweep.add_argument("--gap-index", type=int, default=None)
    sweep.add_argument("--run-all", action="store_true", help="Run all selected schedule-gap tasks sequentially.")
    sweep.add_argument("--schedule-slugs", type=str, default="all",
                       help="Comma-separated slugs or 'all'.")
    sweep.add_argument("--gap-start", type=float, default=0.0)
    sweep.add_argument("--gap-stop", type=float, default=1.0)
    sweep.add_argument("--num-gaps", type=int, default=101)

    combine = subparsers.add_parser("combine-sweep", help="Combine sweep outputs and find hardest gap per schedule.")
    combine.add_argument("--outdir", type=str, default="multi_schedule_results")
    combine.add_argument("--plot", action="store_true")
    combine.add_argument("--regret-scale", type=float, default=1_000_000.0)
    combine.add_argument("--gap-start", type=float, default=0.0)
    combine.add_argument("--gap-stop", type=float, default=1.0)

    history = subparsers.add_parser("history", help="Run regret histories at the hardest gap for each schedule.")
    add_common_sim_args(history)
    history.add_argument("--hardest-csv", type=str, default=None)
    history.add_argument("--history-schedule-index", type=int, default=None,
                         help="Only run this schedule_index from hardest_gaps_by_schedule.csv.")
    history.add_argument("--schedule-slugs", type=str, default="all")
    history.add_argument("--num-checkpoints", type=int, default=250)
    history.add_argument("--linear-early-checkpoints", action="store_true",
                         help="Also record every integer time from 1 to 10000. Slower, but useful for zooming into the start.")
    history.add_argument("--plot", action="store_true")
    history.add_argument("--regret-scale", type=float, default=1_000_000.0)
    history.add_argument("--baseline-mode", choices=["match-final", "raw"], default="match-final")
    history.add_argument("--show-history-bands", action="store_true")

    plot_history = subparsers.add_parser("plot-history", help="Plot existing history CSV files plus baselines.")
    plot_history.add_argument("--outdir", type=str, default="multi_schedule_results")
    plot_history.add_argument("--regret-scale", type=float, default=1_000_000.0)
    plot_history.add_argument("--baseline-mode", choices=["match-final", "raw"], default="match-final")
    plot_history.add_argument("--show-history-bands", action="store_true")

    list_schedules = subparsers.add_parser("list-schedules", help="Print available schedules.")

    return parser


def handle_sweep(args: argparse.Namespace) -> None:
    schedules = selected_schedules(args.schedule_slugs)
    gaps = gap_grid(args.gap_start, args.gap_stop, args.num_gaps)
    config = config_from_args(args)

    if args.task_index is not None:
        total_tasks = len(schedules) * len(gaps)
        if args.task_index < 0 or args.task_index >= total_tasks:
            raise IndexError(f"--task-index must be in [0, {total_tasks - 1}]")
        schedule_index = args.task_index // len(gaps)
        gap_index = args.task_index % len(gaps)
        schedule = schedules[schedule_index]
        run_one_sweep_task(
            schedule=schedule,
            schedule_index=all_schedules().index(schedule),
            gap_index=gap_index,
            gap_arm2=float(gaps[gap_index]),
            args=args,
            config=config,
        )
        return

    if args.schedule_index is not None and args.gap_index is not None:
        if args.schedule_index < 0 or args.schedule_index >= len(schedules):
            raise IndexError(f"--schedule-index must be in [0, {len(schedules) - 1}]")
        if args.gap_index < 0 or args.gap_index >= len(gaps):
            raise IndexError(f"--gap-index must be in [0, {len(gaps) - 1}]")
        schedule = schedules[args.schedule_index]
        run_one_sweep_task(
            schedule=schedule,
            schedule_index=all_schedules().index(schedule),
            gap_index=args.gap_index,
            gap_arm2=float(gaps[args.gap_index]),
            args=args,
            config=config,
        )
        return

    if args.run_all:
        for local_schedule_index, schedule in enumerate(schedules):
            global_schedule_index = all_schedules().index(schedule)
            for gap_index, gap_arm2 in enumerate(gaps):
                run_one_sweep_task(
                    schedule=schedule,
                    schedule_index=global_schedule_index,
                    gap_index=gap_index,
                    gap_arm2=float(gap_arm2),
                    args=args,
                    config=config,
                )
        return

    raise ValueError("For sweep, use --task-index, or --schedule-index with --gap-index, or --run-all.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-schedules":
        for i, schedule in enumerate(all_schedules()):
            print(f"{i}: {schedule.slug:20s} {schedule.label}")
        return

    if args.command == "sweep":
        handle_sweep(args)
        return

    if args.command == "combine-sweep":
        combine_sweep_results(args)
        return

    if args.command == "history":
        run_histories(args)
        return

    if args.command == "plot-history":
        plot_histories(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
