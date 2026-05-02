#!/usr/bin/env python3
"""
Core simulation code for softmax policy-gradient bandits.

This file intentionally contains only the algorithmic pieces:
  - learning-rate schedules eta_t,
  - exact Algorithm-1 updates,
  - blocked aggregate approximation updates,
  - regret and softmax utilities.

No plotting, no CSV management, and no SLURM / Compute Canada logic are kept
here. Those are in run_pg_bandit_experiments.py.

Bandit instance used throughout:
    mu    = (1, 1 - Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)

Only theta_1 and theta_2 are stored explicitly. Arms 3..k stay symmetric, and
sum_a theta_a = 0, so theta_other = -(theta_1 + theta_2) / (k - 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSpec:
    slug: str
    label: str
    kind: str
    power: Optional[float] = None


def all_schedules() -> List[ScheduleSpec]:
    """The six schedules used in the experiments."""
    return [
        ScheduleSpec("inv_t", r"$\eta_t = 1/t$", "power", 1.0),
        ScheduleSpec("inv_t_two_thirds", r"$\eta_t = 1/t^{2/3}$", "power", 2.0 / 3.0),
        ScheduleSpec("log_over_t", r"$\eta_t = \log(t)/t$", "log_over_t"),
        ScheduleSpec("inv_sqrt_t", r"$\eta_t = 1/\sqrt{t}$", "power", 0.5),
        ScheduleSpec("sqrt_log_over_t", r"$\eta_t = \sqrt{\log(t)/t}$", "sqrt_log_over_t"),
        # eta_t = 1/log(t) is singular at t=1. We use log(max(t, 2)).
        ScheduleSpec("inv_log_t", r"$\eta_t = 1/\log(t)$", "inv_log_t"),
    ]


def schedule_by_slug(slug: str) -> ScheduleSpec:
    for schedule in all_schedules():
        if schedule.slug == slug:
            return schedule
    valid = ", ".join(s.slug for s in all_schedules())
    raise KeyError(f"Unknown schedule {slug!r}. Valid schedules: {valid}")


def select_schedules(slugs: Optional[str]) -> List[ScheduleSpec]:
    if slugs is None or slugs.strip().lower() in {"", "all"}:
        return all_schedules()
    return [schedule_by_slug(s.strip()) for s in slugs.split(",") if s.strip()]


def eta_value(schedule: ScheduleSpec, t: np.ndarray | float | int) -> np.ndarray | float:
    """Evaluate eta_t for scalar or numpy-array times t >= 1."""
    x = np.asarray(t, dtype=np.float64)
    x_safe = np.maximum(x, 1.0)

    if schedule.kind == "power":
        value = np.power(x_safe, -float(schedule.power))
    elif schedule.kind == "log_over_t":
        value = np.log(x_safe) / x_safe
    elif schedule.kind == "sqrt_log_over_t":
        value = np.sqrt(np.maximum(np.log(x_safe), 0.0) / x_safe)
    elif schedule.kind == "inv_log_t":
        value = 1.0 / np.log(np.maximum(x_safe, 2.0))
    else:
        raise ValueError(f"Unknown schedule kind: {schedule.kind}")

    if np.isscalar(t):
        return float(value)
    return value


def max_eta_over_interval(schedule: ScheduleSpec, start_time: int, end_time: int) -> float:
    """Conservative maximum eta_t over integer times in [start_time, end_time]."""
    s = float(start_time)
    e = float(end_time)

    if schedule.kind == "power":
        return float(eta_value(schedule, s))

    candidates = [s, e]

    # log(t)/t and sqrt(log(t)/t) peak at t=e on the continuous axis.
    if schedule.kind in {"log_over_t", "sqrt_log_over_t"} and s <= math.e <= e:
        candidates.append(math.e)

    # 1/log(max(t, 2)) is largest at t <= 2.
    if schedule.kind == "inv_log_t" and s <= 2.0 <= e:
        candidates.append(2.0)

    return float(max(eta_value(schedule, c) for c in candidates))


# ---------------------------------------------------------------------------
# Sums of eta_t and eta_t^2 over a block
# ---------------------------------------------------------------------------


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _exact_eta_sums(schedule: ScheduleSpec, start_time: int, block_size: int) -> Tuple[float, float]:
    times = np.arange(start_time, start_time + block_size, dtype=np.float64)
    eta = eta_value(schedule, times)
    return float(np.sum(eta)), float(np.sum(eta * eta))


def _power_sum_euler_maclaurin(start_time: int, block_size: int, power: float) -> float:
    """Approximate sum_{t=start}^{start+B-1} t^{-power}."""
    if block_size <= 0:
        return 0.0
    if block_size == 1:
        return float(start_time) ** (-power)

    a = float(start_time)
    b = float(start_time + block_size - 1)

    if abs(power - 1.0) < 1e-14:
        integral = math.log(b / a)
    else:
        integral = (b ** (1.0 - power) - a ** (1.0 - power)) / (1.0 - power)

    f_a = a ** (-power)
    f_b = b ** (-power)
    fp_a = -power * a ** (-power - 1.0)
    fp_b = -power * b ** (-power - 1.0)
    f3_a = -power * (-power - 1.0) * (-power - 2.0) * a ** (-power - 3.0)
    f3_b = -power * (-power - 1.0) * (-power - 2.0) * b ** (-power - 3.0)

    estimate = integral + 0.5 * (f_a + f_b) + (fp_b - fp_a) / 12.0 - (f3_b - f3_a) / 720.0
    return float(max(0.0, estimate))


def _log_over_t_sum(start_time: int, block_size: int, squared: bool) -> float:
    """Approximate sum log(t)/t or sum (log(t)/t)^2."""
    if block_size <= 0:
        return 0.0
    if block_size == 1:
        val = math.log(max(float(start_time), 1.0)) / max(float(start_time), 1.0)
        return val * val if squared else val

    a = float(start_time)
    b = float(start_time + block_size - 1)

    if not squared:
        la = math.log(max(a, 1.0))
        lb = math.log(max(b, 1.0))
        integral = 0.5 * (lb * lb - la * la)

        def f(x: float) -> float:
            return math.log(max(x, 1.0)) / max(x, 1.0)

        def fp(x: float) -> float:
            if x <= 1.0:
                return 1.0
            return (1.0 - math.log(x)) / (x * x)

        estimate = integral + 0.5 * (f(a) + f(b)) + (fp(b) - fp(a)) / 12.0
        return float(max(0.0, estimate))

    # Integral of log(x)^2 / x^2 is -(log(x)^2 + 2log(x) + 2) / x.
    def antiderivative(x: float) -> float:
        xs = max(x, 1.0)
        lx = math.log(xs)
        return -(lx * lx + 2.0 * lx + 2.0) / xs

    def f2(x: float) -> float:
        xs = max(x, 1.0)
        lx = math.log(xs)
        return (lx / xs) ** 2

    def f2p(x: float) -> float:
        if x <= 1.0:
            return 0.0
        lx = math.log(x)
        return 2.0 * lx * (1.0 - lx) / (x ** 3)

    integral = antiderivative(b) - antiderivative(a)
    estimate = integral + 0.5 * (f2(a) + f2(b)) + (f2p(b) - f2p(a)) / 12.0
    return float(max(0.0, estimate))


def _gauss_cell_sum(schedule: ScheduleSpec, start_time: int, block_size: int, squared: bool) -> float:
    """Approximate sum_t f(t) by integrating f over unit cells around integers."""
    if block_size <= 0:
        return 0.0
    a = max(1.0, float(start_time) - 0.5)
    b = float(start_time + block_size - 1) + 0.5
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * _GAUSS_NODES
    values = eta_value(schedule, x)
    if squared:
        values = values * values
    return float(half * np.dot(_GAUSS_WEIGHTS, values))


def learning_rate_sums(
    schedule: ScheduleSpec,
    start_time: int,
    block_size: int,
    exact_sum_threshold: int = 4096,
) -> Tuple[float, float]:
    """Return sum eta_t and sum eta_t^2 over one block."""
    if block_size <= exact_sum_threshold:
        return _exact_eta_sums(schedule, start_time, block_size)

    if schedule.kind == "power":
        p = float(schedule.power)
        return (
            _power_sum_euler_maclaurin(start_time, block_size, p),
            _power_sum_euler_maclaurin(start_time, block_size, 2.0 * p),
        )

    if schedule.kind == "log_over_t":
        return (
            _log_over_t_sum(start_time, block_size, squared=False),
            _log_over_t_sum(start_time, block_size, squared=True),
        )

    if schedule.kind == "sqrt_log_over_t":
        # eta^2 = log(t)/t has the analytic approximation above.
        return (
            _gauss_cell_sum(schedule, start_time, block_size, squared=False),
            _log_over_t_sum(start_time, block_size, squared=False),
        )

    if schedule.kind == "inv_log_t":
        return (
            _gauss_cell_sum(schedule, start_time, block_size, squared=False),
            _gauss_cell_sum(schedule, start_time, block_size, squared=True),
        )

    raise ValueError(f"Unknown schedule kind: {schedule.kind}")


# ---------------------------------------------------------------------------
# Policy, regret, and block update utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockConfig:
    max_mean_change: float = 0.20
    max_noise_change: float = 0.80
    max_block_size: int = 50_000_000
    block_quantile: float = 0.995
    exact_eta_sum_threshold: int = 4096
    # Critical validation fix: a Gaussian approximation to a one-step
    # categorical update is poor. For small blocks, run literal exact updates.
    exact_small_block_threshold: int = 64


@dataclass
class SimulationOutput:
    schedule_slug: str
    method: str
    gap_arm2: float
    horizon_steps: int
    num_arms: int
    num_trajectories: int
    checkpoint_times: np.ndarray
    sum_regret: np.ndarray
    sumsq_regret: np.ndarray
    sum_pi1: np.ndarray
    sumsq_pi1: np.ndarray
    q10_regret: Optional[np.ndarray]
    q90_regret: Optional[np.ndarray]
    final_regret_samples: Optional[np.ndarray]
    final_pi1_samples: Optional[np.ndarray]
    num_blocks: int

    @property
    def mean_regret(self) -> np.ndarray:
        return self.sum_regret / self.num_trajectories

    @property
    def se_regret(self) -> np.ndarray:
        return standard_errors(self.sum_regret, self.sumsq_regret, self.num_trajectories)

    @property
    def mean_pi1(self) -> np.ndarray:
        return self.sum_pi1 / self.num_trajectories

    @property
    def se_pi1(self) -> np.ndarray:
        return standard_errors(self.sum_pi1, self.sumsq_pi1, self.num_trajectories)


def standard_errors(sums: np.ndarray, sumsq: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.full_like(sums, np.nan, dtype=np.float64)
    variance = (sumsq - sums * sums / n) / (n - 1)
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance / n)


def stable_action_probabilities(
    theta1: np.ndarray,
    theta2: np.ndarray,
    num_other_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax probabilities for arm 1, arm 2, and each symmetric other arm."""
    theta_other = -(theta1 + theta2) / num_other_arms
    max_theta = np.maximum(np.maximum(theta1, theta2), theta_other)

    weight1 = np.exp(theta1 - max_theta)
    weight2 = np.exp(theta2 - max_theta)
    weight_other = np.exp(theta_other - max_theta)

    denom = weight1 + weight2 + num_other_arms * weight_other
    p1 = weight1 / denom
    p2 = weight2 / denom
    p_other_each = weight_other / denom
    return p1, p2, p_other_each


def instantaneous_regret(p1: np.ndarray, p2: np.ndarray, gap_arm2: float) -> np.ndarray:
    """Conditional one-step regret for mu=(1,1-gap,0,...,0)."""
    return (1.0 - p1 - p2) + gap_arm2 * p2


def _high_quantile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    if quantile >= 1.0:
        return float(np.max(values))
    if quantile <= 0.0:
        return float(np.min(values))
    kth = int(math.ceil(quantile * (values.size - 1)))
    return float(np.partition(values, kth)[kth])


def choose_block_size(
    schedule: ScheduleSpec,
    current_time: int,
    remaining_steps: int,
    gap_arm2: float,
    p1: np.ndarray,
    p2: np.ndarray,
    p_other_each: np.ndarray,
    num_other_arms: int,
    block_config: BlockConfig,
) -> int:
    """Adaptive block length for the frozen-policy aggregate approximation."""
    end_time = current_time + remaining_steps - 1
    eta_scale = max_eta_over_interval(schedule, current_time, end_time)
    if eta_scale <= 0.0:
        return 1

    mu2 = 1.0 - gap_arm2
    other_total = num_other_arms * p_other_each
    regret = other_total + gap_arm2 * p2

    mean_g1 = p1 * regret
    mean_g2 = p2 * (regret - gap_arm2)
    mean_g_other = -(mean_g1 + mean_g2) / num_other_arms

    q = block_config.block_quantile
    mean_scale = max(
        _high_quantile(np.abs(mean_g1), q),
        _high_quantile(np.abs(mean_g2), q),
        _high_quantile(np.abs(mean_g_other), q),
    )

    # Second moments of one-step gradient coordinates before multiplying by eta_t.
    ey1_sq = 2.0
    ey2_sq = 1.0 + mu2 * mu2

    second_g1 = p1 * (1.0 - p1) ** 2 * ey1_sq + p2 * p1**2 * ey2_sq
    second_g2 = p1 * p2**2 * ey1_sq + p2 * (1.0 - p2) ** 2 * ey2_sq
    second_g_other = p1 * p_other_each**2 * ey1_sq + p2 * p_other_each**2 * ey2_sq

    noise_scale = max(
        _high_quantile(second_g1, q),
        _high_quantile(second_g2, q),
        _high_quantile(second_g_other, q),
    )

    limit_mean = math.inf
    if mean_scale > 1e-300:
        limit_mean = block_config.max_mean_change / (eta_scale * mean_scale)

    limit_noise = math.inf
    if noise_scale > 1e-300:
        limit_noise = (block_config.max_noise_change / (eta_scale * math.sqrt(noise_scale))) ** 2

    return int(max(1, math.floor(min(
        remaining_steps,
        block_config.max_block_size,
        limit_mean,
        limit_noise,
    ))))


def sample_weighted_reward_sums_gaussian(
    rng: np.random.Generator,
    p1: np.ndarray,
    p2: np.ndarray,
    gap_arm2: float,
    sum_eta: float,
    sum_eta_squared: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian approximation for S_j = sum_t eta_t 1{A_t=j} Y_t, j=1,2."""
    mu1 = 1.0
    mu2 = 1.0 - gap_arm2
    ey1_sq = 2.0
    ey2_sq = 1.0 + mu2 * mu2

    mean1 = p1 * mu1 * sum_eta
    mean2 = p2 * mu2 * sum_eta

    var1 = (p1 * ey1_sq - (p1 * mu1) ** 2) * sum_eta_squared
    var2 = (p2 * ey2_sq - (p2 * mu2) ** 2) * sum_eta_squared
    cov12 = -(p1 * mu1) * (p2 * mu2) * sum_eta_squared

    var1 = np.maximum(var1, 0.0)
    var2 = np.maximum(var2, 0.0)

    z1 = rng.standard_normal(size=p1.shape[0])
    z2 = rng.standard_normal(size=p1.shape[0])

    sd1 = np.sqrt(var1)
    sd2 = np.sqrt(var2)
    denom = sd1 * sd2
    corr = np.divide(cov12, denom, out=np.zeros_like(cov12), where=denom > 1e-300)
    np.clip(corr, -0.999999999, 0.999999999, out=corr)

    s1 = mean1 + sd1 * z1
    s2 = mean2 + sd2 * (corr * z1 + np.sqrt(1.0 - corr * corr) * z2)
    return s1, s2


def apply_gaussian_block_update(
    rng: np.random.Generator,
    schedule: ScheduleSpec,
    current_time: int,
    block_size: int,
    gap_arm2: float,
    theta1: np.ndarray,
    theta2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    block_config: BlockConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one frozen-policy Gaussian aggregate block update."""
    sum_eta, sum_eta_sq = learning_rate_sums(
        schedule,
        current_time,
        block_size,
        block_config.exact_eta_sum_threshold,
    )
    if sum_eta == 0.0 and sum_eta_sq == 0.0:
        return theta1, theta2

    s1, s2 = sample_weighted_reward_sums_gaussian(
        rng=rng,
        p1=p1,
        p2=p2,
        gap_arm2=gap_arm2,
        sum_eta=sum_eta,
        sum_eta_squared=sum_eta_sq,
    )

    theta1 = theta1 + (1.0 - p1) * s1 - p1 * s2
    theta2 = theta2 - p2 * s1 + (1.0 - p2) * s2
    return theta1, theta2


def apply_one_exact_update(
    rng: np.random.Generator,
    schedule: ScheduleSpec,
    time_step: int,
    gap_arm2: float,
    theta1: np.ndarray,
    theta2: np.ndarray,
    cumulative_regret: np.ndarray,
    num_other_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One literal Algorithm-1 update and conditional regret increment."""
    p1, p2, _ = stable_action_probabilities(theta1, theta2, num_other_arms)
    cumulative_regret = cumulative_regret + instantaneous_regret(p1, p2, gap_arm2)

    u = rng.random(theta1.shape[0])
    chose1 = u < p1
    chose2 = (~chose1) & (u < p1 + p2)

    reward = np.zeros(theta1.shape[0], dtype=np.float64)
    n1 = int(np.sum(chose1))
    if n1:
        reward[chose1] = 1.0 + rng.standard_normal(n1)

    n2 = int(np.sum(chose2))
    if n2:
        reward[chose2] = (1.0 - gap_arm2) + rng.standard_normal(n2)

    eta_t = float(eta_value(schedule, time_step))
    if eta_t != 0.0:
        theta1 = theta1 + eta_t * (chose1.astype(np.float64) - p1) * reward
        theta2 = theta2 + eta_t * (chose2.astype(np.float64) - p2) * reward

    return theta1, theta2, cumulative_regret


# ---------------------------------------------------------------------------
# Full simulators
# ---------------------------------------------------------------------------


def _prepare_checkpoints(horizon_steps: int, checkpoints: Optional[Sequence[int]]) -> np.ndarray:
    if checkpoints is None:
        pts = np.asarray([horizon_steps], dtype=np.int64)
    else:
        pts = np.unique(np.asarray(checkpoints, dtype=np.int64))
        pts = pts[(pts >= 1) & (pts <= horizon_steps)]
        if pts.size == 0 or pts[-1] != horizon_steps:
            pts = np.unique(np.concatenate([pts, np.asarray([horizon_steps], dtype=np.int64)]))
    return pts


def log_spaced_checkpoints(horizon_steps: int, count: int, include_early: bool = False) -> np.ndarray:
    if count < 2:
        return np.asarray([horizon_steps], dtype=np.int64)
    pts = np.unique(np.round(np.geomspace(1, horizon_steps, count)).astype(np.int64))
    pts = pts[(pts >= 1) & (pts <= horizon_steps)]
    if include_early:
        early_end = min(horizon_steps, 10_000)
        early = np.arange(1, early_end + 1, dtype=np.int64)
        pts = np.unique(np.concatenate([early, pts]))
    if pts.size == 0 or pts[-1] != horizon_steps:
        pts = np.unique(np.concatenate([pts, np.asarray([horizon_steps], dtype=np.int64)]))
    return pts


def simulate_policy_gradient(
    *,
    schedule_slug: str,
    gap_arm2: float,
    horizon_steps: int,
    num_arms: int = 40,
    num_trajectories: int = 1000,
    random_seed: int = 20260310,
    method: str = "approx",
    checkpoints: Optional[Sequence[int]] = None,
    block_config: Optional[BlockConfig] = None,
    track_quantiles: bool = False,
    return_trajectory_samples: bool = False,
) -> SimulationOutput:
    """Simulate exact or approximate policy gradient.

    method="exact" uses literal round-by-round Algorithm-1 updates.
    method="approx" uses the frozen-policy Gaussian aggregate approximation,
    with exact literal updates for blocks no larger than
    block_config.exact_small_block_threshold.
    """
    if num_arms < 3:
        raise ValueError("This reduced simulator assumes num_arms >= 3.")
    if method not in {"exact", "approx"}:
        raise ValueError("method must be 'exact' or 'approx'.")

    schedule = schedule_by_slug(schedule_slug)
    block_config = block_config or BlockConfig()
    checkpoint_times = _prepare_checkpoints(horizon_steps, checkpoints)

    rng = np.random.default_rng(random_seed)
    num_other_arms = num_arms - 2

    theta1 = np.zeros(num_trajectories, dtype=np.float64)
    theta2 = np.zeros(num_trajectories, dtype=np.float64)
    cumulative_regret = np.zeros(num_trajectories, dtype=np.float64)

    m = checkpoint_times.shape[0]
    sum_regret = np.zeros(m, dtype=np.float64)
    sumsq_regret = np.zeros(m, dtype=np.float64)
    sum_pi1 = np.zeros(m, dtype=np.float64)
    sumsq_pi1 = np.zeros(m, dtype=np.float64)
    q10 = np.full(m, np.nan, dtype=np.float64) if track_quantiles else None
    q90 = np.full(m, np.nan, dtype=np.float64) if track_quantiles else None

    checkpoint_index = 0
    current_time = 1
    num_blocks = 0

    def record_checkpoint(index: int) -> None:
        p1_now, _, _ = stable_action_probabilities(theta1, theta2, num_other_arms)
        sum_regret[index] = float(np.sum(cumulative_regret))
        sumsq_regret[index] = float(np.sum(cumulative_regret * cumulative_regret))
        sum_pi1[index] = float(np.sum(p1_now))
        sumsq_pi1[index] = float(np.sum(p1_now * p1_now))
        if track_quantiles:
            q10[index], q90[index] = np.quantile(cumulative_regret, [0.10, 0.90])

    while current_time <= horizon_steps:
        next_checkpoint = int(checkpoint_times[checkpoint_index])

        if method == "exact":
            theta1, theta2, cumulative_regret = apply_one_exact_update(
                rng=rng,
                schedule=schedule,
                time_step=current_time,
                gap_arm2=gap_arm2,
                theta1=theta1,
                theta2=theta2,
                cumulative_regret=cumulative_regret,
                num_other_arms=num_other_arms,
            )
            num_blocks += 1
            if current_time == next_checkpoint:
                record_checkpoint(checkpoint_index)
                checkpoint_index += 1
            current_time += 1
            continue

        # Approximate mode. Never cross a checkpoint in a Gaussian block.
        p1, p2, p_other_each = stable_action_probabilities(theta1, theta2, num_other_arms)
        remaining_to_horizon = horizon_steps - current_time + 1
        remaining_to_checkpoint = next_checkpoint - current_time + 1
        remaining_steps = min(remaining_to_horizon, remaining_to_checkpoint)

        block_size = choose_block_size(
            schedule=schedule,
            current_time=current_time,
            remaining_steps=remaining_steps,
            gap_arm2=gap_arm2,
            p1=p1,
            p2=p2,
            p_other_each=p_other_each,
            num_other_arms=num_other_arms,
            block_config=block_config,
        )

        if block_size <= block_config.exact_small_block_threshold:
            for _ in range(block_size):
                theta1, theta2, cumulative_regret = apply_one_exact_update(
                    rng=rng,
                    schedule=schedule,
                    time_step=current_time,
                    gap_arm2=gap_arm2,
                    theta1=theta1,
                    theta2=theta2,
                    cumulative_regret=cumulative_regret,
                    num_other_arms=num_other_arms,
                )
                current_time += 1
                num_blocks += 1
                if current_time - 1 == next_checkpoint:
                    record_checkpoint(checkpoint_index)
                    checkpoint_index += 1
                    break
            continue

        # Frozen-policy conditional regret over the block.
        cumulative_regret = cumulative_regret + block_size * instantaneous_regret(p1, p2, gap_arm2)

        theta1, theta2 = apply_gaussian_block_update(
            rng=rng,
            schedule=schedule,
            current_time=current_time,
            block_size=block_size,
            gap_arm2=gap_arm2,
            theta1=theta1,
            theta2=theta2,
            p1=p1,
            p2=p2,
            block_config=block_config,
        )

        current_time += block_size
        num_blocks += 1

        if current_time - 1 == next_checkpoint:
            record_checkpoint(checkpoint_index)
            checkpoint_index += 1

    final_regret_samples = cumulative_regret.copy() if return_trajectory_samples else None
    if return_trajectory_samples:
        p1_final, _, _ = stable_action_probabilities(theta1, theta2, num_other_arms)
        final_pi1_samples = p1_final.copy()
    else:
        final_pi1_samples = None

    return SimulationOutput(
        schedule_slug=schedule.slug,
        method=method,
        gap_arm2=float(gap_arm2),
        horizon_steps=int(horizon_steps),
        num_arms=int(num_arms),
        num_trajectories=int(num_trajectories),
        checkpoint_times=checkpoint_times,
        sum_regret=sum_regret,
        sumsq_regret=sumsq_regret,
        sum_pi1=sum_pi1,
        sumsq_pi1=sumsq_pi1,
        q10_regret=q10,
        q90_regret=q90,
        final_regret_samples=final_regret_samples,
        final_pi1_samples=final_pi1_samples,
        num_blocks=int(num_blocks),
    )
