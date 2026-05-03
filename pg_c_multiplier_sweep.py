#!/usr/bin/env python3
"""
Sweep multiplicative constants c in time-varying policy-gradient stepsizes.

Bandit instance:
    mu    = (1, 1 - Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)

Policy-gradient update:
    theta_{t+1,a} = theta_{t,a} + eta_t * Y_t * (1{A_t=a} - pi_{t,a})

This script tests schedules of the form
    eta_t = c * base_eta_t
where base_eta_t can be 1/t, 1/t^{2/3}, log(t)/t, 1/sqrt(t),
sqrt(log(t)/t), or 1/log(t).  For all log-based schedules, log(t) is
regularized as log(max(t, 2)), matching the short-horizon diagnostic code.

Main experiment idea:
  1. Sweep c and Delta.
  2. For each schedule and c, find the Delta with largest final mean regret.
  3. Plot R_T/T versus c.  If R_T/T stays bounded away from zero and grows
     with c, this supports the finite-horizon almost-linear-regret hypothesis.
  4. At the hardest Delta for each c, plot R_t and R_t/t over time.
  5. Validate exact-vs-approx at T=100k for selected c and Delta values.

Examples:
  # Quick local test, exact update
  python pg_c_multiplier_sweep.py sweep \
      --run-all --method exact --schedule-slugs inv_sqrt_t \
      --c-values 0.25,0.5,1,2 --num-gaps 5 \
      --horizon 2000 --trajectories 200 --workers 4 --outdir test_c_sweep
  python pg_c_multiplier_sweep.py combine-sweep --outdir test_c_sweep
  python pg_c_multiplier_sweep.py plot-sweep --outdir test_c_sweep

  # Large approximate sweep, one SLURM array task
  python pg_c_multiplier_sweep.py sweep \
      --task-index ${SLURM_ARRAY_TASK_ID} \
      --schedule-slugs inv_sqrt_t \
      --c-values 0.125,0.25,0.5,1,2,4,8,16 \
      --num-gaps 101 --horizon 1000000000 --trajectories 50000

  # Exact-vs-approx validation at T=100k
  python pg_c_multiplier_sweep.py validate \
      --schedule-slugs inv_sqrt_t \
      --c-values 0.5,1,2,4,8 \
      --gap-values 0.15,0.2,0.25,0.3 \
      --horizon 100000 --trajectories 1000 --workers 8 --plot
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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------------------------------------------------------
# Schedules and eta sums
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSpec:
    slug: str
    label: str
    kind: str
    power: Optional[float] = None


def all_schedules() -> List[ScheduleSpec]:
    return [
        ScheduleSpec("inv_t", r"$1/t$", "power", 1.0),
        ScheduleSpec("inv_t_two_thirds", r"$1/t^{2/3}$", "power", 2.0 / 3.0),
        ScheduleSpec("log_over_t", r"$\log(t)/t$", "log_over_t"),
        ScheduleSpec("inv_sqrt_t", r"$1/\sqrt{t}$", "power", 0.5),
        ScheduleSpec("sqrt_log_over_t", r"$\sqrt{\log(t)/t}$", "sqrt_log_over_t"),
        ScheduleSpec("inv_log_t", r"$1/\log(t)$", "inv_log_t"),
    ]


def schedule_by_slug(slug: str) -> ScheduleSpec:
    for schedule in all_schedules():
        if schedule.slug == slug:
            return schedule
    valid = ", ".join(s.slug for s in all_schedules())
    raise KeyError(f"Unknown schedule {slug!r}. Valid schedules: {valid}")


def select_schedules(text: Optional[str]) -> List[ScheduleSpec]:
    if text is None or text.strip().lower() in {"", "all"}:
        return all_schedules()
    return [schedule_by_slug(s.strip()) for s in text.split(",") if s.strip()]


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def safe_log(x: np.ndarray | float | int) -> np.ndarray | float:
    """Use log(max(t, 2)) for every log-based schedule."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.log(np.maximum(arr, 2.0))
    if np.isscalar(x):
        return float(out)
    return out


def base_eta(schedule: ScheduleSpec, t: np.ndarray | float | int) -> np.ndarray | float:
    """Base eta_t before multiplying by c."""
    x = np.asarray(t, dtype=np.float64)
    x_safe = np.maximum(x, 1.0)

    if schedule.kind == "power":
        value = np.power(x_safe, -float(schedule.power))
    elif schedule.kind == "log_over_t":
        value = safe_log(x_safe) / x_safe
    elif schedule.kind == "sqrt_log_over_t":
        value = np.sqrt(safe_log(x_safe) / x_safe)
    elif schedule.kind == "inv_log_t":
        value = 1.0 / safe_log(x_safe)
    else:
        raise ValueError(f"Unknown schedule kind: {schedule.kind}")

    if np.isscalar(t):
        return float(value)
    return value


def eta_value(schedule: ScheduleSpec, c_multiplier: float, t: np.ndarray | float | int) -> np.ndarray | float:
    return c_multiplier * base_eta(schedule, t)


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _exact_base_eta_sums(schedule: ScheduleSpec, start_time: int, block_size: int) -> Tuple[float, float]:
    t = np.arange(start_time, start_time + block_size, dtype=np.float64)
    eta = base_eta(schedule, t)
    return float(np.sum(eta)), float(np.sum(eta * eta))


def _power_sum_euler_maclaurin(start_time: int, block_size: int, power: float) -> float:
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


def _gauss_cell_base_sum(schedule: ScheduleSpec, start_time: int, block_size: int, squared: bool) -> float:
    """Approximate sum f(t) by integral over unit cells centered at integers."""
    if block_size <= 0:
        return 0.0
    a = max(1.0, float(start_time) - 0.5)
    b = float(start_time + block_size - 1) + 0.5
    midpoint = 0.5 * (a + b)
    half_width = 0.5 * (b - a)
    x = midpoint + half_width * _GAUSS_NODES
    values = base_eta(schedule, x)
    if squared:
        values = values * values
    return float(half_width * np.dot(_GAUSS_WEIGHTS, values))


def base_eta_sums(schedule: ScheduleSpec, start_time: int, block_size: int, exact_threshold: int) -> Tuple[float, float]:
    """Return sum base_eta_t and sum base_eta_t^2 over one block."""
    if block_size <= exact_threshold:
        return _exact_base_eta_sums(schedule, start_time, block_size)

    if schedule.kind == "power":
        p = float(schedule.power)
        return (
            _power_sum_euler_maclaurin(start_time, block_size, p),
            _power_sum_euler_maclaurin(start_time, block_size, 2.0 * p),
        )

    return (
        _gauss_cell_base_sum(schedule, start_time, block_size, squared=False),
        _gauss_cell_base_sum(schedule, start_time, block_size, squared=True),
    )


def scaled_eta_sums(
    schedule: ScheduleSpec,
    c_multiplier: float,
    start_time: int,
    block_size: int,
    exact_threshold: int,
) -> Tuple[float, float]:
    s1, s2 = base_eta_sums(schedule, start_time, block_size, exact_threshold)
    return c_multiplier * s1, c_multiplier * c_multiplier * s2


def max_base_eta_over_interval(schedule: ScheduleSpec, start_time: int, end_time: int) -> float:
    s = float(start_time)
    e = float(end_time)

    if schedule.kind == "power":
        return float(base_eta(schedule, s))

    candidates = [s, e]

    # log(t)/t and sqrt(log(t)/t), with safe_log, increase initially and peak near e.
    if schedule.kind in {"log_over_t", "sqrt_log_over_t"}:
        peak = math.e
        if s <= peak <= e:
            candidates.append(peak)
        # Because of log(max(t,2)), there is also a kink at t=2.
        if s <= 2.0 <= e:
            candidates.append(2.0)

    if schedule.kind == "inv_log_t":
        if s <= 2.0 <= e:
            candidates.append(2.0)

    return float(max(base_eta(schedule, c) for c in candidates))


# ---------------------------------------------------------------------------
# Simulator core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockConfig:
    max_mean_change: float = 0.20
    max_noise_change: float = 0.80
    max_block_size: int = 50_000_000
    block_quantile: float = 0.995
    exact_eta_sum_threshold: int = 4096
    exact_small_block_threshold: int = 64


@dataclass
class SimResult:
    schedule_slug: str
    c_multiplier: float
    method: str
    regret_mode: str
    gap_arm2: float
    horizon_steps: int
    num_arms: int
    num_trajectories: int
    checkpoint_times: np.ndarray
    sum_regret: np.ndarray
    sumsq_regret: np.ndarray
    sum_pi1: np.ndarray
    sumsq_pi1: np.ndarray
    final_regret_samples: Optional[np.ndarray]
    final_pi1_samples: Optional[np.ndarray]
    num_blocks: int

    @property
    def mean_regret(self) -> np.ndarray:
        return self.sum_regret / self.num_trajectories

    @property
    def se_regret(self) -> np.ndarray:
        return standard_error(self.sum_regret, self.sumsq_regret, self.num_trajectories)

    @property
    def mean_pi1(self) -> np.ndarray:
        return self.sum_pi1 / self.num_trajectories

    @property
    def se_pi1(self) -> np.ndarray:
        return standard_error(self.sum_pi1, self.sumsq_pi1, self.num_trajectories)


def standard_error(sums: np.ndarray, sumsq: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.full_like(sums, np.nan)
    var = (sumsq - sums * sums / n) / (n - 1)
    var = np.maximum(var, 0.0)
    return np.sqrt(var / n)


def stable_action_probabilities(theta1: np.ndarray, theta2: np.ndarray, num_other_arms: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_other = -(theta1 + theta2) / num_other_arms
    max_theta = np.maximum(np.maximum(theta1, theta2), theta_other)
    w1 = np.exp(theta1 - max_theta)
    w2 = np.exp(theta2 - max_theta)
    w_other = np.exp(theta_other - max_theta)
    denom = w1 + w2 + num_other_arms * w_other
    return w1 / denom, w2 / denom, w_other / denom


def instantaneous_regret(p1: np.ndarray, p2: np.ndarray, gap_arm2: float) -> np.ndarray:
    return (1.0 - p1 - p2) + gap_arm2 * p2


def _high_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    if q >= 1.0:
        return float(np.max(values))
    if q <= 0.0:
        return float(np.min(values))
    kth = int(math.ceil(q * (values.size - 1)))
    return float(np.partition(values, kth)[kth])


def choose_block_size(
    schedule: ScheduleSpec,
    c_multiplier: float,
    current_time: int,
    remaining_steps: int,
    gap_arm2: float,
    p1: np.ndarray,
    p2: np.ndarray,
    p_other_each: np.ndarray,
    num_other_arms: int,
    block_config: BlockConfig,
) -> int:
    end_time = current_time + remaining_steps - 1
    eta_scale = c_multiplier * max_base_eta_over_interval(schedule, current_time, end_time)

    if eta_scale <= 0.0:
        return remaining_steps

    mu2 = 1.0 - gap_arm2
    p_other_total = num_other_arms * p_other_each
    regret = p_other_total + gap_arm2 * p2

    mean_g1 = p1 * regret
    mean_g2 = p2 * (regret - gap_arm2)
    mean_g_other = -(mean_g1 + mean_g2) / num_other_arms

    q = block_config.block_quantile
    mean_scale = max(
        _high_quantile(np.abs(mean_g1), q),
        _high_quantile(np.abs(mean_g2), q),
        _high_quantile(np.abs(mean_g_other), q),
    )

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
    sum_eta_sq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian approximation for sum eta_t 1{A_t=j} Y_t, j=1,2."""
    mu1 = 1.0
    mu2 = 1.0 - gap_arm2
    ey1_sq = 2.0
    ey2_sq = 1.0 + mu2 * mu2

    mean1 = p1 * mu1 * sum_eta
    mean2 = p2 * mu2 * sum_eta

    var1 = (p1 * ey1_sq - (p1 * mu1) ** 2) * sum_eta_sq
    var2 = (p2 * ey2_sq - (p2 * mu2) ** 2) * sum_eta_sq
    cov12 = -(p1 * mu1) * (p2 * mu2) * sum_eta_sq

    var1 = np.maximum(var1, 0.0)
    var2 = np.maximum(var2, 0.0)

    z1 = rng.standard_normal(p1.shape[0])
    z2 = rng.standard_normal(p1.shape[0])

    sd1 = np.sqrt(var1)
    sd2 = np.sqrt(var2)
    denom = sd1 * sd2
    corr = np.divide(cov12, denom, out=np.zeros_like(cov12), where=denom > 1e-300)
    np.clip(corr, -0.999999999, 0.999999999, out=corr)

    s1 = mean1 + sd1 * z1
    s2 = mean2 + sd2 * (corr * z1 + np.sqrt(1.0 - corr * corr) * z2)
    return s1, s2


def apply_one_exact_update(
    rng: np.random.Generator,
    schedule: ScheduleSpec,
    c_multiplier: float,
    time_step: int,
    gap_arm2: float,
    theta1: np.ndarray,
    theta2: np.ndarray,
    cumulative_regret: np.ndarray,
    num_other_arms: int,
    regret_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1, p2, _ = stable_action_probabilities(theta1, theta2, num_other_arms)

    if regret_mode == "conditional":
        cumulative_regret = cumulative_regret + instantaneous_regret(p1, p2, gap_arm2)

    u = rng.random(theta1.shape[0])
    chose1 = u < p1
    chose2 = (~chose1) & (u < p1 + p2)
    chose_other = ~(chose1 | chose2)

    if regret_mode == "realized":
        cumulative_regret = cumulative_regret + gap_arm2 * chose2.astype(np.float64) + chose_other.astype(np.float64)

    reward = np.zeros(theta1.shape[0], dtype=np.float64)
    n1 = int(np.sum(chose1))
    if n1:
        reward[chose1] = 1.0 + rng.standard_normal(n1)
    n2 = int(np.sum(chose2))
    if n2:
        reward[chose2] = (1.0 - gap_arm2) + rng.standard_normal(n2)

    eta_t = eta_value(schedule, c_multiplier, time_step)
    if eta_t != 0.0:
        theta1 = theta1 + eta_t * (chose1.astype(np.float64) - p1) * reward
        theta2 = theta2 + eta_t * (chose2.astype(np.float64) - p2) * reward

    return theta1, theta2, cumulative_regret


def apply_gaussian_block_update(
    rng: np.random.Generator,
    schedule: ScheduleSpec,
    c_multiplier: float,
    current_time: int,
    block_size: int,
    gap_arm2: float,
    theta1: np.ndarray,
    theta2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    block_config: BlockConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    sum_eta, sum_eta_sq = scaled_eta_sums(
        schedule,
        c_multiplier,
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
        sum_eta_sq=sum_eta_sq,
    )

    theta1 = theta1 + (1.0 - p1) * s1 - p1 * s2
    theta2 = theta2 - p2 * s1 + (1.0 - p2) * s2
    return theta1, theta2


def checkpoints_for(horizon_steps: int, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([horizon_steps], dtype=np.int64)
    pts = np.unique(np.round(np.geomspace(1, horizon_steps, count)).astype(np.int64))
    pts = pts[(pts >= 1) & (pts <= horizon_steps)]
    if pts.size == 0 or pts[-1] != horizon_steps:
        pts = np.unique(np.concatenate([pts, np.asarray([horizon_steps], dtype=np.int64)]))
    return pts


def simulate_policy_gradient(
    *,
    schedule_slug: str,
    c_multiplier: float,
    gap_arm2: float,
    horizon_steps: int,
    num_arms: int,
    num_trajectories: int,
    random_seed: int,
    method: str,
    regret_mode: str,
    checkpoints: Sequence[int],
    block_config: BlockConfig,
    return_final_samples: bool = False,
) -> SimResult:
    if num_arms < 3:
        raise ValueError("num_arms must be at least 3")
    if method not in {"exact", "approx"}:
        raise ValueError("method must be exact or approx")
    if regret_mode not in {"conditional", "realized"}:
        raise ValueError("regret_mode must be conditional or realized")

    schedule = schedule_by_slug(schedule_slug)
    rng = np.random.default_rng(random_seed)
    num_other_arms = num_arms - 2

    checkpoint_times = np.unique(np.asarray(checkpoints, dtype=np.int64))
    checkpoint_times = checkpoint_times[(checkpoint_times >= 1) & (checkpoint_times <= horizon_steps)]
    if checkpoint_times.size == 0 or checkpoint_times[-1] != horizon_steps:
        checkpoint_times = np.unique(np.concatenate([checkpoint_times, np.asarray([horizon_steps], dtype=np.int64)]))

    theta1 = np.zeros(num_trajectories, dtype=np.float64)
    theta2 = np.zeros(num_trajectories, dtype=np.float64)
    cumulative_regret = np.zeros(num_trajectories, dtype=np.float64)

    m = checkpoint_times.shape[0]
    sum_regret = np.zeros(m, dtype=np.float64)
    sumsq_regret = np.zeros(m, dtype=np.float64)
    sum_pi1 = np.zeros(m, dtype=np.float64)
    sumsq_pi1 = np.zeros(m, dtype=np.float64)

    def record(index: int) -> None:
        p1_now, _, _ = stable_action_probabilities(theta1, theta2, num_other_arms)
        sum_regret[index] = float(np.sum(cumulative_regret))
        sumsq_regret[index] = float(np.sum(cumulative_regret * cumulative_regret))
        sum_pi1[index] = float(np.sum(p1_now))
        sumsq_pi1[index] = float(np.sum(p1_now * p1_now))

    current_time = 1
    checkpoint_index = 0
    num_blocks = 0

    while current_time <= horizon_steps:
        next_checkpoint = int(checkpoint_times[checkpoint_index])

        if method == "exact":
            theta1, theta2, cumulative_regret = apply_one_exact_update(
                rng,
                schedule,
                c_multiplier,
                current_time,
                gap_arm2,
                theta1,
                theta2,
                cumulative_regret,
                num_other_arms,
                regret_mode,
            )
            num_blocks += 1
            if current_time == next_checkpoint:
                record(checkpoint_index)
                checkpoint_index += 1
            current_time += 1
            continue

        p1, p2, p_other_each = stable_action_probabilities(theta1, theta2, num_other_arms)
        remaining_to_horizon = horizon_steps - current_time + 1
        remaining_to_checkpoint = next_checkpoint - current_time + 1
        remaining_steps = min(remaining_to_horizon, remaining_to_checkpoint)

        block_size = choose_block_size(
            schedule,
            c_multiplier,
            current_time,
            remaining_steps,
            gap_arm2,
            p1,
            p2,
            p_other_each,
            num_other_arms,
            block_config,
        )

        if block_size <= block_config.exact_small_block_threshold:
            for _ in range(block_size):
                theta1, theta2, cumulative_regret = apply_one_exact_update(
                    rng,
                    schedule,
                    c_multiplier,
                    current_time,
                    gap_arm2,
                    theta1,
                    theta2,
                    cumulative_regret,
                    num_other_arms,
                    regret_mode,
                )
                num_blocks += 1
                if current_time == next_checkpoint:
                    record(checkpoint_index)
                    checkpoint_index += 1
                    current_time += 1
                    break
                current_time += 1
            continue

        if regret_mode == "conditional":
            cumulative_regret = cumulative_regret + block_size * instantaneous_regret(p1, p2, gap_arm2)
        else:
            # Approximate realized pseudo-regret inside the frozen-policy block.
            count1 = rng.binomial(block_size, p1)
            remaining = block_size - count1
            p2_conditional = np.divide(p2, 1.0 - p1, out=np.zeros_like(p2), where=(1.0 - p1) > 1e-15)
            count2 = rng.binomial(remaining, p2_conditional)
            count_other = block_size - count1 - count2
            cumulative_regret = cumulative_regret + gap_arm2 * count2.astype(np.float64) + count_other.astype(np.float64)

        theta1, theta2 = apply_gaussian_block_update(
            rng,
            schedule,
            c_multiplier,
            current_time,
            block_size,
            gap_arm2,
            theta1,
            theta2,
            p1,
            p2,
            block_config,
        )

        current_time += block_size
        num_blocks += 1

        if current_time - 1 == next_checkpoint:
            record(checkpoint_index)
            checkpoint_index += 1

    final_regret_samples = cumulative_regret.copy() if return_final_samples else None
    if return_final_samples:
        p1_final, _, _ = stable_action_probabilities(theta1, theta2, num_other_arms)
        final_pi1_samples = p1_final.copy()
    else:
        final_pi1_samples = None

    return SimResult(
        schedule_slug=schedule_slug,
        c_multiplier=float(c_multiplier),
        method=method,
        regret_mode=regret_mode,
        gap_arm2=float(gap_arm2),
        horizon_steps=int(horizon_steps),
        num_arms=int(num_arms),
        num_trajectories=int(num_trajectories),
        checkpoint_times=checkpoint_times,
        sum_regret=sum_regret,
        sumsq_regret=sumsq_regret,
        sum_pi1=sum_pi1,
        sumsq_pi1=sumsq_pi1,
        final_regret_samples=final_regret_samples,
        final_pi1_samples=final_pi1_samples,
        num_blocks=int(num_blocks),
    )


# ---------------------------------------------------------------------------
# Parallel and CSV helpers
# ---------------------------------------------------------------------------


def default_workers() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def split_trajectories(total: int, workers: int, chunk_trajectories: Optional[int]) -> List[int]:
    if total <= 0:
        raise ValueError("trajectories must be positive")
    if chunk_trajectories is not None and chunk_trajectories > 0:
        chunks: List[int] = []
        remaining = total
        while remaining > 0:
            chunk = min(int(chunk_trajectories), remaining)
            chunks.append(chunk)
            remaining -= chunk
        return chunks
    workers = max(1, min(int(workers), total))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _simulate_worker(kwargs: dict) -> SimResult:
    return simulate_policy_gradient(**kwargs)


def aggregate_results(results: Sequence[SimResult]) -> SimResult:
    if not results:
        raise ValueError("No results to aggregate")
    first = results[0]
    total_n = sum(r.num_trajectories for r in results)
    sum_regret = np.zeros_like(first.sum_regret)
    sumsq_regret = np.zeros_like(first.sumsq_regret)
    sum_pi1 = np.zeros_like(first.sum_pi1)
    sumsq_pi1 = np.zeros_like(first.sumsq_pi1)
    num_blocks = 0
    final_regret_samples = []
    final_pi1_samples = []

    for r in results:
        if not np.array_equal(r.checkpoint_times, first.checkpoint_times):
            raise ValueError("Cannot aggregate different checkpoint grids")
        sum_regret += r.sum_regret
        sumsq_regret += r.sumsq_regret
        sum_pi1 += r.sum_pi1
        sumsq_pi1 += r.sumsq_pi1
        num_blocks += r.num_blocks
        if r.final_regret_samples is not None:
            final_regret_samples.append(r.final_regret_samples)
        if r.final_pi1_samples is not None:
            final_pi1_samples.append(r.final_pi1_samples)

    return SimResult(
        schedule_slug=first.schedule_slug,
        c_multiplier=first.c_multiplier,
        method=first.method,
        regret_mode=first.regret_mode,
        gap_arm2=first.gap_arm2,
        horizon_steps=first.horizon_steps,
        num_arms=first.num_arms,
        num_trajectories=total_n,
        checkpoint_times=first.checkpoint_times.copy(),
        sum_regret=sum_regret,
        sumsq_regret=sumsq_regret,
        sum_pi1=sum_pi1,
        sumsq_pi1=sumsq_pi1,
        final_regret_samples=np.concatenate(final_regret_samples) if final_regret_samples else None,
        final_pi1_samples=np.concatenate(final_pi1_samples) if final_pi1_samples else None,
        num_blocks=num_blocks,
    )


def simulate_parallel(
    *,
    schedule_slug: str,
    c_multiplier: float,
    gap_arm2: float,
    horizon_steps: int,
    num_arms: int,
    num_trajectories: int,
    random_seed: int,
    method: str,
    regret_mode: str,
    checkpoints: Sequence[int],
    block_config: BlockConfig,
    workers: int,
    chunk_trajectories: Optional[int],
    return_final_samples: bool = False,
) -> SimResult:
    chunks = split_trajectories(num_trajectories, workers, chunk_trajectories)
    print(
        f"[simulate] schedule={schedule_slug} c={c_multiplier:g} gap={gap_arm2:g} "
        f"method={method} regret={regret_mode} T={horizon_steps} N={num_trajectories} chunks={chunks}",
        flush=True,
    )

    kwargs_list = []
    for chunk_id, chunk_n in enumerate(chunks):
        kwargs_list.append(dict(
            schedule_slug=schedule_slug,
            c_multiplier=float(c_multiplier),
            gap_arm2=float(gap_arm2),
            horizon_steps=int(horizon_steps),
            num_arms=int(num_arms),
            num_trajectories=int(chunk_n),
            random_seed=int(random_seed + 9176 * chunk_id),
            method=method,
            regret_mode=regret_mode,
            checkpoints=checkpoints,
            block_config=block_config,
            return_final_samples=return_final_samples,
        ))

    if workers <= 1 or len(kwargs_list) == 1:
        outputs = [_simulate_worker(k) for k in kwargs_list]
    else:
        outputs = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_simulate_worker, k) for k in kwargs_list]
            for fut in as_completed(futures):
                out = fut.result()
                outputs.append(out)
                print(
                    f"  chunk done: n={out.num_trajectories} final_mean={out.mean_regret[-1]:.6e} blocks={out.num_blocks}",
                    flush=True,
                )

    return aggregate_results(outputs)


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
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def c_values_from_args(args: argparse.Namespace) -> np.ndarray:
    explicit = parse_float_list(getattr(args, "c_values", None))
    if explicit is not None:
        return np.asarray(explicit, dtype=np.float64)
    if args.c_grid == "log":
        return np.geomspace(float(args.c_start), float(args.c_stop), int(args.num_c), dtype=np.float64)
    return np.linspace(float(args.c_start), float(args.c_stop), int(args.num_c), dtype=np.float64)


def gap_values_from_args(args: argparse.Namespace) -> np.ndarray:
    explicit = parse_float_list(getattr(args, "gap_values", None))
    if explicit is not None:
        return np.asarray(explicit, dtype=np.float64)
    if int(args.num_gaps) < 2:
        raise ValueError("--num-gaps must be at least 2 unless --gap-values is used")
    return np.linspace(float(args.gap_start), float(args.gap_stop), int(args.num_gaps), dtype=np.float64)


def block_config_from_args(args: argparse.Namespace) -> BlockConfig:
    return BlockConfig(
        max_mean_change=float(args.max_mean_change),
        max_noise_change=float(args.max_noise_change),
        max_block_size=int(args.max_block_size),
        block_quantile=float(args.block_quantile),
        exact_eta_sum_threshold=int(args.exact_eta_sum_threshold),
        exact_small_block_threshold=int(args.exact_small_block_threshold),
    )


# ---------------------------------------------------------------------------
# Sweep over schedule, c, and gap
# ---------------------------------------------------------------------------


def sweep_tasks(args: argparse.Namespace) -> List[Tuple[int, ScheduleSpec, int, float, int, float]]:
    schedules = select_schedules(args.schedule_slugs)
    c_values = c_values_from_args(args)
    gaps = gap_values_from_args(args)

    total = len(schedules) * len(c_values) * len(gaps)
    if args.task_index is not None:
        task_index = int(args.task_index)
        if task_index < 0 or task_index >= total:
            raise IndexError(f"task-index must be in [0, {total - 1}]")
        gap_index = task_index % len(gaps)
        tmp = task_index // len(gaps)
        c_index = tmp % len(c_values)
        schedule_local_index = tmp // len(c_values)
        schedule = schedules[schedule_local_index]
        return [(all_schedules().index(schedule), schedule, c_index, float(c_values[c_index]), gap_index, float(gaps[gap_index]))]

    if args.run_all:
        tasks = []
        for schedule in schedules:
            schedule_index = all_schedules().index(schedule)
            for c_index, c in enumerate(c_values):
                for gap_index, gap in enumerate(gaps):
                    tasks.append((schedule_index, schedule, c_index, float(c), gap_index, float(gap)))
        return tasks

    if args.schedule_index is not None and args.c_index is not None and args.gap_index is not None:
        schedule = schedules[int(args.schedule_index)]
        return [(all_schedules().index(schedule), schedule, int(args.c_index), float(c_values[int(args.c_index)]), int(args.gap_index), float(gaps[int(args.gap_index)]))]

    raise ValueError("Use --task-index, --run-all, or --schedule-index plus --c-index plus --gap-index.")


def run_sweep(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    workers = args.workers or default_workers()
    block_config = block_config_from_args(args)
    checkpoints = [int(args.horizon)]

    for schedule_index, schedule, c_index, c, gap_index, gap in sweep_tasks(args):
        output_path = outdir / "sweep" / schedule.slug / f"c_{c_index:04d}" / f"gap_{gap_index:04d}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists", flush=True)
            continue

        seed = int(args.seed + 10_000_019 * schedule_index + 500_009 * c_index + 1_000_003 * gap_index)
        result = simulate_parallel(
            schedule_slug=schedule.slug,
            c_multiplier=c,
            gap_arm2=gap,
            horizon_steps=int(args.horizon),
            num_arms=int(args.num_arms),
            num_trajectories=int(args.trajectories),
            random_seed=seed,
            method=args.method,
            regret_mode=args.regret_mode,
            checkpoints=checkpoints,
            block_config=block_config,
            workers=workers,
            chunk_trajectories=args.chunk_trajectories,
            return_final_samples=False,
        )

        row = {
            "schedule_index": schedule_index,
            "schedule_slug": schedule.slug,
            "schedule_label": schedule.label,
            "c_index": c_index,
            "c_multiplier": c,
            "gap_index": gap_index,
            "gap_arm2": gap,
            "mean_regret": float(result.mean_regret[-1]),
            "standard_error": float(result.se_regret[-1]),
            "mean_regret_per_round": float(result.mean_regret[-1] / result.horizon_steps),
            "se_regret_per_round": float(result.se_regret[-1] / result.horizon_steps),
            "mean_final_pi1": float(result.mean_pi1[-1]),
            "se_final_pi1": float(result.se_pi1[-1]),
            "num_trajectories": result.num_trajectories,
            "horizon_steps": result.horizon_steps,
            "num_arms": result.num_arms,
            "method": args.method,
            "regret_mode": args.regret_mode,
            "num_blocks_total": result.num_blocks,
            "max_mean_change": block_config.max_mean_change,
            "max_noise_change": block_config.max_noise_change,
            "block_quantile": block_config.block_quantile,
            "exact_small_block_threshold": block_config.exact_small_block_threshold,
        }
        write_csv(output_path, [row])


def combine_sweep(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    paths = sorted((outdir / "sweep").glob("*/c_*/gap_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No sweep files found under {outdir / 'sweep'}")
    rows = [read_one_row(p) for p in paths]
    rows.sort(key=lambda r: (int(r["schedule_index"]), int(r["c_index"]), int(r["gap_index"])))

    combined = outdir / "combined_c_gap_sweep.csv"
    write_csv(combined, rows)

    groups: Dict[Tuple[str, int], List[dict]] = {}
    for row in rows:
        groups.setdefault((row["schedule_slug"], int(row["c_index"])), []).append(row)

    hardest = []
    for key, group in groups.items():
        hardest.append(max(group, key=lambda r: float(r["mean_regret"])))
    hardest.sort(key=lambda r: (int(r["schedule_index"]), int(r["c_index"])))
    hardest_path = outdir / "hardest_gap_by_schedule_and_c.csv"
    write_csv(hardest_path, hardest)

    print("\nHardest gaps by schedule and c:", flush=True)
    for row in hardest:
        print(
            f"  {row['schedule_slug']:22s} c={float(row['c_multiplier']):9.4g} "
            f"Delta={float(row['gap_arm2']):.5g} R/T={float(row['mean_regret_per_round']):.5g}",
            flush=True,
        )


def load_combined_sweep(outdir: pathlib.Path) -> List[dict]:
    path = outdir / "combined_c_gap_sweep.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run combine-sweep first.")
    return read_csv(path)


def load_hardest(outdir: pathlib.Path) -> List[dict]:
    path = outdir / "hardest_gap_by_schedule_and_c.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run combine-sweep first.")
    return read_csv(path)


# ---------------------------------------------------------------------------
# Histories at hardest gaps
# ---------------------------------------------------------------------------


def run_history(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    hardest_rows = load_hardest(outdir)
    wanted_schedules = {s.slug for s in select_schedules(args.schedule_slugs)}
    wanted_c_indices = None
    if args.c_indices and args.c_indices.lower() not in {"", "all"}:
        wanted_c_indices = {int(x.strip()) for x in args.c_indices.split(",") if x.strip()}

    rows_to_run = []
    for row in hardest_rows:
        if row["schedule_slug"] not in wanted_schedules:
            continue
        if wanted_c_indices is not None and int(row["c_index"]) not in wanted_c_indices:
            continue
        rows_to_run.append(row)

    if not rows_to_run:
        raise ValueError("No hardest-gap rows selected for history.")

    checkpoints = checkpoints_for(int(args.horizon), int(args.num_checkpoints))
    workers = args.workers or default_workers()
    block_config = block_config_from_args(args)
    history_dir = outdir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    for row in rows_to_run:
        schedule = schedule_by_slug(row["schedule_slug"])
        c_index = int(row["c_index"])
        c = float(row["c_multiplier"])
        gap = float(row["gap_arm2"])
        output_path = history_dir / f"history_{schedule.slug}_c_{c_index:04d}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists", flush=True)
            continue

        seed = int(args.seed + 50_000_021 * int(row["schedule_index"]) + 500_009 * c_index + 1_000_003 * int(row["gap_index"]))
        result = simulate_parallel(
            schedule_slug=schedule.slug,
            c_multiplier=c,
            gap_arm2=gap,
            horizon_steps=int(args.horizon),
            num_arms=int(args.num_arms),
            num_trajectories=int(args.trajectories),
            random_seed=seed,
            method=args.method,
            regret_mode=args.regret_mode,
            checkpoints=checkpoints,
            block_config=block_config,
            workers=workers,
            chunk_trajectories=args.chunk_trajectories,
            return_final_samples=args.save_final_samples,
        )

        out_rows = []
        for i, t in enumerate(result.checkpoint_times):
            out_rows.append({
                "schedule_index": int(row["schedule_index"]),
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "c_index": c_index,
                "c_multiplier": c,
                "hardest_gap_index": int(row["gap_index"]),
                "hardest_gap_arm2": gap,
                "time": int(t),
                "mean_regret": float(result.mean_regret[i]),
                "standard_error": float(result.se_regret[i]),
                "mean_regret_per_round": float(result.mean_regret[i] / int(t)),
                "se_regret_per_round": float(result.se_regret[i] / int(t)),
                "mean_pi1": float(result.mean_pi1[i]),
                "se_pi1": float(result.se_pi1[i]),
                "num_trajectories": result.num_trajectories,
                "horizon_steps": result.horizon_steps,
                "num_arms": result.num_arms,
                "method": args.method,
                "regret_mode": args.regret_mode,
            })
        write_csv(output_path, out_rows)

        if args.save_final_samples and result.final_regret_samples is not None:
            samples_path = history_dir / f"final_samples_{schedule.slug}_c_{c_index:04d}.csv"
            sample_rows = []
            pi1 = result.final_pi1_samples if result.final_pi1_samples is not None else np.full_like(result.final_regret_samples, np.nan)
            for i, (reg, p1) in enumerate(zip(result.final_regret_samples, pi1)):
                sample_rows.append({"trajectory": i, "final_regret": float(reg), "final_pi1": float(p1)})
            write_csv(samples_path, sample_rows)

    if args.plot:
        plot_history(args)


# ---------------------------------------------------------------------------
# Validation exact vs approximation at short horizon
# ---------------------------------------------------------------------------


def validation_combos(args: argparse.Namespace) -> List[Tuple[ScheduleSpec, int, float, int, float]]:
    schedules = select_schedules(args.schedule_slugs)
    wanted_slugs = {s.slug for s in schedules}

    if getattr(args, "from_hardest", False):
        hardest_path = pathlib.Path(args.outdir) / "hardest_gap_by_schedule_and_c.csv"
        if not hardest_path.exists():
            raise FileNotFoundError(f"Missing {hardest_path}. Run combine-sweep before validate --from-hardest.")
        rows = read_csv(hardest_path)
        explicit_c_values = parse_float_list(getattr(args, "c_values", None))
        combos = []
        for row in rows:
            if row["schedule_slug"] not in wanted_slugs:
                continue
            c = float(row["c_multiplier"])
            if explicit_c_values is not None and not any(abs(c - cv) <= 1e-12 * max(1.0, abs(cv)) for cv in explicit_c_values):
                continue
            combos.append((
                schedule_by_slug(row["schedule_slug"]),
                int(row["c_index"]),
                c,
                int(row["gap_index"]),
                float(row["gap_arm2"]),
            ))
        if not combos:
            raise ValueError("validate --from-hardest selected no rows. Check --schedule-slugs and --c-values.")
        return combos

    c_values = c_values_from_args(args)
    gaps = gap_values_from_args(args)
    combos = []
    for schedule in schedules:
        for c_index, c in enumerate(c_values):
            for gap_index, gap in enumerate(gaps):
                combos.append((schedule, c_index, float(c), gap_index, float(gap)))
    return combos


def run_validation(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir) / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or default_workers()
    block_config = block_config_from_args(args)
    checkpoints = checkpoints_for(int(args.horizon), int(args.num_checkpoints))
    final_rows = []
    history_rows = []

    for schedule, c_index, c, gap_index, gap in validation_combos(args):
        for method in ["exact", "approx"]:
            seed = int(args.seed + (0 if method == "exact" else 80_000_000) + 10_000_019 * all_schedules().index(schedule) + 500_009 * c_index + 1_000_003 * gap_index)
            result = simulate_parallel(
                schedule_slug=schedule.slug,
                c_multiplier=c,
                gap_arm2=gap,
                horizon_steps=int(args.horizon),
                num_arms=int(args.num_arms),
                num_trajectories=int(args.trajectories),
                random_seed=seed,
                method=method,
                regret_mode=args.regret_mode,
                checkpoints=checkpoints,
                block_config=block_config,
                workers=workers,
                chunk_trajectories=args.chunk_trajectories,
                return_final_samples=False,
            )
            final_rows.append({
                "schedule_slug": schedule.slug,
                "schedule_label": schedule.label,
                "c_index": c_index,
                "c_multiplier": c,
                "gap_index": gap_index,
                "gap_arm2": gap,
                "method": method,
                "mean_regret": float(result.mean_regret[-1]),
                "standard_error": float(result.se_regret[-1]),
                "mean_regret_per_round": float(result.mean_regret[-1] / result.horizon_steps),
                "mean_final_pi1": float(result.mean_pi1[-1]),
                "se_final_pi1": float(result.se_pi1[-1]),
                "num_trajectories": result.num_trajectories,
                "horizon_steps": result.horizon_steps,
                "num_arms": result.num_arms,
                "regret_mode": args.regret_mode,
                "num_blocks_total": result.num_blocks,
            })
            for i, t in enumerate(result.checkpoint_times):
                history_rows.append({
                    "schedule_slug": schedule.slug,
                    "schedule_label": schedule.label,
                    "c_index": c_index,
                    "c_multiplier": c,
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

    comparison = []
    grouped: Dict[Tuple[str, int, int], Dict[str, dict]] = {}
    for row in final_rows:
        key = (row["schedule_slug"], int(row["c_index"]), int(row["gap_index"]))
        grouped.setdefault(key, {})[row["method"]] = row
    for key, group in sorted(grouped.items()):
        if "exact" not in group or "approx" not in group:
            continue
        e = group["exact"]
        a = group["approx"]
        comparison.append({
            "schedule_slug": e["schedule_slug"],
            "c_index": e["c_index"],
            "c_multiplier": e["c_multiplier"],
            "gap_index": e["gap_index"],
            "gap_arm2": e["gap_arm2"],
            "exact_mean_regret": e["mean_regret"],
            "approx_mean_regret": a["mean_regret"],
            "approx_minus_exact": float(a["mean_regret"]) - float(e["mean_regret"]),
            "relative_error": (float(a["mean_regret"]) - float(e["mean_regret"])) / float(e["mean_regret"]) if float(e["mean_regret"]) != 0.0 else float("nan"),
            "exact_mean_regret_per_round": e["mean_regret_per_round"],
            "approx_mean_regret_per_round": a["mean_regret_per_round"],
            "exact_mean_final_pi1": e["mean_final_pi1"],
            "approx_mean_final_pi1": a["mean_final_pi1"],
        })
    write_csv(outdir / "validation_comparison.csv", comparison)

    if args.plot:
        plot_validation(args)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")


def plot_sweep(args: argparse.Namespace) -> None:
    _require_matplotlib()
    outdir = pathlib.Path(args.outdir)
    hardest = load_hardest(outdir)
    if args.schedule_slugs and args.schedule_slugs.lower() not in {"", "all"}:
        wanted = {s.slug for s in select_schedules(args.schedule_slugs)}
        hardest = [r for r in hardest if r["schedule_slug"] in wanted]

    groups: Dict[str, List[dict]] = {}
    for row in hardest:
        groups.setdefault(row["schedule_slug"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=180)
    for schedule in all_schedules():
        rows = groups.get(schedule.slug)
        if not rows:
            continue
        rows.sort(key=lambda r: float(r["c_multiplier"]))
        c = np.asarray([float(r["c_multiplier"]) for r in rows])
        y = np.asarray([float(r["mean_regret_per_round"]) for r in rows])
        se = np.asarray([float(r["se_regret_per_round"]) for r in rows])
        ax.plot(c, y, marker="o", linewidth=2.0, label=schedule.label)
        if args.show_bands:
            ax.fill_between(c, y - 2.0 * se, y + 2.0 * se, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"multiplier $c$")
    ax.set_ylabel(r"hardest-gap final regret per round, $R_T/T$")
    ax.set_title("Does increasing c make regret closer to linear?")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path = outdir / "c_sweep_regret_per_round_vs_c.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}", flush=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=180)
    for schedule in all_schedules():
        rows = groups.get(schedule.slug)
        if not rows:
            continue
        rows.sort(key=lambda r: float(r["c_multiplier"]))
        c = np.asarray([float(r["c_multiplier"]) for r in rows])
        gap = np.asarray([float(r["gap_arm2"]) for r in rows])
        ax.plot(c, gap, marker="o", linewidth=2.0, label=schedule.label)
    ax.set_xscale("log")
    ax.set_xlabel(r"multiplier $c$")
    ax.set_ylabel(r"hardest gap $\Delta$")
    ax.set_title("Hardest gap selected by the sweep")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path = outdir / "c_sweep_hardest_gap_vs_c.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}", flush=True)


def load_history_files(outdir: pathlib.Path, schedule_slugs: str = "all") -> List[Tuple[ScheduleSpec, int, float, float, np.ndarray, np.ndarray, np.ndarray, float]]:
    history_dir = outdir / "history"
    paths = sorted(history_dir.glob("history_*_c_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No history files found in {history_dir}")
    wanted = {s.slug for s in select_schedules(schedule_slugs)}
    histories = []
    for path in paths:
        rows = read_csv(path)
        if not rows or rows[0]["schedule_slug"] not in wanted:
            continue
        schedule = schedule_by_slug(rows[0]["schedule_slug"])
        c_index = int(rows[0]["c_index"])
        c = float(rows[0]["c_multiplier"])
        gap = float(rows[0]["hardest_gap_arm2"])
        t = np.asarray([int(r["time"]) for r in rows], dtype=np.float64)
        mean = np.asarray([float(r["mean_regret"]) for r in rows], dtype=np.float64)
        se = np.asarray([float(r["standard_error"]) for r in rows], dtype=np.float64)
        order = np.argsort(t)
        histories.append((schedule, c_index, c, gap, t[order], mean[order], se[order], float(rows[0]["horizon_steps"])))
    return histories


def _plot_histories_one(
    histories: Sequence[Tuple[ScheduleSpec, int, float, float, np.ndarray, np.ndarray, np.ndarray, float]],
    outpath: pathlib.Path,
    y_kind: str,
    xscale: str,
    yscale: str,
    regret_scale: float,
    show_bands: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=180)
    for schedule, c_index, c, gap, t, mean, se, horizon in histories:
        if y_kind == "cumulative":
            y = mean / regret_scale
            band = se / regret_scale
            ylabel = "Average cumulative regret" if regret_scale == 1.0 else rf"Average cumulative regret / ${regret_scale:.0e}$"
        elif y_kind == "per_round":
            y = mean / t
            band = se / t
            ylabel = r"Average regret per round, $R_t/t$"
        else:
            raise ValueError(y_kind)
        label = f"{schedule.label}, c={c:g}, hardest $\Delta={gap:.4g}$"
        ax.plot(t, y, linewidth=2.0, label=label)
        if show_bands:
            ax.fill_between(t, y - 2.0 * band, y + 2.0 * band, alpha=0.10, linewidth=0)

    if y_kind == "cumulative":
        # Raw baselines in same units as regret.
        t_ref = histories[0][4]
        for alpha, label in [(0.5, r"raw $\sqrt{T}$"), (2.0 / 3.0, r"raw $T^{2/3}$"), (1.0, r"raw $T$")]:
            ax.plot(t_ref, np.power(t_ref, alpha) / regret_scale, linestyle="--", linewidth=1.6, alpha=0.8, label=label)

    ax.set_xscale(xscale)
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title("Hardest-gap histories for c-multiplied stepsizes")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpath}", flush=True)


def plot_history(args: argparse.Namespace) -> None:
    _require_matplotlib()
    outdir = pathlib.Path(args.outdir)
    histories = load_history_files(outdir, args.schedule_slugs)
    if not histories:
        raise RuntimeError("No selected histories found")
    histories.sort(key=lambda x: (x[0].slug, x[2]))

    _plot_histories_one(
        histories,
        outdir / "c_sweep_histories_cumulative_linear.png",
        y_kind="cumulative",
        xscale="linear",
        yscale="linear",
        regret_scale=float(args.regret_scale),
        show_bands=args.show_bands,
    )
    _plot_histories_one(
        histories,
        outdir / "c_sweep_histories_cumulative_logx.png",
        y_kind="cumulative",
        xscale="log",
        yscale="linear",
        regret_scale=float(args.regret_scale),
        show_bands=args.show_bands,
    )
    _plot_histories_one(
        histories,
        outdir / "c_sweep_histories_cumulative_loglog.png",
        y_kind="cumulative",
        xscale="log",
        yscale="log",
        regret_scale=float(args.regret_scale),
        show_bands=args.show_bands,
    )
    _plot_histories_one(
        histories,
        outdir / "c_sweep_histories_regret_per_round.png",
        y_kind="per_round",
        xscale="log",
        yscale="linear",
        regret_scale=1.0,
        show_bands=args.show_bands,
    )


def plot_validation(args: argparse.Namespace) -> None:
    _require_matplotlib()
    val_dir = pathlib.Path(args.outdir) / "validation"
    history_path = val_dir / "validation_history.csv"
    final_path = val_dir / "validation_final.csv"
    if not history_path.exists() or not final_path.exists():
        raise FileNotFoundError("Run validate before plot_validation.")

    rows = read_csv(history_path)
    groups: Dict[Tuple[str, int, int, str], List[dict]] = {}
    for row in rows:
        key = (row["schedule_slug"], int(row["c_index"]), int(row["gap_index"]), row["method"])
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=180)
    for key, group in sorted(groups.items()):
        slug, c_index, gap_index, method = key
        group.sort(key=lambda r: int(r["time"]))
        t = np.asarray([int(r["time"]) for r in group], dtype=np.float64)
        mean = np.asarray([float(r["mean_regret"]) for r in group], dtype=np.float64)
        schedule = schedule_by_slug(slug)
        c = float(group[0]["c_multiplier"])
        gap = float(group[0]["gap_arm2"])
        linestyle = "-" if method == "exact" else "--"
        label = f"{schedule.label}, c={c:g}, Δ={gap:g}, {method}"
        ax.plot(t, mean, linestyle=linestyle, linewidth=1.8, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative regret")
    ax.set_title("Exact vs approximate c-multiplied update validation")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=7)
    fig.tight_layout()
    path = val_dir / "validation_exact_vs_approx_history.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}", flush=True)

    final_rows = read_csv(final_path)
    # Plot approximate-minus-exact relative error against c for each schedule/gap.
    grouped_final: Dict[Tuple[str, int, int], Dict[str, dict]] = {}
    for row in final_rows:
        key = (row["schedule_slug"], int(row["c_index"]), int(row["gap_index"]))
        grouped_final.setdefault(key, {})[row["method"]] = row
    comp = []
    for key, group in grouped_final.items():
        if "exact" in group and "approx" in group:
            e = group["exact"]
            a = group["approx"]
            exact = float(e["mean_regret"])
            approx = float(a["mean_regret"])
            comp.append((e["schedule_slug"], float(e["c_multiplier"]), float(e["gap_arm2"]), (approx - exact) / exact if exact else np.nan))
    if comp:
        fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=180)
        for slug in sorted({x[0] for x in comp}):
            for gap in sorted({x[2] for x in comp if x[0] == slug}):
                vals = [(c, err) for s, c, g, err in comp if s == slug and abs(g - gap) < 1e-14]
                vals.sort()
                ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", label=f"{slug}, Δ={gap:g}")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel(r"multiplier $c$")
        ax.set_ylabel("relative error: (approx - exact) / exact")
        ax.set_title("Approximation error at validation horizon")
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        path = val_dir / "validation_relative_error_vs_c.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {path}", flush=True)


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
    parser.add_argument("--outdir", type=str, default="c_multiplier_results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", choices=["exact", "approx"], default="approx")
    parser.add_argument("--regret-mode", choices=["conditional", "realized"], default="conditional")
    parser.add_argument("--max-mean-change", type=float, default=0.20)
    parser.add_argument("--max-noise-change", type=float, default=0.80)
    parser.add_argument("--max-block-size", type=int, default=50_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=4096)
    parser.add_argument("--exact-small-block-threshold", type=int, default=64)


def add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule-slugs", type=str, default="inv_sqrt_t")
    parser.add_argument("--c-values", type=str, default=None)
    parser.add_argument("--c-start", type=float, default=0.125)
    parser.add_argument("--c-stop", type=float, default=16.0)
    parser.add_argument("--num-c", type=int, default=8)
    parser.add_argument("--c-grid", choices=["log", "linear"], default="log")
    parser.add_argument("--gap-values", type=str, default=None)
    parser.add_argument("--gap-start", type=float, default=0.0)
    parser.add_argument("--gap-stop", type=float, default=1.0)
    parser.add_argument("--num-gaps", type=int, default=101)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep multiplicative constants c in eta_t = c * base_eta_t.")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Run final-regret sweep over schedule, c, and gap.")
    add_common_sim_args(sweep)
    add_grid_args(sweep)
    sweep.add_argument("--task-index", type=int, default=None)
    sweep.add_argument("--run-all", action="store_true")
    sweep.add_argument("--schedule-index", type=int, default=None)
    sweep.add_argument("--c-index", type=int, default=None)
    sweep.add_argument("--gap-index", type=int, default=None)

    combine = sub.add_parser("combine-sweep", help="Combine sweep CSVs and choose hardest gap for each schedule and c.")
    combine.add_argument("--outdir", type=str, default="c_multiplier_results")

    plot_sweep_parser = sub.add_parser("plot-sweep", help="Plot R_T/T and hardest gap versus c.")
    plot_sweep_parser.add_argument("--outdir", type=str, default="c_multiplier_results")
    plot_sweep_parser.add_argument("--schedule-slugs", type=str, default="all")
    plot_sweep_parser.add_argument("--show-bands", action="store_true")

    history = sub.add_parser("history", help="Run histories at hardest gaps from c sweep.")
    add_common_sim_args(history)
    history.add_argument("--schedule-slugs", type=str, default="inv_sqrt_t")
    history.add_argument("--c-indices", type=str, default="all", help="Comma-separated c indices or all.")
    history.add_argument("--num-checkpoints", type=int, default=250)
    history.add_argument("--save-final-samples", action="store_true")
    history.add_argument("--plot", action="store_true")
    history.add_argument("--regret-scale", type=float, default=1_000_000.0)
    history.add_argument("--show-bands", action="store_true")

    plot_history_parser = sub.add_parser("plot-history", help="Plot history CSVs in linear, log-x, log-log, and R_t/t views.")
    plot_history_parser.add_argument("--outdir", type=str, default="c_multiplier_results")
    plot_history_parser.add_argument("--schedule-slugs", type=str, default="all")
    plot_history_parser.add_argument("--regret-scale", type=float, default=1_000_000.0)
    plot_history_parser.add_argument("--show-bands", action="store_true")

    validate = sub.add_parser("validate", help="Exact-vs-approx validation for selected c and gap values.")
    add_common_sim_args(validate)
    add_grid_args(validate)
    validate.set_defaults(horizon=100_000, trajectories=1000, method="approx", max_mean_change=0.02, max_noise_change=0.10, max_block_size=1_000_000, block_quantile=1.0)
    validate.add_argument("--num-checkpoints", type=int, default=80)
    validate.add_argument("--plot", action="store_true")
    validate.add_argument("--from-hardest", action="store_true",
                          help="Validate exact-vs-approx at the hardest gaps found by combine-sweep. If --c-values is provided, only those c values are used.")

    list_schedules = sub.add_parser("list-schedules", help="List schedule slugs.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "list-schedules":
        for i, s in enumerate(all_schedules()):
            print(f"{i}: {s.slug:22s} {s.label}")
        return

    if args.command == "sweep":
        run_sweep(args)
    elif args.command == "combine-sweep":
        combine_sweep(args)
    elif args.command == "plot-sweep":
        plot_sweep(args)
    elif args.command == "history":
        run_history(args)
    elif args.command == "plot-history":
        plot_history(args)
    elif args.command == "validate":
        run_validation(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
