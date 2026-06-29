#!/usr/bin/env python3
"""
Experiments for softmax policy-gradient bandits with only

    eta_t = 1 / sqrt(t + 1),  t = 1, 2, ...

The bandit instance is

    mu    = (1, 1 - Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)

where arms 3..K are zero-mean deterministic arms.  If K is the total number
of arms, then m = K - 2 is the number of zero-mean arms.

This script is intentionally narrow.  It produces the four diagnostics requested:

1. sample paths of pi_1, grey individual runs and red average;
2. horizon-wise worst-case regret vs horizon, original and log-log scale;
3. worst-case gap vs predicted scale c * min{sqrt(log(m)/m), 1/log(n)};
4. few-arms comparison to show why a large m matters.

The large-horizon simulator uses a blocked Gaussian aggregate approximation.
For small blocks it falls back to exact round-by-round Algorithm-1 updates.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Matplotlib is imported only in plotting functions so that compute nodes can
# run simulation tasks even if a display backend is unavailable.


# ---------------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    gap_arm2: float
    horizon: int
    checkpoints: np.ndarray
    num_arms: int = 40
    trajectories: int = 10_000
    random_seed: int = 20260310
    method: str = "approx"  # "approx" or "exact"
    exact_small_block_threshold: int = 64
    max_mean_score_change_per_block: float = 0.08
    max_noise_score_change_per_block: float = 0.35
    max_block_size: int = 10_000_000
    block_quantile: float = 0.995


@dataclass
class HistoryResult:
    time: np.ndarray
    mean_regret: np.ndarray
    standard_error_regret: np.ndarray
    mean_pi1: np.ndarray
    mean_pi2: np.ndarray
    mean_pi_other_total: np.ndarray
    num_arms: int
    zero_mean_arms: int
    gap_arm2: float
    trajectories: int


# ---------------------------------------------------------------------------
# Schedule and softmax utilities
# ---------------------------------------------------------------------------


def eta_at_round(round_number: int | np.ndarray) -> float | np.ndarray:
    """eta_t = 1 / sqrt(t + 1), with round_number t = 1, 2, ..."""
    x = np.asarray(round_number, dtype=np.float64)
    value = 1.0 / np.sqrt(x + 1.0)
    if np.isscalar(round_number):
        return float(value)
    return value


def eta_sums_over_block(start_round: int, block_size: int) -> Tuple[float, float]:
    """Return sum eta_t and sum eta_t^2 over a block of rounds.

    Rounds are start_round, ..., start_round + block_size - 1.
    eta_t = 1/sqrt(t+1).
    """
    if block_size <= 0:
        return 0.0, 0.0

    # Exact summation is cheap enough for small and medium blocks and improves
    # agreement with exact updates in validation runs.
    if block_size <= 200_000:
        rounds = np.arange(start_round, start_round + block_size, dtype=np.float64)
        eta = eta_at_round(rounds)
        return float(np.sum(eta)), float(np.sum(eta * eta))

    # Euler--Maclaurin approximation for f(t) = (t+1)^(-p).
    # Sum_{t=a}^{b} f(t) approx integral_a^b f(x) dx + endpoints + derivative correction.
    a = float(start_round)
    b = float(start_round + block_size - 1)

    def power_sum(p: float) -> float:
        # f(x) = (x + 1)^(-p)
        ua = a + 1.0
        ub = b + 1.0

        if abs(p - 1.0) < 1e-14:
            integral = math.log(ub / ua)
        else:
            integral = (ub ** (1.0 - p) - ua ** (1.0 - p)) / (1.0 - p)

        f_a = ua ** (-p)
        f_b = ub ** (-p)
        fp_a = -p * ua ** (-p - 1.0)
        fp_b = -p * ub ** (-p - 1.0)
        estimate = integral + 0.5 * (f_a + f_b) + (fp_b - fp_a) / 12.0
        return float(max(0.0, estimate))

    return power_sum(0.5), power_sum(1.0)


def stable_policy_probabilities(
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta_other: np.ndarray,
    zero_mean_arms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pi_1, pi_2, pi_other_per_arm, and pi_other_total."""
    max_theta = np.maximum(np.maximum(theta1, theta2), theta_other)
    w1 = np.exp(theta1 - max_theta)
    w2 = np.exp(theta2 - max_theta)
    w0 = np.exp(theta_other - max_theta)
    total = w1 + w2 + zero_mean_arms * w0
    pi1 = w1 / total
    pi2 = w2 / total
    pi_other_per_arm = w0 / total
    pi_other_total = zero_mean_arms * pi_other_per_arm
    return pi1, pi2, pi_other_per_arm, pi_other_total


def instantaneous_expected_regret(pi1: np.ndarray, pi2: np.ndarray, gap_arm2: float) -> np.ndarray:
    """Expected one-round pseudo-regret under the current policy."""
    pi_other_total = 1.0 - pi1 - pi2
    return pi_other_total + gap_arm2 * pi2


# ---------------------------------------------------------------------------
# Exact and blocked update kernels
# ---------------------------------------------------------------------------


def exact_updates(
    *,
    theta1: np.ndarray,
    theta2: np.ndarray,
    regret: np.ndarray,
    start_round: int,
    num_rounds: int,
    gap_arm2: float,
    num_arms: int,
    rng: np.random.Generator,
) -> int:
    """Run exact Algorithm-1 updates for a short block.

    Regret is accumulated as conditional expected one-step regret.  This keeps
    the estimated mean regret lower variance while the policy update itself
    still uses sampled actions and rewards.
    """
    zero_mean_arms = num_arms - 2
    theta_other = -(theta1 + theta2) / zero_mean_arms

    for offset in range(num_rounds):
        round_number = start_round + offset
        eta = eta_at_round(round_number)

        pi1, pi2, _, pi_other_total = stable_policy_probabilities(
            theta1, theta2, theta_other, zero_mean_arms
        )
        regret += pi_other_total + gap_arm2 * pi2

        u = rng.random(theta1.shape[0])
        choose_arm1 = u < pi1
        choose_arm2 = (u >= pi1) & (u < pi1 + pi2)

        reward = np.zeros_like(theta1)
        n1 = int(np.sum(choose_arm1))
        n2 = int(np.sum(choose_arm2))
        if n1:
            reward[choose_arm1] = 1.0 + rng.normal(size=n1)
        if n2:
            reward[choose_arm2] = (1.0 - gap_arm2) + rng.normal(size=n2)

        # If a zero-mean deterministic arm is chosen, reward is zero and the
        # update is zero.  So only arm 1 and arm 2 choices matter.
        theta1 += eta * reward * (choose_arm1.astype(np.float64) - pi1)
        theta2 += eta * reward * (choose_arm2.astype(np.float64) - pi2)
        theta_other = -(theta1 + theta2) / zero_mean_arms

    return start_round + num_rounds


def choose_block_size(
    *,
    current_round: int,
    remaining_rounds: int,
    gap_arm2: float,
    pi1: np.ndarray,
    pi2: np.ndarray,
    pi_other_per_arm: np.ndarray,
    zero_mean_arms: int,
    config: SimConfig,
) -> int:
    """Choose a block size for the frozen-policy approximation."""
    if config.method == "exact":
        return 1

    eta_now = eta_at_round(current_round)
    pi_other_total = zero_mean_arms * pi_other_per_arm
    inst_regret = pi_other_total + gap_arm2 * pi2

    # Conditional mean gradient before multiplying by eta.
    mean_grad_1 = pi1 * inst_regret
    mean_grad_2 = pi2 * (inst_regret - gap_arm2)
    mean_grad_other = -(mean_grad_1 + mean_grad_2) / zero_mean_arms
    scale_by_traj = np.maximum.reduce(
        [np.abs(mean_grad_1), np.abs(mean_grad_2), np.abs(mean_grad_other)]
    )
    mean_scale = float(np.quantile(scale_by_traj, config.block_quantile))

    # One-step stochastic scale.  The reward second moments are E[Y_1^2]=2 and
    # E[Y_2^2]=(1-Delta)^2+1.
    mu2 = 1.0 - gap_arm2
    ey1_sq = 2.0
    ey2_sq = mu2 * mu2 + 1.0

    var1 = pi1 * (1.0 - pi1) ** 2 * ey1_sq + pi2 * pi1**2 * ey2_sq
    var2 = pi1 * pi2**2 * ey1_sq + pi2 * (1.0 - pi2) ** 2 * ey2_sq
    var_other = pi1 * pi_other_per_arm**2 * ey1_sq + pi2 * pi_other_per_arm**2 * ey2_sq
    var_scale_by_traj = np.maximum.reduce([var1, var2, var_other])
    var_scale = float(np.quantile(var_scale_by_traj, config.block_quantile))

    limit_by_mean = math.inf
    if mean_scale > 1e-300:
        limit_by_mean = config.max_mean_score_change_per_block / (eta_now * mean_scale)

    limit_by_noise = math.inf
    if var_scale > 1e-300:
        limit_by_noise = (
            config.max_noise_score_change_per_block / (eta_now * math.sqrt(var_scale))
        ) ** 2

    return int(
        max(
            1,
            math.floor(
                min(
                    remaining_rounds,
                    config.max_block_size,
                    limit_by_mean,
                    limit_by_noise,
                )
            ),
        )
    )


def blocked_gaussian_update(
    *,
    theta1: np.ndarray,
    theta2: np.ndarray,
    regret: np.ndarray,
    current_round: int,
    block_size: int,
    gap_arm2: float,
    num_arms: int,
    rng: np.random.Generator,
) -> int:
    """Blocked Gaussian aggregate update with the policy frozen in the block."""
    zero_mean_arms = num_arms - 2
    theta_other = -(theta1 + theta2) / zero_mean_arms
    pi1, pi2, _, pi_other_total = stable_policy_probabilities(theta1, theta2, theta_other, zero_mean_arms)

    # Conditional expected regret over this block.
    regret += block_size * (pi_other_total + gap_arm2 * pi2)

    sum_eta, sum_eta_sq = eta_sums_over_block(current_round, block_size)

    mu1 = 1.0
    mu2 = 1.0 - gap_arm2
    ey1_sq = 2.0
    ey2_sq = mu2 * mu2 + 1.0

    mean_s1 = pi1 * sum_eta * mu1
    mean_s2 = pi2 * sum_eta * mu2
    var_s1 = sum_eta_sq * (pi1 * ey1_sq - (pi1 * mu1) ** 2)
    var_s2 = sum_eta_sq * (pi2 * ey2_sq - (pi2 * mu2) ** 2)
    cov_s12 = sum_eta_sq * (-(pi1 * mu1) * (pi2 * mu2))

    std1 = np.sqrt(np.maximum(var_s1, 0.0))
    z1 = rng.normal(size=theta1.shape[0])
    z2 = rng.normal(size=theta1.shape[0])

    s1 = mean_s1 + std1 * z1

    # Conditional Gaussian draw for S2 given the same z1.
    coeff = np.divide(cov_s12, std1, out=np.zeros_like(cov_s12), where=std1 > 1e-14)
    residual_var = np.maximum(var_s2 - coeff * coeff, 0.0)
    s2 = mean_s2 + coeff * z1 + np.sqrt(residual_var) * z2

    theta1 += (1.0 - pi1) * s1 - pi1 * s2
    theta2 += -pi2 * s1 + (1.0 - pi2) * s2

    return current_round + block_size


# ---------------------------------------------------------------------------
# Simulation drivers
# ---------------------------------------------------------------------------


def make_log_checkpoints(horizon: int, num_horizons: int, min_horizon: int = 1) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if num_horizons <= 1:
        return np.array([horizon], dtype=np.int64)
    min_horizon = max(1, min(min_horizon, horizon))
    raw = np.geomspace(min_horizon, horizon, num=num_horizons)
    checkpoints = np.unique(np.rint(raw).astype(np.int64))
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= horizon)]
    if checkpoints[-1] != horizon:
        checkpoints = np.append(checkpoints, horizon)
    return checkpoints


def parse_horizon_values(text: Optional[str], horizon: int, num_horizons: int) -> np.ndarray:
    if text is None or not text.strip():
        return make_log_checkpoints(horizon, num_horizons)
    values = sorted({int(float(x.strip())) for x in text.split(",") if x.strip()})
    values = [v for v in values if 1 <= v <= horizon]
    if not values:
        raise ValueError("No valid horizon values were provided")
    if values[-1] != horizon:
        values.append(horizon)
    return np.array(values, dtype=np.int64)


def simulate_history(config: SimConfig) -> HistoryResult:
    """Simulate one gap and record history at the requested checkpoints."""
    if config.num_arms < 3:
        raise ValueError("num_arms must be at least 3")
    if config.horizon < int(config.checkpoints[-1]):
        raise ValueError("horizon must be at least the largest checkpoint")

    rng = np.random.default_rng(config.random_seed)
    n = config.trajectories
    zero_mean_arms = config.num_arms - 2

    theta1 = np.zeros(n, dtype=np.float64)
    theta2 = np.zeros(n, dtype=np.float64)
    regret = np.zeros(n, dtype=np.float64)

    checkpoints = np.asarray(config.checkpoints, dtype=np.int64)
    num_checkpoints = len(checkpoints)
    sum_regret = np.zeros(num_checkpoints, dtype=np.float64)
    sumsq_regret = np.zeros(num_checkpoints, dtype=np.float64)
    sum_pi1 = np.zeros(num_checkpoints, dtype=np.float64)
    sum_pi2 = np.zeros(num_checkpoints, dtype=np.float64)
    sum_pi_other_total = np.zeros(num_checkpoints, dtype=np.float64)

    current_round = 1
    checkpoint_index = 0

    while checkpoint_index < num_checkpoints:
        next_checkpoint = int(checkpoints[checkpoint_index])

        if current_round > next_checkpoint:
            theta_other = -(theta1 + theta2) / zero_mean_arms
            pi1, pi2, _, pi_other_total = stable_policy_probabilities(
                theta1, theta2, theta_other, zero_mean_arms
            )
            sum_regret[checkpoint_index] = float(np.sum(regret))
            sumsq_regret[checkpoint_index] = float(np.sum(regret * regret))
            sum_pi1[checkpoint_index] = float(np.sum(pi1))
            sum_pi2[checkpoint_index] = float(np.sum(pi2))
            sum_pi_other_total[checkpoint_index] = float(np.sum(pi_other_total))
            checkpoint_index += 1
            continue

        theta_other = -(theta1 + theta2) / zero_mean_arms
        pi1, pi2, pi_other_per_arm, _ = stable_policy_probabilities(
            theta1, theta2, theta_other, zero_mean_arms
        )

        remaining_to_checkpoint = next_checkpoint - current_round + 1
        block_size = choose_block_size(
            current_round=current_round,
            remaining_rounds=remaining_to_checkpoint,
            gap_arm2=config.gap_arm2,
            pi1=pi1,
            pi2=pi2,
            pi_other_per_arm=pi_other_per_arm,
            zero_mean_arms=zero_mean_arms,
            config=config,
        )
        block_size = min(block_size, remaining_to_checkpoint)

        if config.method == "exact" or block_size <= config.exact_small_block_threshold:
            current_round = exact_updates(
                theta1=theta1,
                theta2=theta2,
                regret=regret,
                start_round=current_round,
                num_rounds=block_size,
                gap_arm2=config.gap_arm2,
                num_arms=config.num_arms,
                rng=rng,
            )
        else:
            current_round = blocked_gaussian_update(
                theta1=theta1,
                theta2=theta2,
                regret=regret,
                current_round=current_round,
                block_size=block_size,
                gap_arm2=config.gap_arm2,
                num_arms=config.num_arms,
                rng=rng,
            )

    mean_regret = sum_regret / n
    variance_regret = np.maximum(sumsq_regret / n - mean_regret * mean_regret, 0.0)
    se_regret = np.sqrt(variance_regret / max(n, 1))

    return HistoryResult(
        time=checkpoints,
        mean_regret=mean_regret,
        standard_error_regret=se_regret,
        mean_pi1=sum_pi1 / n,
        mean_pi2=sum_pi2 / n,
        mean_pi_other_total=sum_pi_other_total / n,
        num_arms=config.num_arms,
        zero_mean_arms=zero_mean_arms,
        gap_arm2=config.gap_arm2,
        trajectories=n,
    )


def _simulate_chunk_worker(args: Tuple[SimConfig, int, int]) -> HistoryResult:
    base_config, chunk_size, seed = args
    chunk_config = SimConfig(
        gap_arm2=base_config.gap_arm2,
        horizon=base_config.horizon,
        checkpoints=base_config.checkpoints,
        num_arms=base_config.num_arms,
        trajectories=chunk_size,
        random_seed=seed,
        method=base_config.method,
        exact_small_block_threshold=base_config.exact_small_block_threshold,
        max_mean_score_change_per_block=base_config.max_mean_score_change_per_block,
        max_noise_score_change_per_block=base_config.max_noise_score_change_per_block,
        max_block_size=base_config.max_block_size,
        block_quantile=base_config.block_quantile,
    )
    return simulate_history(chunk_config)


def simulate_history_parallel(config: SimConfig, workers: int = 1, chunk_trajectories: int = 2500) -> HistoryResult:
    """Split trajectories across processes and combine mean/SE histories."""
    if workers <= 1 or config.trajectories <= chunk_trajectories:
        return simulate_history(config)

    import multiprocessing as mp

    chunks: List[int] = []
    remaining = config.trajectories
    while remaining > 0:
        size = min(chunk_trajectories, remaining)
        chunks.append(size)
        remaining -= size

    args = [
        (config, size, config.random_seed + 1_000_003 * i)
        for i, size in enumerate(chunks)
    ]

    with mp.Pool(processes=workers) as pool:
        results = pool.map(_simulate_chunk_worker, args)

    time = results[0].time
    total_n = sum(r.trajectories for r in results)
    sum_regret = np.zeros_like(results[0].mean_regret)
    sumsq_regret = np.zeros_like(results[0].mean_regret)
    sum_pi1 = np.zeros_like(results[0].mean_pi1)
    sum_pi2 = np.zeros_like(results[0].mean_pi2)
    sum_pi_other = np.zeros_like(results[0].mean_pi_other_total)

    for r in results:
        if not np.array_equal(time, r.time):
            raise RuntimeError("Chunk checkpoints do not match")
        n = r.trajectories
        sum_regret += r.mean_regret * n
        # Reconstruct sumsq from mean and SE: SE^2 = Var/n, Var = E[X^2]-mean^2.
        var = (r.standard_error_regret ** 2) * n
        sumsq_regret += (var + r.mean_regret**2) * n
        sum_pi1 += r.mean_pi1 * n
        sum_pi2 += r.mean_pi2 * n
        sum_pi_other += r.mean_pi_other_total * n

    mean_regret = sum_regret / total_n
    variance_regret = np.maximum(sumsq_regret / total_n - mean_regret * mean_regret, 0.0)
    se_regret = np.sqrt(variance_regret / total_n)

    return HistoryResult(
        time=time,
        mean_regret=mean_regret,
        standard_error_regret=se_regret,
        mean_pi1=sum_pi1 / total_n,
        mean_pi2=sum_pi2 / total_n,
        mean_pi_other_total=sum_pi_other / total_n,
        num_arms=config.num_arms,
        zero_mean_arms=config.num_arms - 2,
        gap_arm2=config.gap_arm2,
        trajectories=total_n,
    )


# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------


def result_to_rows(result: HistoryResult) -> List[Dict[str, float | int]]:
    rows = []
    for i, t in enumerate(result.time):
        rows.append(
            {
                "time": int(t),
                "num_arms": int(result.num_arms),
                "zero_mean_arms": int(result.zero_mean_arms),
                "gap_arm2": float(result.gap_arm2),
                "trajectories": int(result.trajectories),
                "mean_regret": float(result.mean_regret[i]),
                "standard_error_regret": float(result.standard_error_regret[i]),
                "mean_pi1": float(result.mean_pi1[i]),
                "mean_pi2": float(result.mean_pi2[i]),
                "mean_pi_other_total": float(result.mean_pi_other_total[i]),
            }
        )
    return rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_gap_files(outdir: Path, num_arms: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sweep_dir = outdir / "sweep" / f"arms_{num_arms}"
    for path in sorted(sweep_dir.glob("gap_*.csv")):
        for row in read_csv_dicts(path):
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key in {"time", "num_arms", "zero_mean_arms", "trajectories"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise FileNotFoundError(f"No gap CSV files found in {sweep_dir}")
    return rows


def combine_envelope_for_num_arms(outdir: Path, num_arms: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return all sweep rows and hardest gap row for each horizon."""
    rows = load_gap_files(outdir, num_arms)
    by_time: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_time.setdefault(int(row["time"]), []).append(row)

    hardest: List[Dict[str, object]] = []
    for t in sorted(by_time):
        best = max(by_time[t], key=lambda r: float(r["mean_regret"]))
        hardest.append(
            {
                "time": int(best["time"]),
                "num_arms": int(best["num_arms"]),
                "zero_mean_arms": int(best["zero_mean_arms"]),
                "hardest_gap_arm2": float(best["gap_arm2"]),
                "mean_regret": float(best["mean_regret"]),
                "standard_error_regret": float(best["standard_error_regret"]),
                "mean_pi1": float(best["mean_pi1"]),
                "mean_pi2": float(best["mean_pi2"]),
                "mean_pi_other_total": float(best["mean_pi_other_total"]),
            }
        )
    return rows, hardest


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
    sim_kwargs: Dict[str, object],
) -> Path:
    plt = _import_matplotlib()

    checkpoints = make_log_checkpoints(horizon, num_checkpoints)
    config = SimConfig(
        gap_arm2=gap_arm2,
        horizon=horizon,
        checkpoints=checkpoints,
        num_arms=num_arms,
        trajectories=num_paths,
        random_seed=seed,
        method=method,
        **sim_kwargs,
    )

    # For sample paths we need the full pi_1 matrix, so use a custom loop.
    rng = np.random.default_rng(seed)
    zero_mean_arms = num_arms - 2
    theta1 = np.zeros(num_paths, dtype=np.float64)
    theta2 = np.zeros(num_paths, dtype=np.float64)
    regret = np.zeros(num_paths, dtype=np.float64)
    pi1_history = np.zeros((len(checkpoints), num_paths), dtype=np.float64)

    current_round = 1
    checkpoint_index = 0

    while checkpoint_index < len(checkpoints):
        next_checkpoint = int(checkpoints[checkpoint_index])
        if current_round > next_checkpoint:
            theta_other = -(theta1 + theta2) / zero_mean_arms
            pi1, _, _, _ = stable_policy_probabilities(theta1, theta2, theta_other, zero_mean_arms)
            pi1_history[checkpoint_index, :] = pi1
            checkpoint_index += 1
            continue

        theta_other = -(theta1 + theta2) / zero_mean_arms
        pi1, pi2, pi_other_per_arm, _ = stable_policy_probabilities(theta1, theta2, theta_other, zero_mean_arms)
        remaining_to_checkpoint = next_checkpoint - current_round + 1
        block_config = config
        block_size = choose_block_size(
            current_round=current_round,
            remaining_rounds=remaining_to_checkpoint,
            gap_arm2=gap_arm2,
            pi1=pi1,
            pi2=pi2,
            pi_other_per_arm=pi_other_per_arm,
            zero_mean_arms=zero_mean_arms,
            config=block_config,
        )
        block_size = min(block_size, remaining_to_checkpoint)

        if method == "exact" or block_size <= config.exact_small_block_threshold:
            current_round = exact_updates(
                theta1=theta1,
                theta2=theta2,
                regret=regret,
                start_round=current_round,
                num_rounds=block_size,
                gap_arm2=gap_arm2,
                num_arms=num_arms,
                rng=rng,
            )
        else:
            current_round = blocked_gaussian_update(
                theta1=theta1,
                theta2=theta2,
                regret=regret,
                current_round=current_round,
                block_size=block_size,
                gap_arm2=gap_arm2,
                num_arms=num_arms,
                rng=rng,
            )

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=170)
    for j in range(num_paths):
        ax.plot(checkpoints, pi1_history[:, j], color="0.65", alpha=0.35, linewidth=0.8)
    ax.plot(checkpoints, np.mean(pi1_history, axis=1), color="red", linewidth=2.6, label="average")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\pi_t(1)$")
    ax.set_title(rf"Sample paths of $\pi_t(1)$, $\eta_t=1/\sqrt{{t+1}}$, $\Delta={gap_arm2:g}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = outdir / "sample_path_pi1.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_worst_case_regret(hardest_rows: List[Dict[str, object]], outdir: Path, regret_scale: float) -> List[Path]:
    plt = _import_matplotlib()
    t = np.array([float(r["time"]) for r in hardest_rows])
    regret = np.array([float(r["mean_regret"]) for r in hardest_rows]) / regret_scale
    se = np.array([float(r["standard_error_regret"]) for r in hardest_rows]) / regret_scale

    paths: List[Path] = []
    for name, logx, logy in [
        ("worst_case_regret_original_scale.png", False, False),
        ("worst_case_regret_loglog.png", True, True),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=170)
        ax.plot(t, regret, linewidth=2.4, label=r"$\max_\Delta R_t(\Delta)$")
        ax.fill_between(t, regret - 2.0 * se, regret + 2.0 * se, alpha=0.18, linewidth=0)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Horizon")
        ylabel = "Worst-case average cumulative regret"
        if regret_scale != 1.0:
            ylabel += f" / {regret_scale:.0e}"
        ax.set_ylabel(ylabel)
        ax.set_title(r"Worst-case regret envelope, $\eta_t=1/\sqrt{t+1}$")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_worst_gap_vs_prediction(
    hardest_rows: List[Dict[str, object]],
    outdir: Path,
    c_values: Sequence[float],
) -> Path:
    plt = _import_matplotlib()
    t = np.array([float(r["time"]) for r in hardest_rows])
    gap = np.array([float(r["hardest_gap_arm2"]) for r in hardest_rows])
    zero_mean_arms = int(hardest_rows[0]["zero_mean_arms"])

    m = float(zero_mean_arms)
    large_m_term = math.sqrt(math.log(max(m, 2.0)) / m)
    log_term = 1.0 / np.log(np.maximum(t, 2.0))
    base = np.minimum(large_m_term, log_term)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=170)
    ax.plot(t, gap, marker="o", linewidth=2.4, label="empirical hardest gap")
    for c in c_values:
        ax.plot(t, np.minimum(1.0, c * base), linestyle="--", linewidth=1.8, label=rf"$c={c:g}$ prediction")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Horizon n")
    ax.set_ylabel(r"Gap $\Delta$")
    ax.set_title(rf"Hardest gap vs $c\min\{{\sqrt{{\log(m)/m}},1/\log n\}}$, $m={zero_mean_arms}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = outdir / "worst_gap_vs_predicted_gap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_few_arms_comparison(
    hardest_by_arms: Dict[int, List[Dict[str, object]]],
    outdir: Path,
    regret_scale: float,
) -> List[Path]:
    plt = _import_matplotlib()
    paths: List[Path] = []
    for filename, logx, logy in [
        ("few_arms_comparison_original_scale.png", False, False),
        ("few_arms_comparison_loglog.png", True, True),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=170)
        for k, rows in sorted(hardest_by_arms.items()):
            t = np.array([float(r["time"]) for r in rows])
            regret = np.array([float(r["mean_regret"]) for r in rows]) / regret_scale
            m = k - 2
            ax.plot(t, regret, linewidth=2.4, label=f"K={k}, m={m} zero-mean arms")
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Horizon")
        ylabel = "Worst-case average cumulative regret"
        if regret_scale != 1.0:
            ylabel += f" / {regret_scale:.0e}"
        ax.set_ylabel(ylabel)
        ax.set_title("Effect of the number of zero-mean arms")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        path = outdir / filename
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def gap_grid(num_gaps: int, gap_start: float, gap_stop: float) -> np.ndarray:
    if num_gaps <= 1:
        return np.array([gap_start], dtype=np.float64)
    return np.linspace(gap_start, gap_stop, num_gaps, dtype=np.float64)


def common_sim_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "exact_small_block_threshold": args.exact_small_block_threshold,
        "max_mean_score_change_per_block": args.max_mean_change,
        "max_noise_score_change_per_block": args.max_noise_change,
        "max_block_size": args.max_block_size,
        "block_quantile": args.block_quantile,
    }


def run_sweep_one(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    gaps = gap_grid(args.num_gaps, args.gap_start, args.gap_stop)
    if args.gap_index < 0 or args.gap_index >= len(gaps):
        raise ValueError(f"gap-index must be in [0, {len(gaps)-1}]")
    gap = float(gaps[args.gap_index])
    checkpoints = parse_horizon_values(args.horizon_values, args.horizon, args.num_horizons)
    config = SimConfig(
        gap_arm2=gap,
        horizon=int(args.horizon),
        checkpoints=checkpoints,
        num_arms=int(args.num_arms),
        trajectories=int(args.trajectories),
        random_seed=int(args.seed + 10_000 * args.gap_index + 1_000_000 * args.num_arms),
        method=args.method,
        **common_sim_kwargs(args),
    )
    result = simulate_history_parallel(config, workers=args.workers, chunk_trajectories=args.chunk_trajectories)
    rows = result_to_rows(result)
    path = outdir / "sweep" / f"arms_{args.num_arms}" / f"gap_{args.gap_index:04d}.csv"
    write_rows_csv(path, rows)
    print(f"Wrote {path}")


def run_sweep_all(args: argparse.Namespace) -> None:
    gaps = gap_grid(args.num_gaps, args.gap_start, args.gap_stop)
    arms_list = [int(x.strip()) for x in args.num_arms_list.split(",") if x.strip()]
    for num_arms in arms_list:
        for gap_index, gap in enumerate(gaps):
            sub_args = argparse.Namespace(**vars(args))
            sub_args.num_arms = num_arms
            sub_args.gap_index = gap_index
            run_sweep_one(sub_args)


def run_combine_plot(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    main_num_arms = int(args.num_arms)
    arms_list = [int(x.strip()) for x in args.num_arms_list.split(",") if x.strip()]
    if main_num_arms not in arms_list:
        arms_list.insert(0, main_num_arms)

    hardest_by_arms: Dict[int, List[Dict[str, object]]] = {}
    all_rows_combined: List[Dict[str, object]] = []
    for num_arms in arms_list:
        all_rows, hardest = combine_envelope_for_num_arms(outdir, num_arms)
        hardest_by_arms[num_arms] = hardest
        all_rows_combined.extend(all_rows)
        write_rows_csv(outdir / f"combined_envelope_by_gap_arms_{num_arms}.csv", all_rows)
        write_rows_csv(outdir / f"hardest_gap_by_time_arms_{num_arms}.csv", hardest)

    write_rows_csv(outdir / "combined_envelope_by_gap_all_arms.csv", all_rows_combined)

    main_hardest = hardest_by_arms[main_num_arms]
    write_rows_csv(outdir / "hardest_gap_by_time.csv", main_hardest)

    paths: List[Path] = []
    paths.extend(plot_worst_case_regret(main_hardest, outdir, args.regret_scale))
    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]
    paths.append(plot_worst_gap_vs_prediction(main_hardest, outdir, c_values))
    if len(hardest_by_arms) >= 2:
        paths.extend(plot_few_arms_comparison(hardest_by_arms, outdir, args.regret_scale))

    if args.make_sample_path:
        final_row = main_hardest[-1]
        gap_for_sample = float(final_row["hardest_gap_arm2"]) if args.sample_gap is None else float(args.sample_gap)
        sample_kwargs = {
            "exact_small_block_threshold": args.exact_small_block_threshold,
            "max_mean_score_change_per_block": args.max_mean_change,
            "max_noise_score_change_per_block": args.max_noise_change,
            "max_block_size": args.max_block_size,
            "block_quantile": args.block_quantile,
        }
        paths.append(
            plot_sample_paths(
                outdir=outdir,
                gap_arm2=gap_for_sample,
                horizon=args.horizon,
                num_arms=main_num_arms,
                num_paths=args.sample_paths,
                num_checkpoints=args.sample_checkpoints,
                seed=args.seed + 9999,
                method=args.method,
                sim_kwargs=sample_kwargs,
            )
        )

    print("Wrote:")
    for path in paths:
        print(f"  {path}")


def run_sample_path(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    if args.sample_gap is None:
        hardest_path = outdir / "hardest_gap_by_time.csv"
        if not hardest_path.exists():
            raise FileNotFoundError(
                f"{hardest_path} does not exist. Pass --sample-gap or run combine-plot first."
            )
        rows = read_csv_dicts(hardest_path)
        gap = float(rows[-1]["hardest_gap_arm2"])
    else:
        gap = float(args.sample_gap)
    path = plot_sample_paths(
        outdir=outdir,
        gap_arm2=gap,
        horizon=args.horizon,
        num_arms=args.num_arms,
        num_paths=args.sample_paths,
        num_checkpoints=args.sample_checkpoints,
        seed=args.seed,
        method=args.method,
        sim_kwargs=common_sim_kwargs(args),
    )
    print(f"Wrote {path}")


def write_slurm_scripts(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arms_list = [int(x.strip()) for x in args.num_arms_list.split(",") if x.strip()]
    total_tasks = len(arms_list) * args.num_gaps
    arms_bash = " ".join(str(x) for x in arms_list)

    sweep_script = f"""#!/bin/bash
#SBATCH --job-name=inv_sqrt_env
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --array=0-{total_tasks - 1}%{args.array_limit}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ARMS_LIST=({arms_bash})
NUM_GAPS={args.num_gaps}
ARM_CASE=$((SLURM_ARRAY_TASK_ID / NUM_GAPS))
GAP_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_GAPS))
NUM_ARMS=${{ARMS_LIST[$ARM_CASE]}}

python run_inv_sqrt_tplus1_plots.py sweep-one \
  --outdir {args.outdir} \
  --num-arms "$NUM_ARMS" \
  --gap-index "$GAP_INDEX" \
  --num-gaps {args.num_gaps} \
  --gap-start {args.gap_start} \
  --gap-stop {args.gap_stop} \
  --horizon {args.horizon} \
  --num-horizons {args.num_horizons} \
  --trajectories {args.trajectories} \
  --workers "$SLURM_CPUS_PER_TASK" \
  --chunk-trajectories {args.chunk_trajectories} \
  --method {args.method} \
  --max-mean-change {args.max_mean_change} \
  --max-noise-change {args.max_noise_change} \
  --max-block-size {args.max_block_size} \
  --block-quantile {args.block_quantile}
"""

    combine_script = f"""#!/bin/bash
#SBATCH --job-name=inv_sqrt_plot
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs {args.outdir}
module load python scipy-stack || true

python run_inv_sqrt_tplus1_plots.py combine-plot \
  --outdir {args.outdir} \
  --num-arms {args.num_arms} \
  --num-arms-list {args.num_arms_list} \
  --horizon {args.horizon} \
  --regret-scale {args.regret_scale} \
  --c-values {args.c_values} \
  --make-sample-path \
  --sample-paths {args.sample_paths} \
  --sample-checkpoints {args.sample_checkpoints} \
  --method {args.method} \
  --max-mean-change {args.max_mean_change} \
  --max-noise-change {args.max_noise_change} \
  --max-block-size {args.max_block_size} \
  --block-quantile {args.block_quantile}
"""

    sweep_path = outdir / "submit_inv_sqrt_tplus1_envelope_array.sh"
    combine_path = outdir / "combine_plot_inv_sqrt_tplus1.sh"
    sweep_path.write_text(sweep_script)
    combine_path.write_text(combine_script)
    os.chmod(sweep_path, 0o755)
    os.chmod(combine_path, 0o755)
    print(f"Wrote {sweep_path}")
    print(f"Wrote {combine_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", default="inv_sqrt_tplus1_results")
    parser.add_argument("--horizon", type=int, default=10_000_000)
    parser.add_argument("--num-horizons", type=int, default=21)
    parser.add_argument("--horizon-values", default=None)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--num-arms-list", default="40,10", help="Comma-separated K values for main and few-arm comparisons.")
    parser.add_argument("--trajectories", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-trajectories", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--method", choices=["approx", "exact"], default="approx")
    parser.add_argument("--exact-small-block-threshold", type=int, default=64)
    parser.add_argument("--max-mean-change", type=float, default=0.08)
    parser.add_argument("--max-noise-change", type=float, default=0.35)
    parser.add_argument("--max-block-size", type=int, default=10_000_000)
    parser.add_argument("--block-quantile", type=float, default=0.995)
    parser.add_argument("--regret-scale", type=float, default=1_000_000.0)
    parser.add_argument("--c-values", default="0.5,1,2,4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep-one", help="Run one gap for one K. Suitable for a SLURM array task.")
    add_common_args(p)
    p.add_argument("--gap-index", type=int, required=True)
    p.add_argument("--num-gaps", type=int, default=51)
    p.add_argument("--gap-start", type=float, default=0.0)
    p.add_argument("--gap-stop", type=float, default=1.0)
    p.set_defaults(func=run_sweep_one)

    p = sub.add_parser("sweep-all", help="Run all gaps locally for all K values in --num-arms-list.")
    add_common_args(p)
    p.add_argument("--num-gaps", type=int, default=51)
    p.add_argument("--gap-start", type=float, default=0.0)
    p.add_argument("--gap-stop", type=float, default=1.0)
    p.set_defaults(func=run_sweep_all)

    p = sub.add_parser("combine-plot", help="Combine sweep outputs and make the requested plots.")
    add_common_args(p)
    p.add_argument("--make-sample-path", action="store_true")
    p.add_argument("--sample-gap", type=float, default=None)
    p.add_argument("--sample-paths", type=int, default=40)
    p.add_argument("--sample-checkpoints", type=int, default=250)
    p.set_defaults(func=run_combine_plot)

    p = sub.add_parser("sample-path", help="Make only the pi_1 sample-path plot.")
    add_common_args(p)
    p.add_argument("--sample-gap", type=float, default=None)
    p.add_argument("--sample-paths", type=int, default=40)
    p.add_argument("--sample-checkpoints", type=int, default=250)
    p.set_defaults(func=run_sample_path)

    p = sub.add_parser("write-slurm", help="Write Compute Canada / SLURM scripts for the envelope sweep.")
    add_common_args(p)
    p.add_argument("--num-gaps", type=int, default=51)
    p.add_argument("--gap-start", type=float, default=0.0)
    p.add_argument("--gap-stop", type=float, default=1.0)
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--array-limit", type=int, default=100)
    p.add_argument("--mem", default="16G")
    p.add_argument("--sample-paths", type=int, default=40)
    p.add_argument("--sample-checkpoints", type=int, default=250)
    p.set_defaults(func=write_slurm_scripts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
