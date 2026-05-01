#!/usr/bin/env python3
"""
Compare exact Algorithm-1 policy-gradient updates with the blocked Gaussian
aggregate approximation on the lower-bound bandit instance.

Instance:
    mu    = (1, 1 - Delta, 0, ..., 0)
    sigma = (1, 1,         0, ..., 0)
    eta_t = 1 / sqrt(t)

The exact simulator performs the literal round-by-round discrete-time update:
    theta_{t+1,a} = theta_{t,a} + eta_t * (1{A_t=a} - pi_{t,a}) * Y_t.

The approximate simulator is the blocked Gaussian aggregate update: it freezes
the policy inside each block and samples the eta_t-weighted reward sums for arms
1 and 2 from their matching bivariate Gaussian distribution.

This is intended for validation, so the default block-size controls are
conservative. Exact simulation scales as O(horizon * trajectories * gaps), so
use moderate horizons first.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EULER_GAMMA = 0.5772156649015328606


@dataclass(frozen=True)
class Config:
    horizon: int
    num_arms: int
    trajectories: int
    seed: int
    max_mean_change: float
    max_noise_change: float
    max_block_size: int
    block_quantile: float
    exact_eta_sum_threshold: int
    checkpoint_count: int
    save_history: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact PG updates with blocked Gaussian approximation."
    )

    parser.add_argument("--combine", action="store_true",
                        help="Combine final_gap_*.csv/history_gap_*.csv files in --outdir.")
    parser.add_argument("--gap-index", type=int, default=None,
                        help="Run only one gap index. Useful for SLURM arrays.")
    parser.add_argument("--gap-values", type=str, default=None,
                        help="Comma-separated gap values. Overrides gap-start/stop/num-gaps.")
    parser.add_argument("--gap-start", type=float, default=0.0)
    parser.add_argument("--gap-stop", type=float, default=1.0)
    parser.add_argument("--num-gaps", type=int, default=21)

    parser.add_argument("--horizon", type=int, default=100_000)
    parser.add_argument("--num-arms", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--methods", choices=["both", "exact", "approx"], default="both")
    parser.add_argument("--outdir", type=str, default="compare_results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--regret-scale", type=float, default=1_000_000.0)

    # Conservative validation defaults. Increase these for speed; decrease for
    # a stricter approximation check.
    parser.add_argument("--max-mean-change", type=float, default=0.02)
    parser.add_argument("--max-noise-change", type=float, default=0.10)
    parser.add_argument("--max-block-size", type=int, default=1_000_000)
    parser.add_argument("--block-quantile", type=float, default=1.0,
                        help="1.0 uses the worst trajectory; 0.995 is faster but less conservative.")
    parser.add_argument("--exact-eta-sum-threshold", type=int, default=1_000_000)

    parser.add_argument("--checkpoint-count", type=int, default=80)
    parser.add_argument("--no-history", action="store_true")

    return parser.parse_args()


def default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def gap_grid(args: argparse.Namespace) -> np.ndarray:
    if args.gap_values:
        values = [float(x.strip()) for x in args.gap_values.split(",") if x.strip()]
        if not values:
            raise ValueError("--gap-values was provided, but no values were parsed.")
        return np.asarray(values, dtype=np.float64)
    if args.num_gaps < 2:
        raise ValueError("--num-gaps must be at least 2 unless --gap-values is used.")
    return np.linspace(args.gap_start, args.gap_stop, args.num_gaps, dtype=np.float64)


def split_trajectories(total: int, workers: int, chunk_size: Optional[int]) -> List[int]:
    if total <= 0:
        raise ValueError("--trajectories must be positive.")

    if chunk_size is not None and chunk_size > 0:
        chunks: List[int] = []
        remaining = total
        while remaining:
            n = min(chunk_size, remaining)
            chunks.append(n)
            remaining -= n
        return chunks

    workers = max(1, min(workers, total))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def checkpoints_for(horizon: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([horizon], dtype=np.int64)
    pts = np.unique(np.round(np.geomspace(1, horizon, count)).astype(np.int64))
    pts = pts[(pts >= 1) & (pts <= horizon)]
    if pts.size == 0 or pts[-1] != horizon:
        pts = np.unique(np.concatenate([pts, np.asarray([horizon], dtype=np.int64)]))
    return pts


def stable_probs(theta1: np.ndarray, theta2: np.ndarray, num_other_arms: int):
    """Softmax probabilities for arm 1, arm 2, and each identical arm 3..k."""
    theta_other = -(theta1 + theta2) / num_other_arms
    max_theta = np.maximum(np.maximum(theta1, theta2), theta_other)

    w1 = np.exp(theta1 - max_theta)
    w2 = np.exp(theta2 - max_theta)
    w_other = np.exp(theta_other - max_theta)

    denom = w1 + w2 + num_other_arms * w_other
    p1 = w1 / denom
    p2 = w2 / denom
    p_other_each = w_other / denom
    return p1, p2, p_other_each


def instant_regret(p1: np.ndarray, p2: np.ndarray, gap: float) -> np.ndarray:
    """Conditional one-step regret under mu=(1,1-gap,0,...,0)."""
    return (1.0 - p1 - p2) + gap * p2


def aggregate_mean_se(n: int, s: float, ss: float) -> Tuple[float, float]:
    mean = s / n
    if n <= 1:
        return mean, float("nan")
    var = (ss - s * s / n) / (n - 1)
    var = max(0.0, var)
    return mean, math.sqrt(var / n)


def empty_history(num_checkpoints: int) -> Dict[str, np.ndarray]:
    return {
        "sum_regret": np.zeros(num_checkpoints, dtype=np.float64),
        "sumsq_regret": np.zeros(num_checkpoints, dtype=np.float64),
        "sum_pi1": np.zeros(num_checkpoints, dtype=np.float64),
        "sumsq_pi1": np.zeros(num_checkpoints, dtype=np.float64),
    }


def add_history(dst: Dict[str, np.ndarray], src: Dict[str, np.ndarray]) -> None:
    for key in dst:
        dst[key] += src[key]


# ---------------------------------------------------------------------------
# Exact round-by-round Algorithm 1
# ---------------------------------------------------------------------------

def simulate_exact_chunk(
    gap: float,
    chunk_n: int,
    seed: int,
    config: Config,
    checkpoints: np.ndarray,
) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)

    num_other = config.num_arms - 2
    mu2 = 1.0 - gap

    theta1 = np.zeros(chunk_n, dtype=np.float64)
    theta2 = np.zeros(chunk_n, dtype=np.float64)
    cond_regret = np.zeros(chunk_n, dtype=np.float64)
    realized_regret = np.zeros(chunk_n, dtype=np.float64)

    history = empty_history(len(checkpoints)) if config.save_history else None
    checkpoint_pos = 0

    for t in range(1, config.horizon + 1):
        p1, p2, _ = stable_probs(theta1, theta2, num_other)

        cond_regret += instant_regret(p1, p2, gap)

        u = rng.random(chunk_n)
        chose1 = u < p1
        chose2 = (~chose1) & (u < p1 + p2)
        chose_other = ~(chose1 | chose2)

        reward = np.zeros(chunk_n, dtype=np.float64)

        n1 = int(np.sum(chose1))
        if n1:
            reward[chose1] = 1.0 + rng.standard_normal(n1)

        n2 = int(np.sum(chose2))
        if n2:
            reward[chose2] = mu2 + rng.standard_normal(n2)

        eta_t = 1.0 / math.sqrt(float(t))
        theta1 += eta_t * (chose1.astype(np.float64) - p1) * reward
        theta2 += eta_t * (chose2.astype(np.float64) - p2) * reward

        realized_regret += gap * chose2.astype(np.float64) + chose_other.astype(np.float64)

        if config.save_history and t == int(checkpoints[checkpoint_pos]):
            p1_after, _, _ = stable_probs(theta1, theta2, num_other)
            history["sum_regret"][checkpoint_pos] = float(np.sum(cond_regret))
            history["sumsq_regret"][checkpoint_pos] = float(np.sum(cond_regret * cond_regret))
            history["sum_pi1"][checkpoint_pos] = float(np.sum(p1_after))
            history["sumsq_pi1"][checkpoint_pos] = float(np.sum(p1_after * p1_after))
            checkpoint_pos += 1
            if checkpoint_pos >= len(checkpoints):
                checkpoint_pos = len(checkpoints) - 1

    p1_final, _, _ = stable_probs(theta1, theta2, num_other)

    return {
        "method": "exact",
        "n": chunk_n,
        "sum_cond_regret": float(np.sum(cond_regret)),
        "sumsq_cond_regret": float(np.sum(cond_regret * cond_regret)),
        "sum_realized_regret": float(np.sum(realized_regret)),
        "sumsq_realized_regret": float(np.sum(realized_regret * realized_regret)),
        "sum_pi1": float(np.sum(p1_final)),
        "sumsq_pi1": float(np.sum(p1_final * p1_final)),
        "num_blocks": config.horizon,
        "seconds": time.time() - t0,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Blocked Gaussian aggregate approximation
# ---------------------------------------------------------------------------

def harmonic_approx(n: int) -> float:
    if n <= 0:
        return 0.0
    if n <= 100_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    x = float(n)
    x2 = x * x
    return math.log(x) + EULER_GAMMA + 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)


def sum_inv_sqrt_euler_maclaurin(start: int, block_size: int) -> float:
    """Approximate sum_{t=start}^{start+B-1} 1/sqrt(t)."""
    a = float(start)
    b = float(start + block_size - 1)

    # Euler-Maclaurin for integer points a, ..., b.
    integral = 2.0 * (math.sqrt(b) - math.sqrt(a))
    endpoints = 0.5 * (a ** -0.5 + b ** -0.5)
    derivative_correction = (1.0 / 12.0) * (-0.5 * b ** -1.5 + 0.5 * a ** -1.5)
    return integral + endpoints + derivative_correction


def eta_sums(start: int, block_size: int, exact_threshold: int) -> Tuple[float, float]:
    """sum eta_t and sum eta_t^2 for eta_t=1/sqrt(t)."""
    if block_size <= exact_threshold:
        t = np.arange(start, start + block_size, dtype=np.float64)
        inv_sqrt = 1.0 / np.sqrt(t)
        return float(np.sum(inv_sqrt)), float(np.sum(inv_sqrt * inv_sqrt))

    end = start + block_size - 1
    return (
        sum_inv_sqrt_euler_maclaurin(start, block_size),
        harmonic_approx(end) - harmonic_approx(start - 1),
    )


def high_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    if q >= 1.0:
        return float(np.max(values))
    if q <= 0.0:
        return float(np.min(values))
    kth = int(math.ceil(q * (values.size - 1)))
    return float(np.partition(values, kth)[kth])


def choose_block_size(
    current_time: int,
    remaining_steps: int,
    gap: float,
    p1: np.ndarray,
    p2: np.ndarray,
    p_other_each: np.ndarray,
    num_other: int,
    config: Config,
) -> int:
    eta_now = 1.0 / math.sqrt(float(current_time))
    mu2 = 1.0 - gap

    other_total = num_other * p_other_each
    r = other_total + gap * p2

    mean_g1 = p1 * r
    mean_g2 = p2 * (r - gap)
    mean_g_other = -(mean_g1 + mean_g2) / num_other

    q = config.block_quantile
    mean_scale = max(
        high_quantile(np.abs(mean_g1), q),
        high_quantile(np.abs(mean_g2), q),
        high_quantile(np.abs(mean_g_other), q),
    )

    # Per-round second-moment scale for gradient coordinates.
    ey1_sq = 2.0
    ey2_sq = 1.0 + mu2 * mu2

    second_moment_g1 = p1 * (1.0 - p1) ** 2 * ey1_sq + p2 * p1**2 * ey2_sq
    second_moment_g2 = p1 * p2**2 * ey1_sq + p2 * (1.0 - p2) ** 2 * ey2_sq
    second_moment_g_other = p1 * p_other_each**2 * ey1_sq + p2 * p_other_each**2 * ey2_sq

    noise_scale = max(
        high_quantile(second_moment_g1, q),
        high_quantile(second_moment_g2, q),
        high_quantile(second_moment_g_other, q),
    )

    limit_mean = math.inf
    if mean_scale > 1e-300:
        limit_mean = config.max_mean_change / (eta_now * mean_scale)

    limit_noise = math.inf
    if noise_scale > 1e-300:
        limit_noise = (config.max_noise_change / (eta_now * math.sqrt(noise_scale))) ** 2

    return int(max(1, math.floor(min(
        remaining_steps,
        config.max_block_size,
        limit_mean,
        limit_noise,
    ))))


def sample_weighted_reward_sums(
    rng: np.random.Generator,
    p1: np.ndarray,
    p2: np.ndarray,
    gap: float,
    sum_eta: float,
    sum_eta_sq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bivariate Gaussian approximation for sum eta_t 1{A_t=j}Y_t, j=1,2."""
    mu1 = 1.0
    mu2 = 1.0 - gap
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


def simulate_approx_chunk(
    gap: float,
    chunk_n: int,
    seed: int,
    config: Config,
    checkpoints: np.ndarray,
) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)

    num_other = config.num_arms - 2

    theta1 = np.zeros(chunk_n, dtype=np.float64)
    theta2 = np.zeros(chunk_n, dtype=np.float64)
    cond_regret = np.zeros(chunk_n, dtype=np.float64)

    history = empty_history(len(checkpoints)) if config.save_history else None
    checkpoint_pos = 0
    num_blocks = 0
    current_time = 1

    while current_time <= config.horizon:
        p1, p2, p_other_each = stable_probs(theta1, theta2, num_other)

        if config.save_history:
            next_checkpoint = int(checkpoints[checkpoint_pos])
            remaining_to_checkpoint = next_checkpoint - current_time + 1
        else:
            remaining_to_checkpoint = config.horizon - current_time + 1

        remaining_to_horizon = config.horizon - current_time + 1
        remaining = min(remaining_to_horizon, remaining_to_checkpoint)

        block_size = choose_block_size(
            current_time=current_time,
            remaining_steps=remaining,
            gap=gap,
            p1=p1,
            p2=p2,
            p_other_each=p_other_each,
            num_other=num_other,
            config=config,
        )

        cond_regret += block_size * instant_regret(p1, p2, gap)

        sum_eta, sum_eta_sq = eta_sums(
            current_time,
            block_size,
            config.exact_eta_sum_threshold,
        )
        s1, s2 = sample_weighted_reward_sums(
            rng=rng,
            p1=p1,
            p2=p2,
            gap=gap,
            sum_eta=sum_eta,
            sum_eta_sq=sum_eta_sq,
        )

        # Aggregated version of theta_{t+1}=theta_t+eta_t(1_A-pi)Y.
        theta1 += (1.0 - p1) * s1 - p1 * s2
        theta2 += -p2 * s1 + (1.0 - p2) * s2

        current_time += block_size
        num_blocks += 1

        if config.save_history and current_time - 1 == int(checkpoints[checkpoint_pos]):
            p1_after, _, _ = stable_probs(theta1, theta2, num_other)
            history["sum_regret"][checkpoint_pos] = float(np.sum(cond_regret))
            history["sumsq_regret"][checkpoint_pos] = float(np.sum(cond_regret * cond_regret))
            history["sum_pi1"][checkpoint_pos] = float(np.sum(p1_after))
            history["sumsq_pi1"][checkpoint_pos] = float(np.sum(p1_after * p1_after))
            checkpoint_pos += 1
            if checkpoint_pos >= len(checkpoints):
                checkpoint_pos = len(checkpoints) - 1

    p1_final, _, _ = stable_probs(theta1, theta2, num_other)

    return {
        "method": "approx",
        "n": chunk_n,
        "sum_cond_regret": float(np.sum(cond_regret)),
        "sumsq_cond_regret": float(np.sum(cond_regret * cond_regret)),
        "sum_realized_regret": float("nan"),
        "sumsq_realized_regret": float("nan"),
        "sum_pi1": float(np.sum(p1_final)),
        "sumsq_pi1": float(np.sum(p1_final * p1_final)),
        "num_blocks": num_blocks,
        "seconds": time.time() - t0,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Parallel runner and output
# ---------------------------------------------------------------------------

def run_method_gap(
    method: str,
    gap_index: int,
    gap: float,
    chunks: Sequence[int],
    workers: int,
    config: Config,
    checkpoints: np.ndarray,
) -> Tuple[dict, List[dict]]:
    simulator = simulate_exact_chunk if method == "exact" else simulate_approx_chunk

    print(
        f"[run] method={method} gap_index={gap_index} gap={gap:.8g} "
        f"trajectories={sum(chunks)} chunks={len(chunks)} workers={workers}",
        flush=True,
    )

    total_n = 0
    total_cond_s = 0.0
    total_cond_ss = 0.0
    total_real_s = 0.0
    total_real_ss = 0.0
    total_pi_s = 0.0
    total_pi_ss = 0.0
    total_blocks = 0
    max_chunk_seconds = 0.0
    history_acc = empty_history(len(checkpoints)) if config.save_history else None

    start = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_id, chunk_n in enumerate(chunks):
            method_offset = 0 if method == "exact" else 50_000_000
            seed = int(config.seed + method_offset + 1_000_003 * gap_index + 9_176 * chunk_id)
            futures.append(pool.submit(simulator, float(gap), int(chunk_n), seed, config, checkpoints))

        for future in as_completed(futures):
            res = future.result()
            n = int(res["n"])
            total_n += n
            total_cond_s += float(res["sum_cond_regret"])
            total_cond_ss += float(res["sumsq_cond_regret"])
            if method == "exact":
                total_real_s += float(res["sum_realized_regret"])
                total_real_ss += float(res["sumsq_realized_regret"])
            total_pi_s += float(res["sum_pi1"])
            total_pi_ss += float(res["sumsq_pi1"])
            total_blocks += int(res["num_blocks"])
            max_chunk_seconds = max(max_chunk_seconds, float(res["seconds"]))
            if config.save_history and res["history"] is not None:
                add_history(history_acc, res["history"])

            print(
                f"  chunk done: method={method} n={n} "
                f"mean_cond_regret={float(res['sum_cond_regret'])/n:.6e} "
                f"blocks={res['num_blocks']} seconds={res['seconds']:.1f}",
                flush=True,
            )

    mean_cond, se_cond = aggregate_mean_se(total_n, total_cond_s, total_cond_ss)
    mean_pi, se_pi = aggregate_mean_se(total_n, total_pi_s, total_pi_ss)
    if method == "exact":
        mean_real, se_real = aggregate_mean_se(total_n, total_real_s, total_real_ss)
    else:
        mean_real, se_real = float("nan"), float("nan")

    final_row = {
        "gap_index": gap_index,
        "gap_arm2": gap,
        "method": method,
        "mean_conditional_regret": mean_cond,
        "se_conditional_regret": se_cond,
        "mean_realized_regret": mean_real,
        "se_realized_regret": se_real,
        "mean_final_pi1": mean_pi,
        "se_final_pi1": se_pi,
        "num_trajectories": total_n,
        "horizon": config.horizon,
        "num_arms": config.num_arms,
        "eta_schedule": "eta_t = 1/sqrt(t)",
        "mean_blocks_per_chunk": total_blocks / len(chunks),
        "num_chunks": len(chunks),
        "workers": workers,
        "max_chunk_seconds": max_chunk_seconds,
        "wall_seconds": time.time() - start,
        "max_mean_change": config.max_mean_change,
        "max_noise_change": config.max_noise_change,
        "max_block_size": config.max_block_size,
        "block_quantile": config.block_quantile,
    }

    history_rows: List[dict] = []
    if config.save_history and history_acc is not None:
        for i, checkpoint in enumerate(checkpoints):
            mreg, sereg = aggregate_mean_se(
                total_n,
                float(history_acc["sum_regret"][i]),
                float(history_acc["sumsq_regret"][i]),
            )
            mpi, sepi = aggregate_mean_se(
                total_n,
                float(history_acc["sum_pi1"][i]),
                float(history_acc["sumsq_pi1"][i]),
            )
            history_rows.append({
                "gap_index": gap_index,
                "gap_arm2": gap,
                "method": method,
                "time_step": int(checkpoint),
                "mean_conditional_regret": mreg,
                "se_conditional_regret": sereg,
                "mean_pi1": mpi,
                "se_pi1": sepi,
                "num_trajectories": total_n,
            })

    return final_row, history_rows


def write_csv(path: pathlib.Path, rows: List[dict]) -> None:
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


def comparison_rows(final_rows: List[dict]) -> List[dict]:
    by_gap: Dict[int, Dict[str, dict]] = {}
    for row in final_rows:
        by_gap.setdefault(int(row["gap_index"]), {})[row["method"]] = row

    rows: List[dict] = []
    for gap_index in sorted(by_gap):
        group = by_gap[gap_index]
        if "exact" not in group or "approx" not in group:
            continue

        exact = group["exact"]
        approx = group["approx"]

        exact_r = float(exact["mean_conditional_regret"])
        approx_r = float(approx["mean_conditional_regret"])
        exact_r_se = float(exact["se_conditional_regret"])
        approx_r_se = float(approx["se_conditional_regret"])
        r_diff = approx_r - exact_r
        r_den = math.sqrt(exact_r_se * exact_r_se + approx_r_se * approx_r_se)

        exact_pi = float(exact["mean_final_pi1"])
        approx_pi = float(approx["mean_final_pi1"])
        exact_pi_se = float(exact["se_final_pi1"])
        approx_pi_se = float(approx["se_final_pi1"])
        pi_diff = approx_pi - exact_pi
        pi_den = math.sqrt(exact_pi_se * exact_pi_se + approx_pi_se * approx_pi_se)

        rows.append({
            "gap_index": gap_index,
            "gap_arm2": float(exact["gap_arm2"]),
            "exact_mean_conditional_regret": exact_r,
            "approx_mean_conditional_regret": approx_r,
            "regret_difference_approx_minus_exact": r_diff,
            "regret_relative_difference": r_diff / exact_r if exact_r != 0.0 else float("nan"),
            "regret_z_score": r_diff / r_den if r_den > 0.0 else float("nan"),
            "exact_mean_final_pi1": exact_pi,
            "approx_mean_final_pi1": approx_pi,
            "pi1_difference_approx_minus_exact": pi_diff,
            "pi1_z_score": pi_diff / pi_den if pi_den > 0.0 else float("nan"),
        })
    return rows


def combine(outdir: pathlib.Path, do_plot: bool, regret_scale: float) -> None:
    final_paths = sorted(outdir.glob("final_gap_*.csv"))
    if not final_paths:
        raise FileNotFoundError(f"No final_gap_*.csv files found in {outdir}")

    final_rows: List[dict] = []
    for path in final_paths:
        final_rows.extend(read_csv(path))
    final_rows.sort(key=lambda r: (int(r["gap_index"]), r["method"]))
    write_csv(outdir / "combined_final.csv", final_rows)

    comp = comparison_rows(final_rows)
    if comp:
        write_csv(outdir / "comparison_exact_vs_approx.csv", comp)

    history_paths = sorted(outdir.glob("history_gap_*.csv"))
    if history_paths:
        history_rows: List[dict] = []
        for path in history_paths:
            history_rows.extend(read_csv(path))
        history_rows.sort(key=lambda r: (int(r["gap_index"]), r["method"], int(r["time_step"])))
        write_csv(outdir / "combined_history.csv", history_rows)

    if do_plot:
        make_plots(outdir, regret_scale)


def make_plots(outdir: pathlib.Path, regret_scale: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}", file=sys.stderr)
        return

    final_path = outdir / "combined_final.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)

    final = read_csv(final_path)
    labels = {"exact": "Exact Algorithm 1", "approx": "Blocked Gaussian approximation"}

    def arrays(method: str):
        rows = [r for r in final if r["method"] == method]
        rows.sort(key=lambda r: float(r["gap_arm2"]))
        gap = np.asarray([float(r["gap_arm2"]) for r in rows])
        reg = np.asarray([float(r["mean_conditional_regret"]) for r in rows])
        reg_se = np.asarray([float(r["se_conditional_regret"]) for r in rows])
        pi1 = np.asarray([float(r["mean_final_pi1"]) for r in rows])
        pi1_se = np.asarray([float(r["se_final_pi1"]) for r in rows])
        return gap, reg, reg_se, pi1, pi1_se

    present = {r["method"] for r in final}

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for method in ["exact", "approx"]:
        if method not in present:
            continue
        gap, reg, reg_se, _, _ = arrays(method)
        y = reg / regret_scale
        se = reg_se / regret_scale
        ax.plot(gap, y, marker="o", markersize=3.5, linewidth=2.2, label=labels[method])
        ax.fill_between(gap, y - 2.0 * se, y + 2.0 * se, alpha=0.16, linewidth=0)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("Mean conditional regret" if regret_scale == 1 else rf"Mean conditional regret / ${regret_scale:.0e}$")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "regret_exact_vs_approx.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for method in ["exact", "approx"]:
        if method not in present:
            continue
        gap, _, _, pi1, pi1_se = arrays(method)
        ax.plot(gap, pi1, marker="o", markersize=3.5, linewidth=2.2, label=labels[method])
        ax.fill_between(gap, pi1 - 2.0 * pi1_se, pi1 + 2.0 * pi1_se, alpha=0.16, linewidth=0)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"Mean final $\pi_1$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "final_pi1_exact_vs_approx.png", bbox_inches="tight")
    plt.close(fig)

    comp_path = outdir / "comparison_exact_vs_approx.csv"
    if comp_path.exists():
        comp = read_csv(comp_path)
        comp.sort(key=lambda r: float(r["gap_arm2"]))
        gap = np.asarray([float(r["gap_arm2"]) for r in comp])
        rel = np.asarray([float(r["regret_relative_difference"]) for r in comp])
        z = np.asarray([float(r["regret_z_score"]) for r in comp])

        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
        ax.axhline(0.0, linewidth=1.0)
        ax.plot(gap, rel, marker="o", markersize=3.5, linewidth=2.2)
        ax.set_xlabel(r"$\Delta$")
        ax.set_ylabel("(approx - exact) / exact regret")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(outdir / "relative_regret_error.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
        ax.axhline(0.0, linewidth=1.0)
        ax.axhline(2.0, linestyle="--", linewidth=1.0)
        ax.axhline(-2.0, linestyle="--", linewidth=1.0)
        ax.plot(gap, z, marker="o", markersize=3.5, linewidth=2.2)
        ax.set_xlabel(r"$\Delta$")
        ax.set_ylabel("Regret difference z-score")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(outdir / "regret_difference_z_score.png", bbox_inches="tight")
        plt.close(fig)

    hist_path = outdir / "combined_history.csv"
    if hist_path.exists():
        hist = read_csv(hist_path)
        gap_values = sorted({float(r["gap_arm2"]) for r in hist})
        selected_gap = gap_values[len(gap_values) // 2]

        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
        for method in ["exact", "approx"]:
            rows = [
                r for r in hist
                if r["method"] == method and abs(float(r["gap_arm2"]) - selected_gap) < 1e-14
            ]
            if not rows:
                continue
            rows.sort(key=lambda r: int(r["time_step"]))
            t = np.asarray([int(r["time_step"]) for r in rows], dtype=np.float64)
            pi = np.asarray([float(r["mean_pi1"]) for r in rows])
            se = np.asarray([float(r["se_pi1"]) for r in rows])
            ax.plot(t, pi, linewidth=2.2, label=labels[method])
            ax.fill_between(t, pi - 2.0 * se, pi + 2.0 * se, alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Mean $\pi_1$")
        ax.set_title(rf"Checkpoint history at $\Delta={selected_gap:g}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(outdir / "history_pi1_middle_gap.png", bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] wrote plots to {outdir}", flush=True)


def main() -> None:
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        combine(outdir, args.plot, args.regret_scale)
        return

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)

    config = Config(
        horizon=args.horizon,
        num_arms=args.num_arms,
        trajectories=args.trajectories,
        seed=args.seed,
        max_mean_change=args.max_mean_change,
        max_noise_change=args.max_noise_change,
        max_block_size=args.max_block_size,
        block_quantile=args.block_quantile,
        exact_eta_sum_threshold=args.exact_eta_sum_threshold,
        checkpoint_count=args.checkpoint_count,
        save_history=not args.no_history,
    )

    gaps = gap_grid(args)
    if args.gap_index is not None:
        if args.gap_index < 0 or args.gap_index >= len(gaps):
            raise IndexError(f"--gap-index must be in [0, {len(gaps) - 1}]")
        gap_items = [(args.gap_index, float(gaps[args.gap_index]))]
    else:
        gap_items = [(i, float(g)) for i, g in enumerate(gaps)]

    methods = ["exact", "approx"] if args.methods == "both" else [args.methods]
    checkpoints = checkpoints_for(config.horizon, config.checkpoint_count)
    chunks = split_trajectories(config.trajectories, workers, args.chunk_trajectories)

    all_final: List[dict] = []
    all_history: List[dict] = []

    for gap_index, gap in gap_items:
        final_path = outdir / f"final_gap_{gap_index:04d}.csv"
        history_path = outdir / f"history_gap_{gap_index:04d}.csv"

        if final_path.exists() and not args.overwrite:
            print(f"[skip] {final_path} exists. Use --overwrite to recompute.", flush=True)
            continue

        gap_final: List[dict] = []
        gap_history: List[dict] = []

        for method in methods:
            final_row, history_rows = run_method_gap(
                method=method,
                gap_index=gap_index,
                gap=gap,
                chunks=chunks,
                workers=workers,
                config=config,
                checkpoints=checkpoints,
            )
            gap_final.append(final_row)
            gap_history.extend(history_rows)

        write_csv(final_path, gap_final)
        if gap_history:
            write_csv(history_path, gap_history)

        all_final.extend(gap_final)
        all_history.extend(gap_history)

    if args.gap_index is None:
        if all_final:
            write_csv(outdir / "combined_final.csv", all_final)
            comp = comparison_rows(all_final)
            if comp:
                write_csv(outdir / "comparison_exact_vs_approx.csv", comp)
        if all_history:
            write_csv(outdir / "combined_history.csv", all_history)
        if args.plot and all_final:
            make_plots(outdir, args.regret_scale)


if __name__ == "__main__":
    main()
