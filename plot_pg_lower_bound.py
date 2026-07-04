#!/usr/bin/env python3
"""
Reproduce Appendix F-style plots for:
A Diffusion Analysis of Policy Gradient for Stochastic Bandits.

The experiment simulates discrete-time policy gradient, Algorithm 1, on the
lower-bound instance:
    Delta = (0, Delta_2, 1, ..., 1)
    mu_1 = 1
    mu_2 = 1 - Delta_2
    mu_a = 0 for a >= 3
    sigma_1 = sigma_2 = 1
    sigma_a = 0 for a >= 3

The implementation exploits the exact symmetry of arms a >= 3. Since these
arms have identical initial theta, identical mean 0, and zero noise, their
theta values remain equal. Therefore the k-dimensional simulation reduces to
three coordinates:
    theta_1, theta_2, theta_rest
where theta_rest is shared by all arms a >= 3.

It also uses exact geometric skipping: when the sampled action is in the
rest group, the reward is exactly zero and the policy-gradient update is zero.
So we can jump directly to the next time action 1 or action 2 is sampled.

Outputs:
    fig1_logit_scale.png / pdf
    fig1_original_scale.png / pdf
    fig2_logit_scale.png / pdf
    fig2_original_scale.png / pdf
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def eta_configs(delta: float):
    return [
        ("Delta", delta, r"$\eta=\Delta$"),
        ("Delta_over_2", delta / 2.0, r"$\eta=\Delta/2$"),
        ("Delta_over_10", delta / 10.0, r"$\eta=\Delta/10$"),
        ("Delta_over_30", delta / 30.0, r"$\eta=\Delta/30$"),
        ("Delta_over_100", delta / 100.0, r"$\eta=\Delta/100$"),
        ("Delta_squared", delta * delta, r"$\eta=\Delta^2$"),
    ]


def horizon_for_eta_label(label: str) -> int:
    # Chosen to match the visual x-axis ranges in Appendix F:
    # top row roughly up to 10^6, 10^7, 10^7;
    # bottom row roughly up to 10^8, 10^8, 10^9.
    return {
        "Delta": 10**6,
        "Delta_over_2": 10**7,
        "Delta_over_10": 10**7,
        "Delta_over_30": 10**8,
        "Delta_over_100": 10**8,
        "Delta_squared": 10**9,
    }[label]


def make_time_grid(T: int, n_grid: int) -> np.ndarray:
    # Include t=1 and T, log-spaced. We do not include t=0 because x-axis is log scale.
    grid = np.unique(np.round(np.logspace(0, math.log10(T), n_grid)).astype(np.int64))
    grid = grid[grid >= 1]
    if grid[-1] != T:
        grid = np.concatenate([grid, np.array([T], dtype=np.int64)])
    return grid


# ---------------------------------------------------------------------
# Exact symmetry-reduced simulator with geometric skipping
# ---------------------------------------------------------------------

@njit(cache=True)
def _softmax_three(theta1, theta2, thetar, m):
    """
    Softmax probabilities for:
        action 1: p1
        action 2: p2
        rest group total: pr = sum_{a=3}^k p_a
    where there are m = k - 2 identical rest arms.
    """
    mx = theta1
    if theta2 > mx:
        mx = theta2
    if thetar > mx:
        mx = thetar

    e1 = math.exp(theta1 - mx)
    e2 = math.exp(theta2 - mx)
    er = math.exp(thetar - mx)
    z = e1 + e2 + m * er

    return e1 / z, e2 / z, (m * er) / z


@njit(cache=True)
def _sample_geometric_wait(p):
    """
    Return G ~ Geometric(p), supported on {1,2,...}.
    This is the number of rounds until the next informative sample
    from action 1 or action 2.

    Uses inverse CDF:
        P(G > g) = (1-p)^g.
    """
    if p >= 1.0:
        return 1

    u = np.random.random()
    # Avoid log(0).
    if u <= 1e-300:
        u = 1e-300
    return int(math.ceil(math.log(u) / math.log1p(-p)))


@njit(cache=True)
def simulate_one_exact_skip(k, delta, eta, t_grid, seed):
    """
    Exact simulation of Algorithm 1 under the symmetric lower-bound instance,
    using geometric skipping of zero-update rest-arm samples.
    """
    np.random.seed(seed)

    m = k - 2
    T = t_grid[-1]
    out = np.empty(len(t_grid), dtype=np.float64)

    theta1 = 0.0
    theta2 = 0.0
    thetar = 0.0

    t = 0
    idx = 0

    mu1 = 1.0
    mu2 = 1.0 - delta

    while idx < len(t_grid):
        p1, p2, pr = _softmax_three(theta1, theta2, thetar, m)

        # If all requested checkpoints are already at or before current time,
        # write the current policy.
        while idx < len(t_grid) and t_grid[idx] <= t:
            out[idx] = p1
            idx += 1
        if idx >= len(t_grid):
            break

        p12 = p1 + p2

        # Jump over all rounds where the rest group is sampled.
        # Those have exactly zero reward and therefore zero update.
        wait = _sample_geometric_wait(p12)
        next_t = t + wait

        # Any checkpoint before the informative update sees unchanged policy.
        while idx < len(t_grid) and t_grid[idx] < next_t:
            out[idx] = p1
            idx += 1
        if idx >= len(t_grid):
            break

        # If next informative update is beyond T, fill remaining checkpoints.
        if next_t > T:
            while idx < len(t_grid):
                out[idx] = p1
                idx += 1
            break

        # At time next_t, condition on informative sample being action 1 or 2.
        t = next_t

        u = np.random.random()
        choose1 = u < (p1 / p12)

        if choose1:
            y = mu1 + np.random.normal()
            # Full k-dimensional Algorithm 1, reduced by symmetry.
            theta1 += eta * (1.0 - p1) * y
            theta2 += eta * (0.0 - p2) * y
            thetar += eta * (0.0 - pr / m) * y
        else:
            y = mu2 + np.random.normal()
            theta1 += eta * (0.0 - p1) * y
            theta2 += eta * (1.0 - p2) * y
            thetar += eta * (0.0 - pr / m) * y

    return out


# ---------------------------------------------------------------------
# Optional tau-leap approximation for extremely long exploratory runs
# ---------------------------------------------------------------------

def simulate_one_tau_leap(k, delta, eta, t_grid, seed, max_chunk=20000):
    """
    Fast approximate simulator.

    This is not used by default. It approximates many discrete PG updates
    over a block by a Gaussian increment with frozen policy. It is useful for
    very quick previews, but exact_skip is the recommended method for final
    runs.
    """
    rng = np.random.default_rng(seed)
    m = k - 2
    theta1 = 0.0
    theta2 = 0.0
    thetar = 0.0

    mu = np.array([1.0, 1.0 - delta, 0.0])
    sig = np.array([1.0, 1.0, 0.0])

    out = np.empty(len(t_grid), dtype=np.float64)
    t = 0

    for i, target in enumerate(t_grid):
        while t < target:
            h = int(min(max_chunk, target - t))

            p1, p2, pr = _softmax_three(theta1, theta2, thetar, m)
            p = np.array([p1, p2, pr])

            # One-step update vector in reduced coordinates.
            # For group sample, reward is zero, hence update zero.
            updates = []
            probs = []

            # Action 1
            g1 = np.array([1.0 - p1, -p2, -pr / m])
            mean1 = mu[0] * g1
            cov1 = sig[0] ** 2 * np.outer(g1, g1)
            updates.append((mean1, cov1))
            probs.append(p1)

            # Action 2
            g2 = np.array([-p1, 1.0 - p2, -pr / m])
            mean2 = mu[1] * g2
            cov2 = sig[1] ** 2 * np.outer(g2, g2)
            updates.append((mean2, cov2))
            probs.append(p2)

            # Rest group gives exactly zero update.
            probs.append(pr)

            one_mean = probs[0] * updates[0][0] + probs[1] * updates[1][0]

            second = (
                probs[0] * (updates[0][1] + np.outer(updates[0][0], updates[0][0]))
                + probs[1] * (updates[1][1] + np.outer(updates[1][0], updates[1][0]))
            )
            one_cov = second - np.outer(one_mean, one_mean)
            one_cov = 0.5 * (one_cov + one_cov.T)

            mean_block = h * eta * one_mean
            cov_block = h * eta * eta * one_cov

            # Numerical diagonal jitter.
            cov_block += 1e-14 * np.eye(3)

            step = rng.multivariate_normal(mean_block, cov_block)
            theta1 += step[0]
            theta2 += step[1]
            thetar += step[2]
            t += h

        p1, _, _ = _softmax_three(theta1, theta2, thetar, m)
        out[i] = p1

    return out


def worker(args):
    k, delta, eta, t_grid, seed, method = args

    if method == "exact_skip":
        return simulate_one_exact_skip(k, delta, eta, t_grid, seed)
    elif method == "tau_leap":
        return simulate_one_tau_leap(k, delta, eta, t_grid, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


def simulate_panel(k, delta, eta, label, ntraj, n_grid, base_seed, workers, method):
    T = horizon_for_eta_label(label)
    t_grid = make_time_grid(T, n_grid)

    jobs = []
    for r in range(ntraj):
        seed = base_seed + 1000003 * r + 9176 * k + int(round(1e12 * eta)) % 1000003
        jobs.append((k, delta, eta, t_grid, seed, method))

    paths = []
    if workers <= 1:
        for j in jobs:
            paths.append(worker(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, j) for j in jobs]
            for fut in as_completed(futs):
                paths.append(fut.result())

    paths = np.asarray(paths, dtype=np.float64)
    return t_grid, paths


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _logit(p):
    p = np.asarray(p)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _inv_logit(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def prob_tick_label(p):
    if abs(p - 0.5) < 1e-12:
        return r"$1/2$"
    if p < 0.5:
        if p == 1e-1:
            return r"$10^{-1}$"
        if p == 1e-2:
            return r"$10^{-2}$"
        if p == 1e-3:
            return r"$10^{-3}$"
        if p == 1e-4:
            return r"$10^{-4}$"
        return f"{p:g}"
    q = 1.0 - p
    if abs(q - 1e-1) < 1e-12:
        return r"$1-10^{-1}$"
    if abs(q - 1e-2) < 1e-12:
        return r"$1-10^{-2}$"
    if abs(q - 1e-3) < 1e-12:
        return r"$1-10^{-3}$"
    if abs(q - 1e-4) < 1e-12:
        return r"$1-10^{-4}$"
    return f"{p:g}"


def plot_figure(all_results, outpath_base, figure_title, transformed=True):
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharey=False)
    axes = axes.ravel()

    for ax_idx, (ax, result) in enumerate(zip(axes, all_results)):
        label, eta, title, t_grid, paths, k = result

        for r in range(paths.shape[0]):
            ax.plot(
                t_grid,
                paths[r],
                color="0.45",
                alpha=0.18,
                linewidth=0.9,
                zorder=1,
            )

        avg = paths.mean(axis=0)
        ax.plot(
            t_grid,
            avg,
            color="tab:red",
            linewidth=1.8,
            label="Average",
            zorder=5,
        )

        ax.set_xscale("log")
        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
        ax.set_title(title, fontsize=11)

        if transformed:
            ax.set_yscale("function", functions=(_logit, _inv_logit))
            if k == 3:
                ticks = [1e-1, 0.5, 1 - 1e-1, 1 - 1e-2]
                ax.set_ylim(7e-2, 1 - 5e-3)
            else:
                ticks = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4]
                ax.set_ylim(1e-4, 1 - 1e-4)
            ax.set_yticks(ticks)
            ax.set_yticklabels([prob_tick_label(p) for p in ticks])
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks(np.linspace(0, 1, 6))

        if ax_idx in [0, 3]:
            ax.set_ylabel("Optimal Action Probability")
        if ax_idx in [3, 4, 5]:
            ax.set_xlabel("Time")

        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(figure_title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png = outpath_base + ".png"
    pdf = outpath_base + ".pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")


def save_npz(outdir, name, results):
    payload = {}
    for i, result in enumerate(results):
        label, eta, title, t_grid, paths, k = result
        payload[f"label_{i}"] = label
        payload[f"eta_{i}"] = eta
        payload[f"title_{i}"] = title
        payload[f"k_{i}"] = k
        payload[f"t_grid_{i}"] = t_grid
        payload[f"paths_{i}"] = paths

    path = os.path.join(outdir, name)
    np.savez_compressed(path, **payload)
    print(f"Saved {path}")


def run_for_k(k, delta, ntraj, n_grid, seed, workers, method):
    results = []
    for i, (label, eta, title) in enumerate(eta_configs(delta)):
        print(f"Simulating k={k}, eta={title}, horizon={horizon_for_eta_label(label):,}, method={method}")
        t_grid, paths = simulate_panel(
            k=k,
            delta=delta,
            eta=eta,
            label=label,
            ntraj=ntraj,
            n_grid=n_grid,
            base_seed=seed + 77777 * i,
            workers=workers,
            method=method,
        )
        results.append((label, eta, title, t_grid, paths, k))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results_pg_lower_bound")
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--fig1-k", type=int, default=40)
    parser.add_argument("--fig2-k", type=int, default=3)
    parser.add_argument("--ntraj", type=int, default=40)
    parser.add_argument("--n-grid", type=int, default=520)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--method",
        type=str,
        default="exact_skip",
        choices=["exact_skip", "tau_leap"],
        help=(
            "exact_skip is exact for Algorithm 1 under the symmetric lower-bound instance. "
            "tau_leap is a fast approximation for previewing."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Numba available: {NUMBA_AVAILABLE}")
    if not NUMBA_AVAILABLE and args.method == "exact_skip":
        print("WARNING: exact_skip without numba will be very slow.")

    fig1_results = run_for_k(
        k=args.fig1_k,
        delta=args.delta,
        ntraj=args.ntraj,
        n_grid=args.n_grid,
        seed=args.seed,
        workers=args.workers,
        method=args.method,
    )
    save_npz(args.outdir, "fig1_k40_results.npz", fig1_results)

    plot_figure(
        fig1_results,
        os.path.join(args.outdir, "fig1_logit_scale"),
        figure_title=fr"Figure 1 style: $k={args.fig1_k}$, $\Delta_2={args.delta}$, {args.ntraj} trajectories",
        transformed=True,
    )
    plot_figure(
        fig1_results,
        os.path.join(args.outdir, "fig1_original_scale"),
        figure_title=fr"Figure 1 original scale: $k={args.fig1_k}$, $\Delta_2={args.delta}$, {args.ntraj} trajectories",
        transformed=False,
    )

    fig2_results = run_for_k(
        k=args.fig2_k,
        delta=args.delta,
        ntraj=args.ntraj,
        n_grid=args.n_grid,
        seed=args.seed + 999999,
        workers=args.workers,
        method=args.method,
    )
    save_npz(args.outdir, "fig2_k3_results.npz", fig2_results)

    plot_figure(
        fig2_results,
        os.path.join(args.outdir, "fig2_logit_scale"),
        figure_title=fr"Figure 2 style: $k={args.fig2_k}$, $\Delta_2={args.delta}$, {args.ntraj} trajectories",
        transformed=True,
    )
    plot_figure(
        fig2_results,
        os.path.join(args.outdir, "fig2_original_scale"),
        figure_title=fr"Figure 2 original scale: $k={args.fig2_k}$, $\Delta_2={args.delta}$, {args.ntraj} trajectories",
        transformed=False,
    )


if __name__ == "__main__":
    main()
