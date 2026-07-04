#!/usr/bin/env python3

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


SCHEDULE_DELTA_SQ = 0
SCHEDULE_INV_SQRT = 1


def schedule_configs(delta: float):
    return [
        ("eta_delta_squared", SCHEDULE_DELTA_SQ, r"$\eta=\Delta^2$"),
        ("eta_inv_sqrt_t", SCHEDULE_INV_SQRT, r"$\eta_t=1/\sqrt{t}$"),
    ]


def horizon_for_schedule(schedule_id: int, k: int, base_horizon: int, k3_mult: float) -> int:
    T = int(base_horizon)
    if k == 3:
        T = int(math.ceil(k3_mult * T))
    return max(1, T)


def make_time_grid(T: int, n_grid: int) -> np.ndarray:
    grid = np.unique(np.round(np.logspace(0, math.log10(T), n_grid)).astype(np.int64))
    grid = grid[grid >= 1]

    if len(grid) == 0 or grid[0] != 1:
        grid = np.concatenate([np.array([1], dtype=np.int64), grid])

    if grid[-1] != T:
        grid = np.concatenate([grid, np.array([T], dtype=np.int64)])

    return np.unique(grid)


@njit(cache=True)
def _eta_at_round(round_index, schedule_id, delta):
    if schedule_id == SCHEDULE_DELTA_SQ:
        return delta * delta

    return 1.0 / math.sqrt(round_index)


@njit(cache=True)
def _eta_sums(t0, h, schedule_id, delta):
    """
    Sums over rounds r = t0+1, ..., t0+h.
    """
    if schedule_id == SCHEDULE_DELTA_SQ:
        eta = delta * delta
        return eta * h, eta * eta * h

    a = t0 + 1
    b = t0 + h

    if h <= 10000:
        s1 = 0.0
        s2 = 0.0
        for r in range(a, b + 1):
            eta = 1.0 / math.sqrt(r)
            s1 += eta
            s2 += eta * eta
        return s1, s2

    s1 = 2.0 * (math.sqrt(b + 0.5) - math.sqrt(a - 0.5))
    s2 = math.log((b + 0.5) / (a - 0.5))
    return s1, s2


@njit(cache=True)
def _softmax_three(theta1, theta2, thetar, m):
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
    if p >= 1.0:
        return 1

    if p <= 0.0:
        return 9223372036854775807

    u = np.random.random()
    if u <= 1e-300:
        u = 1e-300

    return int(math.ceil(math.log(u) / math.log1p(-p)))


@njit(cache=True)
def _chol3_psd(a00, a01, a02, a11, a12, a22):
    jitter = 1e-18

    for _ in range(12):
        b00 = a00 + jitter
        b11 = a11 + jitter
        b22 = a22 + jitter

        if b00 <= 0.0:
            jitter *= 100.0
            continue

        l00 = math.sqrt(b00)
        l10 = a01 / l00
        l20 = a02 / l00

        d11 = b11 - l10 * l10
        if d11 <= 0.0:
            jitter *= 100.0
            continue

        l11 = math.sqrt(d11)
        l21 = (a12 - l20 * l10) / l11

        d22 = b22 - l20 * l20 - l21 * l21
        if d22 <= 0.0:
            jitter *= 100.0
            continue

        l22 = math.sqrt(d22)

        return l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22

    l00 = math.sqrt(max(a00, 0.0) + jitter)
    l11 = math.sqrt(max(a11, 0.0) + jitter)
    l22 = math.sqrt(max(a22, 0.0) + jitter)

    return l00, 0.0, 0.0, 0.0, l11, 0.0, 0.0, 0.0, l22


@njit(cache=True)
def _block_update(theta1, theta2, thetar, m, delta, schedule_id, t, h):
    """
    Gaussian block approximation with frozen policy.

    This matches the first two moments of the discrete-time PG update over
    a block. It is much faster than exact simulation for horizons like 1e9.
    """
    p1, p2, pr = _softmax_three(theta1, theta2, thetar, m)

    mu1 = 1.0
    mu2 = 1.0 - delta

    ey1sq = 1.0 + mu1 * mu1
    ey2sq = 1.0 + mu2 * mu2

    g10 = 1.0 - p1
    g11 = -p2
    g12 = -pr / m

    g20 = -p1
    g21 = 1.0 - p2
    g22 = -pr / m

    m0 = p1 * mu1 * g10 + p2 * mu2 * g20
    m1 = p1 * mu1 * g11 + p2 * mu2 * g21
    m2 = p1 * mu1 * g12 + p2 * mu2 * g22

    s00 = p1 * ey1sq * g10 * g10 + p2 * ey2sq * g20 * g20
    s01 = p1 * ey1sq * g10 * g11 + p2 * ey2sq * g20 * g21
    s02 = p1 * ey1sq * g10 * g12 + p2 * ey2sq * g20 * g22
    s11 = p1 * ey1sq * g11 * g11 + p2 * ey2sq * g21 * g21
    s12 = p1 * ey1sq * g11 * g12 + p2 * ey2sq * g21 * g22
    s22 = p1 * ey1sq * g12 * g12 + p2 * ey2sq * g22 * g22

    c00 = s00 - m0 * m0
    c01 = s01 - m0 * m1
    c02 = s02 - m0 * m2
    c11 = s11 - m1 * m1
    c12 = s12 - m1 * m2
    c22 = s22 - m2 * m2

    sum_eta, sum_eta2 = _eta_sums(t, h, schedule_id, delta)

    mean0 = sum_eta * m0
    mean1 = sum_eta * m1
    mean2 = sum_eta * m2

    a00 = sum_eta2 * c00
    a01 = sum_eta2 * c01
    a02 = sum_eta2 * c02
    a11 = sum_eta2 * c11
    a12 = sum_eta2 * c12
    a22 = sum_eta2 * c22

    l00, l01, l02, l10, l11, l12, l20, l21, l22 = _chol3_psd(
        a00, a01, a02, a11, a12, a22
    )

    z0 = np.random.normal()
    z1 = np.random.normal()
    z2 = np.random.normal()

    inc0 = mean0 + l00 * z0
    inc1 = mean1 + l10 * z0 + l11 * z1
    inc2 = mean2 + l20 * z0 + l21 * z1 + l22 * z2

    return theta1 + inc0, theta2 + inc1, thetar + inc2


@njit(cache=True)
def _choose_block_h(t, remaining, schedule_id, delta, max_chunk, max_sum_eta):
    eta_now = _eta_at_round(t + 1, schedule_id, delta)

    h_eta = int(max_sum_eta / eta_now)
    if h_eta < 1:
        h_eta = 1

    h = remaining

    if h > max_chunk:
        h = max_chunk

    if h > h_eta:
        h = h_eta

    if h < 1:
        h = 1

    return h


@njit(cache=True)
def simulate_one_hybrid(
    k,
    delta,
    schedule_id,
    t_grid,
    seed,
    exact_until,
    max_chunk,
    max_sum_eta,
):
    """
    Hybrid simulator.

    Up to exact_until:
        exact discrete simulation with geometric skipping of zero-reward rest-arm samples.

    After exact_until:
        block Gaussian approximation matching first two moments of the discrete update.

    The rest arms are symmetry-reduced because actions a >= 3 have identical theta,
    identical mean zero, and zero reward noise.
    """
    np.random.seed(seed)

    m = k - 2
    out = np.empty(len(t_grid), dtype=np.float64)

    theta1 = 0.0
    theta2 = 0.0
    thetar = 0.0

    t = 0
    idx = 0
    T = t_grid[-1]

    while idx < len(t_grid):
        p1, p2, pr = _softmax_three(theta1, theta2, thetar, m)

        while idx < len(t_grid) and t_grid[idx] <= t:
            out[idx] = p1
            idx += 1

        if idx >= len(t_grid):
            break

        next_checkpoint = t_grid[idx]

        if t < exact_until:
            p12 = p1 + p2
            wait = _sample_geometric_wait(p12)
            next_t = t + wait

            limit_t = exact_until

            if next_checkpoint < limit_t:
                limit_t = next_checkpoint

            if T < limit_t:
                limit_t = T

            if next_t > limit_t:
                t = limit_t
                continue

            t = next_t
            eta = _eta_at_round(t, schedule_id, delta)

            u = np.random.random()

            if u < p1 / p12:
                y = 1.0 + np.random.normal()
                theta1 += eta * (1.0 - p1) * y
                theta2 += eta * (-p2) * y
                thetar += eta * (-pr / m) * y
            else:
                y = (1.0 - delta) + np.random.normal()
                theta1 += eta * (-p1) * y
                theta2 += eta * (1.0 - p2) * y
                thetar += eta * (-pr / m) * y

            continue

        remaining = next_checkpoint - t
        h = _choose_block_h(t, remaining, schedule_id, delta, max_chunk, max_sum_eta)

        theta1, theta2, thetar = _block_update(
            theta1,
            theta2,
            thetar,
            m,
            delta,
            schedule_id,
            t,
            h,
        )

        t += h

    return out


def worker(args):
    return simulate_one_hybrid(*args)


def simulate_paths(
    k,
    delta,
    schedule_id,
    t_grid,
    ntraj,
    seed,
    workers,
    exact_until,
    max_chunk,
    max_sum_eta,
):
    jobs = []

    for r in range(ntraj):
        s = int(seed + 1000003 * r + 9176 * k + 104729 * schedule_id)
        jobs.append(
            (
                k,
                delta,
                schedule_id,
                t_grid,
                s,
                exact_until,
                max_chunk,
                max_sum_eta,
            )
        )

    if workers <= 1:
        paths = [worker(j) for j in jobs]
    else:
        paths = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, j) for j in jobs]
            for fut in as_completed(futs):
                paths.append(fut.result())

    return np.asarray(paths, dtype=np.float64)


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
        if abs(p - 1e-1) < 1e-15:
            return r"$10^{-1}$"
        if abs(p - 1e-2) < 1e-15:
            return r"$10^{-2}$"
        if abs(p - 1e-3) < 1e-15:
            return r"$10^{-3}$"
        if abs(p - 1e-4) < 1e-15:
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


def plot_results(results, outbase, title, transformed):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=False)

    for ax_idx, (ax, res) in enumerate(zip(axes, results)):
        label, schedule_id, sched_title, k, t_grid, display_paths, average_paths = res

        for r in range(display_paths.shape[0]):
            ax.plot(
                t_grid,
                display_paths[r],
                color="0.45",
                alpha=0.18,
                linewidth=0.9,
                zorder=1,
            )

        avg = average_paths.mean(axis=0)

        ax.plot(
            t_grid,
            avg,
            color="tab:red",
            linewidth=1.9,
            label="Average",
            zorder=5,
        )

        ax.set_xscale("log")
        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
        ax.set_title(sched_title, fontsize=11)

        if transformed:
            ax.set_yscale("function", functions=(_logit, _inv_logit))

            if k == 3:
                ticks = [1e-1, 0.5, 1 - 1e-1, 1 - 1e-2]
                ax.set_ylim(7e-2, 1 - 5e-3)
            else:
                ticks = [
                    1e-4,
                    1e-3,
                    1e-2,
                    1e-1,
                    0.5,
                    1 - 1e-1,
                    1 - 1e-2,
                    1 - 1e-3,
                    1 - 1e-4,
                ]
                ax.set_ylim(1e-4, 1 - 1e-4)

            ax.set_yticks(ticks)
            ax.set_yticklabels([prob_tick_label(p) for p in ticks])
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks(np.linspace(0, 1, 6))

        if ax_idx == 0:
            ax.set_ylabel("Optimal Action Probability")
            ax.legend(loc="upper left", fontsize=8, frameon=True)

        ax.set_xlabel("Time")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    fig.savefig(outbase + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")

    plt.close(fig)

    print("Saved", outbase + ".png")
    print("Saved", outbase + ".pdf")


def save_npz(outdir, name, results):
    payload = {}

    for i, res in enumerate(results):
        label, schedule_id, sched_title, k, t_grid, display_paths, average_paths = res

        payload[f"label_{i}"] = label
        payload[f"schedule_id_{i}"] = schedule_id
        payload[f"title_{i}"] = sched_title
        payload[f"k_{i}"] = k
        payload[f"t_grid_{i}"] = t_grid
        payload[f"display_paths_{i}"] = display_paths
        payload[f"average_paths_{i}"] = average_paths

    path = os.path.join(outdir, name)
    np.savez_compressed(path, **payload)

    print("Saved", path)


def run_for_k(k, args, seed_offset):
    out = []

    for i, (label, schedule_id, sched_title) in enumerate(schedule_configs(args.delta)):
        T = horizon_for_schedule(
            schedule_id=schedule_id,
            k=k,
            base_horizon=args.horizon,
            k3_mult=args.k3_horizon_mult,
        )

        t_grid = make_time_grid(T, args.n_grid)

        print(f"Simulating k={k}, {label}, T={T:,}")

        display_paths = simulate_paths(
            k=k,
            delta=args.delta,
            schedule_id=schedule_id,
            t_grid=t_grid,
            ntraj=args.ntraj_display,
            seed=args.seed + seed_offset + 100000 * i,
            workers=args.workers,
            exact_until=args.exact_until,
            max_chunk=args.max_chunk,
            max_sum_eta=args.max_sum_eta,
        )

        average_paths = simulate_paths(
            k=k,
            delta=args.delta,
            schedule_id=schedule_id,
            t_grid=t_grid,
            ntraj=args.ntraj_average,
            seed=args.seed + seed_offset + 100000 * i + 50000000,
            workers=args.workers,
            exact_until=args.exact_until,
            max_chunk=args.max_chunk,
            max_sum_eta=args.max_sum_eta,
        )

        out.append(
            (
                label,
                schedule_id,
                sched_title,
                k,
                t_grid,
                display_paths,
                average_paths,
            )
        )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="results_pg_delta2_invsqrt")
    parser.add_argument("--delta", type=float, default=0.002)

    parser.add_argument("--fig1-k", type=int, default=40)
    parser.add_argument("--fig2-k", type=int, default=3)

    parser.add_argument("--horizon", type=int, default=10**9)
    parser.add_argument("--k3-horizon-mult", type=float, default=3.0)

    parser.add_argument("--ntraj-display", type=int, default=40)
    parser.add_argument("--ntraj-average", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=520)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--exact-until", type=int, default=200000)
    parser.add_argument("--max-chunk", type=int, default=25000000)
    parser.add_argument("--max-sum-eta", type=float, default=100.0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Numba available:", NUMBA_AVAILABLE)

    if NUMBA_AVAILABLE:
        warm_grid = np.array([1, 2], dtype=np.int64)
        _ = simulate_one_hybrid(
            3,
            args.delta,
            SCHEDULE_DELTA_SQ,
            warm_grid,
            args.seed,
            2,
            10,
            1.0,
        )

    fig1_results = run_for_k(args.fig1_k, args, seed_offset=0)

    save_npz(
        args.outdir,
        f"fig1_k{args.fig1_k}_delta2_invsqrt_results.npz",
        fig1_results,
    )

    plot_results(
        fig1_results,
        os.path.join(args.outdir, f"fig1_k{args.fig1_k}_logit_scale"),
        title=(
            fr"$k={args.fig1_k}$, $\Delta_2={args.delta}$, "
            fr"gray={args.ntraj_display}, red average={args.ntraj_average}"
        ),
        transformed=True,
    )

    plot_results(
        fig1_results,
        os.path.join(args.outdir, f"fig1_k{args.fig1_k}_original_scale"),
        title=fr"Original scale: $k={args.fig1_k}$, $\Delta_2={args.delta}$",
        transformed=False,
    )

    fig2_results = run_for_k(args.fig2_k, args, seed_offset=999999)

    save_npz(
        args.outdir,
        f"fig2_k{args.fig2_k}_delta2_invsqrt_results.npz",
        fig2_results,
    )

    plot_results(
        fig2_results,
        os.path.join(args.outdir, f"fig2_k{args.fig2_k}_logit_scale"),
        title=(
            fr"$k={args.fig2_k}$, $\Delta_2={args.delta}$, "
            fr"gray={args.ntraj_display}, red average={args.ntraj_average}"
        ),
        transformed=True,
    )

    plot_results(
        fig2_results,
        os.path.join(args.outdir, f"fig2_k{args.fig2_k}_original_scale"),
        title=fr"Original scale: $k={args.fig2_k}$, $\Delta_2={args.delta}$",
        transformed=False,
    )


if __name__ == "__main__":
    main()
