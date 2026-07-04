#!/usr/bin/env python3

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt


SCHEDULE_INV_SQRT = 0
SCHEDULE_SQRT_LOG_OVER_T = 1
SCHEDULE_LOG_OVER_T = 2
SCHEDULE_INV_T = 3


def schedule_configs():
    return [
        (
            "inv_sqrt_1pt",
            SCHEDULE_INV_SQRT,
            r"$\eta_t=1/\sqrt{1+t}$",
        ),
        (
            "sqrt_log_over_t",
            SCHEDULE_SQRT_LOG_OVER_T,
            r"$\eta_t=\sqrt{\log(e+t)/(1+t)}$",
        ),
        (
            "log_over_t",
            SCHEDULE_LOG_OVER_T,
            r"$\eta_t=\log(e+t)/(1+t)$",
        ),
        (
            "inv_t",
            SCHEDULE_INV_T,
            r"$\eta_t=1/(1+t)$",
        ),
    ]


def eta_scalar(t, schedule_id):
    """
    Here t is zero-based: t = 0 for the first update.

    The log schedules use log(e+t), so they are well-defined and nonzero
    at t = 0. This is the stable version of log(t)/t-style schedules.
    """
    x = 1.0 + t

    if schedule_id == SCHEDULE_INV_SQRT:
        return 1.0 / math.sqrt(x)

    if schedule_id == SCHEDULE_SQRT_LOG_OVER_T:
        return math.sqrt(math.log(math.e + t) / x)

    if schedule_id == SCHEDULE_LOG_OVER_T:
        return math.log(math.e + t) / x

    if schedule_id == SCHEDULE_INV_T:
        return 1.0 / x

    raise ValueError(f"Unknown schedule_id={schedule_id}")


def eta_array(t, schedule_id):
    t = np.asarray(t, dtype=np.float64)
    x = 1.0 + t

    if schedule_id == SCHEDULE_INV_SQRT:
        return 1.0 / np.sqrt(x)

    if schedule_id == SCHEDULE_SQRT_LOG_OVER_T:
        return np.sqrt(np.log(np.e + t) / x)

    if schedule_id == SCHEDULE_LOG_OVER_T:
        return np.log(np.e + t) / x

    if schedule_id == SCHEDULE_INV_T:
        return 1.0 / x

    raise ValueError(f"Unknown schedule_id={schedule_id}")


def eta_sums(t, h, schedule_id):
    """
    Approximate
        sum_{u=t}^{t+h-1} eta_u
        sum_{u=t}^{t+h-1} eta_u^2

    For small h, compute exactly. For large h, use 3-point Gauss-Legendre
    quadrature over the block. The block size is controlled elsewhere so
    the approximation remains stable.
    """
    if h <= 10000:
        u = np.arange(t, t + h, dtype=np.float64)
        e = eta_array(u, schedule_id)
        return float(e.sum()), float(np.square(e).sum())

    # 3-point Gauss-Legendre nodes on [0, 1].
    nodes = np.array(
        [0.1127016653792583, 0.5, 0.8872983346207417],
        dtype=np.float64,
    )
    weights = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0], dtype=np.float64)

    x = t + h * nodes
    e = eta_array(x, schedule_id)

    sum_eta = h * float(np.dot(weights, e))
    sum_eta2 = h * float(np.dot(weights, np.square(e)))

    return sum_eta, sum_eta2


def make_time_grid(T, n_grid):
    grid = np.unique(np.round(np.logspace(0, math.log10(T), n_grid)).astype(np.int64))
    grid = grid[grid >= 1]

    if len(grid) == 0 or grid[0] != 1:
        grid = np.concatenate([np.array([1], dtype=np.int64), grid])

    if grid[-1] != T:
        grid = np.concatenate([grid, np.array([T], dtype=np.int64)])

    return np.unique(grid)


def softmax_three(theta1, theta2, thetar, m):
    mx = np.maximum(np.maximum(theta1, theta2), thetar)

    e1 = np.exp(theta1 - mx)
    e2 = np.exp(theta2 - mx)
    er = np.exp(thetar - mx)

    z = e1 + e2 + m * er

    p1 = e1 / z
    p2 = e2 / z
    pr = m * er / z

    return p1, p2, pr


def exact_update_block(theta1, theta2, thetar, m, delta, schedule_id, t, h, rng):
    """
    Exact vectorized simulation of h discrete rounds for all trajectories.
    """
    n = theta1.shape[0]

    for _ in range(h):
        p1, p2, pr = softmax_three(theta1, theta2, thetar, m)

        eta = eta_scalar(t, schedule_id)

        u = rng.random(n)

        choose1 = u < p1
        choose2 = (u >= p1) & (u < p1 + p2)

        y = np.zeros(n, dtype=np.float64)

        n1 = int(choose1.sum())
        n2 = int(choose2.sum())

        if n1 > 0:
            y[choose1] = 1.0 + rng.standard_normal(n1)

        if n2 > 0:
            y[choose2] = (1.0 - delta) + rng.standard_normal(n2)

        ind1 = choose1.astype(np.float64)
        ind2 = choose2.astype(np.float64)

        theta1 += eta * (ind1 - p1) * y
        theta2 += eta * (ind2 - p2) * y
        thetar += eta * (-pr / m) * y

        t += 1

    return theta1, theta2, thetar, t


def choose_block_h(t, remaining, schedule_id, max_chunk, max_sum_eta, max_rel_step):
    eta_now = eta_scalar(t, schedule_id)

    h_eta = max(1, int(max_sum_eta / max(eta_now, 1e-300)))
    h_rel = max(1, int(max_rel_step * max(1.0, 1.0 + t)))

    h = min(int(remaining), int(max_chunk), int(h_eta), int(h_rel))
    return max(1, h)


def gaussian_block_update(theta1, theta2, thetar, m, delta, schedule_id, t, h, rng):
    """
    Frozen-policy Gaussian block approximation.

    Over the block, it matches the first two moments of the discrete-time PG
    update using the policy at the beginning of the block.
    """
    n = theta1.shape[0]

    p1, p2, pr = softmax_three(theta1, theta2, thetar, m)

    mu1 = 1.0
    mu2 = 1.0 - delta

    ey1sq = 1.0 + mu1 * mu1
    ey2sq = 1.0 + mu2 * mu2

    g1 = np.empty((n, 3), dtype=np.float64)
    g2 = np.empty((n, 3), dtype=np.float64)

    g1[:, 0] = 1.0 - p1
    g1[:, 1] = -p2
    g1[:, 2] = -pr / m

    g2[:, 0] = -p1
    g2[:, 1] = 1.0 - p2
    g2[:, 2] = -pr / m

    mean = p1[:, None] * mu1 * g1 + p2[:, None] * mu2 * g2

    second = (
        p1[:, None, None] * ey1sq * g1[:, :, None] * g1[:, None, :]
        + p2[:, None, None] * ey2sq * g2[:, :, None] * g2[:, None, :]
    )

    cov = second - mean[:, :, None] * mean[:, None, :]
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))

    sum_eta, sum_eta2 = eta_sums(t, h, schedule_id)

    block_mean = sum_eta * mean
    block_cov = sum_eta2 * cov

    # Robust PSD square root by eigenvalue clipping.
    evals, evecs = np.linalg.eigh(block_cov)
    evals = np.clip(evals, 0.0, None)

    z = rng.standard_normal((n, 3))
    scaled_z = np.sqrt(evals) * z

    inc = block_mean + np.einsum("nij,nj->ni", evecs, scaled_z)

    theta1 += inc[:, 0]
    theta2 += inc[:, 1]
    thetar += inc[:, 2]

    return theta1, theta2, thetar, t + h


def simulate_paths(
    k,
    delta,
    schedule_id,
    t_grid,
    ntraj,
    seed,
    exact_until,
    max_chunk,
    max_sum_eta,
    max_rel_step,
):
    """
    Simulate all trajectories for one schedule.

    The bandit instance is:
        mu_1 = 1,
        mu_2 = 1 - Delta_2,
        mu_a = 0 for a >= 3,
        sigma_1 = sigma_2 = 1,
        sigma_a = 0 for a >= 3.

    Since arms a >= 3 are symmetric and have zero reward noise, their theta
    coordinates remain identical. The k-dimensional dynamics reduce to:
        theta_1, theta_2, theta_rest.
    """
    rng = np.random.default_rng(seed)

    m = k - 2
    if m <= 0:
        raise ValueError("This script assumes k >= 3.")

    theta1 = np.zeros(ntraj, dtype=np.float64)
    theta2 = np.zeros(ntraj, dtype=np.float64)
    thetar = np.zeros(ntraj, dtype=np.float64)

    paths = np.empty((ntraj, len(t_grid)), dtype=np.float64)

    t = 0

    for j, target in enumerate(t_grid):
        target = int(target)

        while t < target:
            remaining = target - t

            if t < exact_until:
                h = min(remaining, exact_until - t)
                theta1, theta2, thetar, t = exact_update_block(
                    theta1=theta1,
                    theta2=theta2,
                    thetar=thetar,
                    m=m,
                    delta=delta,
                    schedule_id=schedule_id,
                    t=t,
                    h=h,
                    rng=rng,
                )
            else:
                h = choose_block_h(
                    t=t,
                    remaining=remaining,
                    schedule_id=schedule_id,
                    max_chunk=max_chunk,
                    max_sum_eta=max_sum_eta,
                    max_rel_step=max_rel_step,
                )

                theta1, theta2, thetar, t = gaussian_block_update(
                    theta1=theta1,
                    theta2=theta2,
                    thetar=thetar,
                    m=m,
                    delta=delta,
                    schedule_id=schedule_id,
                    t=t,
                    h=h,
                    rng=rng,
                )

        p1, _, _ = softmax_three(theta1, theta2, thetar, m)
        paths[:, j] = p1

    return paths


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


def setup_y_axis(ax, transformed, show_y_ticks):
    if transformed:
        ax.set_yscale("function", functions=(_logit, _inv_logit))

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

        if show_y_ticks:
            ax.set_yticklabels([prob_tick_label(p) for p in ticks])
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    else:
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))

        if not show_y_ticks:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def draw_panel(ax, result, transformed, show_y_ticks, show_legend=False):
    t_grid = result["t_grid"]
    display_paths = result["display_paths"]
    average_paths = result["average_paths"]

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
    ax.set_title(result["title"], fontsize=11)
    ax.set_xlabel("Time")

    setup_y_axis(ax, transformed=transformed, show_y_ticks=show_y_ticks)

    if show_legend:
        ax.legend(loc="upper left", fontsize=8, frameon=True)


def plot_single(result, outbase, main_title, transformed):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.9))

    draw_panel(
        ax=ax,
        result=result,
        transformed=transformed,
        show_y_ticks=True,
        show_legend=True,
    )

    ax.set_ylabel("Optimal Action Probability")
    fig.suptitle(main_title, fontsize=13, y=1.02)
    fig.tight_layout()

    fig.savefig(outbase + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved", outbase + ".png")
    print("Saved", outbase + ".pdf")


def plot_grid(results, layout, outbase, main_title, transformed):
    if layout == "1x4":
        fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), sharey=True)
        axes = np.asarray(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            draw_panel(
                ax=ax,
                result=results[idx],
                transformed=transformed,
                show_y_ticks=(idx == 0),
                show_legend=(idx == 0),
            )

    elif layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.7), sharey=True)
        axes_flat = np.asarray(axes).reshape(-1)

        for idx, ax in enumerate(axes_flat):
            row, col = divmod(idx, 2)

            draw_panel(
                ax=ax,
                result=results[idx],
                transformed=transformed,
                show_y_ticks=(col == 0),
                show_legend=(idx == 0),
            )

    else:
        raise ValueError(f"Unknown layout={layout}")

    fig.suptitle(main_title, fontsize=13, y=1.02)
    fig.supylabel("Optimal Action Probability", fontsize=11)

    fig.tight_layout()

    fig.savefig(outbase + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved", outbase + ".png")
    print("Saved", outbase + ".pdf")


def save_npz(outdir, filename, results):
    payload = {}

    for i, res in enumerate(results):
        payload[f"name_{i}"] = res["name"]
        payload[f"schedule_id_{i}"] = res["schedule_id"]
        payload[f"title_{i}"] = res["title"]
        payload[f"t_grid_{i}"] = res["t_grid"]
        payload[f"display_paths_{i}"] = res["display_paths"]
        payload[f"average_paths_{i}"] = res["average_paths"]

    path = os.path.join(outdir, filename)
    np.savez_compressed(path, **payload)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="results_pg_stepsizes_k40")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--horizon", type=int, default=10**9)

    parser.add_argument("--ntraj-display", type=int, default=40)
    parser.add_argument("--ntraj-average", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=520)

    parser.add_argument("--exact-until", type=int, default=20000)
    parser.add_argument("--max-chunk", type=int, default=25000000)
    parser.add_argument("--max-sum-eta", type=float, default=5.0)
    parser.add_argument("--max-rel-step", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t_grid = make_time_grid(args.horizon, args.n_grid)

    results = []

    for i, (name, schedule_id, title) in enumerate(schedule_configs()):
        print("=" * 80)
        print(f"Simulating schedule: {title}")
        print(f"K={args.k}, Delta_2={args.delta}, horizon={args.horizon:,}")

        display_paths = simulate_paths(
            k=args.k,
            delta=args.delta,
            schedule_id=schedule_id,
            t_grid=t_grid,
            ntraj=args.ntraj_display,
            seed=args.seed + 100000 * i,
            exact_until=args.exact_until,
            max_chunk=args.max_chunk,
            max_sum_eta=args.max_sum_eta,
            max_rel_step=args.max_rel_step,
        )

        average_paths = simulate_paths(
            k=args.k,
            delta=args.delta,
            schedule_id=schedule_id,
            t_grid=t_grid,
            ntraj=args.ntraj_average,
            seed=args.seed + 100000 * i + 50000000,
            exact_until=args.exact_until,
            max_chunk=args.max_chunk,
            max_sum_eta=args.max_sum_eta,
            max_rel_step=args.max_rel_step,
        )

        results.append(
            {
                "name": name,
                "schedule_id": schedule_id,
                "title": title,
                "t_grid": t_grid,
                "display_paths": display_paths,
                "average_paths": average_paths,
            }
        )

    save_npz(args.outdir, "k40_stepsizes_results.npz", results)

    main_title = fr"Policy gradient on lower-bound instance: $K={args.k}$, $\Delta_2={args.delta}$"

    # First requested plot: only eta_t = 1/sqrt(1+t), K = 40.
    inv_sqrt_result = results[0]

    plot_single(
        result=inv_sqrt_result,
        outbase=os.path.join(args.outdir, "k40_inv_sqrt_only_logit_scale"),
        main_title=main_title,
        transformed=True,
    )

    plot_single(
        result=inv_sqrt_result,
        outbase=os.path.join(args.outdir, "k40_inv_sqrt_only_original_scale"),
        main_title=main_title,
        transformed=False,
    )

    # Second requested plot: four schedules, both 1x4 and 2x2 grids.
    plot_grid(
        results=results,
        layout="1x4",
        outbase=os.path.join(args.outdir, "k40_stepsizes_1x4_logit_scale"),
        main_title=main_title,
        transformed=True,
    )

    plot_grid(
        results=results,
        layout="2x2",
        outbase=os.path.join(args.outdir, "k40_stepsizes_2x2_logit_scale"),
        main_title=main_title,
        transformed=True,
    )

    plot_grid(
        results=results,
        layout="1x4",
        outbase=os.path.join(args.outdir, "k40_stepsizes_1x4_original_scale"),
        main_title=main_title,
        transformed=False,
    )

    plot_grid(
        results=results,
        layout="2x2",
        outbase=os.path.join(args.outdir, "k40_stepsizes_2x2_original_scale"),
        main_title=main_title,
        transformed=False,
    )


if __name__ == "__main__":
    main()
