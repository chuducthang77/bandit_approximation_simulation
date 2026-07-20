
#!/usr/bin/env python3

"""

1x4 sample-path plots for pi_t(1).



Bandit instance:

    mu    = (1, 1 - Delta, 0, ..., 0)

    sigma = (1, 1,         0, ..., 0)



Stepsize:

    eta_t = 1 / sqrt(t + 1)



Plots:

    One separate PDF for each K in {3, 4, 5}.

    Each PDF is a 1x4 plot over Delta in {0.2, 0.05, 0.02, 0.002}.



Horizons:

    K = 3:

        Delta = 0.2   -> 10^11

        Delta = 0.05  -> 10^11

        Delta = 0.02  -> 10^11

        Delta = 0.002 -> 10^15



    K = 4 and K = 5:

        all four Delta values -> 10^20

"""



from __future__ import annotations



import argparse

import math

from pathlib import Path

from typing import Sequence, Tuple



import numpy as np



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt





DELTAS = [0.2, 0.05, 0.02, 0.002]



HORIZONS_BY_K = {

    3: [10**11, 10**11, 10**11, 10**15],

    4: [10**20, 10**20, 10**20, 10**20],

    5: [10**20, 10**20, 10**20, 10**20],

}



DEFAULT_OUTDIR = "pi1_four_gaps_results"

DEFAULT_NUM_PATHS = 40

DEFAULT_NUM_CHECKPOINTS = 500

DEFAULT_SEED = 20260310





def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)





def make_log_checkpoints(horizon: int, num_checkpoints: int) -> list[int]:

    """

    Return log-spaced checkpoints as Python ints.



    Python ints are used because horizons such as 10^20 exceed int64.

    """

    horizon = int(horizon)



    if horizon < 1:

        raise ValueError("horizon must be >= 1")



    if num_checkpoints <= 1:

        return [horizon]



    log_horizon = math.log10(horizon)



    raw = []

    for x in np.linspace(0.0, log_horizon, int(num_checkpoints)):

        value = int(round(10.0 ** float(x)))

        value = max(1, min(value, horizon))

        raw.append(value)



    raw.append(horizon)



    checkpoints = sorted(set(raw))



    return checkpoints





def style_axis(ax) -> None:

    # Major grid only, so the plot is not too dense.

    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)

    ax.grid(False, which="minor")





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

        each identical zero-mean arm 3,...,K

    """

    max_score = np.maximum(

        np.maximum(optimal_arm_score, second_arm_score),

        other_arm_score,

    )



    weight_optimal = np.exp(optimal_arm_score - max_score)

    weight_second = np.exp(second_arm_score - max_score)

    weight_other = np.exp(other_arm_score - max_score)



    total_weight = (

        weight_optimal

        + weight_second

        + num_other_arms * weight_other

    )



    optimal_prob = weight_optimal / total_weight

    second_prob = weight_second / total_weight

    other_prob_per_arm = weight_other / total_weight



    return optimal_prob, second_prob, other_prob_per_arm





def eta_value(step_index_zero_based) -> np.ndarray:

    """

    eta_t = 1 / sqrt(t + 1), where t is zero-based.

    """

    step = np.asarray(step_index_zero_based, dtype=np.float64)

    return 1.0 / np.sqrt(step + 1.0)





def harmonic_number(n: int) -> float:

    """

    Approximation to H_n = sum_{j=1}^n 1/j.



    Used because eta_t^2 = 1/(t+1).

    """

    n = int(n)



    if n <= 0:

        return 0.0



    if n < 200_000:

        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))



    gamma = 0.5772156649015328606

    x = float(n)



    return math.log(x) + gamma + 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)





def eta_sums_for_block(start_step: int, block_size: int) -> Tuple[float, float]:

    """

    Return:

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

        math.sqrt(start_step + block_size + 1.0)

        - math.sqrt(start_step + 1.0)

    )



    # Exact identity:

    # sum_{t=a}^{a+B-1} 1/(t+1) = H_{a+B} - H_a.

    sum_eta_squared = (

        harmonic_number(start_step + block_size)

        - harmonic_number(start_step)

    )



    return float(sum_eta), float(sum_eta_squared)





def choose_block_size(

    *,

    current_step: int,

    remaining_steps: int,

    delta: float,

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



    Smaller max_mean_change and max_noise_change are more conservative but slower.

    """

    eta_now = float(eta_value(current_step))



    if eta_now <= 0.0:

        return int(min(remaining_steps, max_block_size))



    delta = float(delta)



    other_total_prob = num_other_arms * other_prob_per_arm

    instantaneous_regret = other_total_prob + delta * second_prob



    mean_gradient_optimal = optimal_prob * instantaneous_regret

    mean_gradient_second = second_prob * (instantaneous_regret - delta)

    mean_gradient_other = (

        -(mean_gradient_optimal + mean_gradient_second)

        / float(num_other_arms)

    )



    gradient_abs = np.maximum.reduce(

        [

            np.abs(mean_gradient_optimal),

            np.abs(mean_gradient_second),

            np.abs(mean_gradient_other),

        ]

    )



    mean_gradient_scale = float(

        np.quantile(gradient_abs, float(block_quantile))

    )



    second_arm_mean = 1.0 - delta



    second_moment_arm1 = 2.0

    second_moment_arm2 = second_arm_mean * second_arm_mean + 1.0



    variance_optimal = (

        optimal_prob

        * (1.0 - optimal_prob) ** 2

        * second_moment_arm1

        + second_prob

        * optimal_prob ** 2

        * second_moment_arm2

    )



    variance_second = (

        optimal_prob

        * second_prob ** 2

        * second_moment_arm1

        + second_prob

        * (1.0 - second_prob) ** 2

        * second_moment_arm2

    )



    variance_other = (

        optimal_prob

        * other_prob_per_arm ** 2

        * second_moment_arm1

        + second_prob

        * other_prob_per_arm ** 2

        * second_moment_arm2

    )



    variance_scale = float(

        np.quantile(

            np.maximum.reduce(

                [variance_optimal, variance_second, variance_other]

            ),

            float(block_quantile),

        )

    )



    limit_by_mean = math.inf



    if mean_gradient_scale > 1e-300:

        limit_by_mean = (

            float(max_mean_change)

            / (eta_now * mean_gradient_scale)

        )



    limit_by_noise = math.inf



    if variance_scale > 1e-300:

        limit_by_noise = (

            float(max_noise_change)

            / (eta_now * math.sqrt(variance_scale))

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





def run_exact_steps(

    *,

    num_steps: int,

    current_step: int,

    random_generator: np.random.Generator,

    delta: float,

    num_other_arms: int,

    optimal_arm_score: np.ndarray,

    second_arm_score: np.ndarray,

    other_arm_score: np.ndarray,

) -> None:

    """

    Literal round-by-round update for small blocks.

    """

    num_paths = optimal_arm_score.size

    second_arm_mean = 1.0 - float(delta)



    for local_step in range(int(num_steps)):

        step = current_step + local_step

        eta = float(eta_value(step))



        (

            optimal_prob,

            second_prob,

            other_prob_per_arm,

        ) = stable_policy_probabilities(

            optimal_arm_score,

            second_arm_score,

            other_arm_score,

            num_other_arms,

        )



        if eta <= 0.0:

            continue



        uniform_draw = random_generator.random(num_paths)



        choose_arm1 = uniform_draw < optimal_prob

        choose_arm2 = (

            (uniform_draw >= optimal_prob)

            & (uniform_draw < optimal_prob + second_prob)

        )



        reward = np.zeros(num_paths, dtype=np.float64)



        count_arm1 = int(np.sum(choose_arm1))

        count_arm2 = int(np.sum(choose_arm2))



        if count_arm1 > 0:

            reward[choose_arm1] = (

                1.0

                + random_generator.normal(size=count_arm1)

            )



        if count_arm2 > 0:

            reward[choose_arm2] = (

                second_arm_mean

                + random_generator.normal(size=count_arm2)

            )



        optimal_arm_score += eta * reward * (

            choose_arm1.astype(np.float64) - optimal_prob

        )



        second_arm_score += eta * reward * (

            choose_arm2.astype(np.float64) - second_prob

        )



        other_arm_score += eta * reward * (-other_prob_per_arm)





def run_gaussian_approx_block(

    *,

    block_size: int,

    current_step: int,

    random_generator: np.random.Generator,

    delta: float,

    num_other_arms: int,

    optimal_arm_score: np.ndarray,

    second_arm_score: np.ndarray,

    other_arm_score: np.ndarray,

) -> None:

    """

    Blocked Gaussian aggregate approximation.



    The policy is frozen inside the block. The weighted reward sums



        S1 = sum eta_t Y_t 1{A_t = 1}

        S2 = sum eta_t Y_t 1{A_t = 2}



    are approximated by a joint Gaussian with matched mean and covariance.

    """

    (

        optimal_prob,

        second_prob,

        other_prob_per_arm,

    ) = stable_policy_probabilities(

        optimal_arm_score,

        second_arm_score,

        other_arm_score,

        num_other_arms,

    )



    sum_eta, sum_eta_squared = eta_sums_for_block(

        current_step,

        block_size,

    )



    if sum_eta == 0.0 and sum_eta_squared == 0.0:

        return



    arm1_mean = 1.0

    arm2_mean = 1.0 - float(delta)



    second_moment_arm1 = 2.0

    second_moment_arm2 = arm2_mean * arm2_mean + 1.0



    mean_s1 = optimal_prob * arm1_mean * sum_eta

    mean_s2 = second_prob * arm2_mean * sum_eta



    var_s1 = (

        optimal_prob * second_moment_arm1

        - (optimal_prob * arm1_mean) ** 2

    ) * sum_eta_squared



    var_s2 = (

        second_prob * second_moment_arm2

        - (second_prob * arm2_mean) ** 2

    ) * sum_eta_squared



    cov_s1_s2 = (

        -(optimal_prob * arm1_mean)

        * (second_prob * arm2_mean)

        * sum_eta_squared

    )



    var_s1 = np.maximum(var_s1, 0.0)

    var_s2 = np.maximum(var_s2, 0.0)



    z1 = random_generator.normal(size=optimal_prob.size)

    z2 = random_generator.normal(size=optimal_prob.size)



    sqrt_var_s1 = np.sqrt(var_s1)

    weighted_reward_arm1 = mean_s1 + sqrt_var_s1 * z1



    coefficient = np.zeros_like(var_s1)

    safe = var_s1 > 1e-300

    coefficient[safe] = cov_s1_s2[safe] / sqrt_var_s1[safe]



    conditional_var_s2 = var_s2 - coefficient ** 2

    conditional_var_s2 = np.maximum(conditional_var_s2, 0.0)



    weighted_reward_arm2 = (

        mean_s2

        + coefficient * z1

        + np.sqrt(conditional_var_s2) * z2

    )



    optimal_arm_score += (

        (1.0 - optimal_prob) * weighted_reward_arm1

        - optimal_prob * weighted_reward_arm2

    )



    second_arm_score += (

        -second_prob * weighted_reward_arm1

        + (1.0 - second_prob) * weighted_reward_arm2

    )



    other_arm_score += (

        -other_prob_per_arm

        * (weighted_reward_arm1 + weighted_reward_arm2)

    )





def simulate_pi1_paths(

    *,

    num_arms: int,

    delta: float,

    horizon: int,

    checkpoints: Sequence[int],

    num_paths: int,

    seed: int,

    max_mean_change: float,

    max_noise_change: float,

    max_block_size: int,

    block_quantile: float,

    exact_small_block_threshold: int,

) -> np.ndarray:

    """

    Simulate pi_t(1) paths for one (K, Delta) pair.

    """

    num_other_arms = int(num_arms) - 2



    if num_other_arms < 1:

        raise ValueError("num_arms must be at least 3")



    checkpoints_list = sorted(

        set(int(x) for x in checkpoints if 1 <= int(x) <= int(horizon))

    )



    if not checkpoints_list or checkpoints_list[-1] != int(horizon):

        checkpoints_list.append(int(horizon))

        checkpoints_list = sorted(set(checkpoints_list))



    random_generator = np.random.default_rng(int(seed))

    num_paths = int(num_paths)



    optimal_arm_score = np.zeros(num_paths, dtype=np.float64)

    second_arm_score = np.zeros(num_paths, dtype=np.float64)

    other_arm_score = np.zeros(num_paths, dtype=np.float64)



    pi1_paths = np.zeros(

        (len(checkpoints_list), num_paths),

        dtype=np.float64,

    )



    current_step = 0

    record_index = 0

    horizon = int(horizon)



    while current_step < horizon:

        next_checkpoint = int(checkpoints_list[record_index])



        remaining_to_horizon = horizon - current_step

        remaining_to_checkpoint = next_checkpoint - current_step



        if remaining_to_checkpoint <= 0:

            p1, _, _ = stable_policy_probabilities(

                optimal_arm_score,

                second_arm_score,

                other_arm_score,

                num_other_arms,

            )



            pi1_paths[record_index, :] = p1

            record_index += 1



            if record_index >= len(checkpoints_list):

                break



            continue



        (

            optimal_prob,

            second_prob,

            other_prob_per_arm,

        ) = stable_policy_probabilities(

            optimal_arm_score,

            second_arm_score,

            other_arm_score,

            num_other_arms,

        )



        block_size = choose_block_size(

            current_step=current_step,

            remaining_steps=remaining_to_horizon,

            delta=delta,

            optimal_prob=optimal_prob,

            second_prob=second_prob,

            other_prob_per_arm=other_prob_per_arm,

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

                delta=delta,

                num_other_arms=num_other_arms,

                optimal_arm_score=optimal_arm_score,

                second_arm_score=second_arm_score,

                other_arm_score=other_arm_score,

            )

        else:

            run_gaussian_approx_block(

                block_size=block_size,

                current_step=current_step,

                random_generator=random_generator,

                delta=delta,

                num_other_arms=num_other_arms,

                optimal_arm_score=optimal_arm_score,

                second_arm_score=second_arm_score,

                other_arm_score=other_arm_score,

            )



        current_step += int(block_size)



        while (

            record_index < len(checkpoints_list)

            and current_step >= int(checkpoints_list[record_index])

        ):

            p1, _, _ = stable_policy_probabilities(

                optimal_arm_score,

                second_arm_score,

                other_arm_score,

                num_other_arms,

            )



            pi1_paths[record_index, :] = p1

            record_index += 1



            if record_index >= len(checkpoints_list):

                break



    return pi1_paths





def case_data_path(outdir: Path, num_arms: int, delta_index: int) -> Path:

    delta = DELTAS[int(delta_index)]

    return outdir / f"case_K{num_arms}_delta_{delta:g}.npz"





def simulate_one_case(args) -> Path:

    outdir = Path(args.outdir)

    ensure_dir(outdir)



    num_arms = int(args.num_arms)

    delta_index = int(args.delta_index)



    if num_arms not in HORIZONS_BY_K:

        raise ValueError("num_arms must be one of 3, 4, 5")



    if delta_index < 0 or delta_index >= len(DELTAS):

        raise ValueError("delta-index must be 0, 1, 2, or 3")



    delta = float(DELTAS[delta_index])

    horizon = int(HORIZONS_BY_K[num_arms][delta_index])



    checkpoints = make_log_checkpoints(horizon, int(args.num_checkpoints))



    seed = (

        int(args.seed)

        + 1_000_003 * num_arms

        + 100_003 * delta_index

    )



    print(

        f"Running K={num_arms}, Delta={delta:g}, "

        f"horizon={horizon}, paths={args.num_paths}"

    )



    pi1_paths = simulate_pi1_paths(

        num_arms=num_arms,

        delta=delta,

        horizon=horizon,

        checkpoints=checkpoints,

        num_paths=int(args.num_paths),

        seed=seed,

        max_mean_change=float(args.max_mean_change),

        max_noise_change=float(args.max_noise_change),

        max_block_size=int(args.max_block_size),

        block_quantile=float(args.block_quantile),

        exact_small_block_threshold=int(args.exact_small_block_threshold),

    )



    output_path = case_data_path(outdir, num_arms, delta_index)



    np.savez_compressed(

        output_path,

        checkpoints_float=np.array(checkpoints, dtype=np.float64),

        pi1_paths=pi1_paths,

        delta=delta,

        horizon_float=float(horizon),

        horizon_str=str(horizon),

        num_arms=num_arms,

        num_paths=int(args.num_paths),

    )



    print(f"Wrote {output_path}")

    return output_path





def plot_one_k(args, num_arms: int) -> Path:

    outdir = Path(args.outdir)

    ensure_dir(outdir)



    fig, axes = plt.subplots(

        1,

        4,

        figsize=(22.0, 5.0),

        dpi=180,

        sharey=True,

    )



    for delta_index, ax in enumerate(axes):

        data_path = case_data_path(outdir, num_arms, delta_index)



        if not data_path.exists():

            raise FileNotFoundError(

                f"Missing {data_path}. Run the corresponding simulation first."

            )



        data = np.load(data_path)

        checkpoints = data["checkpoints_float"].astype(np.float64)

        pi1_paths = np.clip(data["pi1_paths"], 0.0, 1.0)

        delta = float(data["delta"])

        horizon_float = float(data["horizon_float"])



        average_path = np.mean(pi1_paths, axis=1)



        for path_index in range(pi1_paths.shape[1]):

            ax.plot(

                checkpoints,

                pi1_paths[:, path_index],

                color="0.72",

                alpha=0.45,

                linewidth=0.9,

            )



        ax.plot(

            checkpoints,

            average_path,

            color="red",

            linewidth=2.8,

            label="average",

        )



        ax.set_xscale("log")

        ax.set_xlim(1.0, horizon_float)

        ax.set_ylim(0.0, 1.0)



        ax.set_title(rf"$\Delta={delta:g}$", fontsize=16)



        style_axis(ax)



        ax.set_xlabel("")



        if delta_index == 0:

            ax.set_ylabel(r"$\pi_t(1)$", fontsize=15)



        if delta_index == len(axes) - 1:

            ax.legend(loc="lower right", frameon=True)



    # One shared x-axis label. No global title.

    fig.supxlabel("Time", fontsize=16)



    fig.tight_layout(rect=(0.02, 0.04, 1.0, 1.0))



    output_path = outdir / f"pi1_four_gaps_K{num_arms}.pdf"

    fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)



    print(f"Wrote {output_path}")

    return output_path





def plot_all(args) -> None:

    arms_list = [int(x.strip()) for x in args.arms_list.split(",") if x.strip()]



    for num_arms in arms_list:

        plot_one_k(args, num_arms)





def run_all(args) -> None:

    arms_list = [int(x.strip()) for x in args.arms_list.split(",") if x.strip()]



    for num_arms in arms_list:

        for delta_index in range(len(DELTAS)):

            case_args = argparse.Namespace(**vars(args))

            case_args.num_arms = num_arms

            case_args.delta_index = delta_index

            simulate_one_case(case_args)



    plot_all(args)





def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(

        description="Make separate 1x4 PDF plots for K=3, K=4, and K=5."

    )



    subparsers = parser.add_subparsers(dest="command", required=True)



    run_case = subparsers.add_parser("run-case", help="Run one K and one Delta index")

    run_case.add_argument("--outdir", default=DEFAULT_OUTDIR)

    run_case.add_argument("--num-arms", type=int, required=True)

    run_case.add_argument("--delta-index", type=int, required=True)

    run_case.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)

    run_case.add_argument("--num-checkpoints", type=int, default=DEFAULT_NUM_CHECKPOINTS)

    run_case.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run_case.add_argument("--max-mean-change", type=float, default=0.08)

    run_case.add_argument("--max-noise-change", type=float, default=0.35)

    run_case.add_argument("--max-block-size", type=int, default=10**18)

    run_case.add_argument("--block-quantile", type=float, default=0.995)

    run_case.add_argument("--exact-small-block-threshold", type=int, default=64)

    run_case.set_defaults(func=simulate_one_case)



    plot = subparsers.add_parser("plot", help="Create the three separate PDF plots")

    plot.add_argument("--outdir", default=DEFAULT_OUTDIR)

    plot.add_argument("--arms-list", default="3,4,5")

    plot.set_defaults(func=plot_all)



    run_all_parser = subparsers.add_parser("run-all", help="Run all cases sequentially and plot")

    run_all_parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    run_all_parser.add_argument("--arms-list", default="3,4,5")

    run_all_parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)

    run_all_parser.add_argument("--num-checkpoints", type=int, default=DEFAULT_NUM_CHECKPOINTS)

    run_all_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run_all_parser.add_argument("--max-mean-change", type=float, default=0.08)

    run_all_parser.add_argument("--max-noise-change", type=float, default=0.35)

    run_all_parser.add_argument("--max-block-size", type=int, default=10**18)

    run_all_parser.add_argument("--block-quantile", type=float, default=0.995)

    run_all_parser.add_argument("--exact-small-block-threshold", type=int, default=64)

    run_all_parser.set_defaults(func=run_all)



    return parser





def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    args.func(args)





if __name__ == "__main__":

    main()

