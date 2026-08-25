import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
from sklearn.isotonic import isotonic_regression
from python_scripts.math_utils import *
from python_scripts.preprocessing import *

# Parameters from the paper (see Algo 1-2 resp.)
K = 500
DELTA = 5e-3
J = 8

# Loss counted for each metric, and stratification set A (Corollary 2)
METRIC_LOSS = {
    "standard": "standard",
    "FP": "FP",
    "FN": "FN",
    "FPR": "FP",
    "FNR": "FN",
    "PPV": "FP",
    "SE": "FN",
    "SP": "FP",
}
LOWER_BOUNDED = ("PPV", "SE", "SP")


def theta_grid(theta_min, theta_max, k):
    """Build the fixed (Sn-independent) grid G of Algorithms 1-2.

    `theta_max` is excluded, so |G| = k exactly and δ/k is the exact union-bound
    level over G.

    Args:
        theta_min, theta_max (float): Grid endpoints.
        k (int): Grid size.

    Returns:
        np.ndarray: The k thresholds, in increasing order.
    """
    return np.linspace(theta_min, theta_max, k + 1)[:-1]


def emp_errs_count(samples, loss="standard"):
    """Count empirical errors in `samples` for the given loss ('standard', 'FP', 'FN').

    Args:
        samples (pd.DataFrame): Must contain boolean/int columns `y_pred`, `y_true`.
        loss (str): Error type to count.

    Returns:
        int: Number of errors.
    """
    if loss == "standard":
        return (samples.y_pred != samples.y_true).sum()
    elif loss == "FP":
        return ((samples.y_pred == 1) & (samples.y_true == 0)).sum()
    elif loss == "FN":
        return ((samples.y_pred == 0) & (samples.y_true == 1)).sum()
    else:
        raise ValueError("metric must be either 'standard', 'FP' or 'FN'")


def stratify(samples, metric):
    """Restrict `samples` to the stratum A of the metric (Corollary 2).

    Args:
        samples (pd.DataFrame): Samples with `y_pred`, `y_true`.
        metric (str): Metric name.

    Returns:
        pd.DataFrame: The stratified subset (`samples` itself for the L-risks).
    """
    if metric not in METRIC_LOSS:
        raise ValueError(f"Unsupported metric: {metric!r}")
    if metric in ("FPR", "SP"):
        return samples.loc[samples.y_true == 0]
    elif metric in ("FNR", "SE"):
        return samples.loc[samples.y_true == 1]
    elif metric == "PPV":
        return samples.loc[samples.y_pred == 1]
    else:
        return samples


def emp_metric(samples, metric="standard"):
    """Compute an empirical classification metric on `samples`.

    Supports: 'standard', 'FP', 'FN', 'FPR', 'FNR', 'PPV', 'SE', 'SP'.

    Args:
        samples (pd.DataFrame): Must contain `y_pred`, `y_true`.
        metric (str): Metric name.

    Returns:
        float: Metric value, or NaN if the stratum is empty.

    Raises:
        ValueError: If dataset is empty or metric is unknown.
    """
    if samples.shape[0] == 0:
        raise ValueError("no sample in dataset")
    stratum = stratify(samples, metric)
    if stratum.shape[0] == 0:
        return np.nan
    r = emp_errs_count(stratum, loss=METRIC_LOSS[metric]) / stratum.shape[0]
    return 1 - r if metric in LOWER_BOUNDED else r


def satisfied(bound, r_star, metric):
    """Check if the target constraint is satisfied for the metric.

    Args:
        bound (float): Current bound.
        r_star (float): Target level.
        metric (str): Metric name.

    Returns:
        bool: True if constraint is met.
    """
    if metric in LOWER_BOUNDED:
        return bound >= r_star
    else:
        return bound <= r_star


def theta_record(Sn, theta, metric, delta, divisor):
    """Evaluate the metric bound at a single threshold θ.

    Args:
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        theta (float): Confidence threshold.
        metric (str): Metric name.
        delta (float): proba control (reported as-is in the record).
        divisor (int): Union-bound divisor, so the bound is computed at delta/divisor.

    Returns:
        dict: {'theta_star','bound','vacuous','delta','coverage','emp_metric'}.
    """
    selected_samples = Sn.loc[Sn.kappa >= theta]
    stratum = stratify(selected_samples, metric)
    m = stratum.shape[0]
    selected_errs_count = emp_errs_count(stratum, loss=METRIC_LOSS[metric]) if m else 0

    # b = 1 by definition of B^*(.) in Proposition 1, on an empty or fully
    # misclassified stratum; this is the termination condition of Algo 1
    if m == 0 or selected_errs_count == m:
        b = 1
    else:
        b = B_star(delta / divisor, selected_errs_count, m)  # formula in Corollary 2

    return {
        "theta_star": theta,
        "bound": 1 - b if metric in LOWER_BOUNDED else b,
        "vacuous": b == 1,
        "delta": delta,
        "coverage": selected_samples.shape[0] / Sn.shape[0],
        "emp_metric": emp_metric(stratum, metric=metric) if m else np.nan,
    }


def sgp_greedy_search(delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k=K):
    """Scan θ upward and evaluate the bound on the grid (Algorithm 1).

    Args:
        delta (float): proba control.
        r_star (float | None): Target metric level, or None for the full scan.
        Sn (pd.DataFrame): Training set with `kappa`, `y_pred`, `y_true`.
        metric (str): Metric name.
        theta_min, theta_max (float): Grid endpoints.
        k (int): Grid size (Sn-independent), also the Bonferroni divisor

    Returns:
        dict: {'theta_star','bound','delta','coverage','emp_metric'}, or {} if no
            θ satisfies the target
        list[dict]: the same records for every explored θ, in increasing θ
            order -- when `r_star` is None.
    """
    thetas = theta_grid(theta_min, theta_max, k)
    path = []

    for theta in thetas:
        record = theta_record(Sn, theta, metric, delta, k)
        if record["vacuous"]:
            break

        if r_star is None:
            path.append(record)
        elif satisfied(record["bound"], r_star, metric):
            return record

    return path if r_star is None else {}  # {} if we never found satisfactory B..


def sgp_multistart_search(
    delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k=K, J=J, path=None
):
    """Multistart leftward scan with halting, one scan per bin (Algorithm 2).

    The fixed grid is split into `J` consecutive bins.  Each bin is scanned from its
    rightmost θ leftward and halts at the first θ whose bound misses `r_star`, so bin
    j contributes the leftmost θ of its accepted suffix -- or nothing at all if its
    own rightmost θ already fails.  Bounds are computed at δ/J instead of δ/K, hence
    tighter than in Algorithm 1.

    Args:
        delta (float): proba control.
        r_star (float | None): Target metric level, or None to return the bound path.
        Sn (pd.DataFrame): Training set with `kappa`, `y_pred`, `y_true`.
        metric (str): Metric name.
        theta_min, theta_max (float): Grid endpoints.
        k (int): Grid size (Sn-independent).
        J (int): Number of bins (Sn-independent), also the Bonferroni divisor.
        path (list[dict] | None): Precomputed bound path to reuse across targets.

    Returns:
        list[dict]: Θ as {'theta_star','bound','bin','delta','coverage','emp_metric'}
            records, one per contributing bin, in increasing θ order; the first one is
            the coverage-maximizing threshold.  Empty if no bin contributes -- which,
            per Corollary 2, can happen even though some θ in the grid does satisfy
            `r_star` (namely when the whole accepted interval sits strictly inside one
            bin, missing its right endpoint).
        list[dict]: the same records for *every* θ of the grid, in increasing θ order
            -- when `r_star` is None.  Unlike Algorithm 1's path, no θ is skipped:
            failing and vacuous bounds are kept, since they determine where each
            leftward scan halts.
    """
    if path is None:
        thetas = theta_grid(theta_min, theta_max, k)  # same fixed grid as Algo 1
        bins = np.array_split(thetas, J)  # consecutive blocks of G, Sn-independent
        if any(G_j.size < 2 for G_j in bins):
            raise ValueError(f"Each bin needs K_j > 1: {len(thetas)} thetas, J={J}.")

        path = []
        for j, G_j in enumerate(bins):
            for theta in G_j:
                record = theta_record(Sn, theta, metric, delta, J)
                record["bin"] = j
                path.append(record)

    if r_star is None:
        return path

    Theta = []
    for j in range(1 + max(rec["bin"] for rec in path)):
        last = None
        for rec in reversed([rec for rec in path if rec["bin"] == j]):  # leftward
            if rec["vacuous"] or not satisfied(rec["bound"], r_star, metric):
                break  # halt at first failure: this bin is over
            last = rec
        if last is not None:
            Theta.append(last)

    return Theta


def sgp_at_targets(
    train_set,
    test_set,
    delta=DELTA,
    metric_targets=None,
    metric="standard",
    mode="multistart",
    k=K,
    J=J,
    theta_min=0.5,
    theta_max=1,
):
    """Run SGP across multiple target levels and report train/test outcomes.

    The θ grid is scanned once, then reused for every target.

    Args:
        train_set (pd.DataFrame): Training data with `kappa`, `y_pred`, `y_true`.
        test_set (pd.DataFrame): Test data with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        metric_targets (list[float] | None): Target levels r*.
        metric (str): Metric name.
        mode (str): "greedy" for Algorithm 1 (bounds at δ/k, first admissible θ of the
            upward scan) or "multistart" for Algorithm 2 (bounds at δ/J, leftward scan
            per bin halting at the first failure).
        k (int): Grid size.
        J (int): Number of bins -- "multistart" only.
        theta_min, theta_max (float): Grid endpoints.

    Returns:
        pd.DataFrame: One row per retained target with bounds, θ*, and coverages.
            In "multistart" mode a target may be dropped even though some grid θ
            satisfies it, per Corollary 2.
    """
    if mode not in ("greedy", "multistart"):
        raise ValueError(f"Unsupported mode: {mode!r}")
    if metric_targets is None:
        metric_targets = [i / 100 for i in range(1, 15)]

    search = sgp_greedy_search if mode == "greedy" else sgp_multistart_search
    kwargs = {} if mode == "greedy" else {"J": J}

    path = search(
        delta,
        None,
        train_set,
        metric,
        theta_min=theta_min,
        theta_max=theta_max,
        k=k,
        **kwargs,
    )

    n_test = test_set.shape[0]
    test_cache = {}
    results = []

    for r_star in metric_targets:
        if mode == "greedy":
            # min of O, i.e. first admissible θ of the upward scan
            sgp_dico = next(
                (rec for rec in path if satisfied(rec["bound"], r_star, metric)), None
            )
        else:
            # Θ depends on r*: replay the per-bin halting rule on the cached bounds
            Theta = search(delta, r_star, train_set, metric, path=path)
            sgp_dico = Theta[0] if Theta else None  # leftmost accepted θ

        if sgp_dico is None:
            continue

        theta_star = sgp_dico["theta_star"]
        if theta_star not in test_cache:
            covered_test_set = test_set.loc[test_set.kappa >= theta_star]
            n_covered = covered_test_set.shape[0]
            test_cache[theta_star] = (
                emp_metric(covered_test_set, metric=metric) if n_covered else np.nan,
                n_covered / n_test,
            )
        test_metric, test_coverage = test_cache[theta_star]

        results.append(
            {
                "metric_target": r_star,
                "metric_bound": sgp_dico["bound"],
                "theta_star": theta_star,
                "train_metric": sgp_dico["emp_metric"],
                "train_coverage": sgp_dico["coverage"],
                "test_metric": test_metric,
                "test_coverage": test_coverage,
            }
        )

    return pd.DataFrame(
        results,
        columns=[
            "metric_target",
            "metric_bound",
            "theta_star",
            "train_metric",
            "train_coverage",
            "test_metric",
            "test_coverage",
        ],
    )


def sgp_at_targets_on_imbalanced_sets(
    proportions_of_1,
    metric_targets,
    sgp_df,
    delta=DELTA,
    k=K,
    J=J,
    metric="standard",
    mode="multistart",
    theta_min=0.5,
    theta_max=1,
):
    """Evaluate SGP at multiple class-1 proportions.

    Args:
        proportions_of_1 (list[float]): Desired positive-class rates.
        metric_targets (list[float]): Target levels r*.
        sgp_df (pd.DataFrame): Base dataset with `y_true`, `kappa`.
        delta (float): Confidence level.
        k (int): Grid size.
        J (int): Number of bins -- "multistart" only.
        metric (str): Metric name.
        mode (str): Search mode, see `sgp_at_targets`.
        theta_min, theta_max (float): Grid endpoints.

    Returns:
        pd.DataFrame: Results with proportion, bounds, θ*, and metrics.
    """
    all_propor_dfs = []
    imbalanced_datasets = generate_imbalanced_datasets(sgp_df, proportions_of_1, seed=0)

    for proportion_1, imbalanced_set in zip(proportions_of_1, imbalanced_datasets):

        train_set_ = imbalanced_set.iloc[: int(imbalanced_set.shape[0] / 2)]
        test_set_ = imbalanced_set.iloc[int(imbalanced_set.shape[0] / 2) :]

        results = sgp_at_targets(
            train_set_,
            test_set_,
            delta=delta,
            metric_targets=metric_targets,
            metric=metric,
            mode=mode,
            k=k,
            J=J,
            theta_min=theta_min,
            theta_max=theta_max,
        )
        results["proportion_1"] = proportion_1
        all_propor_dfs.append(results)

    return pd.concat(all_propor_dfs).reset_index(drop=True)


def bound_evo_w_theta(metric, Sn, delta, theta_min=0.5, theta_max=1, k=K):
    """Trace the metric bound as a function of θ.

    Args:
        metric (str): Metric name.
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        theta_min, theta_max (float): Grid endpoints.
        k (int): Grid size.

    Returns:
        (np.ndarray, list[float]): (thetas, bounds) with NaNs for invalid regions.
    """
    thetas = theta_grid(theta_min, theta_max, k)

    path = sgp_greedy_search(
        delta, None, Sn, metric, theta_min=theta_min, theta_max=theta_max, k=k
    )

    bounds = [np.nan if rec["vacuous"] else rec["bound"] for rec in path]
    bounds += [np.nan] * (len(thetas) - len(bounds))
    return thetas, bounds


def reachable_bounds(
    metrics_list, Sn, delta=DELTA, mode="greedy", theta_min=0.5, theta_max=1, k=K, J=J
):
    """Compute θ/coverage grids and bounds for a list of metrics.

    Args:
        metrics_list (list[str]): Metrics to evaluate.
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        theta_min, theta_max (float): Grid endpoints.
        k (int): grid size.

    Returns:
        dict: {'thetas','coverages', metric->bounds}, all aligned on the same grid.
    """
    res_dico = {}

    # thetas and coverages coordinates
    thetas = theta_grid(theta_min, theta_max, k)
    res_dico["thetas"] = list(thetas)
    res_dico["coverages"] = [
        Sn.loc[Sn.kappa >= theta].shape[0] / Sn.shape[0] for theta in thetas
    ]
    # metrics bounds with respect to thetas
    for metric in metrics_list:
        _, bounds = bound_evo_w_theta(
            metric,
            Sn,
            delta,
            theta_min=theta_min,
            theta_max=theta_max,
            k=k,
        )
        res_dico[metric] = bounds

    return res_dico


def pos_propor_w_theta(Sn, k=K, theta_min=0.5, theta_max=1):
    """Compute positive-class proportion among samples selected by θ.

    Args:
        Sn (pd.DataFrame): Dataset with `kappa`, `y_true`.
        k (int): grid size.
        theta_min, theta_max (float): Grid endpoints.

    Returns:
        (np.ndarray, list[float]): (thetas, positive proportions), NaN if no sample.
    """
    pos_propor, thetas = [], theta_grid(theta_min, theta_max, k)

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        n = selected_samples.shape[0]
        pos_propor.append(selected_samples.y_true.sum() / n if n else np.nan)

    return thetas, pos_propor


def runtime(sim_df, mode="multistart", k=K, J=J, theta_min=0.5, theta_max=1):
    """Measure wall-time (seconds) for SGP search mode on `sim_df`.

    Args:
        sim_df (pd.DataFrame): Simulated Dataset for timing.
        mode (str): Search mode, see `sgp_at_targets`.
        k (int): grid size.
        J (int): Number of bins -- "multistart" only.
        theta_min, theta_max (float): Grid endpoints.

    Returns:
        float: Elapsed seconds.
    """
    search = sgp_greedy_search if mode == "greedy" else sgp_multistart_search
    kwargs = {} if mode == "greedy" else {"J": J}

    t0 = time.perf_counter()

    res = search(
        delta=DELTA,
        r_star=0.05,
        Sn=sim_df,
        metric="standard",
        theta_min=theta_min,
        theta_max=theta_max,
        k=k,
        **kwargs,
    )

    return time.perf_counter() - t0


def joint_control(
    metrics_and_targets,
    sgp_df,
    delta=DELTA,
    theta_min=0.5,
    theta_max=1,
    plot=False,
    k=K,
):
    """Find θ intervals satisfying multiple metric targets (optionally plot).

    Args:
        metrics_and_targets (dict): {metric: target}.
        sgp_df (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        theta_min, theta_max (float): Grid endpoints.
        plot (bool): If True, plot bounds and feasible θ segments.
        k (int): grid size.

    Returns:
        dict | None: If not plotting, {'theta_intervals', 'best_theta'}.
    """
    y_proj = -0.01
    projection_handles = []

    # Use a colormap to assign a unique color to each metric
    colors = plt.get_cmap("tab10").colors
    segments_per_metric = {key: [] for key in metrics_and_targets.keys()}

    if plot:
        plt.figure()

    for i, (metric, target) in enumerate(metrics_and_targets.items()):
        thetas, bounds = bound_evo_w_theta(
            metric,
            sgp_df,
            delta,
            theta_min=theta_min,
            theta_max=theta_max,
            k=k,
        )
        color = colors[i % len(colors)]
        sign = ">" if metric in LOWER_BOUNDED else "<"

        mask = np.array([satisfied(b, target, metric) for b in bounds])
        segments = get_segments(thetas, mask)
        segments_per_metric[metric] = segments

        if plot:
            plt.plot(
                thetas, bounds, color=color, label=f"{metric} bound", linewidth=1.5
            )
            plt.axhline(y=target, color=color, linestyle="--", label=f"{metric} target")
            plt.xlabel(r"$\theta$")
            plt.ylabel("Metric bounds")
            plt.tick_params(axis="y")

            for x_start, x_end in segments:
                plt.hlines(y_proj * i, x_start, x_end, colors=color, linewidth=2)

            # Legend handle for projections
            projection_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2,
                    label=r"$\theta$ " + f"/ {metric} {sign} {target}",
                )
            )

    if plot:
        plt.ylim(bottom=y_proj - 0.02)
        plt.legend(handles=projection_handles, loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        intersected_intervals = compute_all_interval_intersections(segments_per_metric)
        return {
            "theta_intervals": intersected_intervals,
            "best_theta": best_theta(intersected_intervals),
        }


def mean_abs_diff(u, v):
    """Mean absolute difference ignoring NaNs (pairwise).

    Args:
        u, v (array-like): Sequences to compare.

    Returns:
        float: Mean |u - v| over valid pairs, or NaN if none.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Mask to filter out nan entries
    not_masked = ~np.isnan(u) & ~np.isnan(v)

    if not np.any(not_masked):
        return np.nan

    return np.mean(np.abs(u[not_masked] - v[not_masked]))


def ABC(
    ds, metric, mode="multistart", theta_min=0.5, theta_max=1, k=K, delta=DELTA, J=J
):
    """Compute average absolute gap between bound and test metric vs θ.

    Args:
        ds (pd.DataFrame): Dataset split in half into train/test.
        metric (str): One of {'standard','FP','FN','FPR','FNR'}.
        mode (str): Search mode, see `bound_evo_w_theta`.
        theta_min, theta_max (float): Grid endpoints.
        k (int): grid size.
        delta (float): Confidence level.
        J (int): Number of bins -- "multistart" only.

    Returns:
        float: Mean absolute difference between bound and empirical metric.
    """
    train_set = ds.iloc[: int(len(ds) / 2)]
    test_set = ds.iloc[int(len(ds) / 2) :]

    thetas, bounds = bound_evo_w_theta(
        metric,
        train_set,
        delta,
        theta_min=theta_min,
        theta_max=theta_max,
        k=k,
    )
    emp_metrics = []
    for theta in thetas:
        selected_set = test_set.loc[test_set.kappa >= theta]
        emp_metrics.append(
            emp_metric(selected_set, metric=metric) if selected_set.shape[0] else np.nan
        )

    return mean_abs_diff(bounds, emp_metrics)


def our_bound(selected_samples, metric, delta=DELTA, k=K):
    """
    Compute our guaranteed conditional metric bound (to be compared to external reference)

    Args:
        selected_samples: samples with confidence higher than threshold
        metric: one of the selective metrics 'standard', 'FPR', 'FNR' etc...
        delta: probability control
        k: grid size, i.e. the union-bound divisor

    Returns:
        float: bound from proposition 2-3, NaN if vacuous
    """
    stratum = stratify(selected_samples, metric)
    m = stratum.shape[0]
    selected_errs_count = emp_errs_count(stratum, loss=METRIC_LOSS[metric]) if m else 0

    if m == 0 or selected_errs_count == m:
        return np.nan  # b = 1 by definition of B^*(.) in Proposition 1

    b = B_star(delta / k, selected_errs_count, m)  # see formula in Corollary 2
    return 1 - b if metric in LOWER_BOUNDED else b


def eq11_bound(selected_samples, metric, delta=DELTA, detailed=False):
    """
    Compute conditional metric bound with Eq. (11) from (Balsubramani et al., 2019)

    Args:
        selected_samples: samples with confidence higher than threshold
        metric: 'FPR', 'FNR' etc..
        delta: probability control

    Returns:
        float: bound from Eq. (11)
    """
    if metric == "FPR":
        a = (selected_samples.y_pred * (1 - selected_samples.y_true)).sum() / (
            1 - selected_samples.y_true
        ).sum()
        b = np.sqrt(-2 * np.log(delta) / (1 - selected_samples.y_true).sum())

    elif metric == "FNR":
        a = (
            (1 - selected_samples.y_pred) * selected_samples.y_true
        ).sum() / selected_samples.y_true.sum()
        b = np.sqrt(-2 * np.log(delta) / selected_samples.y_true.sum())

    else:  # PPV
        a = (
            selected_samples.y_pred * selected_samples.y_true
        ).sum() / selected_samples.y_pred.sum()
        b = np.sqrt(-2 * np.log(delta) / selected_samples.y_pred.sum())

    if detailed:
        return a, b
    return a - b if metric == "PPV" else a + b


def run_one_seed_all_targets(
    sgp_df,
    s,
    metric_targets,
    delta=DELTA,
    theta_min=0.5,
    theta_max=1,
    metric="standard",
    mode="multistart",
    delta_test=0.05,
):
    """One split, one grid pass, every target read off it.

    A target counts as failed only when the held-out set gives significant evidence
    that the population metric exceeds r*, i.e. when the one-sided Clopper-Pearson
    limit on the stratum, at level `delta_test`, still misses r*.

    Returns:
        (np.ndarray, np.ndarray): (valid, failed) 0/1 arrays aligned with
        `metric_targets`, summable across seeds.  `failed` is 0 on splits where the
        search returned nothing, so sums divide by num_seed.
    """
    train_set, test_set = train_test_split(sgp_df, seed=s, p_train=0.75)
    results = sgp_at_targets(
        train_set,
        test_set,
        delta=delta,
        metric_targets=metric_targets,
        metric=metric,
        mode=mode,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    valid = np.zeros(len(metric_targets), dtype=np.int64)
    failed = np.zeros(len(metric_targets), dtype=np.int64)
    if results.shape[0] == 0:
        return valid, failed

    # metric_target round-trips through the frame untouched, so equality holds
    position = {t: i for i, t in enumerate(metric_targets)}
    for t, theta_star, test_metric in zip(
        results.metric_target, results.theta_star, results.test_metric
    ):
        i = position[t]
        valid[i] = 1

        if np.isnan(test_metric):
            continue

        if delta_test is None:
            failed[i] = (
                bool(test_metric < t)
                if metric in LOWER_BOUNDED
                else bool(t < test_metric)
            )
            continue

        # CP lower limit on the held-out error rate of the stratum
        stratum = stratify(test_set.loc[test_set.kappa >= theta_star], metric)
        m_cov = stratum.shape[0]
        c_cov = emp_errs_count(stratum, loss=METRIC_LOSS[metric]) if m_cov else 0
        if m_cov == 0:
            continue
        lower = beta.ppf(delta_test, c_cov, m_cov - c_cov + 1) if c_cov else 0.0

        if metric in LOWER_BOUNDED:
            failed[i] = bool((1 - lower) < t)  # CP upper limit on the metric
        else:
            failed[i] = bool(t < lower)

    return valid, failed


def bound_path(Sn, metric, delta=DELTA, theta_min=0, theta_max=1, k=K, J=J):
    path = sgp_multistart_search(
        delta, None, Sn, metric, theta_min=theta_min, theta_max=theta_max, k=k, J=J
    )
    b = np.array([r["bound"] for r in path if not r["vacuous"]], dtype=float)
    return -b if metric in LOWER_BOUNDED else b


def unimodal_fit(b):
    best, fit = np.inf, None
    for m in range(-1, len(b)):
        left = isotonic_regression(b[: m + 1], increasing=False) if m >= 0 else []
        right = (
            isotonic_regression(b[m + 1 :], increasing=True) if m + 1 < len(b) else []
        )
        cand = np.concatenate([left, right])
        sse = ((b - cand) ** 2).sum()
        if sse < best:
            best, fit = sse, cand
    return fit


def ushape_area(b):
    return float(np.abs(b - unimodal_fit(b)).sum() / K)


def errs_mask(samples, loss="standard"):
    if loss == "standard":
        return samples.y_pred != samples.y_true
    elif loss == "FP":
        return (samples.y_pred == 1) & (samples.y_true == 0)
    else:
        return (samples.y_pred == 0) & (samples.y_true == 1)


def h1_transport(Sn, metric):
    st = stratify(Sn, metric).sort_values("kappa", ascending=False)
    L = errs_mask(st, METRIC_LOSS[metric]).values.astype(int)
    n_err, n_ok = int(L.sum()), int((L == 0).sum())
    if n_err == 0 or n_ok == 0:
        return 0.0, np.nan, True
    cost = float(np.abs(np.cumsum(L) - np.cumsum(np.sort(L))).sum())
    return cost, 1 - cost / (n_ok * n_err), bool(cost == 0)


def accepted_mask(path, r_star, metric):
    """Grid thresholds whose bound meets r*, using Algorithm 2's own accept test.

    Args:
        path (list[dict]): Raw bound path over the whole grid, at level delta/J.
        r_star (float): Target level.
        metric (str): Metric name.

    Returns:
        np.ndarray: Boolean mask over the grid, in increasing theta order.
    """
    return np.array(
        [
            (not rec["vacuous"]) and satisfied(rec["bound"], r_star, metric)
            for rec in path
        ],
        dtype=bool,
    )


def oracle_index(path, r_star, metric):
    """Index of min{theta in G : B(theta) <= r*}, right-hand side of Corollary 2.

    Returns:
        int | None: None when no threshold of the grid meets r*.
    """
    idx = np.flatnonzero(accepted_mask(path, r_star, metric))
    return int(idx[0]) if idx.size else None


def is_contiguous(mask):
    """True iff the accepted set is an interval of the grid.

    Quasiconvexity (Lemma 1, under Hypothesis 1) implies contiguity, so this is
    the observable mechanism through which a violation of Hypothesis 1 can break
    Corollary 2.
    """
    idx = np.flatnonzero(mask)
    return bool(idx.size == 0 or idx.size == idx[-1] - idx[0] + 1)


def completeness_record(path, r_star, metric, delta=DELTA):
    """Compare the threshold returned by Algorithm 2 with the best bound on G of Corollary 2.

    Status is 'exact' when the two coincide, 'suboptimal' when they do not, and
    'empty' when Algorithm 2 returns nothing.  Corollary 2 assumes Theta non-empty,
    so 'empty' runs are not counted: they are a power loss driven by r* and J, and
    they occur under Hypothesis 1 as well, whenever the accepted set sits strictly
    inside a bin and misses its right endpoint.

    Returns:
        dict | None: None when r* is unattainable on the grid (Algorithm 2 then
            returns nothing either, so the comparison is vacuous).
    """
    mask = accepted_mask(path, r_star, metric)
    i_oracle = oracle_index(path, r_star, metric)
    if i_oracle is None:
        return None

    Theta = sgp_multistart_search(delta, r_star, None, metric, path=path)
    pos = {rec["theta_star"]: i for i, rec in enumerate(path)}

    if not Theta:
        status, i_algo2, gap_steps, coverage_loss = "empty", None, np.nan, np.nan
    else:
        i_algo2 = pos[Theta[0]["theta_star"]]
        status = "exact" if i_algo2 == i_oracle else "suboptimal"
        gap_steps = i_algo2 - i_oracle
        coverage_loss = path[i_oracle]["coverage"] - Theta[0]["coverage"]

    return {
        "r_star": r_star,
        "status": status,
        "theta_oracle": path[i_oracle]["theta_star"],
        "theta_algo2": None if i_algo2 is None else Theta[0]["theta_star"],
        "gap_steps": gap_steps,
        "coverage_loss": coverage_loss,
        "n_accepted": int(mask.sum()),
        "contiguous": is_contiguous(mask),
    }


def run_one(n, ranking, seed, metrics, r_stars, theta_min, theta_max, delta):
    """All records for one simulated dataset, shared across metrics and targets."""
    out = []
    sim = simulate_sgp_dataset(n, ranking=ranking, seed=seed)
    for metric in metrics:
        path = sgp_multistart_search(
            delta, None, sim, metric, theta_min=theta_min, theta_max=theta_max
        )
        _, auc_emp, _ = h1_transport(sim, metric)
        for r_star in r_stars:
            rec = completeness_record(path, r_star, metric, delta=delta)
            if rec is None:  # no threshold of the grid meets r*
                continue
            rec.update(metric=metric, n=n, ranking=ranking, seed=seed, auc_emp=auc_emp)
            out.append(rec)
    return out


LOSS_LABEL = {"standard": "L_01", "FP": "L_FP", "FN": "L_FN"}


def discordance_table(datasets, metrics=("standard", "FP", "FN")):
    """Discordance rate of Hypothesis 1, one row per (dataset, loss).

    Args:
        datasets (dict): {name: DataFrame with `kappa`, `y_pred`, `y_true`}.
        metrics (tuple[str]): Metrics whose loss/stratum to evaluate.
    """
    rows = []
    for ds_name, Sn in datasets.items():
        for metric in metrics:
            A = stratify(Sn, metric)
            L = errs_mask(A, METRIC_LOSS[metric]).values.astype(np.int64)
            n, n_err = L.size, int(L.sum())
            n_ok = n - n_err
            cost, auc_emp, holds = h1_transport(Sn, metric)
            mixed, total = n_ok * n_err, n * (n - 1) // 2
            rows.append(
                {
                    "dataset": ds_name,
                    "loss": LOSS_LABEL.get(metric, metric),
                    "n": n,
                    "n_err": n_err,
                    "kappa_ties": int(n - A.kappa.nunique()),
                    "discordant_pairs": int(cost),
                    "discordance_pct": 100 * (1 - auc_emp) if mixed else np.nan,
                    "discordance_pct_all_pairs": (
                        100 * cost / total if total else np.nan
                    ),
                    "auc_emp": auc_emp,
                    "h1_holds": holds,
                }
            )
    return pd.DataFrame(rows)
