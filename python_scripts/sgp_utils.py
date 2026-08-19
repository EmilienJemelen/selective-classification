import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import math
import scipy.special
from scipy.stats import beta, binom
from scipy.optimize import brentq
import random as rd
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights
from tqdm import tqdm
import pickle
import torch.optim.lr_scheduler as lr_scheduler
from scipy.special import gammaln
from collections import defaultdict
from pathlib import Path
from collections import Counter
from python_scripts.math_utils import *
from python_scripts.preprocessing import *

# Parameters from the paper (see Algo 1-2 resp.)
K = 100
DELTA = 5e-3


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


def emp_metric(samples, metric="standard"):
    """Compute an empirical classification metric on `samples`.

    Supports: 'standard', 'FP', 'FN', 'FPR', 'FNR', 'PPV', 'SE', 'SP'.

    Args:
        samples (pd.DataFrame): Must contain `y_pred`, `y_true`.
        metric (str): Metric name.

    Returns:
        float: Metric value.

    Raises:
        ValueError: If dataset is empty or metric is unknown.
    """
    if samples.shape[0] == 0:
        raise ValueError("no sample in dataset")
    if metric == "standard":
        return emp_errs_count(samples) / samples.shape[0]
    elif metric == "FP":
        return emp_errs_count(samples, loss="FP") / samples.shape[0]
    elif metric == "FN":
        return emp_errs_count(samples, loss="FN") / samples.shape[0]
    elif metric == "FPR":
        return emp_errs_count(samples, loss="FP") / (1 - samples.y_true).sum()
    elif metric == "FNR":
        return emp_errs_count(samples, loss="FN") / samples.y_true.sum()
    elif metric == "PPV":
        return (samples.y_pred * samples.y_true).sum() / samples.y_pred.sum()
    elif metric == "SE":
        return (samples.y_pred * samples.y_true).sum() / samples.y_true.sum()
    elif metric == "SP":
        return ((1 - samples.y_pred) * (1 - samples.y_true)).sum() / (
            1 - samples.y_true
        ).sum()
    else:
        raise ValueError(
            "metric must be in 'standard', 'FP','FN','FPR','FNR','PPV','SE','SP'"
        )


def upper_bound_denominator(metric, selected_samples, delta, n):
    """Denominator term for upper bounds of ratio metrics.

    Applies to: FPR, FNR, PPV, SE, SP.

    Args:
        metric (str): Metric name.
        selected_samples (pd.DataFrame): Selected subset with `y_pred`, `y_true`.
        delta (float): Confidence level.
        n (int): Total sample size.

    Returns:
        float: Denominator value.
    """
    d2 = np.sqrt(n * np.log(2 / delta) / 2) / selected_samples.shape[0]
    if metric == "PPV":
        d1 = selected_samples.y_pred.sum() / selected_samples.shape[0]
    else:
        d1 = selected_samples.y_true.sum() / selected_samples.shape[0]

    if metric in ["FPR", "SP"]:
        return max(1e-10, 1 - d1 - d2)
    else:  # FNR, SE, PPV
        return max(1e-10, d1 - d2)


def bound(b, selected_samples, delta, metric, n):
    """Transform risk bound `b` into a metric-specific bound.

    Args:
        b (float): Base bound B*.
        selected_samples (pd.DataFrame): Selected subset.
        delta (float): Confidence level.
        metric (str): Target metric.
        n (int): Total sample size.

    Returns:
        float: Metric bound
    """
    if metric in ["standard", "FP", "FN"]:
        B = b
    elif metric in ["FPR", "FNR"]:
        B = min(1, b / upper_bound_denominator(metric, selected_samples, delta, n))
    else:  # PPV, SE, SP
        B = max(0, 1 - b / upper_bound_denominator(metric, selected_samples, delta, n))

    return B


def satisfied(bound, r_star, metric):
    """Check if the target constraint is satisfied for the metric.

    Args:
        bound (float): Current bound.
        r_star (float): Target level.
        metric (str): Metric name.

    Returns:
        bool: True if constraint is met.
    """
    if metric in ["standard", "FP", "FN", "FPR", "FNR"]:
        return True if bound <= r_star else False
    else:
        return True if bound >= r_star else False


def sgp_greedy_search(delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k=K):
    """Scan θ upward and evaluate the bound on the grid.

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
        list[dict]: the same records for every admissible θ, in increasing θ
            order -- when `r_star` is None.
    """
    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }
    if metric not in metric_loss_mapping:
        raise ValueError(f"Unsupported metric: {metric!r}")

    thetas = np.linspace(theta_min, theta_max, k)[:-1]
    path = []

    for theta in thetas:
        if metric in ("standard", "FP", "FN"):
            selected_samples = Sn.loc[Sn.kappa >= theta]
        elif metric in ("FPR", "SP"):
            selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_true == 0)]
        elif metric in ("FNR", "SE"):
            selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_true == 1)]
        else:  # PPV
            selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_pred == 1)]

        n = selected_samples.shape[0]
        selected_errs_count = emp_errs_count(
            selected_samples, loss=metric_loss_mapping[metric]
        )

        # no mistake on the selected subset => none at later θ either, so b* is
        # stuck at 1 - delta^(1/n).  (Also covers the empty-selection case.)
        if n == 0 or selected_errs_count == 0:
            break

        b = B_star(delta / k, selected_errs_count, n)  # see formula in Corollary 2
        if selected_errs_count == n:
            b = 1  # by definition of B^*(.) in Proposition 1
        if b == 1:
            break

        if metric in ("SP", "SE", "PPV"):
            b = 1 - b

        if r_star is not None and not satisfied(b, r_star, metric):
            continue

        record = {
            "theta_star": theta,
            "bound": b,
            "delta": delta,
            "coverage": n / Sn.shape[0],
            "emp_metric": emp_metric(selected_samples, metric=metric),
        }
        if r_star is not None:
            return record
        path.append(record)

    return path if r_star is None else {}  # {} if we never found satisfactory B..


def sgp_multistart_search(
    delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k=K, J=8, path=None
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
    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }
    if metric not in metric_loss_mapping:
        raise ValueError(f"Unsupported metric: {metric!r}")

    if path is None:
        thetas = np.linspace(theta_min, theta_max, k)[:-1]  # same fixed grid as Algo 1
        bins = np.array_split(thetas, J)  # consecutive blocks of G, Sn-independent
        if any(G_j.size < 2 for G_j in bins):
            raise ValueError(f"Each bin needs K_j > 1: {len(thetas)} thetas, J={J}.")

        path = []
        for j, G_j in enumerate(bins):
            for theta in G_j:
                if metric in ("standard", "FP", "FN"):
                    selected_samples = Sn.loc[Sn.kappa >= theta]
                elif metric in ("FPR", "SP"):
                    selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_true == 0)]
                elif metric in ("FNR", "SE"):
                    selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_true == 1)]
                else:  # PPV
                    selected_samples = Sn.loc[(Sn.kappa >= theta) & (Sn.y_pred == 1)]

                n = selected_samples.shape[0]
                selected_errs_count = (
                    emp_errs_count(selected_samples, loss=metric_loss_mapping[metric])
                    if n
                    else 0
                )

                # vacuous bound on an empty or fully misclassified stratum, by
                # definition of B^*(.) in Proposition 1
                if n == 0 or selected_errs_count == n:
                    b = 1
                else:
                    b = B_star(delta / J, selected_errs_count, n)  # Corollary 2

                if metric in ("SP", "SE", "PPV"):
                    b = 1 - b

                path.append(
                    {
                        "theta_star": theta,
                        "bound": b,
                        "bin": j,
                        "delta": delta,
                        "coverage": n / Sn.shape[0],
                        "emp_metric": (
                            emp_metric(selected_samples, metric=metric) if n else np.nan
                        ),
                    }
                )

    if r_star is None:
        return path

    Theta = []
    for j in range(1 + max(rec["bin"] for rec in path)):
        last = None
        for rec in reversed([rec for rec in path if rec["bin"] == j]):  # leftward
            if not satisfied(rec["bound"], r_star, metric):
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
    J=8,
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
            # the path holds admissible θ only, in increasing order
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
            covered_test_set = test_set.loc[test_set.kappa > theta_star]
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
    metric="standard",
):
    """Evaluate SGP at multiple class-1 proportions.

    Args:
        proportions_of_1 (list[float]): Desired positive-class rates.
        metric_targets (list[float]): Target levels r*.
        sgp_df (pd.DataFrame): Base dataset with `y_true`, `kappa`.
        delta (float): Confidence level.
        k (int): Grid size.
        metric (str): Metric name.

    Returns:
        pd.DataFrame: Results with proportion, bounds, θ*, and metrics.
    """
    all_propor_dfs = pd.DataFrame()
    imbalanced_datasets = generate_imbalanced_datasets(sgp_df, proportions_of_1, seed=0)

    for proportion_1, imbalanced_set in zip(proportions_of_1, imbalanced_datasets):

        train_set_ = imbalanced_set.iloc[: int(imbalanced_set.shape[0] / 2)]
        train_set_ = (
            train_set_.sort_values("kappa", ascending=True)
            .reset_index(drop=True)
            .copy()
        )
        test_set_ = imbalanced_set.iloc[int(imbalanced_set.shape[0] / 2) :]

        results = sgp_at_targets(
            train_set_,
            test_set_,
            delta=delta,
            metric_targets=metric_targets,
            metric=metric,
            k=k,
        )
        results["proportion_1"] = proportion_1
        all_propor_dfs = pd.concat([all_propor_dfs, results]).reset_index(drop=True)

    return all_propor_dfs


def bound_evo_w_theta(
    metric, Sn, delta, mode="multistart", theta_min=0.5, theta_max=1, k=K, J=8
):
    """Trace the metric bound as a function of θ.

    Args:
        mode (str): "greedy" for Algorithm 1 bounds (δ/k, scan truncated at the
            termination condition) or "multistart" for Algorithm 2 bounds (δ/J,
            whole grid, so the right arm of the U-shape stays visible).
        J (int): Number of bins -- "multistart" only.

    Returns:
        (np.ndarray, list[float]): (thetas, bounds) with NaNs for invalid regions.
    """
    thetas = np.linspace(theta_min, theta_max, k)[:-1]

    if mode == "greedy":
        path = sgp_greedy_search(
            delta, None, Sn, metric, theta_min=theta_min, theta_max=theta_max, k=k
        )
    elif mode == "multistart":
        path = sgp_multistart_search(
            delta,
            None,
            Sn,
            metric,
            theta_min=theta_min,
            theta_max=theta_max,
            k=k,
            J=J,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    # vacuous bounds on an empty stratum
    bounds = [rec["bound"] if rec["coverage"] > 0 else np.nan for rec in path]
    bounds += [np.nan] * (len(thetas) - len(bounds))
    return thetas, bounds


def reachable_bounds(metrics_list, Sn, delta=DELTA, theta_min=0.5, theta_max=1, k=K):
    """Compute θ/coverage grids and bounds for a list of metrics.

    Args:
        metrics_list (list[str]): Metrics to evaluate.
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        k (int): grid size.

    Returns:
        dict: {'thetas','coverages', metric->bounds}.
    """
    res_dico = {}

    # thetas and coverages coordinates
    thetas = np.linspace(theta_min, theta_max, k)
    res_dico["thetas"] = sorted(thetas)
    res_dico["coverages"] = sorted(
        [Sn.loc[Sn.kappa >= theta].shape[0] / Sn.shape[0] for theta in thetas],
        reverse=True,
    )
    # metrics bounds with respect to thetas
    for metric in metrics_list:
        _, bounds = bound_evo_w_theta(
            metric, Sn, delta, theta_min=theta_min, theta_max=theta_max, k=k
        )
        res_dico[metric] = bounds

    return res_dico


def pos_propor_w_theta(Sn, k=K, theta_min=0.5, theta_max=1):
    """Compute positive-class proportion among samples selected by θ.

    Args:
        Sn (pd.DataFrame): Dataset with `kappa`, `y_true`.
        k (int): grid size.

    Returns:
        (np.ndarray, list[float]): (thetas, positive proportions).
    """
    Sn = Sn.sort_values("kappa", ascending=True)
    pos_propor, thetas = [], np.linspace(theta_min, theta_max, k)

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        pos_propor.append(selected_samples.y_true.sum() / selected_samples.shape[0])

    return thetas, pos_propor


def runtime(sim_df, k: int = K, theta_min=0.5, theta_max=1):
    """Measure wall-time (seconds) for SGP search mode on `sim_df`.

    Args:
        sim_df (pd.DataFrame): Simulated Dataset for timing.
        k (int): grid size.

    Returns:
        int: Elapsed seconds.
    """
    t0 = datetime.now()

    res = sgp_greedy_search(
        delta=DELTA,
        r_star=None,
        Sn=sim_df,
        metric="standard",
        theta_min=theta_min,
        theta_max=theta_max,
        k=k,
    )

    t1 = datetime.now()
    return (t1 - t0).seconds


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
        plot (bool): If True, plot bounds and feasible θ segments.
        k (int): grid size.

    Returns:
        dict | None: If not plotting, {'theta_intervals', 'best_theta'}.
    """
    metric_sign_mapping = {
        "standard": "<",
        "FP": "<",
        "FN": "<",
        "FPR": "<",
        "FNR": "<",
        "PPV": ">",
        "SE": ">",
        "SP": ">",
    }
    y_proj = -0.01
    projection_handles = []

    # Use a colormap to assign a unique color to each metric
    num_metrics = len(metrics_and_targets)
    color_map = cm.get_cmap("tab10", num_metrics)
    colors = [color_map(i) for i in range(num_metrics)]
    segments_per_metric = {key: [] for key in metrics_and_targets.keys()}

    if plot:
        plt.figure()

    for i, (metric, target) in enumerate(metrics_and_targets.items()):
        thetas, bounds = bound_evo_w_theta(
            metric, sgp_df, delta, theta_min=theta_min, theta_max=theta_max, k=k
        )
        color = colors[i]

        if metric_sign_mapping[metric] == ">":
            mask = np.array(bounds) > target
        else:
            mask = np.array(bounds) < target

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
                    label=r"$\theta$ "
                    + f"/ {metric} {metric_sign_mapping[metric]} {target}",
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
    u = np.asarray(u)
    v = np.asarray(v)

    # Mask to filter out nan entries
    not_masked = ~np.isnan(u) & ~np.isnan(v)

    # If no valid entries, return nan
    if not np.any(not_masked):
        return np.nan

    diffs = np.abs(u[not_masked] - v[not_masked])
    return np.mean(diffs)


def ABC(ds, metric, theta_min=0.5, theta_max=1, k=30, delta=DELTA):
    """Compute average absolute gap between bound and test metric vs θ.

    Args:
        ds (pd.DataFrame): Dataset split in half into train/test.
        metric (str): One of {'standard','FP','FN','FPR','FNR'}.
        k (int): grid size.
        delta (float): Confidence level.

    Returns:
        float: Mean absolute difference between bound and empirical metric.
    """
    train_set = ds.iloc[: int(len(ds) / 2)]
    train_set = (
        train_set.sort_values("kappa", ascending=True).reset_index(drop=True).copy()
    )
    test_set = ds.iloc[int(len(ds) / 2) :]

    thetas, bounds = bound_evo_w_theta(
        metric, train_set, delta, theta_min=theta_min, theta_max=theta_max, k=k
    )
    emp_metrics = []
    for theta in thetas:
        try:
            selected_set = test_set.loc[test_set.kappa >= theta].copy()
            emp_metrics.append(emp_metric(selected_set, metric=metric))
        except ValueError:
            emp_metrics.append(np.nan)

    return mean_abs_diff(bounds, emp_metrics)


def our_bound(selected_samples, metric, delta=DELTA, k=K):
    """
    Compute our guaranteed conditional metric bound (to be compared to external reference)

    Args:
        selected_samples: samples with confidence higher than threshold
        metric: one of the selective metrics 'standard', 'FPR', 'FNR' etc...
        delta: probability control
        n: size of Sn, the original dataset

    Returns:
        float: bound from proposition 2-3
    """

    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }

    if metric in ("FPR", "SP"):
        selected_samples = selected_samples.loc[(selected_samples.y_true == 0)]
    elif metric in ("FNR", "SE"):
        selected_samples = selected_samples.loc[(selected_samples.y_true == 1)]
    elif metric == "PPV":
        selected_samples = selected_samples.loc[(selected_samples.y_pred == 1)]

    selected_errs_count = emp_errs_count(
        selected_samples, loss=metric_loss_mapping[metric]
    )

    n = selected_samples.shape[0]
    if selected_errs_count == 0:
        # no mistake on selected subset => no mistake as next iters, so b* is stuck at 1-delta^(1/n)
        return np.nan

    b = B_star(delta / k, selected_errs_count, n)  # see formula in Corollary 2
    if (n == 0) or (selected_errs_count == n):
        b = 1  # by definition of B^*(.) in Proposition 1.
    if b == 1:
        return np.nan

    if metric in ["SP", "SE", "PPV"]:
        b = 1 - b

    return b


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
        return a - b

    if detailed:
        return a, b
    return a + b


def run_one_seed_all_targets(
    sgp_df,
    s,
    metric_targets,
    delta=DELTA,
    theta_min=0.5,
    theta_max=1,
    metric="standard",
    mode="multistart",
    eps=0,
    delta_test=0.05,
):
    """One split, one grid pass, every target read off it.

    A target counts as failed only when the held-out set gives significant evidence
    that the population metric exceeds r*, i.e. when its one-sided Clopper-Pearson
    lower confidence limit is above r* + eps.

    Returns:
        (np.ndarray, np.ndarray): (valid, failed) 0/1 arrays aligned with
        `metric_targets`, summable across seeds.  `failed` is 0 on splits where the
        search returned nothing, so sums divide by num_seed.
    """

    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }

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
            failed[i] = bool(t + eps < test_metric)
            continue

        # significance-based check: CP lower limit on the held-out estimate
        covered_test_set = test_set.loc[test_set.kappa >= theta_star]
        n_cov = covered_test_set.shape[0]
        c_cov = emp_errs_count(covered_test_set, loss=metric_loss_mapping[metric])
        lower = beta.ppf(delta_test, c_cov, n_cov - c_cov + 1) if c_cov else 0.0
        failed[i] = bool(t + eps < lower)

    return valid, failed


######## conformal bounds (Angelopoulos)
mapping = {
    "FNR": lambda y, p, c: (y == c, p != c),
    "FPR": lambda y, p, c: (y != c, p == c),
    "FDR": lambda y, p, c: (p == c, y != c),
    "error": lambda y, p, c: (np.ones(len(y), bool), y != p),
}


def conformal_bound(
    metric="FNR",
    theta=0.5,
    calibration_set=None,
    delta=DELTA / K,
    positive_class=1,
    min_subgroup=25,
):
    """See LTT sec 3.2. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control
    Angelopoulos et al. (2022)"""
    d = calibration_set
    cond, err = mapping[metric](d["y_true"].values, d["y_pred"].values, positive_class)
    sub = cond & (d["kappa"].values >= theta)
    m, k = int(sub.sum()), int((err & sub).sum())
    if m < min_subgroup or k == m:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, m - k))
