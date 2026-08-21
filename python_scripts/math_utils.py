import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import math
import scipy.special
from scipy.stats import beta, binom, norm, rankdata
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


def sotfmax(x):
    """
    x a vector of floats, returns softmaxes for these points
    """
    return np.exp(x) / np.exp(x).sum()


def binomial_log(n, j):
    """
    Return the log of the binomial coefficient "n choose j".

    Computed as:
        log(n! / (j! * (n - j)!))
      = gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)

    Parameters
    ----------
    n : int
        Total number of trials.
    j : int
        Number of successes.

    Returns
    -------
    float
        Logarithm of the binomial coefficient.
    """
    return gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)


def binom_sum(b, e, n):
    """
    binomial sum of term b in [0,1]
    with binomial coefs "j among n"
    for j in 0,1,...,e
    it is the proba of doing at most e errors among n Bernoulli iid experiences with error proba b
    """
    if e < n:
        v = np.array(
            [
                np.exp(binomial_log(n, j) + j * np.log(b) + (n - j) * np.log(1 - b))
                for j in range(e + 1)
            ]
        )
        return np.sum(v)
    elif e == n:
        return 1
    else:
        raise ValueError


def B_star(delta, e, n, b1=0, b2=1, eps=1e-5):
    """
    b_star recursive dichotomy
    approximate solution up to eps
    """

    if (e == n) or (n == 0):
        return 1

    return float(beta.ppf(1 - delta, e + 1, n - e))


def integers_log_spacing(start, end, num_points=40):
    """
    Returns a list of integers between `start` and `end`, spaced so that more points
    are concentrated toward the start of the range (log-distributed shape).

    Parameters:
    - start (int): Start of the range.
    - end (int): End of the range.
    - num_points (int): Number of points to sample (default is 40).

    Returns:
    - list[int]: Integers biased toward the beginning of the range.
    """
    if start >= end:
        raise ValueError("Start must be less than end.")
    # Generate a range from 0 (dense) to 1 (sparse)
    lin = np.linspace(0, 1, num_points)
    # Apply inverse exponential shape (log-bias toward the start)
    log_bias = 1 - (1 - lin) ** 4  # This compresses more values at the start
    # Scale to range
    values = start + (end - start) * log_bias
    values = np.round(values).astype(int)
    # Remove duplicates
    values = np.unique(np.clip(values, start, end))

    return values.tolist()


def integers_exp_spacing(start, end, num_points=40):
    """
    Returns a list of integers between `start` and `end`, spaced so that more points
    are concentrated toward the high end of the range.

    Parameters:
    - start (int): Start of the range.
    - end (int): End of the range.
    - num_points (int): Number of points to sample (default is 40).

    Returns:
    - list[int]: Integers biased toward the end of the range.
    """
    if start >= end:
        raise ValueError("Start must be less than end.")
    # Generate a range of indices from 0 to 1
    lin = np.linspace(0, 1, num_points)
    # Exponential bias toward 1 (end of the range)
    exp_bias = lin**4  # Tune the exponent for more/less bias
    # Scale to actual range
    values = start + (end - start) * exp_bias
    values = np.round(values).astype(int)
    # Remove duplicates
    values = np.unique(np.clip(values, start, end))

    return values.tolist()


def simulate_sgp_dataset(n, high_conf_propor=0.7, ranking=1.0, err_rate=0.19, seed=42):
    """Binary predictions with Beta-mixture confidence; errors concentrate at low kappa.

    `ranking` sets the kappa_f ranking quality (0 = none, np.inf = Hypothesis 1 holds)
    without changing the error rate nor the kappa distribution.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    match = rng.random(n) < high_conf_propor
    kappa = np.where(
        match,
        beta.rvs(9, 1, size=n, random_state=rng),
        beta.rvs(3, 2, size=n, random_state=rng),
    )
    z = norm.ppf(rankdata(-kappa) / (n + 1))
    score = z if np.isinf(ranking) else ranking * z + rng.normal(size=n)
    err = score >= np.quantile(score, 1 - err_rate)
    return pd.DataFrame(
        {"y_true": y_true, "y_pred": np.where(err, 1 - y_true, y_true), "kappa": kappa}
    )
