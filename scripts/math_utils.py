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
    return np.exp(x)/np.exp(x).sum()



def binomial_log(m, j):
    return gammaln(m + 1) - gammaln(j + 1) - gammaln(m - j + 1)



def binom_sum(b,e,m):
    """
    binomial sum of term b in [0,1]
    with binomial coefs "j among m"
    for j in 0,1,...,e
    it is the proba of doing at most e errors among m Bernoulli iid experiences with error proba b
    """
    if e < m:
        v = np.array([
            np.exp(
                binomial_log(m, j)
                +j*np.log(b)
                +(m-j)*np.log(1-b)) for j in range(e+1)])
        return np.sum(v)
    elif e == m:
        return 1
    else:
        raise ValueError
    


def B_star(delta, e, m, eps=1e-6, b1=0, b2=1):
    """
    b_star recursive computation by dichotomy search over [0,1], given probability delta
    approximate solution at eps (in terms of FUN images)
    """
    if e==m:
        raise ValueError
    if e==0:
        return 1-delta**(1/m)
    
    b = (b1+b2)/2 # middle of segment
    if abs(binom_sum(b,e,m) - delta) < eps:
        return b
    elif binom_sum(b,e,m) <= delta - eps:
        return B_star(delta, e, m, b1=b1, b2=b)
    else:
        return B_star(delta, e, m, b1=b, b2=b2)
    


def integers_log_spacing(start, end, num_points = 40):
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
    log_bias = 1 - (1 - lin)**4  # This compresses more values at the start

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
