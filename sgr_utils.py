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


def filter_classes(dataset, allowed_classes):
    """
    Function to filter CIFAR dataset by class labels
    """
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_classes]
    return Subset(dataset, indices)



def prepare_sgr_dico(dataloader, model, device, T):
    """
    Prepare dataframe containing exactly the required features to train SGR module
    true class of each sample, model predicted class, 
    softmax response (SR) or any other confidence function output 
    """
    sgr_dico = {'y_true' : np.array([]),
                'y_pred' : np.array([]),
                'SR' : np.array([])}
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            batch_preds = model(images)
            softmax_values = F.softmax(batch_preds/T, dim=1) 
            softmax_responses = torch.max(softmax_values, dim=1)[0].cpu().numpy()
            _, predicted_classes = torch.max(batch_preds, 1)
            predicted_classes = predicted_classes.cpu().numpy()

            sgr_dico['y_true'] = np.concatenate((sgr_dico['y_true'], labels.cpu().numpy()))
            sgr_dico['y_pred'] = np.concatenate((sgr_dico['y_pred'], predicted_classes))
            sgr_dico['SR'] = np.concatenate((sgr_dico['SR'], softmax_responses))

    return sgr_dico



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



def emp_errs_count(samples, metric = 'standard'):
    if metric == 'standard':
        return (samples.y_pred != samples.y_true).sum()
    elif (metric == 'FP') or (metric == 'FPR'):
        return ((samples.y_pred == 1) & (samples.y_true == 0)).sum()
    elif (metric == 'FN') or (metric == 'FNR'):
        return ((samples.y_pred == 0) & (samples.y_true == 1)).sum()
    else:
        raise ValueError("metric must be either 'standard', 'FP' or 'FN'")
    


def emp_risk(samples, metric = 'standard'):
    if samples.shape[0] == 0:
        raise ValueError('no sample in dataset')
    if metric == 'standard':
        return emp_errs_count(samples)/samples.shape[0]
    elif metric == 'FP':
        return emp_errs_count(samples, metric = 'FP')/samples.shape[0]
    elif metric == 'FN':
        return emp_errs_count(samples, metric = 'FN')/samples.shape[0]
    elif metric == 'FPR': 
        return emp_errs_count(samples, metric = 'FP')/(samples.y_true == 0).sum()
    elif metric == 'FPR': 
        return emp_errs_count(samples, metric = 'FN')/samples.y_true.sum()
    else:
        raise ValueError("metric must be either 'standard', 'FP' or 'FN'")



def upper_bound_denominator(metric, selected_samples, delta):
    """
    denominator of upper bound for rates metrics (FPR, FNR, VPP)
    """
    if metric == 'FPR':
        d1 = selected_samples.y_true.sum()/selected_samples.shape[0]
        d2 = np.sqrt(-np.log(delta/2)/(2*selected_samples.shape[0]))
        return(1-d1-d2)
    elif metric == 'FNR':
        # a écrire
        return
    elif metric == 'PPV':
        # a écrere
        return
    else:
        raise ValueError('metric should be FPR or FNR or PPV')



def SGR(delta, r_star, Sm, k, metric='standard', 
        tolerance=2e-3, union=True):
    """
    Selection with Guaranteed Risk (SGR) algorithm
    """
    
    m = Sm.shape[0]
    zmin = 0
    zmax = m
    desired_prob = delta/k if union else delta

    for i in range(k+1):
        
        z = int((zmin+zmax)/2)
        theta = Sm.SR[z]
        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, metric = metric)

        bound = B_star(desired_prob, 
                        selected_errs_count,
                        selected_samples.shape[0])
         
        if metric in ['FPR','FNR','PPV']:
            bound = bound/upper_bound_denominator(metric, selected_samples,delta)
            
        if bound < r_star:
            zmax = z
        else:
            zmin = z

        if ((selected_errs_count == 0) and (bound > r_star)):
            # algo can't get bound any lower if already 0 mistakes => r* can't be guaranteed
            return {}

        if abs(bound - r_star) < tolerance:
            return {'theta_star' : theta,
                'bound' : bound,
                'delta' : delta,
                'coverage' : selected_samples.shape[0]/m,
                'risk' : emp_risk(selected_samples, metric = metric)}
        
    return {}



def SGR_at_risks(train_set,test_set, k, delta = 0.001, 
                 desired_risks = [i/100 for i in range(1,15)], 
                 metric = 'standard', union = True):
    """
    Compute SGR risk bound and actual risks on training and test sets, for different target risks (r_star)
    wp of exceeding r_star < delta
    """
    results = []
    for r_star in desired_risks:

        sgr_dico = SGR(delta, r_star, train_set, k, metric = metric, union = union)
        if sgr_dico != {}:
            theta_star = sgr_dico['theta_star']
            covered_test_set = test_set.loc[test_set.SR > theta_star]
            if covered_test_set.shape[0] > 0:
                test_risk = emp_risk(covered_test_set, metric = metric)
            else:
                test_risk = np.nan
            results.append({'desired_risk' : r_star,
                            'risk_bound' : sgr_dico['bound'],
                            'train_risk' : sgr_dico['risk'],
                            'train_coverage' : sgr_dico['coverage'],
                            'test_risk' : test_risk,
                            'test_coverage' : covered_test_set.shape[0]/test_set.shape[0]})
    
    return pd.DataFrame(results)



def sample_with_proportion(df, label_col, proportion_1, sample_size):
    """
    Sample a balanced dataset from a DataFrame according to a specified class proportion.

    Args:
        df (pd.DataFrame): The input DataFrame containing labeled data.
        label_col (str): Name of the column containing binary class labels (0 or 1).
        proportion_1 (float): Desired proportion of class 1 samples in the final sample (between 0 and 1).
        sample_size (int): Total number of samples to draw.

    Returns:
        pd.DataFrame: A new DataFrame containing the sampled data, shuffled and balanced according to the given proportion.
    """
    # Separate classes
    ones = df[df[label_col] == 1]
    zeros = df[df[label_col] == 0]

    # Calculate how many 1s and 0s you need
    n_ones = int(sample_size * proportion_1)
    n_zeros = sample_size - n_ones

    # Sample from each class
    sampled_ones = ones.sample(n=n_ones, random_state=42)
    sampled_zeros = zeros.sample(n=n_zeros, random_state=42)

    # Concatenate and shuffle
    sampled_df = pd.concat([sampled_ones, sampled_zeros]).sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled_df



def compute_mean_std(dataset_root):
    """
    Compute the mean and standard deviation of an image dataset for normalization purposes.

    Args:
        dataset_root (str): Root directory path of the image dataset (organized in class-specific subfolders).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors for the dataset (per channel).
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=dataset_root, transform=transform)
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std



def split_balanced_dataset(root, transform=None, train_size=0.3, val_size=0.2, seed=42):
    """
    Split a binary image classification dataset into balanced train, validation, and test subsets.

    Args:
        root (str): Path to the root directory of the dataset (organized in class-specific subfolders).
        transform (callable, optional): Transform to apply to the images. Defaults to None.
        train_size (float): Proportion of each class to allocate to the training set.
        val_size (float): Proportion of each class to allocate to the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Subset, Subset, Subset]: Train, validation, and test subsets of the dataset.
    """
    dataset = ImageFolder(root=root, transform=transform)
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)

    # Get indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Make sure we only have two classes (binary case)
    assert len(class_indices) == 2, "Expected binary classification dataset (2 classes)."

    # Find minimum number of samples across the two classes
    min_class_len = min(len(idxs) for idxs in class_indices.values())

    # Compute counts per split
    n_train = int(train_size * min_class_len)
    n_val = int(val_size * min_class_len)

    # Split per class
    train_indices, val_indices, test_indices = [], [], []

    for cls, idxs in class_indices.items():
        idxs = rng.permutation(idxs)
        train = idxs[:n_train]
        val = idxs[n_train:n_train + n_val]
        test = idxs[n_train + n_val:]
        
        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)

    # Shuffle final splits
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )



def count_labels(dataset):
    """
    Count the number of samples for each label (0 and 1) in a dataset.

    Args:
        dataset (Dataset or Subset): A PyTorch Dataset or Subset (e.g., from ImageFolder or a split subset).

    Returns:
        dict: A dictionary mapping class labels (0 and 1) to their respective counts.
    """
    targets = []
    
    # Handle Subset datasets
    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    counts = Counter(targets)
    return {label: counts.get(label, 0) for label in range(2)}



def integers_log_spacing(start, end, num_points = 40):
    """
    Returns a list of integers between `start` and `end` using roughly logarithmic spacing.

    Parameters:
    - start (int): Start of the range.
    - end (int): End of the range.
    - num_points (int): Number of points to sample (default is 40).

    Returns:
    - list[int]: Logarithmically spaced integers within the range.
    """
    all_ints = np.arange(start-1, end+1)

    # Create log-spaced indices over the range of indices (not the values themselves)
    log_indices = np.logspace(0, np.log10(len(all_ints)), num=num_points, base=10, dtype=int)
    # Remove duplicates and clip to valid range
    log_indices = np.unique(np.clip(log_indices, 0, len(all_ints) - 1))
    # Select from the original array
    fewer_ints = all_ints[log_indices]

    return fewer_ints.tolist()