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



def generate_imbalanced_datasets(balanced_dataset, proportions, label_col='y_true', seed=None):
    """
    Create imbalanced datasets by downsampling class-1 samples while keeping all class-0 samples.

    Assumes the input dataset is initially class-balanced and all values in `v` are <= 0.5.

    Args:
        train_set (pd.DataFrame): Input DataFrame with binary labels.
        proportions (list of float): Target proportions of class-1 samples in the output datasets.
        label_col (str): Name of the label column. Default is 'y_true'.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list of pd.DataFrame: List of datasets with adjusted class-1 proportions.
    """
    np.random.seed(seed)
    
    # Split the dataset
    df_pos = balanced_dataset[balanced_dataset[label_col] == 1]
    df_neg = balanced_dataset[balanced_dataset[label_col] == 0]
    
    N0 = len(df_neg)
    datasets = []

    for p in proportions:
        N1 = int(p * balanced_dataset.shape[0])
        N1 = min(N1, len(df_pos))  # Just in case

        df_pos_sampled = df_pos.sample(n=N1, random_state=seed, replace=False)
        df_combined = pd.concat([df_neg, df_pos_sampled], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)

        datasets.append(df_combined)

    return datasets



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



