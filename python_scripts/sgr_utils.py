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

from python_scripts.math_utils import *
from python_scripts.preprocessing import *



def emp_errs_count(samples, loss = 'standard'):
    if loss == 'standard':
        return (samples.y_pred != samples.y_true).sum()
    elif loss == 'FP':
        return ((samples.y_pred == 1) & (samples.y_true == 0)).sum()
    elif loss == 'FN':
        return ((samples.y_pred == 0) & (samples.y_true == 1)).sum()
    else:
        raise ValueError("metric must be either 'standard', 'FP' or 'FN'")
    


def emp_metric(samples, metric = 'standard'):
    if samples.shape[0] == 0:
        raise ValueError('no sample in dataset')
    if metric == 'standard':
        return emp_errs_count(samples)/samples.shape[0]
    elif metric == 'FP':
        return emp_errs_count(samples, loss = 'FP')/samples.shape[0]
    elif metric == 'FN':
        return emp_errs_count(samples, loss = 'FN')/samples.shape[0]
    elif metric == 'FPR': 
        return emp_errs_count(samples, loss = 'FP')/(1- samples.y_true).sum()
    elif metric == 'FNR': 
        return emp_errs_count(samples, loss = 'FN')/samples.y_true.sum()
    elif metric == 'PPV':
        return (samples.y_pred * samples.y_true).sum()/samples.y_pred.sum()
    elif metric == 'SE':
        return (samples.y_pred * samples.y_true).sum()/samples.y_true.sum()
    elif metric == 'SP':
        return ((1 - samples.y_pred) * (1 - samples.y_true)).sum()/(1 - samples.y_true).sum()
    else:
        raise ValueError("metric must be in 'standard', 'FP','FN','FPR','FNR','PPV','SE','SP'")



def upper_bound_denominator(metric, selected_samples, delta, m):
    """
    denominator of upper bound for metrics FPR, FNR, VPP, SE, SP
    """
    d2 = np.sqrt(-m*np.log(delta/2))/selected_samples.shape[0]
    if (metric == 'PPV'):
        d1 = selected_samples.y_pred.sum()/selected_samples.shape[0]
    else:
        d1 = selected_samples.y_true.sum()/selected_samples.shape[0]
    
    if metric in ['FPR', 'SP']:
        return 1-d1-d2
    else: # FNR, SE, PPV
        return d1-d2



def bound(b, selected_samples, delta, metric, m):
    if metric in ['standard', 'FP', 'FN']:
        return b
    elif metric in ['FPR', 'FNR']:
        return b/abs(upper_bound_denominator(metric, selected_samples, delta, m))
    else: # PPV, SE, SP
        return 1 - b/abs(upper_bound_denominator(metric, selected_samples, delta ,m))



def terminal_condition(selected_errs_count, bound, r_star, xi, metric):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return True if ((selected_errs_count == 0) and (bound > r_star + xi)) else False
    elif metric in ['PPV', 'SE', 'SP']:
        return True if ((selected_errs_count == 0) and (bound < r_star - xi)) else False
    else:
        raise ValueError('invalid metric')



def decrease_theta(bound, r_star, metric):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return True if bound < r_star else False
    elif metric in ['PPV', 'SE', 'SP']:
        return True if bound > r_star else False
    else:
        raise ValueError('invalid metric')



def SGR_dicho(delta, r_star, Sm, k, metric, xi=1e-3, union=False):
    """
    General Selection with Guaranteed Risk (SGR) algorithm
    """
    
    m = Sm.shape[0]
    zmin = 0
    zmax = m
    desired_prob = delta/k if union else delta

    for i in range(k):
        
        z = int((zmin+zmax)/2)
        theta = Sm.SR[z]
        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric)

        b = B_star(desired_prob, 
                   selected_errs_count,
                   selected_samples.shape[0])

        if (selected_samples.shape[0]==0) or ((selected_errs_count==0) and (b > r_star + xi)): #terminal condition
            return {}

        if b < r_star:
            zmax = z
        else:
            zmin = z

    return {'theta_star' : theta,
            'bound' : b,
            'delta' : delta,
            'coverage' : selected_samples.shape[0]/m,
            'emp_metric' : emp_metric(selected_samples, metric = metric)}



def satisfaction(bound, r_star, metric, xi=5e-3):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return True if (bound <= r_star + xi) else False
    else:
        return True if (bound >= r_star - xi) else False

        

def SGR_greedy_search(delta, r_star, Sm, metric, xi=1e-3, steps=100):
    """
    Greedy search for LOWEST theta with bound close enough (xi) to r*
    """
    metric_loss_mapping = {'standard': 'standard',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'} 
    Sm = Sm.sort_values('SR', ascending=True)
    kappas = np.array(Sm.SR)
    
    for theta in np.linspace(kappas[0], kappas[-1], steps):

        try:
            if selected_samples.shape[0] == 0:
                return {}
        except:
            pass

        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])

        b = B_star(delta, 
                   selected_errs_count,
                   selected_samples.shape[0])    
        if b==np.inf:
            continue
            
        B = bound(b, selected_samples, delta, metric, m=Sm.shape[0])
        
        if satisfaction(B, r_star, metric, xi=xi):
            return {'theta_star' : theta,
                    'bound' : B,
                    'delta' : delta,
                    'coverage' : selected_samples.shape[0]/Sm.shape[0],
                    'emp_metric' : emp_metric(selected_samples, metric = metric)}
        
    return {} # if we never found satisfactory B..



def SGR_at_targets(train_set,test_set, k, delta = 0.001, 
                   metric_targets = [i/100 for i in range(1,15)], 
                   metric = 'standard', union=False, 
                   mode='greedy', steps=100):
    """
    Compute SGR metric bound and empirical metric on training and test sets, for different metric targets (r_star)
    wp of exceeding r_star < delta
    """
    results = []
    for r_star in metric_targets:

        if mode == 'dicho':
            sgr_dico = SGR_dicho(delta, r_star, train_set, k, metric = metric, union = union)
        elif mode == 'greedy':
            sgr_dico = SGR_greedy_search(delta, r_star, train_set, metric, xi=1e-3, steps=steps)
        else:
            raise ValueError('mode should be either "greedy" or "dicho"')
        
        if sgr_dico != {}:
            theta_star = sgr_dico['theta_star']
            covered_test_set = test_set.loc[test_set.SR > theta_star]
            if covered_test_set.shape[0] > 0:
                test_metric = emp_metric(covered_test_set, metric = metric)
            else:
                test_metric = np.nan
            results.append({'metric_target' : r_star,
                            'metric_bound' : sgr_dico['bound'],
                            'theta_star' : theta_star,
                            'train_metric' : sgr_dico['emp_metric'],
                            'train_coverage' : sgr_dico['coverage'],
                            'test_metric' : test_metric,
                            'test_coverage' : covered_test_set.shape[0]/test_set.shape[0]})
    
    return pd.DataFrame(results)



def SGR_at_targets_on_imbalanced_sets(proportions_of_1, metric_targets, 
                                      sgr_df, delta, mode='dicho',
                                      greedy_search_steps_num=50, metric='standard'):
    """
    Run SGR on datasets with varying class-1 proportions.

    Creates imbalanced datasets from `sgr_df`, runs SGR on each, and collects results.

    Args:
        proportions_of_1 (list of float): Target class-1 proportions.
        metric_targets (dict): Target metric values.
        sgr_df (pd.DataFrame): Input balanced dataset with 'y_true' and 'SR' columns.
        greedy_search_steps_num (int): Number of greedy search steps.
        delta (float): control proba
        metric: chosen metric

    Returns:
        pd.DataFrame: Results with class-1 proportions.
    """
    all_propor_dfs = pd.DataFrame()
    imbalanced_datasets = generate_imbalanced_datasets(sgr_df, proportions_of_1, seed=42)

    for proportion_1, imbalanced_set in zip(proportions_of_1, imbalanced_datasets):

        train_set_ = imbalanced_set.iloc[:2*int(imbalanced_set.shape[0]/3)]
        train_set_ = train_set_.sort_values('SR', ascending=True).reset_index(drop=True).copy()
        test_set_ = imbalanced_set.iloc[2*int(imbalanced_set.shape[0]/3):]

        results = SGR_at_targets(train_set_, test_set_, k=int(np.log2(train_set_.shape[0])),
                                delta=delta, metric_targets=metric_targets, metric=metric,
                                mode=mode, steps=greedy_search_steps_num)
        results['proportion_1'] = proportion_1
        all_propor_dfs = pd.concat([all_propor_dfs, results]).reset_index(drop=True)

    return all_propor_dfs



def bound_evo_w_theta(metric, Sm, delta, steps=100):

    metric_loss_mapping = {'standard': 'standard',
                           'FP': 'FP', 'FN': 'FN',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'}
    Sm = Sm.sort_values('SR', ascending=True)
    kappas = np.array(Sm.SR)
    bounds, thetas = [], []

    for theta in np.linspace(kappas[0], kappas[-1], steps):

        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])

        b = B_star(delta, 
                   selected_errs_count,
                   selected_samples.shape[0])        
        B = bound(b, selected_samples, delta, metric, m=Sm.shape[0])
        if selected_errs_count == 0:
            return thetas, bounds
        thetas.append(theta)
        bounds.append(B)

    return thetas, bounds



def pos_propor_w_theta(Sm, steps=100):

    Sm = Sm.sort_values('SR', ascending=True)
    kappas = np.array(Sm.SR)
    pos_propor, thetas = [], np.linspace(kappas[0], kappas[-1], steps)

    for theta in thetas:

        selected_samples = Sm.loc[Sm.SR >= theta]
        pos_propor.append(selected_samples.y_true.sum()/selected_samples.shape[0])

    return thetas, pos_propor



def runtime(sim_df, mode:str='dicho', greedy_steps:int=20):
    """
    study of time to run dicho search vs greedy search (with different step_num, more steps mean more acute results)
    """
    t0 = datetime.now()
    if mode=='dicho':
        res = SGR_dicho(delta=1e-3, r_star=0.05, 
                        Sm=sim_df, k=int(np.log2(sim_df.shape[0])),
                        metric='standard', xi=1e-3, union=False)
    elif mode=='greedy':
        res = SGR_greedy_search(delta=1e-3, r_star=0.05, Sm=sim_df, 
                                metric='standard', steps=greedy_steps)
    else:
        raise ValueError('mode should either be dicho or greedy')
    t1 = datetime.now()
    return (t1-t0).seconds