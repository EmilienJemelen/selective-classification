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

from scripts.math_utils import *
from scripts.preprocessing import *



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



def upper_bound_denominator(metric, selected_samples, delta):
    """
    denominator of upper bound for metrics FPR, FNR, VPP, SE, SP
    """
    d2 = np.sqrt(-np.log(delta/2)/(2*selected_samples.shape[0]))
    if (metric == 'PPV'):
        d1 = selected_samples.y_pred.sum()/selected_samples.shape[0]
    else:
        d1 = selected_samples.y_true.sum()/selected_samples.shape[0]
    
    if metric in ['FPR', 'SP']:
        return 1-d1-d2
    else: # FNR, SE, PPV
        return d1-d2



def bound(b, selected_samples, delta, metric):
    if metric in ['standard', 'FP', 'FN']:
        return b
    elif metric in ['FPR', 'FNR']:
        return b/upper_bound_denominator(metric, selected_samples, delta)
    else: # PPV, SE, SP
        return 1 - b/upper_bound_denominator(metric, selected_samples, delta)



def terminal_condition(selected_errs_count, bound, r_star, tolerance, metric):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return True if ((selected_errs_count == 0) and (bound > r_star + tolerance)) else False
    elif metric in ['PPV', 'SE', 'SP']:
        return True if ((selected_errs_count == 0) and (bound < r_star - tolerance)) else False
    else:
        raise ValueError('invalid metric')



def decrease_theta(bound, r_star, metric):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return True if bound < r_star else False
    elif metric in ['PPV', 'SE', 'SP']:
        return True if bound > r_star else False
    else:
        raise ValueError('invalid metric')



def SGR_dicho(delta, r_star, Sm, k, metric, tolerance=1e-2, union=False):
    """
    General Selection with Guaranteed Risk (SGR) algorithm
    """
    
    m = Sm.shape[0]
    zmin = 0
    zmax = m
    desired_prob = delta/k if union else delta
    metric_loss_mapping = {'standard': 'standard',
                           'FP': 'FP', 'FN': 'FN',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'}

    for i in range(k+1):
        
        z = int((zmin+zmax)/2)
        theta = Sm.SR[z]
        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])

        b = B_star(desired_prob, 
                   selected_errs_count,
                   selected_samples.shape[0])
         
        B = bound(b, selected_samples, delta, metric)
        
        if terminal_condition(selected_errs_count, B, r_star, tolerance, metric):
        # algo can't get bound any closer if already 0 mistakes => r* can't be guaranteed
            return {}    
        if decrease_theta(B, r_star, metric):
            zmax = z
        else:
            zmin = z

    if B > 0 and B < 1: 
        # FNR bound can be negative (and so SE bound > 1) if proportion of 1s in selected samples close to zero:
        # in this case 
        return {'theta_star' : theta,
                'bound' : B,
                'delta' : delta,
                'coverage' : selected_samples.shape[0]/m,
                'emp_metric' : emp_metric(selected_samples, metric = metric)}
    else:
        return {}



def SGR_dicho_at_risks(train_set,test_set, k, delta = 0.001, 
                       metric_targets = [i/100 for i in range(1,15)], 
                       metric = 'standard', union=False):
    """
    Compute SGR metric bound and empirical metric on training and test sets, for different metric targets (r_star)
    wp of exceeding r_star < delta
    """
    results = []
    for r_star in metric_targets:

        sgr_dico = SGR_dicho(delta, r_star, train_set, k, metric = metric, union = union)
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



###
### UNFINISHED BUSINESS BELOW
###





def SGR_greedy(delta, r_star, k, Sm, metric='standard', tolerance=1e-2, union=False):
    """
    Selection with Guaranteed Metric (SGM) greedy search algorithm
    """
    
    m = Sm.shape[0]
    desired_prob = delta/k if union else delta
    
    best_bound = np.inf
    best_theta = 0
    best_coverage = 0
    emp_metric = 0

    metric_loss_mapping = {'standard': 'standard',
                           'FP': 'FP', 'FN': 'FN',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'}

    for theta in Sm.SR: # SR col must be ranked ascendingly
        
        selected_samples = Sm.loc[Sm.SR >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])

        bound = B_star(desired_prob, 
                       selected_errs_count,
                       selected_samples.shape[0])
         
        if metric in ['FPR','FNR']:
            bound = bound/upper_bound_denominator(metric, selected_samples,delta)
        elif metric in ['PPV', 'SE', 'SP']:
            bound = 1 - bound/upper_bound_denominator(metric, selected_samples,delta)
        else: # metric is standard 0/1 
            pass 

        if abs(bound - r_star) < abs(best_bound - r_star):
            best_bound = bound
            best_theta = theta
            best_coverage = selected_samples.shape[0]/m
            emp_metric = emp_metric(selected_samples, metric = metric)
            
    if abs(best_bound - r_star) < tolerance:
        return {'theta_star' : best_theta,
                'bound' : best_bound,
                'delta' : delta,
                'coverage' : best_coverage,
                'emp_metric' : emp_metric}
    else:
        return {}