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
        B = b
    elif metric in ['FPR', 'FNR']:
        B = b/abs(upper_bound_denominator(metric, selected_samples, delta, m))
    else: # PPV, SE, SP
        B = 1 - b/abs(upper_bound_denominator(metric, selected_samples, delta ,m))
    if (B>=1) or (B<=0):
        return np.nan
    else:
        return B


def terminal_condition(selected_errs_count, bound, r_star, metric, tolerance=1e-2):
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



def satisfied(bound, r_star, metric, tolerance):
    if metric in ['standard', 'FP', 'FN', 'FPR', 'FNR']:
        return (True if bound < r_star + tolerance else False)
    else:
        return (True if bound > r_star - tolerance else False)



def sgp_dicho(delta, r_star, Sn, k, metric, union=False, tolerance=1e-3):
    """
    General Selection with Guaranteed Performances (sgp) algorithm
    """
    
    m = Sn.shape[0]
    zmin = 0
    zmax = m
    desired_prob = delta/k if union else delta

    for i in range(k):
        
        z = int((zmin+zmax)/2)
        theta = Sn.kappa[z]
        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric)

        b = B_star(desired_prob, 
                   selected_errs_count,
                   selected_samples.shape[0])

        if (selected_samples.shape[0]==0) or ((selected_errs_count==0) and (b >= r_star)): #terminal condition
            return {}

        if b < r_star:
            zmax = z
        else:
            zmin = z

    if not satisfied(b, r_star, metric, tolerance):
        return {}

    return {'theta_star' : theta,
            'bound' : b,
            'delta' : delta,
            'coverage' : selected_samples.shape[0]/m,
            'emp_metric' : emp_metric(selected_samples, metric = metric)}


        
def sgp_greedy_search(delta, r_star, Sn, metric, steps=100, tolerance=1e-3):
    """
    Greedy search for LOWEST theta with bound close enough to r* 
    """
    metric_loss_mapping = {'standard': 'standard',
                           'FP':'FP', 'FN':'FN',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'} 
    Sn = Sn.sort_values('kappa', ascending=True)
    kappas = np.array(Sn.kappa)
    
    for theta in np.linspace(kappas[0], kappas[-1], steps):

        try:
            if selected_samples.shape[0] == 0:
                return {}
        except:
            pass

        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])

        if selected_errs_count == 0: 
            # no mistake on selected subset => no mistake as next iters, so b* is stuck at 1-delta^(1/n)
            return {}

        b = B_star(delta, 
                   selected_errs_count,
                   selected_samples.shape[0])    
        if b==1:
            return {}
            
        B = bound(b, selected_samples, delta, metric, m=Sn.shape[0])
        if np.isnan(B):
            return {}

        if satisfied(B, r_star, metric, tolerance):
            return {'theta_star' : theta,
                    'bound' : B,
                    'delta' : delta,
                    'coverage' : selected_samples.shape[0]/Sn.shape[0],
                    'emp_metric' : emp_metric(selected_samples, metric = metric)}
        
    return {} # if we never found satisfactory B..



def sgp_at_targets(train_set,test_set, k=None, delta = 0.001, 
                   metric_targets = [i/100 for i in range(1,15)], 
                   metric = 'standard', union=False, 
                   mode='greedy', steps=100):
    """
    Compute sgp metric bound and empirical metric on training and test sets, for different metric targets (r_star)
    wp of exceeding r_star < delta
    """
    results = []
    for r_star in metric_targets:

        if mode == 'dicho':
            sgp_dico = sgp_dicho(delta, r_star, train_set, k, metric = metric, union = union)
        elif mode == 'greedy':
            sgp_dico = sgp_greedy_search(delta, r_star, train_set, metric, steps=steps)
        else:
            raise ValueError('mode should be either "greedy" or "dicho"')
        
        if sgp_dico != {}:
            theta_star = sgp_dico['theta_star']
            covered_test_set = test_set.loc[test_set.kappa > theta_star]
            if covered_test_set.shape[0] > 0:
                test_metric = emp_metric(covered_test_set, metric = metric)
            else:
                test_metric = np.nan
            results.append({'metric_target' : r_star,
                            'metric_bound' : sgp_dico['bound'],
                            'theta_star' : theta_star,
                            'train_metric' : sgp_dico['emp_metric'],
                            'train_coverage' : sgp_dico['coverage'],
                            'test_metric' : test_metric,
                            'test_coverage' : covered_test_set.shape[0]/test_set.shape[0]})
    
    return pd.DataFrame(results)



def sgp_at_targets_on_imbalanced_sets(proportions_of_1, metric_targets, 
                                      sgp_df, delta, mode='dicho',
                                      greedy_search_steps_num=50, metric='standard'):
    """
    Run sgp on datasets with varying class-1 proportions.

    Creates imbalanced datasets from `sgp_df`, runs sgp on each, and collects results.

    Args:
        proportions_of_1 (list of float): Target class-1 proportions.
        metric_targets (dict): Target metric values.
        sgp_df (pd.DataFrame): Input balanced dataset with 'y_true' and 'kappa' columns.
        greedy_search_steps_num (int): Number of greedy search steps.
        delta (float): control proba
        metric: chosen metric

    Returns:
        pd.DataFrame: Results with class-1 proportions.
    """
    all_propor_dfs = pd.DataFrame()
    imbalanced_datasets = generate_imbalanced_datasets(sgp_df, proportions_of_1, seed=0)

    for proportion_1, imbalanced_set in zip(proportions_of_1, imbalanced_datasets):

        train_set_ = imbalanced_set.iloc[:int(imbalanced_set.shape[0]/2)]
        train_set_ = train_set_.sort_values('kappa', ascending=True).reset_index(drop=True).copy()
        test_set_ = imbalanced_set.iloc[int(imbalanced_set.shape[0]/2):]

        results = sgp_at_targets(train_set_, test_set_, k=int(np.log2(train_set_.shape[0])),
                                delta=delta, metric_targets=metric_targets, metric=metric,
                                mode=mode, steps=greedy_search_steps_num)
        results['proportion_1'] = proportion_1
        all_propor_dfs = pd.concat([all_propor_dfs, results]).reset_index(drop=True)

    return all_propor_dfs



def bound_evo_w_theta(metric, Sn, delta, steps=100):

    metric_loss_mapping = {'standard': 'standard',
                           'FP': 'FP', 'FN': 'FN',
                           'FPR': 'FP', 'FNR': 'FN',
                           'PPV': 'FP', 'SE': 'FN',
                           'SP': 'FP'}
    Sn = Sn.sort_values('kappa', ascending=True)
    kappas = sorted(np.array(Sn.kappa))
    bounds, thetas = [], np.linspace(kappas[0], kappas[-1], steps)

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = metric_loss_mapping[metric])
        if (selected_errs_count==0):
            break

        b = B_star(delta, 
                   selected_errs_count,
                   selected_samples.shape[0])
        if b==1:
            break
        
        B = bound(b, selected_samples, delta, metric, m=Sn.shape[0])
        if np.isnan(B):
            break
        bounds.append(B) 

    while len(bounds) < len(thetas):
        bounds.append(np.nan)

    return thetas, bounds



def reachable_bounds(metrics_list, Sn, delta, steps=100):
    res_dico = {}

    # thetas and coverages coordinates
    kappas = sorted(np.array(Sn.kappa))
    thetas = np.linspace(kappas[0], kappas[-1], steps)
    res_dico['thetas'] =  sorted(thetas)
    res_dico['coverages'] = sorted([Sn.loc[Sn.kappa >= theta].shape[0]/Sn.shape[0] for theta in thetas],reverse=True)
    # metrics bounds with respect to thetas
    for metric in metrics_list:
        _, bounds = bound_evo_w_theta(metric, Sn, delta, steps=steps)
        res_dico[metric] = bounds

    return res_dico



def pos_propor_w_theta(Sn, steps=100):

    Sn = Sn.sort_values('kappa', ascending=True)
    kappas = np.array(Sn.kappa)
    pos_propor, thetas = [], np.linspace(kappas[0], kappas[-1], steps)

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        pos_propor.append(selected_samples.y_true.sum()/selected_samples.shape[0])

    return thetas, pos_propor



def runtime(sim_df, mode:str='dicho', greedy_steps:int=20):
    """
    study of time to run dicho search vs greedy search (with different step_num, more steps mean more acute results)
    """
    t0 = datetime.now()
    if mode=='dicho':
        res = sgp_dicho(delta=1e-3, r_star=0.05, 
                        Sn=sim_df, k=int(np.log2(sim_df.shape[0])),
                        metric='standard', union=False)
    elif mode=='greedy':
        res = sgp_greedy_search(delta=1e-3, r_star=0.05, Sn=sim_df, 
                                metric='standard', steps=greedy_steps)
    else:
        raise ValueError('mode should either be dicho or greedy')
    t1 = datetime.now()
    return (t1-t0).seconds




def joint_control(metrics_and_targets, sgp_df, delta, plot=False, steps=100):

    """
    fun to compute g_theta satisfying target bounds for a set of metrics given by the user
    if plot==True, then the function plots each theta interval satisfying target bound, for each metric in the specified set
    """

    metric_sign_mapping = {'standard': '<',
                           'FP': '<', 'FN': '<',
                           'FPR': '<', 'FNR': '<',
                           'PPV': '>', 'SE': '>',
                           'SP': '>'}
    y_proj = -0.01
    projection_handles = []

    # Use a colormap to assign a unique color to each metric
    num_metrics = len(metrics_and_targets)
    color_map = cm.get_cmap('tab10', num_metrics)
    colors = [color_map(i) for i in range(num_metrics)]
    segments_per_metric = {key: [] for key in metrics_and_targets.keys()}

    if plot:
        plt.figure()

    for i, (metric, target) in enumerate(metrics_and_targets.items()):
        thetas, bounds = bound_evo_w_theta(metric, sgp_df, delta, steps=steps)
        color = colors[i]

        if metric_sign_mapping[metric] == '>':
                mask = (np.array(bounds) > target)
        else:
            mask = (np.array(bounds) < target)

        segments = get_segments(thetas, mask)
        segments_per_metric[metric] = segments

        if plot:
            plt.plot(thetas, bounds, color=color, label=f'{metric} bound', linewidth=1.5)
            plt.axhline(y=target, color=color, linestyle='--', label=f'{metric} target')
            plt.xlabel(r'$\theta$')
            plt.ylabel('Metric bounds')
            plt.tick_params(axis='y')

            for (x_start, x_end) in segments:
                plt.hlines(y_proj*i, x_start, x_end, colors=color, linewidth=2)

            # Legend handle for projections
            projection_handles.append(
                Line2D([0], [0], color=color, linewidth=2, label=r'$\theta$ '+ f'/ {metric} {metric_sign_mapping[metric]} {target}')
            )

    if plot:
        plt.ylim(bottom=y_proj - 0.02)
        plt.legend(handles=projection_handles, loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        intersected_intervals = compute_all_interval_intersections(segments_per_metric)
        return {'theta_intervals' : intersected_intervals,
                'best_theta' : best_theta(intersected_intervals)}



def mean_abs_diff(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Mask to filter out nan entries
    not_masked = ~np.isnan(u) & ~np.isnan(v)
    
    # If no valid entries, return nan
    if not np.any(not_masked):
        return np.nan
    
    diffs = np.abs(u[not_masked] - v[not_masked])
    return np.mean(diffs)



def ABC(ds, metric, steps=30, delta=1e-3):
    """
    metric must be in 'standard', 'FP', 'FN', 'FPR', 'FNR'
    """
    train_set = ds.iloc[:int(len(ds)/2)]
    train_set = train_set.sort_values('kappa', ascending=True).reset_index(drop=True).copy()
    test_set = ds.iloc[int(len(ds)/2):]

    thetas, bounds = bound_evo_w_theta(metric, train_set, delta, steps=steps)
    emp_metrics = []
    for theta in thetas:
        try:
            selected_set = test_set.loc[test_set.kappa >= theta].copy()
            emp_metrics.append(emp_metric(selected_set, metric=metric))
        except ValueError:
            emp_metrics.append(np.nan)

    return mean_abs_diff(bounds, emp_metrics)