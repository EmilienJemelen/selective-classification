import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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



def prepare_sgr_dico(dataloader, model, device, T):
    
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
    

'''
def B_star(delta, e, m, eps=1e-6, b1=0, b2=1):
    """
    b_star iterative computation by dichotomy search over [0,1], given probability delta
    approximate solution at eps (in terms of FUN images)
    """
    b = (b1+b2)/2 # middle of segment
    while abs(binom_sum(b,e,m) - delta) >= eps:
        if binom_sum(b,e,m) <= delta - eps:
            b2 = b
        else:
            b1 = b
        b = (b1+b2)/2
    
    return b
'''


def B_star(delta, e, m, eps=1e-6, b1=0, b2=1):
    """
    b_star recursive computation by dichotomy search over [0,1], given probability delta
    approximate solution at eps (in terms of FUN images)
    """
    if e==m:
        raise ValueError
    b = (b1+b2)/2 # middle of segment
    if abs(binom_sum(b,e,m) - delta) < eps:
        return b
    elif binom_sum(b,e,m) <= delta - eps:
        return B_star(delta, e, m, b1=b1, b2=b)
    else:
        return B_star(delta, e, m, b1=b, b2=b2)



def emp_errs_count(samples, loss = 'standard'):
    if loss == 'standard':
        return (samples.y_pred != samples.y_true).sum()
    elif loss == 'typeI':
        return ((samples.y_pred == 1) & (samples.y_true == 0)).sum()
    elif loss == 'typeII':
        return ((samples.y_pred == 0) & (samples.y_true == 1)).sum()
    else:
        raise ValueError("loss must be either 'standard', 'typeI' or 'typeII'")
    


def emp_risk(samples, loss = 'standard'):
    if loss == 'standard':
        return emp_errs_count(samples)/samples.shape[0]
    elif loss == 'typeI':
        return emp_errs_count(samples, loss = 'typeI')/samples.y_pred.sum()
    elif loss == 'typeII':
        return emp_errs_count(samples, loss = 'typeII')/(samples.y_pred == 0).sum()
    else:
        raise ValueError("loss must be either 'standard', 'typeI' or 'typeII'")



def SGR(delta, r_star, Sm, loss = 'standard'):
    """
    Selection with Guaranteed Risk (SGR) algorithm
    from Geifman el Yaniv 2017
    """
    
    m = Sm.shape[0]
    zmin = 0
    zmax = m
    for i in range(int(np.log(m)/np.log(2))):
        
        z = int((zmin+zmax)/2)
        theta = Sm.SR[z]
        selected_samples = Sm.loc[Sm.SR > theta]
        selected_errs_count = emp_errs_count(selected_samples, loss = loss)

        if selected_errs_count == 0: # if no error, then we stop since algo is stuck anyways...
            try:
                return {'theta_star' : old_theta,
                        'b_star' : old_b_star,
                        'delta' : delta,
                        'coverage' : old_selected_samples.shape[0]/m,
                        'risk' : emp_risk(old_selected_samples, loss = loss)}
            except:
                return {}

        else:
            b_star = B_star(delta/int(np.log(m)/np.log(2)), 
                            selected_errs_count,
                            selected_samples.shape[0])
            if b_star < r_star:
                zmax = z
            else:
                zmin = z

            old_b_star = b_star
            old_theta = theta # saving old theta before overwriting
            old_selected_samples = selected_samples # same idea
        
    return {'theta_star' : theta,
            'b_star' : b_star,
            'delta' : delta,
            'coverage' : selected_samples.shape[0]/m,
            'risk' : emp_risk(selected_samples, loss = loss)}



def SGR_at_risks(train_set,test_set, delta = 0.001, desired_risks = [i/100 for i in range(1,15)], loss = 'standard'):
    """
    Compute SGR risk bound and actual risks on training and test sets, for different target risks (r_star)
    wp of exceeding r_star < delta
    """
    results = []
    for r_star in desired_risks:

        sgr_dico = SGR(delta, r_star, train_set, loss = loss)   
        if sgr_dico != {}:
            theta_star = sgr_dico['theta_star']
            covered_test_set = test_set.loc[test_set.SR > theta_star]
            results.append({'desired_risk' : r_star,
                            'risk_bound' : sgr_dico['b_star'],
                            'train_risk' : sgr_dico['risk'],
                            'train_coverage' : sgr_dico['coverage'],
                            'test_risk' : emp_risk(covered_test_set, loss = loss),
                            'test_coverage' : covered_test_set.shape[0]/test_set.shape[0]})
    
    return pd.DataFrame(results)



def sample_with_proportion(df, label_col, proportion_1, sample_size):
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