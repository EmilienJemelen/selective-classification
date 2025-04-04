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
    v = np.array([
        np.exp(
            binomial_log(m, j)
            +j*np.log(b)
            +(m-j)*np.log(1-b)) for j in range(e+1)])
    return np.sum(v)



def B_star(delta, e, m, eps=1e-6, b1=0, b2=1):
    """
    b_star recursive computation by dichotomy search over [0,1], given probability delta
    approximate solution at eps (in terms of FUN images)
    """
    b = (b1+b2)/2 # middle of segment
    if abs(binom_sum(b,e,m) - delta) < eps:
        return b
    elif binom_sum(b,e,m) <= delta - eps:
        return B_star(delta, e, m, b1=b1, b2=b)
    else:
        return B_star(delta, e, m, b1=b, b2=b2)
    


def SGR(delta, r_star, Sm):
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
        selected_errs_count = (selected_samples.y_pred != selected_samples.y_true).sum()

        b_star = B_star(delta/int(np.log(m)/np.log(2)), 
                        selected_errs_count,
                        selected_samples.shape[0])
        if b_star < r_star:
            zmax = z
        else:
            zmin = z

    return {'theta_star' : theta,
            'b_star' : b_star,
            'delta' : delta,
            'coverage' : Sm.loc[Sm.SR > theta].shape[0]/m}