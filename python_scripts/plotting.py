import os
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math
import scipy.special
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)



def metric_plots(results, ylabel: str, 
                 xlim1: list = [0, 1], xlim2: list = [0, 1],
                 ylim: list = [0, 1],
                 title : str =  None):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Coverage plot
    axs[0].plot(results.test_coverage, results.metric_target, label='Target', c='red')
    axs[0].plot(results.test_coverage, results.metric_bound, label='Guaranteed', linestyle='--')
    axs[0].plot(results.train_coverage, results.train_metric, label='On training set', linestyle='--')
    axs[0].plot(results.test_coverage, results.test_metric, label='On test set')
    axs[0].set_xlabel('Coverage')
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlim(xlim1[0], xlim1[1])
    axs[0].set_ylim(ylim[0], ylim[1])
    #axs[0].legend()
    axs[0].grid()

    # Subplot 2: Theta plot
    axs[1].plot(results.theta_star, results.metric_target, label='Target', c='red')
    axs[1].plot(results.theta_star, results.metric_bound, label='Guaranteed', linestyle='--')
    axs[1].plot(results.theta_star, results.train_metric, label='On training set', linestyle='--')
    axs[1].plot(results.theta_star, results.test_metric, label='On test set')
    axs[1].set_xlabel(r'$\theta^*$')
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim(xlim2[0], xlim2[1])
    axs[1].set_ylim(ylim[0], ylim[1])
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    if title:
        plt.title(title, loc = 'center')
    plt.show()




def metric_plots_with_imbalance(all_propor_dfs, proportions,
                                ylabel: str, ylim: list = [0, 1],
                                xlim1: list = [0, 1], xlim2: list = [0, 1],
                                title : str= None):
    
    # Set up colormaps
    cmap_blue = cm.get_cmap('Blues')
    cmap_orange = cm.get_cmap('Oranges')
    cmap_gray = cm.get_cmap('Grays')

    # Normalize for colorbar
    norm = mcolors.Normalize(vmin=1, vmax=50)
    sm = cm.ScalarMappable(cmap=cmap_gray, norm=norm)
    sm.set_array([])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Track for legend proxies
    proxy_blue = plt.Line2D([0], [0], color=cmap_blue(0.8), label='Guaranteed', linestyle='--')
    proxy_orange = plt.Line2D([0], [0], color=cmap_orange(0.8), label='On test set')

    for proportion_1 in proportions:
        norm_value = (10+proportion_1*100) / 60
        color_blue = cmap_blue(norm_value)
        color_orange = cmap_orange(norm_value)

        results = all_propor_dfs.loc[all_propor_dfs.proportion_1 == proportion_1]

        # Coverage subplot
        axs[0].plot(results.test_coverage, results.metric_bound, color=color_blue, linestyle='--')
        axs[0].plot(results.test_coverage, results.test_metric, color=color_orange)

        # Theta* subplot
        axs[1].plot(results.theta_star, results.metric_bound, color=color_blue, linestyle='--')
        axs[1].plot(results.theta_star, results.test_metric, color=color_orange)

    # Labels and limits
    axs[0].set_xlabel('Coverage')
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlim(xlim1)
    axs[0].set_ylim(ylim)
    axs[0].grid(True)
    
    axs[1].set_xlabel(r'$\theta^*$')
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim(xlim2)
    axs[1].set_ylim(ylim)
    axs[1].grid(True)
    axs[1].legend(handles=[proxy_blue, proxy_orange])

    # Shared colorbar
    cbar = fig.colorbar(sm, ax=axs[1], shrink=0.95)
    cbar.set_label('Proportion of 1s (%)')

    plt.tight_layout()
    if title:
        plt.title(title, loc = 'center')
    plt.show()

