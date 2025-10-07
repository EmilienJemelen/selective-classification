import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import pandas as pd 


def accuracy_threshold(Y_hat, Y, values, metric_name="mesure", num_thresholds=1000, color='blue', display=True):
    """
    Trace la courbe accuracy en fonction du seuil sur une métrique donnée.
    Y_hat : prédictions (tensor ou array)
    Y : labels réels (tensor ou array)
    values : vecteur par échantillon de la métrique utilisée comme seuil (variance, predictive entropy, etc.)
    metric_name : nom de la métrique (sera utilisé dans le titre)
    num_thresholds : nombre de seuils à tester
    display : si True, affiche la courbe, sinon ne l'affiche pas
    """
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    if torch.is_tensor(values): values = values.cpu().numpy()

    thresholds = np.linspace(values.min(), values.max(), num_thresholds)
    accuracies = []

    counts = []  # Ajout pour compter les échantillons sélectionnés

    for thresh in thresholds:
        mask = values < thresh
        if mask.sum() == 0:
            accuracies.append(np.nan)
        else:
            acc = (Y_hat[mask] == Y[mask]).mean()
            accuracies.append(acc)

    accuracies = np.array(accuracies)
    min_accuracy = np.nanmin(accuracies)

    if display:
        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, accuracies, color=color, linewidth=1.5)
        plt.axhline(y=min_accuracy, color='red', linestyle='--', label=f"Accuracy min = {min_accuracy:.4f}")
        plt.xlabel(f"Seuil sur {metric_name}")
        plt.ylabel("Accuracy (Y_hat = Y)")
        plt.title(f"Accuracy en fonction du seuil de {metric_name}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()

    return thresholds, accuracies


def isotonic_regression(thresholds, accuracies, color='seagreen', display=True):
    """
    Affiche la courbe d'accuracy originale et sa correction isotone décroissante.
    display : si True, affiche la courbe, sinon ne l'affiche pas
    """
    mask = ~np.isnan(accuracies) # ignorer les NaN
    if mask.sum() == 0:
        print("Pas de valeurs valides pour la régression isotone.")
        return None
    iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip') # décroissante, appelée fonction antitone 
    iso_accuracies = iso_reg.fit_transform(thresholds[mask], accuracies[mask]) # renvoie les valeurs corrigées aux seuils valides uniquement (sans NaN)

    if display:
        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, accuracies, label='Accuracy originale', color=color, linewidth=1.5)
        plt.plot(thresholds[mask], iso_accuracies, label='Accuracy monotone (régression isotone)', color='brown', linewidth=1.5)
        plt.fill_between(thresholds[mask], iso_accuracies, accuracies[mask], color='red', alpha=0.3, label='Violations de monotonie')
        min_iso = np.min(iso_accuracies)
        plt.axhline(y=min_iso, color='red', linestyle='--', label=f"Min régression isotone = {min_iso:.4f}")
        plt.xlabel('Seuil')
        plt.ylabel('Accuracy')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Correction monotone de la fonction d'accuracy par régression isotone")
        plt.legend()
        plt.grid(True)
        plt.show()

    return iso_accuracies # retourne les valeurs corrigées


def monotonic_rearrangement(arr, thresholds=None, accuracies=None, color='deeppink', display=True):
    """
    Applique le réarrangement monotone décroissant à la liste arr.
    Si thresholds et accuracies sont fournis, affiche la comparaison avant/après
    display : si True, affiche la courbe, sinon ne l'affiche pas
    """
    arr = np.array(arr).copy()
    for i in range(1, len(arr)):
        if np.isnan(arr[i-1]): # ignorer NaN
            continue
        if arr[i] > arr[i-1]: # si la valeur courante est plus grande que la précédente, on la remplace
            arr[i] = arr[i-1]
    if display and thresholds is not None and accuracies is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, accuracies, label='Accuracy originale', color=color, linewidth=1.5)
        plt.plot(thresholds, arr, label='Accuracy monotone (réarrangement)', color='brown', linewidth=1.5)
        plt.fill_between(thresholds, arr, accuracies, color='red', alpha=0.3, label='Violations de monotonie')
        min_val = np.nanmin(arr)
        plt.axhline(y=min_val, color='red', linestyle='--', label=f"Min = {min_val:.4f}")
        plt.xlabel('Seuil')
        plt.ylabel('Accuracy')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Correction monotone de l'accuracy par réarrangement monotone")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return arr # renvoie le tableau corrigé


def monotonicity_penalty(thresholds, accuracies, method='isotonic'):
    """
    Calcule la somme des aires de violation de monotonie entre la courbe d'accuracy
    et sa version monotone (isotone ou réarrangement).
    method : 'isotonic' ou 'rearrangement'
    """
    mask = ~np.isnan(accuracies)
    x = thresholds[mask]
    y1 = accuracies[mask]

    if method == 'isotonic':
        iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
        y2 = iso_reg.fit_transform(x, y1)
    elif method == 'rearrangement':
        arr = y1.copy()
        for i in range(1, len(arr)):
            if arr[i] > arr[i-1]:
                arr[i] = arr[i-1]
        y2 = arr
    else:
        raise ValueError("method doit être 'isotonic' ou 'rearrangement'")

    penalty = np.trapz(np.abs(y1 - y2), x)  # aire totale entre les deux courbes
    return penalty