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

# on regarde la distribution des incertitudes et on regarde le quantile des seuils : 2% des seuils d'incertitude etc 
# y reste entre 0 et 1 : je fais une transformation linéaire pour que y soit entre 0 et 1, si le minimum est 0.2, je fais y = (y - 0.2) / (1 - 0.2) 
# l'étape de transformation de y se fait à la fin du calcul des quantiles
# pour x : pour quantile in 1 à 100 (boucle for), on calcule le quantile d'ordre q des incertitudes, on calcule l'accuracy sur les échantillons dont l'incertitude est inférieure à ce quantile (q%), on trace le graphe

def accuracy_threshold(Y_hat, Y, values, metric_name="mesure", num_quantiles=100, color='blue', display=True):
    """
    Trace la courbe accuracy en fonction du quantile sur une métrique donnée.
    Les abscisses correspondent aux quantiles (de 0 à 1), donc x=0.1 signifie 10% des valeurs les plus faibles.
    L'ordonnée (accuracy) est normalisée linéairement entre 0 et 1 (min=0).
    Renvoie quantiles, thresholds (valeurs de la métrique correspondant aux quantiles) et accuracies (originales).
    """
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    if torch.is_tensor(values): values = values.cpu().numpy()

    quantiles = np.linspace(0, 1, num_quantiles)
    thresholds = np.quantile(values, quantiles)
    accuracies = []

    for thresh in thresholds:
        mask = values < thresh
        if mask.sum() == 0:
            accuracies.append(np.nan)
        else:
            acc = (Y_hat[mask] == Y[mask]).mean()
            accuracies.append(acc)

    accuracies = np.array(accuracies)
    # Normalisation linéaire des accuracies entre 0 et 1, min=0
    min_acc = np.nanmin(accuracies)
    max_acc = np.nanmax(accuracies)
    if max_acc > min_acc:
        accuracies_norm = (accuracies - min_acc) / (max_acc - min_acc)
    else:
        accuracies_norm = accuracies.copy()

    if display:
        plt.figure(figsize=(7, 4))
        plt.plot(quantiles, accuracies_norm, color=color, linewidth=1.5, label="Accuracy normalisée")
        plt.xlabel(f"Quantiles de {metric_name} (proportion d'échantillons inclus)")
        plt.ylabel("Accuracy normalisée")
        plt.title(f"Accuracy en fonction du quantile de {metric_name}")
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)
        plt.show()
    return quantiles, thresholds, accuracies  # retourne les quantiles, thresholds et accuracy originale (y)
    
def isotonic_regression(quantiles, accuracies, color='seagreen', display=True):
    """
    Affiche la courbe d'accuracy originale normalisée, sa correction isotone décroissante normalisée,
    et colore l'aire entre les deux courbes en fonction des quantiles.
    Les abscisses (quantiles) sont comprises entre 0 et 1.
    display : si True, affiche la courbe, sinon ne l'affiche pas
    """
    mask = ~np.isnan(accuracies)  # ignorer les NaN
    if mask.sum() == 0:
        print("Pas de valeurs valides pour la régression isotone.")
        return None

    # Normalisation linéaire des accuracies entre 0 et 1, min=0
    acc = accuracies[mask]
    min_acc = np.nanmin(acc)
    max_acc = np.nanmax(acc)
    if max_acc > min_acc:
        acc_norm = (acc - min_acc) / (max_acc - min_acc)
    else:
        acc_norm = acc.copy()

    iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
    iso_accuracies = iso_reg.fit_transform(quantiles[mask], acc)
    # Normalisation de la courbe isotone sur la même échelle
    min_iso = np.nanmin(iso_accuracies)
    max_iso = np.nanmax(iso_accuracies)
    if max_iso > min_iso:
        iso_norm = (iso_accuracies - min_iso) / (max_iso - min_iso)
    else:
        iso_norm = iso_accuracies.copy()

    if display:
        plt.figure(figsize=(7, 4))
        plt.plot(quantiles[mask], acc_norm, label='Accuracy normalisée', color='brown', linewidth=1.5)
        plt.plot(quantiles[mask], iso_norm, label='Régression isotone normalisée', color=color, linewidth=1.5)
        plt.fill_between(quantiles[mask], acc_norm, iso_norm, color='red', alpha=0.3, label="Aire entre les courbes")
        plt.xlabel("Quantiles")
        plt.ylabel("Accuracy normalisée")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Accuracy normalisée et régression isotone")
        plt.legend()
        plt.grid(True)
        plt.show()

    return iso_accuracies


def monotonic_rearrangement(arr, quantiles=None, accuracies=None, color='deeppink', display=True):
    """
    Applique le réarrangement monotone décroissant à la liste arr (en fonction des quantiles).
    Si quantiles et accuracies sont fournis, affiche la comparaison avant/après, normalisées, et colore l'aire entre les deux courbes.
    display : si True, affiche la courbe, sinon ne l'affiche pas
    """
    arr = np.array(arr).copy()
    for i in range(1, len(arr)):
        if np.isnan(arr[i-1]):
            continue
        if arr[i] > arr[i-1]:
            arr[i] = arr[i-1]

    if display and quantiles is not None and accuracies is not None:
        # Normalisation linéaire des accuracies entre 0 et 1, min=0
        mask = ~np.isnan(accuracies)
        acc = np.array(accuracies)[mask]
        q = np.array(quantiles)[mask]
        arr_masked = arr[mask]

        min_acc = np.nanmin(acc)
        max_acc = np.nanmax(acc)
        if max_acc > min_acc:
            acc_norm = (acc - min_acc) / (max_acc - min_acc)
        else:
            acc_norm = acc.copy()

        min_arr = np.nanmin(arr_masked)
        max_arr = np.nanmax(arr_masked)
        if max_arr > min_arr:
            arr_norm = (arr_masked - min_arr) / (max_arr - min_arr)
        else:
            arr_norm = arr_masked.copy()

        plt.figure(figsize=(7, 4))
        plt.plot(q, acc_norm, label='Accuracy normalisée', color='brown', linewidth=1.5)
        plt.plot(q, arr_norm, label='Réarrangement monotone normalisé', color=color, linewidth=1.5)
        plt.fill_between(q, acc_norm, arr_norm, color='red', alpha=0.3, label="Aire entre les courbes")
        plt.xlabel("Quantiles")
        plt.ylabel("Accuracy normalisée")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Accuracy normalisée et réarrangement monotone")
        plt.legend()
        plt.grid(True)
        plt.show()
    return arr

def monotonicity_penalty(quantiles, accuracies, method='isotonic'):
    """
    Calcule la somme des aires de violation de monotonie entre la courbe d'accuracy
    et sa version monotone (isotone ou réarrangement), en fonction des quantiles.
    method : 'isotonic' ou 'rearrangement'
    """
    mask = ~np.isnan(accuracies)
    x = quantiles[mask]
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
