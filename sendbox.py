
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

def penalty_linear(accuracies):
    """
    Calcule la pénalité linéaire pour les montées de la courbe d'accuracy.
    La pénalité est la somme des valeurs absolues des montées (différences positives entre points consécutifs).
    """
    penalty = 0.0
    for i in range(1, len(accuracies)):
        if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
            continue
        diff = accuracies[i] - accuracies[i-1]
        if diff > 0:
            penalty += abs(diff)
    return penalty

def penalty_quadratic(accuracies):
    """
    Calcule la pénalité quadratique pour les montées de la courbe d'accuracy.
    La pénalité est la somme des carrés des montées (différences positives entre points consécutifs).
    """
    penalty = 0.0
    for i in range(1, len(accuracies)):
        if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
            continue
        diff = accuracies[i] - accuracies[i-1]
        if diff > 0:
            penalty += diff ** 2
    return penalty

def penalty_LOM(accuracies):
    """
    Calcule la pénalité LOM (Loss of Monotonicity) pour les montées de la courbe d'accuracy.
    La pénalité est le rapport entre la somme des montées et la somme des variations.
    0 signifie parfaitement monotone décroissante, 1 signifie totalement non monotone.
    """
    num = 0.0
    denom = 0.0
    for i in range(1, len(accuracies)):
        if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
            continue
        diff = accuracies[i] - accuracies[i-1]
        if diff > 0:
            num += diff
        denom += abs(diff)
    return num / denom if denom > 0 else 0.0

def penalty_count_violations(accuracies):
    """
    Calcule la pénalité par comptage des violations pour les montées de la courbe d'accuracy.
    La pénalité est le nombre de montées (différences positives entre points consécutifs).
    Retourne aussi le ratio de montées par rapport au nombre total de transitions.
    0 signifie parfaitement monotone décroissante, 1 signifie totalement non monotone.
    """
    count = 0
    for i in range(1, len(accuracies)):
        if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
            continue
        if accuracies[i] > accuracies[i-1]:
            count += 1
    return count, count / (len(accuracies)-1)

thresholds, accuracies = accuracy_threshold(Y_hat, Y, metric_values['variance_max'], metric_name="variance max", color="seagreen")

print("Pénalité linéaire :", penalty_linear(accuracies))
print("Pénalité quadratique :", penalty_quadratic(accuracies))
print("Indice LOM :", penalty_LOM(accuracies))
violations, prop_violations = penalty_count_violations(accuracies)
print(f"Nombre de violations : {violations} | Proportion : {prop_violations:.3f}")

# Tracer plusieurs courbes d'accuracy pénalisée selon différentes pénalités

plt.figure(figsize=(7, 4))
plt.plot(thresholds, accuracies, label="Accuracy brute", color="seagreen", linewidth=2)

# Courbe accuracy - pénalité linéaire cumulée
penalty_lin = np.zeros_like(accuracies)
for i in range(1, len(accuracies)):
    if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
        penalty_lin[i] = penalty_lin[i-1]
    else:
        diff = accuracies[i] - accuracies[i-1]
        penalty_lin[i] = penalty_lin[i-1] + (diff if diff > 0 else 0)
plt.plot(thresholds, accuracies - penalty_lin, label="Accuracy - pénalité linéaire cumulée", color="blue")

# Courbe accuracy - pénalité quadratique cumulée
penalty_quad = np.zeros_like(accuracies)
for i in range(1, len(accuracies)):
    if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
        penalty_quad[i] = penalty_quad[i-1]
    else:
        diff = accuracies[i] - accuracies[i-1]
        penalty_quad[i] = penalty_quad[i-1] + (diff**2 if diff > 0 else 0)
plt.plot(thresholds, accuracies - penalty_quad, label="Accuracy - pénalité quadratique cumulée", color="red")

# Courbe accuracy - proportion de violations (chaque point = nb violations jusqu'ici / i)
penalty_viol = np.zeros_like(accuracies)
count = 0
for i in range(1, len(accuracies)):
    if np.isnan(accuracies[i-1]) or np.isnan(accuracies[i]):
        penalty_viol[i] = penalty_viol[i-1]
    else:
        if accuracies[i] > accuracies[i-1]:
            count += 1
        penalty_viol[i] = count / i
plt.plot(thresholds, accuracies - penalty_viol, label="Accuracy - proportion violations", color="purple")

plt.xlabel("Seuil sur variance max")
plt.ylabel("Accuracy pénalisée")
plt.title("Accuracy pénalisée selon différentes mesures de pénalité (cumulées)")
plt.legend()
plt.grid(True)
plt.show()

