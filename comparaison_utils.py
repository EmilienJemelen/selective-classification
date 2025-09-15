from reproducibility import set_global_seed
set_global_seed(42)
from utils import dico_layers
from models import SimpleCNN
from comparaison_models import CNN_MCdropout, CNN_MCdropout_beforeReLU, CNN_MCdropout_torch
from mc_dropout_utils import mc_predict_mean_probs, generate_mc_outputs
import torch
import torch.nn.functional as F

def get_all_mc_models(base_model):
    model_after_relu = CNN_MCdropout(base_model, dico_layers=dico_layers)
    model_before_relu = CNN_MCdropout_beforeReLU(base_model, dico_layers=dico_layers)
    model_torch = CNN_MCdropout_torch(base_model, dico_layers=dico_layers)
    return model_after_relu, model_before_relu, model_torch

def eval_mc_metrics_on_models(base_model, X, labels, T=1000, metrics=None, verbose=True):
    if metrics is None:
        metrics = ["mc_estimate", "variance_predicted", "variance_max", "predictive_entropy_predicted", "predictive_entropy_max", "relative_norm"]
    models = get_all_mc_models(base_model)
    names = ["MCdropout_afterReLU", "MCdropout_beforeReLU", "MCdropout_torch"]
    results = {}
    for name, model in zip(names, models):
        print(f"\n--- {name} ---")
        _, _, metrics_dict, _, _ = generate_mc_outputs(model, X, T=T, metrics=metrics, labels=labels, verbose=verbose)
        for k, v in metrics_dict.items():
            print(f"{k}: {getattr(v, 'shape', v) if not isinstance(v, float) else f'{v:.6f}'}")
        results[name] = metrics_dict
    return results

def test_mc_dropout_batch(base_model, X, Y, T=1000, device=None, verbose=True):
    models = get_all_mc_models(base_model)
    names = ["MCdropout_afterReLU", "MCdropout_beforeReLU", "MCdropout_torch"]
    Y_hats, times = {}, {}
    for name, model in zip(names, models):
        if device:
            model, X_, Y_ = model.to(device), X.to(device), Y.to(device)
        else:
            X_, Y_ = X, Y
        probs, t = mc_predict_mean_probs(model, X_, T=T, verbose=verbose)
        Y_hat = probs.argmax(1)
        acc = (Y_hat == Y_).float().mean().item()
        print(f"{name} (t={t:.2f}s): {Y_hat.tolist()} | Acc: {acc:.4f}")
        Y_hats[name] = Y_hat.cpu() if hasattr(Y_hat, "cpu") else Y_hat
        times[name] = t
    print("Classes vraies:", Y.cpu().tolist() if hasattr(Y, "cpu") else Y.tolist())
    return Y_hats, times

def test_accuracy_per_class(Y_hat, Y, num_classes=10):
    """
    Calcule l'accuracy pour chaque classe.
    Pour chaque classe c, on sélectionne les indices où Y == c,
    puis on calcule la proportion de bonnes prédictions parmi ces indices.
    """
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    accs = {}
    for c in range(num_classes):
        idx = (Y == c)  # masque booléen : True là où la vraie classe est c
        if idx.sum() == 0:
            accs[c] = float('nan')
        else:
            # (Y_hat[idx] == Y[idx]) : tableau booléen des bonnes prédictions pour la classe c
            # .mean() : proportion de bonnes prédictions (accuracy) pour la classe c
            accs[c] = (Y_hat[idx] == Y[idx]).mean()
    return accs

def test_metrics_per_class(Y_hat, Y, metrics_dict, num_classes=10):
    """
    Pour chaque métrique (présente dans metrics_dict), retourne la liste des valeurs pour chaque classe.
    metrics_dict : dict {metric_name: tensor/array de taille [batch]}
    Retourne un dict {metric_name: {classe: liste_des_valeurs_pour_cette_classe}}
    """
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    results = {}
    for metric, values in metrics_dict.items():
        if torch.is_tensor(values): values = values.cpu().numpy()
        per_class = {}
        for c in range(num_classes):
            idx = (Y == c)
            per_class[c] = values[idx]  # tableau des valeurs pour la classe c (pas de moyenne)
        results[metric] = per_class
    return results

# Suggestions d'améliorations possibles :

# 1. Ajoute une fonction pour afficher joliment les résultats par classe (accuracy ou métriques)
def print_per_class_results(results, label="Accuracy"):
    """
    Affiche joliment les résultats par classe (dict {classe: valeur}).
    """
    for c, v in results.items():
        print(f"{label} classe {c}: {v:.4f}" if not isinstance(v, float) or not (v != v) else f"{label} classe {c}: nan")

# 2. Ajoute une fonction pour obtenir la matrice de confusion
def confusion_matrix(Y_hat, Y, num_classes=10):
    """
    Retourne la matrice de confusion (numpy array).
    La matrice de confusion est un tableau de taille [num_classes, num_classes] où
    l'élément [i, j] indique le nombre d'échantillons de la vraie classe i prédits comme classe j.
    - Ligne = vraie classe
    - Colonne = classe prédite
    """
    import numpy as np
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(Y, Y_hat):
        cm[t, p] += 1
    return cm