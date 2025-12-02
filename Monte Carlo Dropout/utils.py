import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def mc_predict_mean_probs(model, X, T=1000, verbose=True):
    """
    Effectue T passes Monte Carlo Dropout sur le modèle pour le batch X,
    retourne la moyenne des probabilités softmax sur T passes.
    Permet d'estimer l'incertitude du modèle via dropout activé à l'inférence.
    """
    model.train()
    probs_list = []
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(T), disable=not verbose):
            logits_t = model(X)
            probs_t = F.softmax(logits_t, dim=1)
            probs_list.append(probs_t.unsqueeze(0))
    elapsed = time.time() - start_time
    probs_mc = torch.cat(probs_list, dim=0)
    model.eval()
    if verbose:
        print(f"Temps total: {elapsed:.2f} s  |  Temps moyen par passe: {elapsed/T:.4f} s")
    return probs_mc.mean(0), elapsed

def predictive_entropy_multi_class(mean_probs):
    mean_probs = torch.clamp(mean_probs, min=1e-12, max=1-1e-12)  # éviter log(0)
    return -(mean_probs * torch.log(mean_probs)).sum(dim=1)  # entropie par échantillon

def generate_mc_outputs(model, X, T=1000, metrics=["mc_estimate"], labels=None, verbose=True):
    """
    Effectue T passes Monte Carlo Dropout sur le modèle pour le batch X,
    calcule les métriques d'incertitude spécifiées.
    Retourne les sorties de chaque passe, la moyenne des probabilités softmax,
    un dictionnaire des valeurs de métriques, et les temps de calcul.
    """
    model.train()
    outputs = []
    mean_probs = None 

    start_forward = time.time()
    with torch.no_grad():
        for _ in tqdm(range(T), disable=not verbose):
            out = model(X)
            outputs.append(out.unsqueeze(0))
    elapsed_forward = time.time() - start_forward

    outputs = torch.cat(outputs, dim=0)  # [T, batch, num_classes]
    results = {}
    elapsed_metrics = {}

    # Calcul de la classe prédite initialement (premier passage sans dropout)
    with torch.no_grad():
        first_logits = model(X)
        first_probs = torch.softmax(first_logits, dim=1)
        initial_pred = first_probs.argmax(dim=1)  # [batch], classe prédite pour chaque échantillon
        
    all_probs = torch.softmax(outputs, dim=2)  
    mean_probs = all_probs.mean(dim=0) 

    for metric in metrics:
        start_metric = time.time()

        if metric == "mc_estimate":
            results[metric] = mean_probs

        elif metric == "variance_predicted": # var des probas softmax de la classe prédite initialement
            idx = initial_pred.unsqueeze(0).expand(T, -1) # matrice avec T lignes égales à initial_pred
            selected_probs = all_probs.gather(2, idx.unsqueeze(2)).squeeze(2) # pour chaque batch, on prend la colonne de la classe prédite initialement
            var_pred_class = selected_probs.var(dim=0) 
            results["variance_predicted_mean"] = var_pred_class.mean().item()
            results["variance_predicted"] = var_pred_class

        elif metric == "variance_max": # variance des probas max (toutes classes confondues) 
            max_probs, _ = all_probs.max(dim=2)  # shape [T, batch]
            var_max = max_probs.var(dim=0)  # shape [batch]
            results["variance_max_mean"] = var_max.mean().item()
            results["variance_max"] = var_max

        elif metric == "predictive_entropy_binary_predicted": # PE pour la classe prédite initialement      
            idx = initial_pred.unsqueeze(1)          
            selected_mean_probs = mean_probs.gather(1, idx).squeeze(1) # on sélectionne la proba moyenne de la classe prédite
            # entropie binaire pour la classe prédite (p*log(p) + (1-p)*log(1-p))
            entropies_pred = -(selected_mean_probs * (selected_mean_probs + 1e-12).log() +
                               (1 - selected_mean_probs) * ((1 - selected_mean_probs + 1e-12).log()))
            results["predictive_entropy_binary_predicted_mean"] = entropies_pred.mean().item()
            results["predictive_entropy_binary_predicted"] = entropies_pred

        elif metric == "predictive_entropy_binary_max": # PE de la probabilité max (toutes classes confondues)
            max_probs, _ = mean_probs.max(dim=1)  # shape [batch]
            # entropie binaire associée à p_max
            entropies_max = -(max_probs * (max_probs + 1e-12).log() +
                          (1 - max_probs) * ((1 - max_probs + 1e-12).log()))
            results["predictive_entropy_binary_max_mean"] = entropies_max.mean().item()
            results["predictive_entropy_binary_max"] = entropies_max

        elif metric == "predictive_entropy_predicted":
            # Entropie complète de la distribution moyenne
            entropies_multi_pred = predictive_entropy_multi_class(mean_probs)
            results["predictive_entropy_predicted_mean"] = entropies_multi_pred.mean().item()
            results["predictive_entropy_predicted"] = entropies_multi_pred

        elif metric == "predictive_entropy_max":
            # Entropie MOYENNE des entropies par passe (plus sensible aux variations)
            entropies_per_pass = torch.zeros(all_probs.size(1), device=all_probs.device)
            for t in range(T):
                entropies_per_pass += predictive_entropy_multi_class(all_probs[t])
            entropies_multi_max = entropies_per_pass / T
            results["predictive_entropy_max_mean"] = entropies_multi_max.mean().item()
            results["predictive_entropy_max"] = entropies_multi_max

        elif metric == "relative_norm":
            if labels is None:
                raise ValueError("labels doivent être fournis pour relative_norm")
            labels_onehot = F.one_hot(labels, num_classes=mean_probs.size(1)).float()
            diff_norm = torch.norm(mean_probs - labels_onehot, dim=1)
            denom = torch.max(torch.norm(mean_probs, dim=1), torch.norm(labels_onehot, dim=1))
            relative_norm = diff_norm / (denom + 1e-12)
            results[metric + "_mean"] = relative_norm.mean().item()
            results[metric] = relative_norm

        else:
            raise ValueError(f"Métrique {metric} non reconnue")

        elapsed_metrics[metric] = time.time() - start_metric

    model.eval()

    if verbose:
        print(f"Temps forward pass: {elapsed_forward:.2f}s  |  Temps moyen par passe: {elapsed_forward/T:.4f}s")
        for m, t in elapsed_metrics.items():
            print(f"Temps calcul métrique '{m}': {t:.6f}s")

    return outputs, mean_probs, results, elapsed_forward, elapsed_metrics
