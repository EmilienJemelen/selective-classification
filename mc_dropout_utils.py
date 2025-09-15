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

    outputs = torch.cat(outputs, dim=0)
    results = {}
    elapsed_metrics = {}

    with torch.no_grad():
        first_logits = model(X)
        first_probs = torch.softmax(first_logits, dim=1)
        initial_pred = first_probs.argmax(dim=1)

    all_probs = torch.softmax(outputs, dim=2)
    mean_probs = all_probs.mean(dim=0)

    for metric in metrics:
        start_metric = time.time()

        if metric == "mc_estimate":
            results[metric] = mean_probs

        elif metric == "variance_predicted":
            idx = initial_pred.unsqueeze(0).expand(T, -1)
            selected_probs = all_probs.gather(2, idx.unsqueeze(2)).squeeze(2)
            var_pred_class = selected_probs.var(dim=0)
            results["variance_predicted_mean"] = var_pred_class.mean().item()
            results["variance_predicted"] = var_pred_class

        elif metric == "variance_max":
            max_probs, _ = all_probs.max(dim=2)
            var_max = max_probs.var(dim=0)
            results["variance_max_mean"] = var_max.mean().item()
            results["variance_max"] = var_max

        elif metric == "predictive_entropy_predicted":
            idx = initial_pred.unsqueeze(1)
            selected_mean_probs = mean_probs.gather(1, idx).squeeze(1)
            entropies_pred = -(selected_mean_probs * (selected_mean_probs + 1e-12).log() +
                               (1 - selected_mean_probs) * ((1 - selected_mean_probs + 1e-12).log()))
            results["predictive_entropy_predicted_mean"] = entropies_pred.mean().item()
            results["predictive_entropy_predicted"] = entropies_pred

        elif metric == "predictive_entropy_max":
            max_probs, _ = mean_probs.max(dim=1)
            entropies = -(max_probs * (max_probs + 1e-12).log() +
                          (1 - max_probs) * ((1 - max_probs + 1e-12).log()))
            results["predictive_entropy_max_mean"] = entropies.mean().item()
            results["predictive_entropy_max"] = entropies

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

def accuracy_threshold(Y_hat, Y, values, metric_name="mesure", num_thresholds=1000, color='blue'):
    """
    Trace la courbe accuracy en fonction du seuil sur une métrique donnée.
    Y_hat : prédictions (tensor ou array)
    Y : labels réels (tensor ou array)
    values : vecteur par échantillon de la métrique utilisée comme seuil (variance, predictive entropy, etc.)
    metric_name : nom de la métrique (sera utilisé dans le titre)
    num_thresholds : nombre de seuils à tester
    color : couleur de la courbe
    """
    if torch.is_tensor(Y_hat): Y_hat = Y_hat.cpu().numpy()
    if torch.is_tensor(Y): Y = Y.cpu().numpy()
    if torch.is_tensor(values): values = values.cpu().numpy()

    thresholds = np.linspace(values.min(), values.max(), num_thresholds)
    accuracies = []

    for thresh in thresholds:
        mask = values < thresh
        if mask.sum() == 0:
            accuracies.append(np.nan)
        else:
            acc = (Y_hat[mask] == Y[mask]).mean()
            accuracies.append(acc)

    accuracies = np.array(accuracies)
    min_accuracy = np.nanmin(accuracies)

    plt.figure(figsize=(7,4))
    plt.plot(thresholds, accuracies, color=color, linewidth=2)
    plt.axhline(y=min_accuracy, color='red', linestyle='--', label=f"Accuracy min = {min_accuracy:.4f}")
    plt.xlabel(f"Seuil sur {metric_name}")
    plt.ylabel("Accuracy (Y_hat = Y)")
    plt.title(f"Accuracy en fonction du seuil de {metric_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
