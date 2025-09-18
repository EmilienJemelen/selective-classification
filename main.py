import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN, CNN_MCdropout
from train import train_model, evaluate
from utils import generate_mc_outputs
import config
from dico import dico_layers
from reproducibility import set_global_seed
from accuracy import accuracy_threshold

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, valloader

def main():
    set_global_seed(42)
    device = config.DEVICE
    save_path = "best_model.pt"
    trainloader, testloader, valloader = load_data(config.BATCH_SIZE)
    
    base_model = SimpleCNN()
    if os.path.exists(save_path):
        print("Chargement du modèle sauvegardé")
        base_model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Pas de modèle sauvegardé, on entraîne le modèle")
        base_model = train_model(base_model, trainloader, valloader, device, epochs=20, save_path=save_path)
        base_model.load_state_dict(torch.load(save_path, map_location=device))  # recharge les meilleurs poids

    print(f"Utilisation du dico_layers : {dico_layers}")
    model = CNN_MCdropout(base_model, dico_layers=dico_layers).to(device)

    test_loss, test_acc = evaluate(model, testloader, device)
    print(f"Final Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

    # Tester sur un batch
    X, Y = next(iter(valloader))
    X, Y = X.to(device), Y.to(device)
    T = config.MC_T
    
    user_metrics = input(
        "Quelles métriques voulez-vous calculer ? (mc_estimate, variance_predicted, variance_max, predictive_entropy_predicted, predictive_entropy_max, relative_norm)\n"
        "Vous pouvez en choisir plusieurs, séparées par des virgules : ")
    user_metrics = [m.strip() for m in user_metrics.split(",")]
    outputs, mean_probs, metric_values, elapsed_forward, elapsed_metrics = generate_mc_outputs(
        model, X, T, metrics=user_metrics, labels=Y
    )
    print(f"Liste des métriques choisies par l'utilisateur : {user_metrics}")
    for metric in user_metrics:
        print(f"Métrique choisie : {metric}")
        print(f"Résultat : {metric_values[metric]}\n")

    # Appel accuracy_threshold pour variance_predicted et predictive_entropy_predicted si présents
    Y_hat = mean_probs.argmax(1)
    for metric in ["variance_predicted", "variance_max", "predictive_entropy_predicted", "predictive_entropy_max"]:
        if metric in metric_values:
            accuracy_threshold(Y_hat, Y, metric_values[metric], metric_name=metric, num_thresholds=1000)
    

if __name__ == "__main__":
    main()