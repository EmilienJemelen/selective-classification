import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN, CNN_MCdropout
from train import train_model, evaluate
from utils import generate_mc_outputs
import config

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader

def main():
    device = config.DEVICE
    trainloader, valloader = load_data(config.BATCH_SIZE)
    
    model = SimpleCNN().to(device)
    if os.path.exists(config.MODEL_PATH):
        print(f"Chargement du modèle depuis {config.MODEL_PATH}")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Aucun modèle trouvé. Entraînement en cours...")
        model = train_model(model, trainloader, valloader, device, config.EPOCHS, config.MODEL_PATH)
        print("Entraînement terminé et modèle sauvegardé.")
    
    val_loss, val_acc = evaluate(model, valloader, device)
    print(f"Validation Accuracy : {val_acc:.4f} - Loss : {val_loss:.4f}")

    model = CNN_MCdropout(model).to(device)
    
    # Tester sur un batch
    X, Y = next(iter(valloader))
    X, Y = X.to(device), Y.to(device)
    
    user_metrics = input(
    "Quelles métriques voulez-vous calculer ? (mc_estimate, variance, predictive_entropy, relative_norm)\n"
    "Vous pouvez en choisir plusieurs, séparées par des virgules : ")
    user_metrics = [m.strip() for m in user_metrics.split(",")]
    outputs, mean_probs, metric_values = generate_mc_outputs(model, X, T=1000, metrics=user_metrics, labels=Y)
    print(f"Liste des métriques choisies par l\'utilisateur : {user_metrics}")
    for metric in user_metrics:
        print(f"Métrique choisie : {metric}")
        print(f"Résultat : {metric_values[metric]}\n")
    

if __name__ == "__main__":
    main()