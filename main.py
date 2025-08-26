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
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, valloader

def main():
    device = config.DEVICE
    save_path = "best_model.pt"
    trainloader, testloader, valloader = load_data(config.BATCH_SIZE)
    
    # base_model = SimpleCNN().to(device)
    # if os.path.exists(config.MODEL_PATH):
    #     print(f"Chargement du modèle depuis {config.MODEL_PATH}")
    #     base_model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    # else:
    #     print("Aucun modèle trouvé. Entraînement en cours...")
    #     base_model = train_model(base_model, trainloader, valloader, device, config.EPOCHS, config.MODEL_PATH)
    #     print("Entraînement terminé et modèle sauvegardé.")
    
    # val_loss, val_acc = evaluate(base_model, valloader, device)
    # print(f"Validation Accuracy : {val_acc:.4f} - Loss : {val_loss:.4f}")

    # model = CNN_MCdropout(base_model).to(device)
    
    # Vérifie si les poids existent déjà
    base_model = SimpleCNN()
    if os.path.exists(save_path):
        print("Chargement du modèle sauvegardé")
        base_model.load_state_dict(torch.load(save_path, map_location=device))  # même architecture que celle qui a sauvegardé
    else:
        print("Pas de modèle sauvegardé, on entraîne le modèle")
        base_model = train_model(base_model, trainloader, valloader, device, epochs=20, save_path=save_path)
        base_model.load_state_dict(torch.load(save_path, map_location=device))  # recharge les meilleurs poids

    # Choix des couches à masquer par l'utilisateur
    user_layers = input(
        "Sur quelles couches voulez-vous appliquer le MC Dropout ? "
        "(choisissez parmi conv1, conv2, conv3, fc1, séparées par des virgules) : ")
    mc_layers = [layer.strip() for layer in user_layers.split(',') if layer.strip() in ['conv1','conv2','conv3','fc1']]

    model = CNN_MCdropout(base_model, mc_layers=mc_layers, p1=0.1, p2=0.1, p3=0.1, p4=0.1).to(device)

    test_loss, test_acc = evaluate(model, testloader, device)
    print(f"Final Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

    # Tester sur un batch
    X, Y = next(iter(valloader))
    X, Y = X.to(device), Y.to(device)
    T = config.MC_T
    
    user_metrics = input(
    "Quelles métriques voulez-vous calculer ? (mc_estimate, variance, predictive_entropy, relative_norm)\n"
    "Vous pouvez en choisir plusieurs, séparées par des virgules : ")
    user_metrics = [m.strip() for m in user_metrics.split(",")]
    outputs, mean_probs, metric_values = generate_mc_outputs(model, X, T, metrics=user_metrics, labels=Y)
    print(f"Liste des métriques choisies par l\'utilisateur : {user_metrics}")
    for metric in user_metrics:
        print(f"Métrique choisie : {metric}")
        print(f"Résultat : {metric_values[metric]}\n")
    

if __name__ == "__main__":
    main()