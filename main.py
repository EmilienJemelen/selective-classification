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
    
    model = SimpleCNN()
    model = train_model(model, trainloader, valloader, device, config.EPOCHS, config.MODEL_PATH)
    
    # Charger meilleur modèle et activer MC Dropout
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    mc_model = CNN_MCdropout(model).to(device)
    
    # Tester sur un batch
    X, Y = next(iter(valloader))
    X, Y = X.to(device), Y.to(device)
    
    outputs, mean_probs, mc_metric = generate_mc_outputs(mc_model, X, config.MC_T, config.MC_METRIC)
    print("MC Dropout metric:", mc_metric)

if __name__ == "__main__":
    main()