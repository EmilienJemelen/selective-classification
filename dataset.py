import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size=128, val_ratio=0.1):
    # Images 32x32, normalisation CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int((1 - val_ratio) * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, valloader, testloader, classes