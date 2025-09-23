import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_resnet18(batch_size=128, val_ratio=0.1):
    """
    Charge les DataLoaders pour CIFAR10, avec un prétraitement adapté à ResNet18 pré-entraîné sur ImageNet :
    - Redimensionne les images à 224x224 (taille attendue par ResNet18 standard).
    - Applique la normalisation ImageNet.
    - Retourne les DataLoaders pour train, validation et test, ainsi que la liste des classes CIFAR10.
    - N'instancie ni ne retourne de modèle ResNet18.
    """

    # Images 224x224, normalisation ImageNet pour ResNet18 pré-entraîné
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet18 attend 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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