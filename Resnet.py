import torch
import numpy as np
import random
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

# Fixer le seed pour la reproductibilité
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Charger un ResNet18 pré-entraîné standard (ImageNet, 10 classes)
# Pour CIFAR-10, il est courant d'adapter la première couche (conv1) car les images sont 32x32.
# Ici, seule la dernière couche est modifiée pour 10 classes.
# Adapter la première couche pour CIFAR-10

resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet18.maxpool = nn.Identity()
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

# class ResNet18(nn.Module):
#     def __init__(self, num_classes=1000, pretrained=True):
#         super(ResNet18, self).__init__()
#         self.model = models.resnet18(pretrained=pretrained)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# def get_resnet18(pretrained=True, num_classes=1000):
#     return ResNet18(num_classes=num_classes, pretrained=pretrained)