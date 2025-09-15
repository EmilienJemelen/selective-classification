import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_MCdropout(nn.Module):
    def __init__(self, model, dico_layers=None):
        super().__init__()
        self.model = model
        self.dico_layers = dico_layers or {}

        for name, layer in list(self.model._modules.items()):
            if isinstance(layer, nn.Conv2d) and name in self.dico_layers:
                p = self.dico_layers[name]
                self.model._modules[name] = nn.Sequential(layer, nn.ReLU(), nn.Dropout2d(p))
            elif isinstance(layer, nn.Linear) and name in self.dico_layers:
                p = self.dico_layers[name]
                self.model._modules[name] = nn.Sequential(layer, nn.ReLU(), nn.Dropout(p))

    def forward(self, x):
        return self.model.forward(x)