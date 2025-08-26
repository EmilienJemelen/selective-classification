import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 128)
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
    def __init__(self, model, mc_layers=None, p1=0.1, p2=0.1, p3=0.1, p4=0.1):
        super().__init__()
        self.model = model
        self.mc_layers = mc_layers or [] # si None, aucune couche à masquer
        self.p1 = p1  # dropout sur conv1 output
        self.p2 = p2  # dropout sur conv2 output
        self.p3 = p3  # dropout sur conv3 output
        self.p4 = p4  # dropout sur fc1 output
        # pas de dropout sur la dernière couche
        self.ps = {'conv1': p1, 'conv2': p2, 'conv3': p3, 'fc1': p4}
    
    def dropout_mask(self, x, p, active=True):
        if not self.training or p == 0.0 or not active:
            return torch.ones_like(x)
        mask = (torch.rand_like(x) > p).float() / (1 - p)
        return mask

    def forward(self, x):
        x = F.relu(self.model.conv1(x))
        x = self.dropout_mask(x, self.ps['conv1'], 'conv1' in self.mc_layers) * x
        x = self.model.pool(x)

        x = F.relu(self.model.conv2(x))
        x = self.dropout_mask(x, self.ps['conv2'], 'conv2' in self.mc_layers) * x
        x = self.model.pool(x)

        x = F.relu(self.model.conv3(x))
        x = self.dropout_mask(x, self.ps['conv3'], 'conv3' in self.mc_layers) * x
        x = self.model.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = self.dropout_mask(x, self.ps['fc1'], 'fc1' in self.mc_layers) * x
        x = self.model.fc2(x)
        return x  #pas encore normalisé