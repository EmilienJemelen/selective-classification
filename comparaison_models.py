import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MCdropout(nn.Module):
    def __init__(self, model, dico_layers=None):
        super().__init__()
        self.model = model
        self.dico_layers = dico_layers or {}
        self.ps = {'conv1': dico_layers.get('conv1', 0.0),
                   'conv2': dico_layers.get('conv2', 0.0),
                   'conv3': dico_layers.get('conv3', 0.0),
                   'fc1': dico_layers.get('fc1', 0.0)}

    def dropout_mask(self, x, p, active=True):
        if not self.training or p == 0.0 or not active:
            return torch.ones_like(x)
        mask = (torch.rand_like(x) > p).float() / (1 - p)
        return mask

    def forward(self, x):
        x = F.relu(self.model.conv1(x))
        x = self.dropout_mask(x, self.ps['conv1'], 'conv1' in self.dico_layers) * x
        x = self.model.pool(x)
        x = F.relu(self.model.conv2(x))
        x = self.dropout_mask(x, self.ps['conv2'], 'conv2' in self.dico_layers) * x
        x = self.model.pool(x)
        x = F.relu(self.model.conv3(x))
        x = self.dropout_mask(x, self.ps['conv3'], 'conv3' in self.dico_layers) * x
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = self.dropout_mask(x, self.ps['fc1'], 'fc1' in self.dico_layers) * x
        x = self.model.fc2(x)
        return x

class CNN_MCdropout_beforeReLU(nn.Module):
    def __init__(self, model, dico_layers=None):
        super().__init__()
        self.model = model
        self.dico_layers = dico_layers or {}
        self.ps = {'conv1': dico_layers.get('conv1', 0.0),
                   'conv2': dico_layers.get('conv2', 0.0),
                   'conv3': dico_layers.get('conv3', 0.0),
                   'fc1': dico_layers.get('fc1', 0.0)}

    def dropout_mask(self, x, p, active=True):
        if not self.training or p == 0.0 or not active:
            return torch.ones_like(x)
        mask = (torch.rand_like(x) > p).float() / (1 - p)
        return mask

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.dropout_mask(x, self.ps['conv1'], 'conv1' in self.dico_layers) * x
        x = F.relu(x)
        x = self.model.pool(x)
        x = self.model.conv2(x)
        x = self.dropout_mask(x, self.ps['conv2'], 'conv2' in self.dico_layers) * x
        x = F.relu(x)
        x = self.model.pool(x)
        x = self.model.conv3(x)
        x = self.dropout_mask(x, self.ps['conv3'], 'conv3' in self.dico_layers) * x
        x = F.relu(x)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        x = self.dropout_mask(x, self.ps['fc1'], 'fc1' in self.dico_layers) * x
        x = F.relu(x)
        x = self.model.fc2(x)
        return x

class CNN_MCdropout_torch(nn.Module):
    def __init__(self, model, dico_layers=None):
        super().__init__()
        self.model = model
        self.dico_layers = dico_layers or {}
        p1 = dico_layers.get('conv1', 0.0)
        p2 = dico_layers.get('conv2', 0.0)
        p3 = dico_layers.get('conv3', 0.0)
        p4 = dico_layers.get('fc1', 0.0)

        if 'conv1' in self.dico_layers and isinstance(self.model.conv1, nn.Conv2d):
            self.model.conv1 = nn.Sequential(self.model.conv1, nn.ReLU(), nn.Dropout2d(p1))
        else:
            self.model.conv1 = nn.Sequential(self.model.conv1, nn.ReLU())

        if 'conv2' in self.dico_layers and isinstance(self.model.conv2, nn.Conv2d):
            self.model.conv2 = nn.Sequential(self.model.conv2, nn.ReLU(), nn.Dropout2d(p2))
        else:
            self.model.conv2 = nn.Sequential(self.model.conv2, nn.ReLU())

        if 'conv3' in self.dico_layers and isinstance(self.model.conv3, nn.Conv2d):
            self.model.conv3 = nn.Sequential(self.model.conv3, nn.ReLU(), nn.Dropout2d(p3))
        else:
            self.model.conv3 = nn.Sequential(self.model.conv3, nn.ReLU())

        if 'fc1' in self.dico_layers and isinstance(self.model.fc1, nn.Linear):
            self.model.fc1 = nn.Sequential(self.model.fc1, nn.ReLU(), nn.Dropout(p4))
        else:
            self.model.fc1 = nn.Sequential(self.model.fc1, nn.ReLU())

    def forward(self, x):
        return self.model.forward(x)
