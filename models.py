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
    def __init__(self, model, p1=0.1, p2=0.1, p3=0.1, p4=0.1):
        super().__init__()
        self.model = model
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def dropout_mask(self, x, p):
        if not self.training or p == 0.0:
            return torch.ones_like(x)
        mask = (torch.rand_like(x) > p).float() / (1 - p)
        return mask
    
    def forward(self, x):
        #Bloc 1
        x = F.relu(self.model.conv1(x))
        x = x * self.dropout_mask(x, self.p1)
        x = self.model.pool(x)
        #Bloc 2
        x = F.relu(self.model.conv2(x))
        x = x * self.dropout_mask(x, self.p2)
        x = self.model.pool(x)
        #Bloc 3
        x = F.relu(self.model.conv3(x))
        x = x * self.dropout_mask(x, self.p3)
        x = self.model.pool(x)
        #Flatten pour les fully conected
        x = x.view(x.size(0), -1)
        #Fully connected layers
        x = F.relu(self.model.fc1(x))
        x = x * self.dropout_mask(x, self.p4)
        x = self.model.fc2(x)
        return x