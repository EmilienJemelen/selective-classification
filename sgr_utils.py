import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import scipy.special
import random as rd
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights
from tqdm import tqdm
import pickle
import torch.optim.lr_scheduler as lr_scheduler





def prepare_sgr_dico(dataloader, model, device, T):
    
    sgr_dico = {'y_true' : np.array([]),
                'y_pred' : np.array([]),
                'SR' : np.array([])}
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            batch_preds = model(images)
            softmax_values = F.softmax(batch_preds/T, dim=1) 
            softmax_responses = torch.max(softmax_values, dim=1)[0].cpu().numpy()
            _, predicted_classes = torch.max(batch_preds, 1)
            predicted_classes = predicted_classes.cpu().numpy()

            sgr_dico['y_true'] = np.concatenate((sgr_dico['y_true'], labels.cpu().numpy()))
            sgr_dico['y_pred'] = np.concatenate((sgr_dico['y_pred'], predicted_classes))
            sgr_dico['SR'] = np.concatenate((sgr_dico['SR'], softmax_responses))

    return sgr_dico



