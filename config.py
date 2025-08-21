import torch
# Hyperparameters et chemins
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pt"
MC_T = 1000
MC_METRIC = "variance"  # ou "entropy"