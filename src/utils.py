# src/utils.py
import torch
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        self.device = None

    def normalize(self, x):
        if torch.is_tensor(x):
            if self.device is None: self.device = x.device
            return (x - torch.tensor(self.mean, device=self.device)) / torch.tensor(self.std, device=self.device)
        return (x - self.mean) / self.std

    def denormalize(self, x):
        if torch.is_tensor(x):
            return x * torch.tensor(self.std, device=self.device) + torch.tensor(self.mean, device=self.device)
        return x * self.std + self.mean