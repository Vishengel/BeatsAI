import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.input_size, 50),
            nn.ReLU(),
            nn.Linear(100, config.n_classes)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x