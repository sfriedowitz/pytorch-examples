import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Output
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
