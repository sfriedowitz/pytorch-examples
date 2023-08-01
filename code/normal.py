import torch
from torch import nn
import torch.distributions as td


class DeepNormal(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()

        self.jitter = 1e-6
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
        )
        self.std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> td.Normal:
        embedding = self.embedding(x)
        mean = self.mean(embedding)
        std = self.std(embedding) + self.jitter
        return td.Normal(mean, std)
