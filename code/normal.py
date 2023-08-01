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


class DeepMultivariateNormal(nn.Module):
    """Parameterize the MVN by a learned mean, standard deviation, and correlations across labels.

    Constrain the correlations to [-1, 1] and learn only the lower triangular portion.
    """

    def __init__(self, input_size: int, hidden_size: int, label_size: int, dropout: float = 0.2):
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
            nn.Linear(hidden_size, label_size),
        )
        self.std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, label_size),
            nn.Softplus(),
        )

        correlation_size = int(label_size * (label_size - 1) / 2)
        self.correlation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, correlation_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> td.MultivariateNormal:
        # Compute shared embedding vector
        embedding = self.embedding(x)

        # Compute flattened mean vector
        mean = self.mean(embedding)
        mean = mean.flatten()

        # Compute std as a flattened vector
        std = self.std(embedding) + self.jitter
        std = std.flatten().reshape(-1, 1)

        # Compute corr and embed into upper/lower 2nd diagonals of matrix
        distr_dimension = torch.numel(std)

        corr_vec = self.correlation(embedding)
        corr_spaced = torch.zeros(distr_dimension - 1)
        corr_spaced[::2] = corr_vec.flatten()

        # Construct correlation matrix of size (batch * labels) x (batch * labels)
        corr_diag = torch.eye(distr_dimension)
        corr_upper = torch.diag_embed(corr_spaced, offset=1)
        corr_lower = torch.diag_embed(corr_spaced, offset=-1)
        corr = corr_diag + corr_upper + corr_lower

        # Rescale to cov matrix
        cov = std.T * corr * std

        return td.MultivariateNormal(mean, cov)
