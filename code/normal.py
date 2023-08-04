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

        self.event_dim = label_size
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
        self.variance = nn.Sequential(
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

        # Compute std and corr and fill into a matrix
        var = self.variance(embedding) + self.jitter
        corr = self.correlation(embedding)

        # Fill var and corr into a tril matrix for MVN
        # var down the diagonal, corr in lower tril portion without main diagonal
        tril_row, tril_col = torch.tril_indices(self.event_dim, self.event_dim, offset=-1)

        cov = torch.diag_embed(var)
        cov[:, tril_row, tril_col] = corr

        return td.MultivariateNormal(loc=mean, scale_tril=cov)
