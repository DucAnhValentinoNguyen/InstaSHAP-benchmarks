import torch
import torch.nn as nn

class AdditiveSurrogate(nn.Module):
    """
    Simple InstaSHAP-style additive neural network:
    Each feature j has its own small MLP g_j(x_j),
    then outputs are summed across features.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim

        self.subnets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(in_dim)
        ])

    def forward(self, x):
        # x shape: (batch, features)
        contributions = []
        for j, subnet in enumerate(self.subnets):
            contrib_j = subnet(x[:, j:j+1])  # (batch, 1)
            contributions.append(contrib_j)

        contribs = torch.cat(contributions, dim=1)  # (batch, features)
        return contribs.sum(dim=1, keepdim=True)     # (batch, 1)

    def feature_contribs(self, x):
        """
        Return per-feature contributions g_j(x_j).
        Shape: (batch, features)
        """
        contributions = []
        for j, subnet in enumerate(self.subnets):
            contrib_j = subnet(x[:, j:j+1])  # (batch, 1)
            contributions.append(contrib_j)

        return torch.cat(contributions, dim=1)  # (batch, features)
