"""Router head: query embedding -> 2 logits (0 = GCN dense tower, 1 = SPLADE)."""

from torch import nn


class RouterHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)
