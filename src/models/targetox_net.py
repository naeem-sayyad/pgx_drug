"""PyTorch model definition for the TargetTox classifier."""

from __future__ import annotations

import torch
from torch import nn


class TargetToxNet(nn.Module):
    """Simple feed-forward network for toxicity classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return logits.squeeze(-1)

