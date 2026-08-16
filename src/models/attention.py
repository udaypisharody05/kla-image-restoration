"""Lightweight squeeze-and-excitation channel attention (not full RCAN).

``feature map -> global average pool -> 1x1 reduce -> ReLU -> 1x1 expand ->
sigmoid -> channel-wise multiply``. A single reusable block, not an
architecture of its own -- optionally inserted into ``ResidualSRNet``'s
residual blocks via ``--channel-attention`` (see ``src/models/residual_sr.py``).
"""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel gate with a configurable reduction ratio."""

    def __init__(self, num_features: int, reduction: int = 8) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be a positive integer")
        reduced_features = max(1, num_features // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(num_features, reduced_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(reduced_features, num_features, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.pool(x)
        gate = self.relu(self.reduce(gate))
        gate = self.sigmoid(self.expand(gate))
        return x * gate
