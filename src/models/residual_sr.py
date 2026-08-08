"""A small residual CNN baseline for noisy LR -> clean 2x GT restoration.

Noisy LR -> initial convolution -> feature maps -> residual blocks -> convolution
-> PixelShuffle x2 -> restored output. No pretrained weights, no downloads,
no normalization layers; kept deliberately small and easy to debug.
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a local identity skip connection."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return residual + out


class ResidualSRNet(nn.Module):
    """Lightweight residual CNN with PixelShuffle upsampling for a fixed scale factor."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.conv_body_out = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1
        )
        self.upsample_conv = nn.Conv2d(
            num_features, out_channels * scale * scale, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shape [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        features = self.conv_in(x)
        body_out = self.conv_body_out(self.body(features))
        features = features + body_out
        upsampled = self.upsample_conv(features)
        return self.pixel_shuffle(upsampled)
