"""ResidualSR architecture, trimmed to exactly what the packaged champion
checkpoint (``models/residualsr_final_ema.pt``) uses.

This is a self-contained copy of the runtime path of the training
repository's ``src/models/residual_sr.py::ResidualSRNet`` (grayscale
in/out, plain residual blocks, PixelShuffle upsampling). The champion
checkpoint was trained with every optional variant (channel attention,
multiscale/RDB blocks, denoise stem) disabled, so those variants are not
reproduced here -- omitting them does not change the state-dict keys or
parameter count for this checkpoint (verified: 38 tensors, 630,724
parameters, ``load_state_dict(..., strict=True)`` succeeds).
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
        num_features: int = 64,
        num_blocks: int = 8,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.conv_body_out = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
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
