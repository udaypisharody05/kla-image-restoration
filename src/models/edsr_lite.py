"""EDSR-style "lite" residual super-resolution network.

Follows the EDSR (Lim et al., 2017) design principles at a scale appropriate
for ~3,200 grayscale training pairs and a single ~8GB laptop GPU: no
BatchNorm, residual blocks with a fixed residual scale, a long/global skip
over the residual body, and learned PixelShuffle upsampling followed by a
separate reconstruction convolution -- unlike ``ResidualSRNet``, which fuses
upsampling and reconstruction into a single conv+PixelShuffle step, EDSR-lite
keeps the two separate (feature-space upsample, then a dedicated
image-space reconstruction conv), matching classical EDSR structure.

Deliberately not the full research-scale EDSR (32 blocks x 256 features,
~43M parameters) -- see EXPERIMENT_LOG.md's Experiment 9 entry for the chosen
configuration and why.
"""

import torch
from torch import nn


class EDSRResidualBlock(nn.Module):
    """EDSR-style block: Conv3x3 -> ReLU -> Conv3x3, added back with a fixed residual scale.

    No BatchNorm (EDSR's key departure from SRResNet -- batch statistics were
    found to hurt this kind of regression task and normalization is removed
    entirely). The residual scale (typically well below 1.0) stabilizes
    training in deeper, BatchNorm-free residual stacks; it is a fixed
    constant, not a learned parameter, since nothing here motivates the extra
    complexity of making it trainable.
    """

    def __init__(self, num_features: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + self.residual_scale * residual


class EDSRLite(nn.Module):
    """Initial conv -> deep residual body (global skip) -> learned upsampling -> reconstruction conv.

    ```
    Noisy LR
      -> 3x3 conv (in_channels -> num_features)
      -> num_blocks x EDSRResidualBlock (no BatchNorm, fixed residual_scale)
      -> 3x3 conv ("conv_after_body")
      -> + (long/global residual connection back to the post-conv_in features)
      -> 3x3 conv (num_features -> num_features * scale^2) -> PixelShuffle(scale)
      -> 3x3 reconstruction conv (num_features -> out_channels)
      -> restored grayscale output
    ```
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        scale: int = 2,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.residual_scale = residual_scale

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[EDSRResidualBlock(num_features, residual_scale) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample_conv = nn.Conv2d(
            num_features, num_features * scale * scale, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv_reconstruction = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shape [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        features = self.conv_in(x)
        body_out = self.conv_after_body(self.body(features))
        features = features + body_out  # long/global residual connection over the body
        upsampled = self.pixel_shuffle(self.upsample_conv(features))
        return self.conv_reconstruction(upsampled)
