"""NAFNet-SR-Lite: a compact NAFNet-inspired architecture adapted for grayscale
2x super-resolution.

Implements the core NAFNet ("Simple Baselines for Image Restoration", Chen et al.
2022) block design locally, in plain PyTorch -- no third-party NAFNet package, no
pretrained weights:

- ``LayerNorm2d``: channel-wise layer normalization (normalizes each spatial
  position across channels), NAFNet's normalization choice -- not BatchNorm, no
  running statistics, no dependence on batch composition.
- ``SimpleGate``: splits channels into two equal halves and multiplies them
  element-wise. This *is* the block's nonlinearity -- there is no ReLU/GELU
  anywhere inside a ``NAFBlock``.
- ``NAFBlock``: a gated conv branch (1x1 channel expansion -> 3x3 depthwise conv
  -> SimpleGate -> simplified channel attention -> 1x1 projection, added back
  through a learnable per-channel scale ``beta``) followed by a gated
  feed-forward branch (1x1 expansion -> SimpleGate -> 1x1 projection, added back
  through a learnable per-channel scale ``gamma``), each branch normalized by its
  own ``LayerNorm2d`` before it runs. ``beta``/``gamma`` are initialized to zero
  so every ``NAFBlock`` starts as a near-identity map, matching NAFNet's original
  initialization.

Canonical NAFNet operates at a single, fixed resolution (image restoration, not
super-resolution). This task needs 2x upsampling, so ``NAFNetSR`` reuses the same
shallow-conv / long-skip / PixelShuffle-upsample / reconstruction-conv skeleton
already used by ``ResidualSRNet``/``EDSRLite`` -- only the feature-processing body
(the residual blocks) is replaced by a stack of ``NAFBlock``s.
"""

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for ``[N,C,H,W]`` tensors.

    Normalizes each spatial position across the channel dimension, then applies a
    learned per-channel scale/shift.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Splits channels into two equal halves and multiplies them element-wise.

    This is NAFNet's entire in-block "activation" -- a multiplicative gate, not an
    approximation of ReLU/GELU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_half, second_half = x.chunk(2, dim=1)
        return first_half * second_half


class NAFBlock(nn.Module):
    """One NAFNet block: a gated conv branch + a gated feed-forward branch.

    ``dw_expand``/``ffn_expand`` are the channel-expansion factors used inside
    each branch before ``SimpleGate`` halves the channel count back down again --
    fixed at NAFNet's published defaults (2, 2) rather than exposed as CLI knobs,
    matching this project's convention of pinning architecture-internal constants
    when nothing signals they need per-run tuning (as with EDSRLite's
    ``residual_scale``).
    """

    def __init__(self, num_channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        conv_channels = num_channels * dw_expand
        ffn_channels = num_channels * ffn_expand

        self.norm1 = LayerNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, conv_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1, groups=conv_channels
        )
        self.simple_gate1 = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(conv_channels // 2, conv_channels // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(conv_channels // 2, num_channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.norm2 = LayerNorm2d(num_channels)
        self.conv4 = nn.Conv2d(num_channels, ffn_channels, kernel_size=1)
        self.simple_gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.simple_gate1(y)
        y = y * self.channel_attention(y)
        y = self.conv3(y)
        x = residual + y * self.beta

        residual = x
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.simple_gate2(y)
        y = self.conv5(y)
        return residual + y * self.gamma


class NAFNetSR(nn.Module):
    """Shallow conv -> NAF feature body (long/global residual) -> PixelShuffle x2
    upsample -> reconstruction conv.

    ```
    Noisy LR
      -> 3x3 conv (in_channels -> num_features)
      -> num_blocks x NAFBlock
      -> 3x3 conv ("conv_after_body")
      -> + (long/global residual connection back to the post-conv_in features)
      -> 3x3 conv (num_features -> num_features * scale^2) -> PixelShuffle(scale)
      -> 3x3 reconstruction conv (num_features -> out_channels)
      -> restored grayscale output
    ```

    No pooling or strided convolution anywhere in the body, so spatial dimensions
    (including odd sizes) pass through the body unchanged before upsampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 32,
        num_blocks: int = 8,
        scale: int = 2,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[NAFBlock(num_features, dw_expand, ffn_expand) for _ in range(num_blocks)]
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
