"""Optional lightweight pre-trunk denoising stem (Phase 3A).

Motivated directly by the Experiment 22 degradation forensics
(``results/degradation_analysis/degradation_report.md``): the measured
NoisyLR corruption is dominated by strongly signal-dependent noise (residual
std 0.0897, ~2.8x the champion pipeline's validation L1), not by blur or a
resampling-kernel mismatch. That is "meaningful noise before upsampling" in
the sense Phase 3A asks about, which is what justifies trying an explicit
pre-trunk denoising stage here -- as one candidate mechanism among others
(Experiment 25's noise-conditioning channel is a different, already-prepared
response to the same finding; this is not a replacement for it, and the two
are independently selectable).

``Input -> Conv3x3 -> {2..4} lightweight gated restoration blocks -> Conv3x3
(residual) -> existing ResidualSR trunk``, matching the project spec's
prescribed shape. Deliberately NOT a full NAFNet: no LayerNorm, no channel
attention inside the stem, no multi-stage encoder/decoder -- a single-scale
stack of simplified-gate blocks operating at input resolution, kept small
enough that parameter growth stays modest (see ``ResidualSRNet``'s
``denoise_stem*`` arguments and its test-verified parameter counts).
"""

import torch
from torch import nn


class SimpleGateBlock(nn.Module):
    """A single lightweight gated restoration block (simplified-NAFNet style).

    ``conv1`` doubles the channel count so the subsequent element-wise
    "simple gate" (``a * b``, splitting the doubled channels in half) can act
    as a cheap multiplicative nonlinearity in place of ReLU -- the core idea
    NAFNet uses to drop activation functions entirely. No normalization
    layers or channel attention are added, keeping this smaller than a real
    NAFNet block.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        gated = self.conv1(x)
        a, b = gated.chunk(2, dim=1)
        out = self.conv2(a * b)
        return residual + out


class DenoiseStem(nn.Module):
    """``Conv3x3 -> N x SimpleGateBlock -> Conv3x3``, applied as a residual
    correction: ``output = input + stem(input)``.

    Operates directly on the raw model input (same channel count as
    ``ResidualSRNet.in_channels``, so it composes transparently with
    Experiment 25's noise-conditioning wrapper's extra sigma channel), before
    ``conv_in``. The residual formulation means a stem initialized with small
    random weights starts close to the identity map, rather than needing to
    learn identity from scratch.
    """

    def __init__(self, in_channels: int = 1, stem_features: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        self.conv_in = nn.Conv2d(in_channels, stem_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[SimpleGateBlock(stem_features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(stem_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_in(x)
        features = self.blocks(features)
        correction = self.conv_out(features)
        return x + correction
