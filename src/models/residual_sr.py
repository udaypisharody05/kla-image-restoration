"""A small residual CNN baseline for noisy LR -> clean 2x GT restoration.

Noisy LR -> initial convolution -> feature maps -> residual blocks -> convolution
-> PixelShuffle x2 -> restored output. No pretrained weights, no downloads,
no normalization layers; kept deliberately small and easy to debug.

Two independent, OPTIONAL, default-OFF variants can be inserted into the
residual blocks without changing anything about the default architecture:

- ``channel_attention`` wraps each block's output in a lightweight
  squeeze-and-excitation gate (``src/models/attention.py::ChannelAttention``).
- ``multiscale_block`` replaces ``ResidualBlock`` with ``MultiScaleBlock``
  (a local 3x3 branch fused with a dilated 3x3 branch, for a larger receptive
  field without a large kernel).
- ``denoise_stem`` (Phase 3A, ``src/models/denoise_stem.py::DenoiseStem``)
  inserts a small ``Conv3x3 -> gated blocks -> Conv3x3`` residual denoising
  stage before ``conv_in``, motivated by the Experiment 22 forensics finding
  that the NoisyLR corruption is dominated by signal-dependent noise.

All three default to ``False``/unused, so ``ResidualSRNet(in_channels=1,
out_channels=1, num_features=64, num_blocks=8, scale=2)`` -- every historical
call site -- constructs the exact same graph and state-dict keys as before
any variant existed, and every existing checkpoint remains loadable
unmodified. They are independently selectable (never auto-combined) so each
can be ablated on its own; nothing prevents requesting several together.
"""

import torch
from torch import nn

from .attention import ChannelAttention
from .denoise_stem import DenoiseStem


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a local identity skip connection.

    ``use_attention=False`` (the default) creates no attention submodule at
    all -- not even a disabled one -- so the parameter count, state-dict
    keys, and forward computation are byte-identical to before
    ``ChannelAttention`` support existed.
    """

    def __init__(self, num_features: int, use_attention: bool = False, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.attention = ChannelAttention(num_features, reduction) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.attention is not None:
            out = self.attention(out)
        return residual + out


class MultiScaleBlock(nn.Module):
    """Alternative residual block with a larger receptive field, without a
    large kernel or a large parameter increase.

    A local 3x3 branch and a dilated 3x3 (dilation=2) branch run in parallel
    over the same input, are concatenated, and fused back down to
    ``num_features`` with a single 1x1 convolution before the residual add --
    mirroring ``ResidualBlock``'s "nonlinearity then one more linear layer,
    then add" shape (branches replace ``conv1``, the 1x1 fuse replaces
    ``conv2``) while keeping the added parameter cost to just that 1x1 fuse
    layer (dilation changes a 3x3 kernel's receptive field, not its
    parameter count). Same optional ``use_attention``/``reduction`` hook as
    ``ResidualBlock``.
    """

    def __init__(self, num_features: int, use_attention: bool = False, reduction: int = 8) -> None:
        super().__init__()
        self.branch_local = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.branch_dilated = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=2, dilation=2
        )
        self.relu = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(num_features * 2, num_features, kernel_size=1)
        self.attention = ChannelAttention(num_features, reduction) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local_branch = self.relu(self.branch_local(x))
        dilated_branch = self.relu(self.branch_dilated(x))
        out = self.fuse(torch.cat([local_branch, dilated_branch], dim=1))
        if self.attention is not None:
            out = self.attention(out)
        return residual + out


class ResidualDenseBlock(nn.Module):
    """Lightweight Residual Dense Block (Phase 10), inspired by RDN (Zhang et
    al. 2018) but deliberately NOT a full RDN: a single dense-connectivity
    block dropped in as a third alternative to ``ResidualBlock``/
    ``MultiScaleBlock``, reusing ``ResidualSRNet``'s existing stem and
    upsampling head unchanged.

    ``num_layers`` 3x3 convolutions, each seeing the concatenation of the
    block's input and every previous layer's output (dense feature reuse,
    the defining RDB idea), followed by a 1x1 "local feature fusion" back
    down to ``num_features`` and a residual add. A small ``growth_rate``
    (channels added per layer) keeps the concatenated channel count -- and
    therefore the parameter count -- modest: with the defaults
    (``growth_rate=16, num_layers=3``) at the champion's 64-feature width,
    this block has *fewer* parameters than ``ResidualBlock`` x8 combined
    (see ``tests/test_model_unit.py`` for the exact verified count), because
    dense reuse needs less raw width per layer to represent similar capacity.
    """

    def __init__(
        self,
        num_features: int,
        growth_rate: int = 16,
        num_layers: int = 3,
        use_attention: bool = False,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        if growth_rate < 1:
            raise ValueError("growth_rate must be a positive integer")
        if num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.layers = nn.ModuleList()
        channels = num_features
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1))
            channels += growth_rate
        self.relu = nn.ReLU(inplace=True)
        self.local_feature_fusion = nn.Conv2d(channels, num_features, kernel_size=1)
        self.attention = ChannelAttention(num_features, reduction) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        features = [x]
        for layer in self.layers:
            out = self.relu(layer(torch.cat(features, dim=1)))
            features.append(out)
        fused = self.local_feature_fusion(torch.cat(features, dim=1))
        if self.attention is not None:
            fused = self.attention(fused)
        return residual + fused


class ResidualSRNet(nn.Module):
    """Lightweight residual CNN with PixelShuffle upsampling for a fixed scale factor."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 2,
        channel_attention: bool = False,
        attention_reduction: int = 8,
        multiscale_block: bool = False,
        rdb_block: bool = False,
        rdb_growth_rate: int = 16,
        rdb_num_layers: int = 3,
        denoise_stem: bool = False,
        denoise_stem_features: int = 32,
        denoise_stem_blocks: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        if multiscale_block and rdb_block:
            raise ValueError(
                "multiscale_block and rdb_block are alternative residual-block replacements "
                "and are mutually exclusive; enable at most one at a time."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.channel_attention = channel_attention
        self.attention_reduction = attention_reduction
        self.multiscale_block = multiscale_block
        self.rdb_block = rdb_block
        self.rdb_growth_rate = rdb_growth_rate
        self.rdb_num_layers = rdb_num_layers
        self.denoise_stem = denoise_stem
        self.denoise_stem_features = denoise_stem_features
        self.denoise_stem_blocks = denoise_stem_blocks

        # False (the default) constructs no stem submodule at all -- not even
        # a disabled one -- so parameter count and state-dict keys are
        # byte-identical to before this existed, mirroring how
        # channel_attention=False builds no ChannelAttention submodule.
        self.stem = (
            DenoiseStem(in_channels, denoise_stem_features, denoise_stem_blocks)
            if denoise_stem
            else None
        )

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        if rdb_block:
            self.body = nn.Sequential(
                *[
                    ResidualDenseBlock(
                        num_features,
                        growth_rate=rdb_growth_rate,
                        num_layers=rdb_num_layers,
                        use_attention=channel_attention,
                        reduction=attention_reduction,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            block_cls = MultiScaleBlock if multiscale_block else ResidualBlock
            self.body = nn.Sequential(
                *[
                    block_cls(num_features, use_attention=channel_attention, reduction=attention_reduction)
                    for _ in range(num_blocks)
                ]
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
        if self.stem is not None:
            x = self.stem(x)
        features = self.conv_in(x)
        body_out = self.conv_body_out(self.body(features))
        features = features + body_out
        upsampled = self.upsample_conv(features)
        return self.pixel_shuffle(upsampled)
