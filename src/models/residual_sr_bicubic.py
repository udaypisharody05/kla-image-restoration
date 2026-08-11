"""ResidualSRNet's learned branch plus a fixed, non-trainable global bicubic skip.

Same topology as ``src.models.residual_sr.ResidualSRNet`` (reuses
``ResidualBlock`` directly -- no duplicated block logic, no extra blocks,
channels, normalization, activations, or residual scaling), except the final
prediction is::

    prediction = fixed_bicubic_upsample(LR) + learned_residual_branch(LR)

instead of the learned branch's raw output alone. The bicubic term has no
trainable parameters and is never clipped before being added -- clipping (if
any) remains entirely the caller's responsibility, exactly like every other
architecture in this project (``src/tta.py``, ``src/metrics.py``).

**Bicubic implementation note:** this project's historical bicubic *baseline*
(``src.baseline.bicubic_upscale``) resizes a 2D numpy array via
``PIL.Image.resize(..., resample=Image.Resampling.BICUBIC)`` on the CPU --
useful for a one-off classical comparison, but not something that can run
efficiently inside a batched GPU training loop, and not differentiable (not
that a fixed skip needs to be differentiable, but it does need to run once per
forward pass on-device without a CPU round trip). ``fixed_bicubic_upsample``
below instead uses ``torch.nn.functional.interpolate(mode="bicubic")``, a
different implementation (different convolution kernel/anti-aliasing
behavior) from PIL's. The two are **not** bit-identical -- this is a
documented, deliberate difference, not an accidental divergence, and
``src.baseline.bicubic_upscale`` itself is completely untouched by this file.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .residual_sr import ResidualBlock


def fixed_bicubic_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Deterministic, parameter-free bicubic upsample for use inside a forward pass.

    Stays on ``x``'s device/dtype, preserves batch/channel dimensions, and
    produces exactly ``scale``x the spatial dimensions. See the module
    docstring for how this differs from the project's PIL-based bicubic
    baseline.
    """
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


class ResidualSRBicubic(nn.Module):
    """``ResidualSRNet``'s exact learned branch, added to a fixed bicubic upsample.

    ```
    Noisy LR ---------------------------> fixed_bicubic_upsample --------+
      |                                                                  |
      v                                                                  |
    3x3 conv (in_channels -> num_features)                              |
      |                                                                  |
      v                                                                  |
    num_blocks x ResidualBlock                                          |
      |                                                                  |
      v                                                                  |
    3x3 conv ("conv_body_out") -> + (local skip back to post-conv_in)   |
      |                                                                  |
      v                                                                  |
    3x3 conv (num_features -> out_channels * scale^2) -> PixelShuffle   |
      |                                                                  |
      v                                                                  |
    learned residual -----------------------------------------------> + -> raw prediction
    ```

    Identical convolutional layout to ``ResidualSRNet`` -- same
    ``conv_in``/``body``/``conv_body_out``/``upsample_conv``/``pixel_shuffle``
    modules, same parameter count, only the final `forward` step differs
    (add the bicubic term instead of returning the learned branch alone).
    Requires ``in_channels == out_channels`` so the bicubic term's channel
    count matches the learned residual's.
    """

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
        if in_channels != out_channels:
            raise ValueError(
                "ResidualSRBicubic requires in_channels == out_channels "
                f"(got in_channels={in_channels}, out_channels={out_channels}) "
                "so the bicubic skip's channel count matches the learned residual's."
            )
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
        residual = self.pixel_shuffle(upsampled)
        bicubic = fixed_bicubic_upsample(x, self.scale)
        return bicubic + residual
