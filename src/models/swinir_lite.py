"""SwinIR-lite: a compact SwinIR-inspired architecture for grayscale 2x
super-resolution, built on windowed self-attention (Liu et al. 2021's Swin
Transformer / Liang et al. 2021's SwinIR) rather than convolution alone.

Implemented locally in plain PyTorch -- no third-party Swin/SwinIR package, no
pretrained weights. Core components:

- ``window_partition``/``window_reverse``: split a ``[B,H,W,C]`` feature map
  into non-overlapping ``window_size x window_size`` windows and back. Exact
  inverses of each other (no information lost or duplicated).
- ``WindowAttention``: standard multi-head self-attention computed *within*
  each window only (not globally), plus a learned relative position bias
  (Swin's core spatial-awareness mechanism inside a window).
- ``SwinTransformerBlock``: LayerNorm -> (optional cyclic shift) -> window
  partition -> window multi-head self-attention -> window reverse -> (undo
  shift) -> residual, followed by LayerNorm -> MLP -> residual. Blocks
  alternate between regular windows (``shift_size=0``) and shifted windows
  (``shift_size=window_size//2``) so information can flow across window
  boundaries -- the shifted variant uses a precomputed attention mask so a
  window straddling the cyclic-shift wrap-around never attends across the
  seam.
- ``SwinIRLite``: adapts this (same-resolution) transformer body to 2x
  super-resolution using the same shallow-conv / long-skip /
  PixelShuffle-upsample skeleton already used by
  ``ResidualSRNet``/``EDSRLite``/``NAFNetSR`` -- only the feature-processing
  body is replaced by a stack of ``SwinTransformerBlock``s operating on
  tokens instead of a stack of convolutional blocks.
"""

import torch
import torch.nn.functional as F
from torch import nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """``[B,H,W,C]`` -> ``[B * (H/ws) * (W/ws), ws, ws, C]`` non-overlapping windows.

    Requires ``H`` and ``W`` to be exact multiples of ``window_size`` --
    callers are responsible for padding beforehand (see ``SwinIRLite.forward``).
    """
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """Exact inverse of ``window_partition``: windows -> ``[B,H,W,C]``."""
    num_windows_total = windows.shape[0]
    batch = num_windows_total // ((height // window_size) * (width // window_size))
    x = windows.view(
        batch, height // window_size, width // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention restricted to a single window, with a learned
    relative position bias indexed by each pair of positions' offset within
    the window (Swin's mechanism for giving attention spatial awareness
    without global position embeddings)."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        num_relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        )  # [2, ws, ws]
        coords_flat = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``[num_windows*B, N, C]`` where ``N = window_size**2``.

        ``mask``, when given (shifted-window blocks only): ``[num_windows, N, N]``,
        additive (0 for allowed pairs, a large negative number for pairs that must
        not attend to each other across the cyclic-shift seam).
        """
        batch_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [batch_windows, heads, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(tokens, tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    """LayerNorm -> (shifted) window attention -> residual, then LayerNorm -> MLP -> residual.

    ``shift_size=0`` gives ordinary (non-overlapping, axis-aligned) windows;
    ``shift_size=window_size//2`` cyclically shifts the feature map before
    partitioning so window boundaries move, letting information cross the
    previous blocks' boundaries -- the standard Swin shifted-window trick. The
    caller (``SwinIRLite``) alternates the two across the block stack.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if not (0 <= shift_size < window_size):
            raise ValueError(f"shift_size ({shift_size}) must be in [0, window_size)")
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def _shifted_window_attention_mask(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Standard Swin mask: gives each of the 9 shift-induced regions of the
        feature map a distinct id, then masks out attention between tokens
        from different ids inside the same (post-shift) window -- otherwise
        the cyclic shift would let a window mix tokens that were not spatially
        adjacent before shifting."""
        img_mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        region_id = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                img_mask[:, height_slice, width_slice, :] = region_id
                region_id += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """``x``: ``[B, height*width, C]`` (``height``/``width`` must both be
        exact multiples of ``window_size`` -- the caller pads beforehand)."""
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(f"Expected {height * width} tokens, got {length}")

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._shifted_window_attention_mask(height, width, x.device)
        else:
            shifted = x
            attn_mask = None

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted
        x = x.view(batch, height * width, channels)

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinIRLite(nn.Module):
    """Shallow conv -> residual Swin Transformer body (long/global residual)
    -> PixelShuffle x2 upsample -> reconstruction conv.

    ```
    Noisy LR
      -> 3x3 conv (in_channels -> embed_dim)
      -> reflect-pad spatial dims up to a multiple of window_size (if needed)
      -> depth x SwinTransformerBlock (alternating regular/shifted windows)
      -> LayerNorm -> 3x3 conv ("conv_after_body")
      -> crop back to the original (unpadded) spatial size
      -> + (long/global residual connection back to the post-conv_in features)
      -> 3x3 conv (embed_dim -> embed_dim * scale^2) -> PixelShuffle(scale)
      -> 3x3 reconstruction conv (embed_dim -> out_channels)
      -> restored grayscale output
    ```

    Any input spatial size at least ``window_size`` in each dimension is
    supported: sizes that are not exact multiples of ``window_size`` are
    reflect-padded internally before the transformer body and cropped back to
    the exact original size before upsampling, so validation images are never
    cropped or otherwise altered -- only compute-internal padding is added and
    then discarded.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 48,
        depth: int = 6,
        num_heads: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.window_size = window_size
        self.embed_dim = embed_dim

        self.conv_in = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if block_index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for block_index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.upsample_conv = nn.Conv2d(
            embed_dim, embed_dim * scale * scale, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv_reconstruction = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shape [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        if height < self.window_size or width < self.window_size:
            raise ValueError(
                f"Input spatial size ({height}x{width}) must be at least "
                f"window_size ({self.window_size}) in each dimension"
            )

        features = self.conv_in(x)  # [B, embed_dim, H, W]

        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        padded = F.pad(features, (0, pad_width, 0, pad_height), mode="reflect")
        padded_height, padded_width = padded.shape[-2:]

        tokens = padded.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        for block in self.blocks:
            tokens = block(tokens, padded_height, padded_width)
        tokens = self.norm(tokens)
        body_out = tokens.transpose(1, 2).view(batch, self.embed_dim, padded_height, padded_width)
        body_out = self.conv_after_body(body_out)
        if pad_height or pad_width:
            body_out = body_out[:, :, :height, :width]

        features = features + body_out  # long/global residual connection over the body
        upsampled = self.pixel_shuffle(self.upsample_conv(features))
        return self.conv_reconstruction(upsampled)
