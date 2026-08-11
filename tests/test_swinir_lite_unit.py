"""Fast, dataset-free tests for the SwinIR-lite architecture (src/models/swinir_lite.py)."""

import pytest
import torch
from torch import nn

from src.models.swinir_lite import (
    SwinIRLite,
    SwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_reverse,
)
from src.tta import predict_x8


def _exp13_config() -> dict:
    """Chosen Experiment 13 configuration -- see EXPERIMENT_LOG.md for the CUDA
    memory/runtime search that led to this choice."""
    return {
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 60,
        "depth": 6,
        "num_heads": 6,
        "window_size": 8,
        "mlp_ratio": 2.0,
        "scale": 2,
    }


# --- SwinIRLite: shapes, finiteness, gradients ---


def test_accepts_grayscale_input() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape[1] == 1


def test_output_is_exactly_2x_spatial_dimensions() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2)
    output = model(torch.randn(2, 1, 16, 24))
    assert output.shape == (2, 1, 32, 48)


def test_96x96_training_crop_size_works() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=8, scale=2)
    output = model(torch.randn(1, 1, 96, 96))
    assert output.shape == (1, 1, 192, 192)


def test_128x128_full_validation_image_size_works() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=8, scale=2)
    output = model(torch.randn(1, 1, 128, 128))
    assert output.shape == (1, 1, 256, 256)


def test_non_multiple_of_window_size_input_still_produces_exact_2x_output() -> None:
    """Internal padding is added and then cropped back off before upsampling,
    so any input size at least window_size still yields exactly 2x output --
    validation images are never silently cropped."""
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=8, scale=2)
    output = model(torch.randn(1, 1, 20, 30))
    assert output.shape == (1, 1, 40, 60)


def test_input_smaller_than_window_size_raises_clear_error() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=8, scale=2)
    with pytest.raises(ValueError, match="window_size"):
        model(torch.randn(1, 1, 4, 4))


def test_forward_output_is_finite() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert torch.isfinite(output).all()


def test_backward_gradients_are_finite() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    loss = nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rejects_mismatched_input_channels() -> None:
    model = SwinIRLite(in_channels=1, embed_dim=8, depth=2, num_heads=2, window_size=4)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(1, 3, 16, 16))


def test_output_channels_match_configured_out_channels() -> None:
    model = SwinIRLite(
        in_channels=1, out_channels=3, embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2
    )
    output = model(torch.randn(2, 1, 12, 12))
    assert output.shape == (2, 3, 24, 24)


def test_rejects_embed_dim_not_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SwinIRLite(embed_dim=10, depth=2, num_heads=3, window_size=4)


def test_exp13_config_matches_exact_expected_param_count() -> None:
    model = SwinIRLite(**_exp13_config())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Verified by direct instantiation (see final report), not derived here.
    assert param_count == 348_421


def test_no_batchnorm_anywhere_in_the_model() -> None:
    model = SwinIRLite(**_exp13_config())
    normalization_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    assert not any(isinstance(module, normalization_types) for module in model.modules())


def test_x8_tta_works_on_swinir_lite() -> None:
    model = SwinIRLite(embed_dim=8, depth=2, num_heads=2, window_size=4, scale=2)
    result = predict_x8(model, torch.rand(1, 1, 16, 16))
    assert result.shape == (1, 1, 32, 32)
    assert torch.isfinite(result).all()


# --- window_partition / window_reverse ---


def test_window_partition_reverse_round_trips_exactly() -> None:
    x = torch.randn(2, 16, 24, 3)
    windows = window_partition(x, window_size=4)
    restored = window_reverse(windows, window_size=4, height=16, width=24)
    assert torch.equal(x, restored)


def test_window_partition_shape() -> None:
    x = torch.randn(2, 16, 16, 5)
    windows = window_partition(x, window_size=8)
    # (16/8)*(16/8)=4 windows per batch item, batch=2 -> 8 total
    assert windows.shape == (8, 8, 8, 5)


def test_window_partition_reverse_round_trips_for_single_window() -> None:
    x = torch.randn(1, 8, 8, 4)
    windows = window_partition(x, window_size=8)
    assert windows.shape == (1, 8, 8, 4)
    restored = window_reverse(windows, window_size=8, height=8, width=8)
    assert torch.equal(x, restored)


# --- WindowAttention ---


def test_window_attention_output_shape_matches_input() -> None:
    attn = WindowAttention(dim=8, window_size=4, num_heads=2)
    x = torch.randn(3, 16, 8)  # [num_windows*B, N=window_size^2, C]
    output = attn(x)
    assert output.shape == x.shape


def test_window_attention_output_is_finite() -> None:
    attn = WindowAttention(dim=8, window_size=4, num_heads=2)
    output = attn(torch.randn(2, 16, 8))
    assert torch.isfinite(output).all()


def test_window_attention_rejects_dim_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        WindowAttention(dim=10, window_size=4, num_heads=3)


def test_window_attention_with_mask_changes_output() -> None:
    """A non-trivial attention mask (blocking some pairs) must change the
    result relative to no mask -- otherwise the mask isn't wired in."""
    torch.manual_seed(0)
    attn = WindowAttention(dim=8, window_size=4, num_heads=2)
    x = torch.randn(1, 16, 8)
    unmasked = attn(x)
    mask = torch.zeros(1, 16, 16)
    mask[:, :8, 8:] = -100.0
    mask[:, 8:, :8] = -100.0
    masked = attn(x, mask=mask)
    assert not torch.allclose(unmasked, masked)


# --- SwinTransformerBlock ---


def test_regular_window_block_preserves_token_shape() -> None:
    block = SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=0, mlp_ratio=2.0)
    x = torch.randn(1, 64, 8)  # 8x8 tokens
    output = block(x, height=8, width=8)
    assert output.shape == x.shape


def test_shifted_window_block_preserves_token_shape() -> None:
    block = SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=2, mlp_ratio=2.0)
    x = torch.randn(1, 64, 8)
    output = block(x, height=8, width=8)
    assert output.shape == x.shape


def test_shifted_window_attention_mask_has_expected_shape() -> None:
    block = SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=2, mlp_ratio=2.0)
    mask = block._shifted_window_attention_mask(height=8, width=8, device=torch.device("cpu"))
    num_windows = (8 // 4) * (8 // 4)
    tokens_per_window = 4 * 4
    assert mask.shape == (num_windows, tokens_per_window, tokens_per_window)


def test_shift_size_must_be_less_than_window_size() -> None:
    with pytest.raises(ValueError, match="shift_size"):
        SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=4, mlp_ratio=2.0)


def test_block_output_is_finite() -> None:
    block = SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=2, mlp_ratio=2.0)
    output = block(torch.randn(2, 64, 8), height=8, width=8)
    assert torch.isfinite(output).all()
