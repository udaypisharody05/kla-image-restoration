"""Fast, dataset-free tests for the EDSR-lite architecture (src/models/edsr_lite.py)."""

import pytest
import torch
from torch import nn

from src.models.edsr_lite import EDSRLite, EDSRResidualBlock


def _exp9_config() -> dict:
    return {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 16,
        "scale": 2,
        "residual_scale": 0.1,
    }


def test_accepts_grayscale_input() -> None:
    model = EDSRLite(**_exp9_config())
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape[1] == 1


def test_output_is_exactly_2x_spatial_dimensions() -> None:
    model = EDSRLite(**_exp9_config())
    output = model(torch.randn(2, 1, 20, 24))
    assert output.shape == (2, 1, 40, 48)


@pytest.mark.parametrize(
    ("height", "width"), [(8, 8), (16, 24), (32, 32), (96, 96), (128, 128)]
)
def test_multiple_spatial_input_sizes_work(height: int, width: int) -> None:
    model = EDSRLite(num_features=8, num_blocks=2, scale=2)  # small for speed
    output = model(torch.randn(1, 1, height, width))
    assert output.shape == (1, 1, height * 2, width * 2)


def test_experiment_9_config_matches_exact_expected_param_count() -> None:
    model = EDSRLite(**_exp9_config())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Verified by direct instantiation (see final report), not derived here.
    assert param_count == 1_367_553


def test_no_batchnorm_anywhere_in_the_model() -> None:
    model = EDSRLite(**_exp9_config())
    normalization_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    assert not any(isinstance(module, normalization_types) for module in model.modules())


def test_residual_block_structure() -> None:
    block = EDSRResidualBlock(num_features=16, residual_scale=0.1)
    assert isinstance(block.conv1, nn.Conv2d)
    assert isinstance(block.conv2, nn.Conv2d)
    assert isinstance(block.relu, nn.ReLU)
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv2.kernel_size == (3, 3)
    assert block.conv1.in_channels == block.conv1.out_channels == 16
    normalization_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    assert not any(isinstance(module, normalization_types) for module in block.modules())


def test_residual_block_matches_formula_input_plus_scale_times_residual() -> None:
    torch.manual_seed(0)
    block = EDSRResidualBlock(num_features=4, residual_scale=0.1)
    x = torch.randn(2, 4, 8, 8)
    expected_residual = block.conv2(block.relu(block.conv1(x)))
    expected = x + 0.1 * expected_residual
    actual = block(x)
    assert torch.allclose(actual, expected)


def test_residual_scale_of_zero_reduces_block_to_pure_identity() -> None:
    block = EDSRResidualBlock(num_features=4, residual_scale=0.0)
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(block(x), x)


def test_different_residual_scales_produce_different_outputs() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 4, 8, 8)
    torch.manual_seed(42)
    block_small = EDSRResidualBlock(num_features=4, residual_scale=0.1)
    torch.manual_seed(42)
    block_large = EDSRResidualBlock(num_features=4, residual_scale=1.0)
    # Same initial weights (same seed), different scale -> different output.
    assert not torch.equal(block_small(x), block_large(x))


def test_forward_output_is_finite() -> None:
    model = EDSRLite(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert torch.isfinite(output).all()


def test_backward_gradients_are_finite() -> None:
    model = EDSRLite(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rejects_mismatched_input_channels() -> None:
    model = EDSRLite(in_channels=1, num_features=8, num_blocks=2)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(1, 3, 16, 16))


def test_output_channels_match_configured_out_channels() -> None:
    model = EDSRLite(in_channels=1, out_channels=3, num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 12, 12))
    assert output.shape == (2, 3, 24, 24)
