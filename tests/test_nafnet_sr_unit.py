"""Fast, dataset-free tests for the NAFNet-SR architecture (src/models/nafnet_sr.py)."""

import pytest
import torch
from torch import nn

from src.models.nafnet_sr import LayerNorm2d, NAFBlock, NAFNetSR, SimpleGate
from src.tta import predict_x8


def _exp12_config() -> dict:
    """Chosen Experiment 12 configuration -- see EXPERIMENT_LOG.md for justification
    (width/depth were revised down from an initial 96f/12b pick after the CUDA
    sanity check showed that config needs ~13GB, exceeding the 8GB target GPU)."""
    return {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "dw_expand": 2,
        "ffn_expand": 2,
    }


# --- NAFNetSR: shapes, finiteness, gradients ---


def test_accepts_grayscale_input() -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape[1] == 1


def test_output_is_exactly_2x_spatial_dimensions() -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 20, 24))
    assert output.shape == (2, 1, 40, 48)


@pytest.mark.parametrize(
    ("height", "width"), [(8, 8), (9, 9), (16, 24), (32, 32), (96, 96)]
)
def test_multiple_spatial_input_sizes_work(height: int, width: int) -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)  # small for speed
    output = model(torch.randn(1, 1, height, width))
    assert output.shape == (1, 1, height * 2, width * 2)


def test_forward_output_is_finite() -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert torch.isfinite(output).all()


def test_backward_gradients_are_finite() -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rejects_mismatched_input_channels() -> None:
    model = NAFNetSR(in_channels=1, num_features=8, num_blocks=2)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(1, 3, 16, 16))


def test_output_channels_match_configured_out_channels() -> None:
    model = NAFNetSR(in_channels=1, out_channels=3, num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 12, 12))
    assert output.shape == (2, 3, 24, 24)


def test_experiment_12_config_matches_exact_expected_param_count() -> None:
    model = NAFNetSR(**_exp12_config())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Verified by direct instantiation (see final report), not derived here.
    assert param_count == 432_129


def test_no_batchnorm_anywhere_in_the_model() -> None:
    model = NAFNetSR(**_exp12_config())
    normalization_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    assert not any(isinstance(module, normalization_types) for module in model.modules())


# --- SimpleGate ---


def test_simple_gate_splits_and_multiplies() -> None:
    gate = SimpleGate()
    x = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    first_half, second_half = x.chunk(2, dim=1)
    expected = first_half * second_half
    assert torch.equal(gate(x), expected)


def test_simple_gate_halves_channel_count() -> None:
    gate = SimpleGate()
    x = torch.randn(1, 8, 4, 4)
    assert gate(x).shape == (1, 4, 4, 4)


def test_simple_gate_has_no_relu_or_gelu_submodules() -> None:
    gate = SimpleGate()
    assert not any(isinstance(module, (nn.ReLU, nn.GELU)) for module in gate.modules())
    assert list(gate.parameters()) == []  # a pure split-and-multiply, no learnable weights


def test_simple_gate_is_not_equivalent_to_relu_or_identity() -> None:
    """Confirms the gate is a genuine multiplicative nonlinearity, not silently
    reducible to a known activation."""
    gate = SimpleGate()
    x = torch.tensor([[[[1.0]], [[-2.0]], [[3.0]], [[4.0]]]])  # [1,4,1,1]
    result = gate(x)
    expected = torch.tensor([[[[1.0 * 3.0]], [[-2.0 * 4.0]]]])  # first_half * second_half
    assert torch.equal(result, expected)
    assert not torch.equal(result, torch.relu(x[:, :2]))


# --- NAFBlock ---


def test_naf_block_preserves_shape() -> None:
    block = NAFBlock(num_channels=8)
    x = torch.randn(2, 8, 10, 12)
    assert block(x).shape == x.shape


def test_naf_block_no_batchnorm() -> None:
    block = NAFBlock(num_channels=8)
    normalization_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    assert not any(isinstance(module, normalization_types) for module in block.modules())


def test_naf_block_uses_layernorm2d() -> None:
    block = NAFBlock(num_channels=8)
    assert isinstance(block.norm1, LayerNorm2d)
    assert isinstance(block.norm2, LayerNorm2d)


def test_naf_block_is_near_identity_at_zero_initialized_beta_gamma() -> None:
    """beta/gamma are initialized to zero (NAFNet's stabilizing init trick), so at
    construction time each residual branch contributes nothing -- the block's
    forward pass must equal its input exactly before any training occurs."""
    block = NAFBlock(num_channels=4)
    assert torch.equal(block.beta, torch.zeros_like(block.beta))
    assert torch.equal(block.gamma, torch.zeros_like(block.gamma))
    x = torch.randn(1, 4, 6, 6)
    assert torch.equal(block(x), x)


def test_naf_block_residual_path_changes_output_once_beta_gamma_are_nonzero() -> None:
    block = NAFBlock(num_channels=4)
    with torch.no_grad():
        block.beta.fill_(1.0)
        block.gamma.fill_(1.0)
    x = torch.randn(1, 4, 6, 6)
    assert not torch.equal(block(x), x)


def test_naf_block_output_is_finite() -> None:
    block = NAFBlock(num_channels=8)
    with torch.no_grad():
        block.beta.fill_(1.0)
        block.gamma.fill_(1.0)
    output = block(torch.randn(2, 8, 10, 10))
    assert torch.isfinite(output).all()


# --- LayerNorm2d ---


def test_layer_norm_2d_normalizes_over_channel_dimension() -> None:
    norm = LayerNorm2d(num_channels=4)
    x = torch.randn(2, 4, 5, 5) * 10 + 3  # arbitrary scale/shift
    output = norm(x)
    mean_over_channels = output.mean(dim=1)
    std_over_channels = output.std(dim=1, unbiased=False)
    assert torch.allclose(mean_over_channels, torch.zeros_like(mean_over_channels), atol=1e-4)
    assert torch.allclose(std_over_channels, torch.ones_like(std_over_channels), atol=1e-2)


def test_layer_norm_2d_is_not_batchnorm() -> None:
    norm = LayerNorm2d(num_channels=4)
    assert not isinstance(norm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    assert not hasattr(norm, "running_mean")


# --- x8 TTA compatibility (src/tta.py is unmodified; this just confirms NAFNetSR
# works through it automatically, like every other architecture already does) ---


def test_predict_x8_works_with_nafnet_sr() -> None:
    model = NAFNetSR(num_features=8, num_blocks=2, scale=2)
    result = predict_x8(model, torch.rand(2, 1, 16, 16))
    assert result.shape == (2, 1, 32, 32)
    assert torch.isfinite(result).all()
