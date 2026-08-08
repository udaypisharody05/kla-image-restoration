"""Fast, dataset-free tests for the small residual CNN restoration model."""

import pytest
import torch

from src.models import ResidualSRNet


@pytest.mark.parametrize(
    ("batch_size", "channels", "height", "width"),
    [(1, 1, 16, 16), (5, 1, 20, 24)],
)
def test_forward_output_shape_matches_2x_upscale(
    batch_size: int, channels: int, height: int, width: int
) -> None:
    model = ResidualSRNet(
        in_channels=channels, out_channels=channels, num_features=8, num_blocks=2, scale=2
    )
    output = model(torch.randn(batch_size, channels, height, width))
    assert output.shape == (batch_size, channels, height * 2, width * 2)


def test_output_channels_match_configured_out_channels() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=3, num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 12, 12))
    assert output.shape == (2, 3, 24, 24)


def test_output_contains_finite_values() -> None:
    model = ResidualSRNet(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert torch.isfinite(output).all()


def test_rejects_mismatched_input_channels() -> None:
    model = ResidualSRNet(in_channels=1, num_features=8, num_blocks=2)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(1, 3, 16, 16))


def test_gradients_flow_through_the_full_model() -> None:
    model = ResidualSRNet(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
