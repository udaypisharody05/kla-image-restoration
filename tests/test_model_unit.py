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


# --- Experiment 3 capacity configuration (32/4 baseline vs 64/8 wider/deeper) ---


def test_experiment_1_2_capacity_config_still_works() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=32, num_blocks=4, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 84_708


def test_experiment_3_capacity_config_constructs_and_produces_expected_shape() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    batch = torch.randn(3, 1, 64, 64)
    output = model(batch)
    assert output.shape == (3, 1, 128, 128)
    assert torch.isfinite(output).all()


def test_experiment_3_capacity_config_backward_pass_succeeds() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    output = model(torch.randn(2, 1, 64, 64))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 128, 128))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_experiment_3_capacity_config_matches_exact_expected_param_count() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Verified by direct instantiation (see final report), not derived here.
    assert param_count == 630_724


def test_different_capacity_configs_produce_different_param_counts() -> None:
    exp2_params = sum(
        p.numel()
        for p in ResidualSRNet(num_features=32, num_blocks=4).parameters()
        if p.requires_grad
    )
    exp3_params = sum(
        p.numel()
        for p in ResidualSRNet(num_features=64, num_blocks=8).parameters()
        if p.requires_grad
    )
    assert exp2_params == 84_708
    assert exp3_params == 630_724
    assert exp3_params > exp2_params
