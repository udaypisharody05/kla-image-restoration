"""Fast, dataset-free tests for the small residual CNN restoration model."""

import pytest
import torch

from src.models import EDSRLite, NAFNetSR, ResidualSRNet, build_model, build_model_config


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


# --- Experiment 9: model factory (build_model_config/build_model) ---


def test_residual_sr_config_omits_architecture_key_for_legacy_compatibility() -> None:
    """Historical model_config (Exp 1-8) never had an "architecture" key; the
    factory must reproduce that exact shape for residual_sr so old-vs-new
    dict-equality resume checks keep working with no special-casing."""
    config = build_model_config("residual_sr", num_features=64, num_blocks=8, scale=2)
    assert "architecture" not in config
    assert config == {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }


def test_edsr_lite_config_includes_architecture_and_residual_scale() -> None:
    config = build_model_config(
        "edsr_lite", num_features=64, num_blocks=16, scale=2, residual_scale=0.1
    )
    assert config == {
        "architecture": "edsr_lite",
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 16,
        "scale": 2,
        "residual_scale": 0.1,
    }


def test_build_model_reconstructs_residual_sr() -> None:
    config = build_model_config("residual_sr", num_features=8, num_blocks=2, scale=2)
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_build_model_reconstructs_edsr_lite() -> None:
    config = build_model_config(
        "edsr_lite", num_features=8, num_blocks=2, scale=2, residual_scale=0.1
    )
    model = build_model(config)
    assert isinstance(model, EDSRLite)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_build_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model({"architecture": "transformer_sr", "in_channels": 1})


def test_legacy_checkpoint_model_config_without_architecture_key_loads_as_residual_sr() -> None:
    """Simulates a real Exp1-8 checkpoint's model_config verbatim."""
    legacy_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }
    model = build_model(legacy_config)
    assert isinstance(model, ResidualSRNet)


def test_experiment_6_style_model_config_still_works_through_factory() -> None:
    """Experiment 6's exact real model_config, reconstructed through build_model."""
    exp6_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }
    model = build_model(exp6_config)
    assert isinstance(model, ResidualSRNet)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 630_724


# --- Experiment 12: NAFNet-SR model factory support ---


def test_nafnet_sr_config_includes_architecture_and_expansion_factors() -> None:
    config = build_model_config(
        "nafnet_sr", num_features=64, num_blocks=8, scale=2, dw_expand=2, ffn_expand=2
    )
    assert config == {
        "architecture": "nafnet_sr",
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "dw_expand": 2,
        "ffn_expand": 2,
    }


def test_build_model_reconstructs_nafnet_sr() -> None:
    config = build_model_config("nafnet_sr", num_features=8, num_blocks=2, scale=2)
    model = build_model(config)
    assert isinstance(model, NAFNetSR)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_experiment_12_style_model_config_still_works_through_factory() -> None:
    """Experiment 12's exact intended model_config, reconstructed through build_model."""
    exp12_config = {
        "architecture": "nafnet_sr",
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "dw_expand": 2,
        "ffn_expand": 2,
    }
    model = build_model(exp12_config)
    assert isinstance(model, NAFNetSR)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 432_129
