"""Fast, dataset-free tests for the small residual CNN restoration model."""

import pytest
import torch

from src.models import (
    ChannelAttention,
    DenoiseStem,
    EDSRLite,
    MultiScaleBlock,
    NAFNetSR,
    ResidualDenseBlock,
    ResidualSRBicubic,
    ResidualSRNet,
    SimpleGateBlock,
    SwinIRLite,
    build_model,
    build_model_config,
)
from src.models.residual_sr import ResidualBlock


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


# --- Experiment 13: SwinIR-lite model factory support ---


def test_swinir_lite_config_includes_architecture_and_transformer_fields() -> None:
    config = build_model_config(
        "swinir_lite", embed_dim=60, depth=6, num_heads=6, window_size=8, mlp_ratio=2.0, scale=2
    )
    assert config == {
        "architecture": "swinir_lite",
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 60,
        "depth": 6,
        "num_heads": 6,
        "window_size": 8,
        "mlp_ratio": 2.0,
        "scale": 2,
    }


def test_build_model_reconstructs_swinir_lite() -> None:
    config = build_model_config(
        "swinir_lite", embed_dim=8, depth=2, num_heads=2, window_size=4, mlp_ratio=2.0, scale=2
    )
    model = build_model(config)
    assert isinstance(model, SwinIRLite)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_experiment_13_style_model_config_still_works_through_factory() -> None:
    """Experiment 13's exact intended model_config, reconstructed through build_model."""
    exp13_config = {
        "architecture": "swinir_lite",
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 60,
        "depth": 6,
        "num_heads": 6,
        "window_size": 8,
        "mlp_ratio": 2.0,
        "scale": 2,
    }
    model = build_model(exp13_config)
    assert isinstance(model, SwinIRLite)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 348_421


# --- Experiment 17: bicubic-residual model factory support ---


def test_residual_sr_bicubic_config_includes_architecture_identifier() -> None:
    config = build_model_config("residual_sr_bicubic", num_features=64, num_blocks=8, scale=2)
    assert config == {
        "architecture": "residual_sr_bicubic",
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }


def test_build_model_reconstructs_residual_sr_bicubic() -> None:
    config = build_model_config("residual_sr_bicubic", num_features=8, num_blocks=2, scale=2)
    model = build_model(config)
    assert isinstance(model, ResidualSRBicubic)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_experiment_17_style_model_config_still_works_through_factory() -> None:
    """Experiment 17's exact intended model_config, reconstructed through build_model."""
    exp17_config = {
        "architecture": "residual_sr_bicubic",
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }
    model = build_model(exp17_config)
    assert isinstance(model, ResidualSRBicubic)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 630_724


# --- Channel attention (lightweight squeeze-and-excitation, optional) ---


def test_channel_attention_module_preserves_shape_and_is_finite() -> None:
    attention = ChannelAttention(num_features=16, reduction=8)
    x = torch.randn(2, 16, 12, 12)
    output = attention(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_channel_attention_gate_is_a_channelwise_multiplicative_scale() -> None:
    """Zeroing the input must zero the output regardless of the gate's bias
    terms (the operation is x * sigmoid(...), never an additive term)."""
    attention = ChannelAttention(num_features=8, reduction=4)
    output = attention(torch.zeros(1, 8, 6, 6))
    assert torch.equal(output, torch.zeros(1, 8, 6, 6))


def test_channel_attention_rejects_non_positive_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        ChannelAttention(num_features=8, reduction=0)


def test_channel_attention_backward_pass_finite_gradients() -> None:
    attention = ChannelAttention(num_features=8, reduction=4)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    output = attention(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_residual_block_default_creates_no_attention_submodule() -> None:
    """use_attention=False (the default) must not even construct a disabled
    attention module -- this is what keeps state_dict keys and parameter
    count identical to before channel attention existed."""
    block = ResidualBlock(num_features=16)
    assert block.attention is None


def test_residual_block_with_attention_creates_the_submodule() -> None:
    block = ResidualBlock(num_features=16, use_attention=True, reduction=4)
    assert isinstance(block.attention, ChannelAttention)


def test_residual_sr_default_construction_is_unaffected_by_attention_support() -> None:
    """The exact historical call site (Experiment 3's 64F/8B config) must
    still produce the exact historical parameter count -- proves adding the
    optional channel_attention/multiscale_block kwargs changed nothing about
    default construction."""
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 630_724
    assert model.body[0].attention is None


def test_residual_sr_with_channel_attention_forward_shape() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, channel_attention=True)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()


def test_residual_sr_channel_attention_increases_param_count_modestly() -> None:
    base = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    attention_model = ResidualSRNet(
        in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2,
        channel_attention=True, attention_reduction=8,
    )
    base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    attention_params = sum(p.numel() for p in attention_model.parameters() if p.requires_grad)
    assert attention_params > base_params
    # "modest": well under a 5% increase at the champion 64F/8B capacity.
    assert (attention_params - base_params) / base_params < 0.05


def test_residual_sr_channel_attention_backward_pass_finite_gradients() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, channel_attention=True)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_build_model_config_residual_sr_omits_attention_keys_when_disabled() -> None:
    """Channel-attention-off must reproduce the exact historical dict -- an
    old checkpoint's model_config still compares equal for resume."""
    config = build_model_config("residual_sr", num_features=64, num_blocks=8, scale=2)
    assert "channel_attention" not in config
    assert "attention_reduction" not in config


def test_build_model_config_residual_sr_includes_attention_keys_when_enabled() -> None:
    config = build_model_config(
        "residual_sr", num_features=64, num_blocks=8, scale=2,
        channel_attention=True, attention_reduction=4,
    )
    assert config == {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "channel_attention": True,
        "attention_reduction": 4,
    }


def test_build_model_reconstructs_residual_sr_with_channel_attention() -> None:
    config = build_model_config(
        "residual_sr", num_features=8, num_blocks=2, scale=2, channel_attention=True
    )
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    assert model.body[0].attention is not None
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_legacy_model_config_without_attention_keys_still_builds_plain_residual_sr() -> None:
    legacy_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 64, "num_blocks": 8, "scale": 2,
    }
    model = build_model(legacy_config)
    assert model.body[0].attention is None
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 630_724


# --- Multi-scale receptive-field block (optional, local+dilated 3x3 fusion) ---


def test_multiscale_block_preserves_shape_and_is_finite() -> None:
    block = MultiScaleBlock(num_features=16)
    x = torch.randn(2, 16, 12, 12)
    output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_multiscale_block_default_creates_no_attention_submodule() -> None:
    block = MultiScaleBlock(num_features=16)
    assert block.attention is None


def test_multiscale_block_with_attention_creates_the_submodule() -> None:
    block = MultiScaleBlock(num_features=16, use_attention=True, reduction=4)
    assert isinstance(block.attention, ChannelAttention)


def test_multiscale_block_backward_pass_finite_gradients() -> None:
    block = MultiScaleBlock(num_features=8)
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    output = block(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_multiscale_block_dilated_branch_has_dilation_2() -> None:
    block = MultiScaleBlock(num_features=8)
    assert block.branch_dilated.dilation == (2, 2)
    assert block.branch_local.dilation == (1, 1)


def test_residual_sr_with_multiscale_block_forward_shape() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, multiscale_block=True)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert isinstance(model.body[0], MultiScaleBlock)


def test_residual_sr_multiscale_block_increases_param_count_without_ballooning() -> None:
    base = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    multiscale_model = ResidualSRNet(
        in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2,
        multiscale_block=True,
    )
    base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    multiscale_params = sum(p.numel() for p in multiscale_model.parameters() if p.requires_grad)
    assert multiscale_params > base_params
    # "lightweight...without creating a large model": well under a 25% increase.
    assert (multiscale_params - base_params) / base_params < 0.25


def test_residual_sr_multiscale_block_backward_pass_finite_gradients() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, multiscale_block=True)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_build_model_config_residual_sr_omits_multiscale_key_when_disabled() -> None:
    config = build_model_config("residual_sr", num_features=64, num_blocks=8, scale=2)
    assert "multiscale_block" not in config


def test_build_model_config_residual_sr_includes_multiscale_key_when_enabled() -> None:
    config = build_model_config(
        "residual_sr", num_features=64, num_blocks=8, scale=2, multiscale_block=True
    )
    assert config == {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "multiscale_block": True,
    }


def test_build_model_reconstructs_residual_sr_with_multiscale_block() -> None:
    config = build_model_config(
        "residual_sr", num_features=8, num_blocks=2, scale=2, multiscale_block=True
    )
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    assert isinstance(model.body[0], MultiScaleBlock)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_legacy_model_config_without_multiscale_key_still_builds_plain_residual_sr() -> None:
    legacy_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 64, "num_blocks": 8, "scale": 2,
    }
    model = build_model(legacy_config)
    assert not isinstance(model.body[0], MultiScaleBlock)
    assert isinstance(model.body[0], ResidualBlock)


# --- Channel attention and multi-scale block are independently selectable ---


def test_channel_attention_alone_does_not_enable_multiscale_block() -> None:
    config = build_model_config("residual_sr", channel_attention=True)
    assert "multiscale_block" not in config
    model = build_model(config)
    assert isinstance(model.body[0], ResidualBlock)
    assert not isinstance(model.body[0], MultiScaleBlock)


def test_multiscale_block_alone_does_not_enable_channel_attention() -> None:
    config = build_model_config("residual_sr", multiscale_block=True)
    assert "channel_attention" not in config
    model = build_model(config)
    assert model.body[0].attention is None


def test_channel_attention_and_multiscale_block_can_be_explicitly_combined() -> None:
    """Not auto-combined (the two tests above), but nothing prevents an
    explicit ablation of both together."""
    model = ResidualSRNet(
        num_features=16, num_blocks=2, scale=2, channel_attention=True, multiscale_block=True
    )
    assert isinstance(model.body[0], MultiScaleBlock)
    assert model.body[0].attention is not None
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


# --- Residual Dense Block (Phase 10, lightweight RDN-inspired variant) -----


def test_rdb_preserves_shape_and_is_finite() -> None:
    block = ResidualDenseBlock(num_features=16, growth_rate=8, num_layers=3)
    x = torch.randn(2, 16, 12, 12)
    output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_rdb_default_creates_no_attention_submodule() -> None:
    block = ResidualDenseBlock(num_features=16)
    assert block.attention is None


def test_rdb_with_attention_creates_the_submodule() -> None:
    block = ResidualDenseBlock(num_features=16, use_attention=True, reduction=4)
    assert isinstance(block.attention, ChannelAttention)


def test_rdb_backward_pass_finite_gradients() -> None:
    block = ResidualDenseBlock(num_features=8, growth_rate=4, num_layers=2)
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    output = block(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rdb_dense_layers_grow_channel_count() -> None:
    block = ResidualDenseBlock(num_features=16, growth_rate=8, num_layers=3)
    # layer i consumes 16 + i*8 input channels (dense concatenation).
    assert block.layers[0].in_channels == 16
    assert block.layers[1].in_channels == 24
    assert block.layers[2].in_channels == 32
    assert block.local_feature_fusion.in_channels == 16 + 3 * 8


def test_rdb_rejects_non_positive_growth_rate_or_num_layers() -> None:
    with pytest.raises(ValueError, match="growth_rate"):
        ResidualDenseBlock(num_features=16, growth_rate=0)
    with pytest.raises(ValueError, match="num_layers"):
        ResidualDenseBlock(num_features=16, num_layers=0)


def test_residual_sr_with_rdb_block_forward_shape() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, rdb_block=True)
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert isinstance(model.body[0], ResidualDenseBlock)


def test_residual_sr_rdb_block_stays_under_one_million_params_at_champion_capacity() -> None:
    model = ResidualSRNet(
        in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2, rdb_block=True
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 1_000_000


def test_residual_sr_rdb_block_backward_pass_finite_gradients() -> None:
    model = ResidualSRNet(num_features=16, num_blocks=2, scale=2, rdb_block=True)
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_residual_sr_rdb_block_and_multiscale_block_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ResidualSRNet(num_features=16, num_blocks=2, scale=2, rdb_block=True, multiscale_block=True)


def test_build_model_config_residual_sr_omits_rdb_keys_when_disabled() -> None:
    config = build_model_config("residual_sr", num_features=64, num_blocks=8, scale=2)
    assert "rdb_block" not in config
    assert "rdb_growth_rate" not in config
    assert "rdb_num_layers" not in config


def test_build_model_config_residual_sr_includes_rdb_keys_when_enabled() -> None:
    config = build_model_config(
        "residual_sr", num_features=64, num_blocks=8, scale=2, rdb_block=True,
        rdb_growth_rate=16, rdb_num_layers=3,
    )
    assert config == {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "rdb_block": True,
        "rdb_growth_rate": 16,
        "rdb_num_layers": 3,
    }


def test_build_model_reconstructs_residual_sr_with_rdb_block() -> None:
    config = build_model_config("residual_sr", num_features=8, num_blocks=2, scale=2, rdb_block=True)
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    assert isinstance(model.body[0], ResidualDenseBlock)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_legacy_model_config_without_rdb_key_still_builds_plain_residual_sr() -> None:
    legacy_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 64, "num_blocks": 8, "scale": 2,
    }
    model = build_model(legacy_config)
    assert not isinstance(model.body[0], ResidualDenseBlock)
    assert isinstance(model.body[0], ResidualBlock)


# --- Denoise stem (Phase 3A, optional pre-trunk denoising) -----------------


def test_simple_gate_block_preserves_shape_and_is_finite() -> None:
    block = SimpleGateBlock(channels=16)
    x = torch.randn(2, 16, 12, 12)
    output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_simple_gate_block_backward_pass_finite_gradients() -> None:
    block = SimpleGateBlock(channels=8)
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    output = block(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_denoise_stem_preserves_input_shape_and_channels() -> None:
    stem = DenoiseStem(in_channels=1, stem_features=16, num_blocks=2)
    x = torch.randn(2, 1, 24, 24)
    output = stem(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_denoise_stem_two_channel_input_for_noise_conditioning_compatibility() -> None:
    """Must accept in_channels=2 so it composes with Experiment 25's
    noise-conditioning wrapper (which concatenates a sigma channel onto LR)."""
    stem = DenoiseStem(in_channels=2, stem_features=16, num_blocks=2)
    output = stem(torch.randn(2, 2, 24, 24))
    assert output.shape == (2, 2, 24, 24)


def test_denoise_stem_rejects_non_positive_num_blocks() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        DenoiseStem(num_blocks=0)


def test_denoise_stem_backward_pass_finite_gradients() -> None:
    stem = DenoiseStem(in_channels=1, stem_features=8, num_blocks=2)
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    output = stem(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_residual_sr_default_construction_is_unaffected_by_denoise_stem_support() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=32, num_blocks=4, scale=2)
    assert model.stem is None


def test_residual_sr_with_denoise_stem_forward_shape() -> None:
    model = ResidualSRNet(
        num_features=16, num_blocks=2, scale=2, denoise_stem=True,
        denoise_stem_features=8, denoise_stem_blocks=2,
    )
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert isinstance(model.stem, DenoiseStem)


def test_residual_sr_denoise_stem_increases_param_count_modestly() -> None:
    base = ResidualSRNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2)
    stem_model = ResidualSRNet(
        in_channels=1, out_channels=1, num_features=64, num_blocks=8, scale=2,
        denoise_stem=True, denoise_stem_features=32, denoise_stem_blocks=2,
    )
    base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    stem_params = sum(p.numel() for p in stem_model.parameters() if p.requires_grad)
    assert stem_params > base_params
    assert (stem_params - base_params) / base_params < 0.25


def test_residual_sr_denoise_stem_backward_pass_finite_gradients() -> None:
    model = ResidualSRNet(
        num_features=16, num_blocks=2, scale=2, denoise_stem=True,
        denoise_stem_features=8, denoise_stem_blocks=2,
    )
    output = model(torch.randn(2, 1, 16, 16))
    loss = torch.nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_build_model_config_residual_sr_omits_denoise_stem_keys_when_disabled() -> None:
    config = build_model_config("residual_sr", num_features=64, num_blocks=8, scale=2)
    assert "denoise_stem" not in config
    assert "denoise_stem_features" not in config
    assert "denoise_stem_blocks" not in config


def test_build_model_config_residual_sr_includes_denoise_stem_keys_when_enabled() -> None:
    config = build_model_config(
        "residual_sr", num_features=64, num_blocks=8, scale=2, denoise_stem=True,
        denoise_stem_features=32, denoise_stem_blocks=2,
    )
    assert config == {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
        "denoise_stem": True,
        "denoise_stem_features": 32,
        "denoise_stem_blocks": 2,
    }


def test_build_model_reconstructs_residual_sr_with_denoise_stem() -> None:
    config = build_model_config(
        "residual_sr", num_features=8, num_blocks=2, scale=2, denoise_stem=True,
        denoise_stem_features=8, denoise_stem_blocks=2,
    )
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    assert isinstance(model.stem, DenoiseStem)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)


def test_legacy_model_config_without_denoise_stem_key_still_builds_stemless_residual_sr() -> None:
    legacy_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 64, "num_blocks": 8, "scale": 2,
    }
    model = build_model(legacy_config)
    assert model.stem is None


def test_denoise_stem_and_rdb_block_can_be_explicitly_combined() -> None:
    """Independent mechanisms (pre-trunk stem vs. residual block type) --
    nothing prevents ablating both together."""
    model = ResidualSRNet(
        num_features=16, num_blocks=2, scale=2, rdb_block=True,
        denoise_stem=True, denoise_stem_features=8, denoise_stem_blocks=2,
    )
    assert isinstance(model.stem, DenoiseStem)
    assert isinstance(model.body[0], ResidualDenseBlock)
    output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
