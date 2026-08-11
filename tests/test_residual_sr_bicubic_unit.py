"""Fast, dataset-free tests for the bicubic-residual architecture
(src/models/residual_sr_bicubic.py)."""

import pytest
import torch
from torch import nn

from src.models.residual_sr_bicubic import ResidualSRBicubic, fixed_bicubic_upsample
from src.tta import predict_x8


def _exp17_config() -> dict:
    """Chosen Experiment 17 configuration -- identical width/depth to Exp6."""
    return {"in_channels": 1, "out_channels": 1, "num_features": 64, "num_blocks": 8, "scale": 2}


# --- Construction, shapes, parameter count ---


def test_construction_succeeds_with_exp17_config() -> None:
    model = ResidualSRBicubic(**_exp17_config())
    assert isinstance(model, nn.Module)


def test_exp17_config_matches_exact_expected_param_count() -> None:
    model = ResidualSRBicubic(**_exp17_config())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Identical topology to ResidualSRNet (the bicubic skip has no parameters).
    assert param_count == 630_724


def test_96x96_training_crop_size_produces_192x192_output() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(1, 1, 96, 96))
    assert output.shape == (1, 1, 192, 192)


def test_128x128_full_validation_image_produces_256x256_output() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(1, 1, 128, 128))
    assert output.shape == (1, 1, 256, 256)


def test_batch16_behavior() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(16, 1, 96, 96))
    assert output.shape == (16, 1, 192, 192)


def test_output_is_exactly_2x_spatial_dimensions() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 20, 24))
    assert output.shape == (2, 1, 40, 48)


def test_rejects_mismatched_input_channels() -> None:
    model = ResidualSRBicubic(in_channels=1, num_features=8, num_blocks=2)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(1, 3, 16, 16))


def test_rejects_in_channels_not_equal_out_channels() -> None:
    with pytest.raises(ValueError, match="in_channels == out_channels"):
        ResidualSRBicubic(in_channels=1, out_channels=3, num_features=8, num_blocks=2)


# --- Finiteness / gradients ---


def test_forward_output_is_finite() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    assert torch.isfinite(output).all()


def test_backward_gradients_are_finite() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    output = model(torch.randn(2, 1, 16, 16))
    loss = nn.functional.l1_loss(output, torch.randn(2, 1, 32, 32))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# --- Bicubic skip: parameter-free, device/dtype, parity, no clipping ---


def test_bicubic_skip_has_no_parameters() -> None:
    """fixed_bicubic_upsample is a plain function (F.interpolate) -- confirm the
    model's own parameter count includes nothing beyond the learned branch by
    comparing against a model with the same learned-branch topology
    (ResidualSRNet) and identical num_features/num_blocks."""
    from src.models import ResidualSRNet

    bicubic_model = ResidualSRBicubic(num_features=16, num_blocks=3, scale=2)
    plain_model = ResidualSRNet(num_features=16, num_blocks=3, scale=2)
    bicubic_params = sum(p.numel() for p in bicubic_model.parameters() if p.requires_grad)
    plain_params = sum(p.numel() for p in plain_model.parameters() if p.requires_grad)
    assert bicubic_params == plain_params


def test_fixed_bicubic_upsample_preserves_batch_and_channel_dims() -> None:
    x = torch.rand(3, 1, 16, 24)
    result = fixed_bicubic_upsample(x, scale=2)
    assert result.shape == (3, 1, 32, 48)


def test_fixed_bicubic_upsample_matches_input_device_and_dtype() -> None:
    x = torch.rand(1, 1, 8, 8, dtype=torch.float32)
    result = fixed_bicubic_upsample(x, scale=2)
    assert result.device == x.device
    assert result.dtype == x.dtype


def test_zero_learned_residual_produces_exact_bicubic_output() -> None:
    """Zeroing the final conv layer's weight/bias makes the learned residual
    branch output exactly zero -- under that condition the model's output
    must equal its own fixed bicubic upsample exactly. This is the key parity
    check that the global skip is wired correctly."""
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    with torch.no_grad():
        model.upsample_conv.weight.zero_()
        model.upsample_conv.bias.zero_()
    x = torch.rand(2, 1, 16, 16)
    output = model(x)
    expected = fixed_bicubic_upsample(x, scale=2)
    assert torch.allclose(output, expected, atol=1e-6)


def test_summed_output_is_not_clamped() -> None:
    """Feeding an input far outside [0,1] must produce output far outside
    [0,1] too -- confirms nothing clips the bicubic+residual sum."""
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    x = torch.full((1, 1, 8, 8), 5.0)
    output = model(x)
    assert output.max().item() > 1.0


def test_negative_input_also_survives_unclamped() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    x = torch.full((1, 1, 8, 8), -5.0)
    output = model(x)
    assert output.min().item() < 0.0


# --- x8 TTA compatibility (model-level; no double-adding bicubic) ---


def test_x8_tta_works_on_residual_sr_bicubic() -> None:
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    result = predict_x8(model, torch.rand(1, 1, 16, 16))
    assert result.shape == (1, 1, 32, 32)
    assert torch.isfinite(result).all()


def test_x8_tta_averages_the_complete_model_output_not_bicubic_twice() -> None:
    """predict_x8 must call the *whole* model (bicubic skip included) once per
    transform and average those complete outputs -- not add an extra bicubic
    term on top. Cross-checked against a manual x8 computation using the same
    primitives as src/tta.py."""
    from src.tta import d4_transforms, forward_transform, inverse_transform

    torch.manual_seed(0)
    model = ResidualSRBicubic(num_features=8, num_blocks=2, scale=2)
    x = torch.rand(1, 1, 16, 16)

    manual_predictions = []
    with torch.no_grad():
        for flip, k in d4_transforms():
            transformed = forward_transform(x, flip, k)
            output = model(transformed)
            manual_predictions.append(inverse_transform(output, flip, k))
    expected = torch.stack(manual_predictions, dim=0).mean(dim=0)

    actual = predict_x8(model, x)
    assert torch.allclose(actual, expected, atol=1e-6)
