"""Fast, dataset-free tests for x8 geometric self-ensemble TTA (src/tta.py)."""

import pytest
import torch
from torch import nn

from src.models import ResidualSRNet
from src.tta import d4_transforms, forward_transform, inverse_transform, predict_x8


# --- The 8 D4 transforms themselves ---


def test_exactly_8_unique_transforms_exist() -> None:
    transforms = list(d4_transforms())
    assert len(transforms) == 8
    assert len(set(transforms)) == 8  # no duplicates


@pytest.mark.parametrize("flip,k", list(d4_transforms()))
def test_transform_followed_by_inverse_reproduces_original_exactly(flip: bool, k: int) -> None:
    x = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
    transformed = forward_transform(x, flip, k)
    restored = inverse_transform(transformed, flip, k)
    assert torch.equal(x, restored)


def test_batch_dimension_is_preserved_by_transforms() -> None:
    x = torch.rand(5, 1, 8, 8)
    for flip, k in d4_transforms():
        assert forward_transform(x, flip, k).shape[0] == 5


def test_channel_dimension_is_preserved_by_transforms() -> None:
    x = torch.rand(2, 3, 8, 8)
    for flip, k in d4_transforms():
        assert forward_transform(x, flip, k).shape[1] == 3


@pytest.mark.parametrize("size", [7, 8, 9, 16])  # odd and even square sizes
def test_odd_and_even_spatial_dimensions_round_trip_exactly(size: int) -> None:
    x = torch.rand(1, 1, size, size)
    for flip, k in d4_transforms():
        transformed = forward_transform(x, flip, k)
        assert transformed.shape[-2:] == (size, size)  # square in -> square out
        assert torch.equal(inverse_transform(transformed, flip, k), x)


def test_rectangular_non_square_inputs_still_round_trip_exactly() -> None:
    """Not used by this project's square crops, but the math must not silently
    assume squareness -- rotations swap H/W for a rectangle."""
    x = torch.rand(1, 1, 5, 9)
    for flip, k in d4_transforms():
        transformed = forward_transform(x, flip, k)
        assert torch.equal(inverse_transform(transformed, flip, k), x)


# --- predict_x8 ---


def test_x8_prediction_returns_expected_2x_spatial_dimensions() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    prediction = predict_x8(model, torch.rand(2, 1, 16, 16))
    assert prediction.shape == (2, 1, 32, 32)


def test_equivariant_model_gives_same_result_with_and_without_x8_tta() -> None:
    """A nearest-neighbor 2x upsample commutes exactly with every D4 transform,
    so x8-averaging 8 identical values must reproduce the single-pass result."""

    class NearestUpsample2x(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    model = NearestUpsample2x()
    x = torch.rand(3, 1, 8, 10)  # batch>1, non-square, to stress the claim
    single_pass = model(x)
    tta_result = predict_x8(model, x)
    assert torch.allclose(single_pass, tta_result, atol=1e-6)


def test_predict_x8_averages_raw_predictions_matching_manual_computation() -> None:
    """Cross-checks predict_x8's internal averaging against an independent,
    manually-computed average using the same primitives."""
    torch.manual_seed(0)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    x = torch.rand(2, 1, 8, 8)

    manual_predictions = []
    with torch.no_grad():
        for flip, k in d4_transforms():
            transformed = forward_transform(x, flip, k)
            output = model(transformed)
            manual_predictions.append(inverse_transform(output, flip, k))
    expected = torch.stack(manual_predictions, dim=0).mean(dim=0)

    actual = predict_x8(model, x)
    assert torch.allclose(actual, expected, atol=1e-6)


class _ConstantModel(nn.Module):
    """Always returns a fixed constant value, regardless of input -- used to prove
    predict_x8 never clips individual branch outputs before averaging."""

    def __init__(self, value: float, out_size: int) -> None:
        super().__init__()
        self.value = value
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape[0], x.shape[1]
        return torch.full((n, c, self.out_size, self.out_size), self.value)


def test_no_individual_output_clipping_occurs_before_averaging() -> None:
    """If any branch were clipped to [0,1] before averaging, this constant-2.0
    model's x8 average would come out <= 1.0. It must not."""
    model = _ConstantModel(value=2.0, out_size=16)
    result = predict_x8(model, torch.rand(1, 1, 8, 8))
    assert torch.allclose(result, torch.full_like(result, 2.0))
    assert result.max().item() > 1.0


def test_negative_raw_values_also_survive_averaging_unclipped() -> None:
    model = _ConstantModel(value=-0.5, out_size=16)
    result = predict_x8(model, torch.rand(1, 1, 8, 8))
    assert torch.allclose(result, torch.full_like(result, -0.5))
    assert result.min().item() < 0.0


def test_x8_output_is_finite() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    result = predict_x8(model, torch.rand(2, 1, 16, 16))
    assert torch.isfinite(result).all()


def test_x8_works_on_cpu() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2).to("cpu")
    x = torch.rand(1, 1, 8, 8, device="cpu")
    result = predict_x8(model, x)
    assert result.device.type == "cpu"


def test_x8_works_with_batch_greater_than_one() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    result = predict_x8(model, torch.rand(4, 1, 16, 16))
    assert result.shape[0] == 4


def test_predict_x8_does_not_leave_model_in_training_mode_if_it_started_eval() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    model.eval()
    predict_x8(model, torch.rand(1, 1, 8, 8))
    assert not model.training


def test_predict_x8_restores_training_mode_if_model_was_training() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    model.train()
    predict_x8(model, torch.rand(1, 1, 8, 8))
    assert model.training


def test_predict_x8_produces_no_gradients() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    x = torch.rand(1, 1, 8, 8)
    result = predict_x8(model, x)
    assert not result.requires_grad
