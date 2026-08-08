"""Fast, dataset-free tests for torch-tensor PSNR/SSIM metric utilities."""

import math

import pytest
import torch

from src.metrics import psnr, ssim


def test_psnr_known_constant_difference() -> None:
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.full((1, 1, 8, 8), 0.5)
    assert psnr(prediction, target) == pytest.approx(6.020599913, abs=1e-6)


def test_psnr_identical_images_is_infinite() -> None:
    image = torch.rand(2, 1, 8, 8)
    assert math.isinf(psnr(image, image.clone()))


def test_psnr_random_batch_is_finite() -> None:
    value = psnr(torch.rand(4, 1, 16, 16), torch.rand(4, 1, 16, 16))
    assert math.isfinite(value)


def test_ssim_random_batch_is_finite_and_bounded() -> None:
    value = ssim(torch.rand(3, 1, 16, 16), torch.rand(3, 1, 16, 16))
    assert math.isfinite(value)
    assert value <= 1.0


def test_ssim_identical_images_is_approximately_one() -> None:
    image = torch.rand(2, 1, 16, 16)
    assert ssim(image, image.clone()) == pytest.approx(1.0)


def test_metrics_accept_unbatched_chw_tensor() -> None:
    image = torch.rand(1, 16, 16)
    assert math.isfinite(psnr(image, image.clone() * 0.9))
    assert math.isfinite(ssim(image, image.clone() * 0.9))


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        psnr(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16))


def test_psnr_clips_out_of_range_predictions_by_default() -> None:
    # Matches src.baseline.metric_arrays' default clip_prediction=True convention.
    target = torch.zeros(1, 1, 8, 8)
    out_of_range = torch.full((1, 1, 8, 8), 5.0)
    clamped_equivalent = torch.ones(1, 1, 8, 8)
    assert psnr(out_of_range, target) == pytest.approx(psnr(clamped_equivalent, target))


def test_ssim_clips_out_of_range_predictions_by_default() -> None:
    target = torch.zeros(1, 1, 8, 8)
    out_of_range = torch.full((1, 1, 8, 8), -5.0)
    clamped_equivalent = torch.zeros(1, 1, 8, 8)
    assert ssim(out_of_range, target) == pytest.approx(ssim(clamped_equivalent, target))


def test_clip_prediction_false_uses_raw_values() -> None:
    target = torch.zeros(1, 1, 8, 8)
    out_of_range = torch.full((1, 1, 8, 8), 5.0)
    clipped = psnr(out_of_range, target, clip_prediction=True)
    unclipped = psnr(out_of_range, target, clip_prediction=False)
    assert clipped != pytest.approx(unclipped)


def test_target_is_never_clipped() -> None:
    # Only the prediction is clipped, matching src.baseline.metric_arrays.
    prediction = torch.zeros(1, 1, 8, 8)
    out_of_range_target = torch.full((1, 1, 8, 8), 5.0)
    assert psnr(prediction, out_of_range_target) == pytest.approx(
        psnr(prediction, out_of_range_target, clip_prediction=False)
    )
