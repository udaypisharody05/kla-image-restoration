"""Portable tests for deterministic splits and the bicubic baseline."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluate_baseline import evaluate_validation_baseline
from src.baseline import (
    bicubic_upscale,
    evaluate_metrics,
    lpips_input_tensor,
    metric_arrays,
    peak_signal_noise_ratio,
    structural_similarity_index,
)
from src.splits import split_pairs


def test_split_is_reproducible_complete_and_disjoint() -> None:
    pairs = tuple(range(100))
    first_train, first_validation = split_pairs(pairs, val_fraction=0.2, seed=42)
    second_train, second_validation = split_pairs(pairs, val_fraction=0.2, seed=42)
    other_train, other_validation = split_pairs(pairs, val_fraction=0.2, seed=7)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 80 and len(first_validation) == 20
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train) | set(first_validation) == set(pairs)
    assert other_validation != first_validation
    assert set(other_train) | set(other_validation) == set(pairs)


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.0, 1.1])
def test_split_rejects_invalid_validation_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        split_pairs(tuple(range(10)), val_fraction=fraction)


def test_bicubic_is_float_deterministic_and_does_not_mutate_input() -> None:
    source = np.linspace(-0.2, 1.2, 128 * 128, dtype=np.float32).reshape(128, 128)
    original = source.copy()
    first = bicubic_upscale(source)
    second = bicubic_upscale(source)

    assert first.shape == (256, 256)
    assert np.issubdtype(first.dtype, np.floating)
    assert np.array_equal(source, original)
    assert np.array_equal(first, second)
    assert first.min() < 0.0 and first.max() > 1.0


def test_psnr_known_cases() -> None:
    reference = np.zeros((8, 8), dtype=np.float32)
    assert math.isinf(peak_signal_noise_ratio(reference, reference))
    changed = np.full((8, 8), 0.5, dtype=np.float32)
    assert peak_signal_noise_ratio(changed, reference) == pytest.approx(
        6.020599913, abs=1e-8
    )


def test_ssim_identical_and_different_images() -> None:
    reference = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    assert structural_similarity_index(reference, reference) == pytest.approx(1.0)
    assert structural_similarity_index(1.0 - reference, reference) < 1.0


def test_metric_clipping_is_explicit_and_does_not_mutate_prediction() -> None:
    prediction = np.array([[-0.5, 0.5], [1.5, 0.25]], dtype=np.float32)
    target = np.zeros((2, 2), dtype=np.float32)
    original = prediction.copy()
    clipped, _ = metric_arrays(prediction, target, clip_prediction=True)
    raw, _ = metric_arrays(prediction, target, clip_prediction=False)

    assert np.array_equal(prediction, original)
    assert clipped.min() == 0.0 and clipped.max() == 1.0
    assert raw.min() == -0.5 and raw.max() == 1.5


def test_evaluate_metrics_returns_standard_values() -> None:
    target = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    metrics = evaluate_metrics(target.copy(), target)
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["lpips"] is None


# --- LPIPS input preparation (grayscale->RGB + range conversion) ---
# Uses torch directly (already a hard project dependency) rather than the
# optional 'lpips' package, so these run regardless of whether lpips/its
# pretrained weights are installed -- only LPIPSMetric itself needs those.


def test_lpips_input_tensor_replicates_grayscale_to_three_channels() -> None:
    array = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    tensor = lpips_input_tensor(array, torch)
    assert tensor.shape == (1, 3, 4, 4)
    # All 3 channels must be identical copies of the same grayscale content.
    assert torch.equal(tensor[0, 0], tensor[0, 1])
    assert torch.equal(tensor[0, 1], tensor[0, 2])


def test_lpips_input_tensor_maps_0_1_range_to_minus1_1() -> None:
    array = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    tensor = lpips_input_tensor(array, torch)
    expected = torch.tensor([[-1.0, 1.0], [0.0, -0.5]])
    assert torch.allclose(tensor[0, 0], expected, atol=1e-6)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


def test_lpips_input_tensor_clips_out_of_range_values_before_scaling() -> None:
    array = np.array([[-0.5, 1.5]], dtype=np.float32)
    tensor = lpips_input_tensor(array, torch)
    # -0.5 clips to 0.0 -> -1.0; 1.5 clips to 1.0 -> 1.0.
    assert torch.allclose(tensor[0, 0], torch.tensor([[-1.0, 1.0]]), atol=1e-6)


def test_lpips_input_tensor_does_not_mutate_input_array() -> None:
    array = np.array([[0.2, 0.8]], dtype=np.float32)
    original = array.copy()
    lpips_input_tensor(array, torch)
    assert np.array_equal(array, original)


def test_lpips_input_tensor_rejects_non_2d_array() -> None:
    with pytest.raises(ValueError, match="2D"):
        lpips_input_tensor(np.zeros((2, 2, 2), dtype=np.float32), torch)


def test_lpips_input_tensor_deterministic_for_same_input() -> None:
    array = np.linspace(-0.3, 1.3, 64, dtype=np.float32).reshape(8, 8)
    first = lpips_input_tensor(array, torch)
    second = lpips_input_tensor(array, torch)
    assert torch.equal(first, second)


def test_synthetic_pipeline_runs_and_reports_timing(
    synthetic_dataset_dir: Path,
) -> None:
    result = evaluate_validation_baseline(
        synthetic_dataset_dir, val_fraction=0.5, seed=42
    )
    assert result["train_samples"] == 1
    assert result["validation_samples"] == 1
    assert result["evaluated_samples"] == 1
    assert math.isfinite(result["psnr_db"])
    assert math.isfinite(result["ssim"])
    assert result["mean_interpolation_ms"] >= 0.0
