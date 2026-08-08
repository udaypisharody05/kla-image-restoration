"""Exact alignment and value-preservation tests for paired preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.transforms import (
    PairedRandomCrop,
    aligned_paired_crop,
    apply_paired_geometry,
    create_training_transform,
)


def _coordinate_pair(height: int = 128, width: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    low_resolution = (y * 1000.0 + x - 100.0).unsqueeze(0)
    ground_truth = low_resolution.repeat_interleave(2, -2).repeat_interleave(2, -1)
    return low_resolution, ground_truth


def _assert_exact_2x_alignment(
    low_resolution: torch.Tensor, ground_truth: torch.Tensor
) -> None:
    expected = low_resolution.repeat_interleave(2, -2).repeat_interleave(2, -1)
    assert torch.equal(ground_truth, expected)


def _write_pairs(root: Path, count: int = 4) -> list[ImagePair]:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    low_resolution, ground_truth = _coordinate_pair()
    pairs = []
    for index in range(count):
        input_path = input_dir / f"{index:06d}.npy"
        target_path = target_dir / f"{index:06d}.npy"
        np.save(input_path, low_resolution.numpy()[0])
        np.save(target_path, ground_truth.numpy()[0])
        pairs.append(ImagePair(f"{index:06d}", input_path, target_path))
    return pairs


def test_aligned_crop_maps_exact_coordinates_at_final_legal_origin() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    cropped_lr, cropped_gt = aligned_paired_crop(
        low_resolution, ground_truth, crop_size=64, y=64, x=64, scale=2
    )
    assert cropped_lr.shape == (1, 64, 64)
    assert cropped_gt.shape == (1, 128, 128)
    assert torch.equal(cropped_lr, low_resolution[:, 64:128, 64:128])
    assert torch.equal(cropped_gt, ground_truth[:, 128:256, 128:256])
    _assert_exact_2x_alignment(cropped_lr, cropped_gt)


def test_crop_equal_to_full_image_and_oversized_crop() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    full_lr, full_gt = aligned_paired_crop(
        low_resolution, ground_truth, crop_size=128, y=0, x=0
    )
    assert torch.equal(full_lr, low_resolution)
    assert torch.equal(full_gt, ground_truth)
    with pytest.raises(ValueError, match="exceeds"):
        PairedRandomCrop(crop_size=129)(low_resolution, ground_truth)


def test_seeded_random_crop_is_reproducible_and_aligned() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    first = PairedRandomCrop(64, generator=torch.Generator().manual_seed(42))
    second = PairedRandomCrop(64, generator=torch.Generator().manual_seed(42))
    first_lr, first_gt = first(low_resolution, ground_truth)
    second_lr, second_gt = second(low_resolution, ground_truth)
    assert torch.equal(first_lr, second_lr)
    assert torch.equal(first_gt, second_gt)
    _assert_exact_2x_alignment(first_lr, first_gt)


@pytest.mark.parametrize(
    ("horizontal", "vertical"), [(True, False), (False, True), (True, True)]
)
def test_paired_flips_preserve_exact_alignment(
    horizontal: bool, vertical: bool
) -> None:
    low_resolution, ground_truth = _coordinate_pair(8, 10)
    transformed_lr, transformed_gt = apply_paired_geometry(
        low_resolution,
        ground_truth,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
    )
    _assert_exact_2x_alignment(transformed_lr, transformed_gt)


@pytest.mark.parametrize("rotation_k", [1, 2, 3])
def test_right_angle_rotations_preserve_exact_alignment(rotation_k: int) -> None:
    low_resolution, ground_truth = _coordinate_pair(8, 10)
    transformed_lr, transformed_gt = apply_paired_geometry(
        low_resolution, ground_truth, rotation_k=rotation_k
    )
    assert torch.equal(
        transformed_lr, torch.rot90(low_resolution, rotation_k, (-2, -1))
    )
    _assert_exact_2x_alignment(transformed_lr, transformed_gt)


def test_combined_training_transform_is_seeded_aligned_and_not_frozen() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    first = create_training_transform(crop_size=64, seed=123)
    second = create_training_transform(crop_size=64, seed=123)
    first_lr, first_gt = first(low_resolution, ground_truth)
    matching_lr, matching_gt = second(low_resolution, ground_truth)
    later_lr, later_gt = first(low_resolution, ground_truth)
    assert torch.equal(first_lr, matching_lr)
    assert torch.equal(first_gt, matching_gt)
    _assert_exact_2x_alignment(first_lr, first_gt)
    _assert_exact_2x_alignment(later_lr, later_gt)
    assert not torch.equal(first_lr, later_lr)


def test_geometric_preprocessing_preserves_raw_value_multiset() -> None:
    low_resolution = torch.tensor(
        [[[-0.1, 0.2], [1.5, 0.7]]], dtype=torch.float32
    )
    ground_truth = low_resolution.repeat_interleave(2, -2).repeat_interleave(2, -1)
    transformed_lr, transformed_gt = apply_paired_geometry(
        low_resolution,
        ground_truth,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_k=3,
    )
    assert torch.equal(torch.sort(transformed_lr.flatten()).values, torch.sort(low_resolution.flatten()).values)
    assert transformed_lr.dtype == transformed_gt.dtype == torch.float32
    assert float(transformed_lr.min()) == pytest.approx(-0.1)
    assert float(transformed_lr.max()) == pytest.approx(1.5)
    _assert_exact_2x_alignment(transformed_lr, transformed_gt)


def test_dataset_training_and_validation_batch_shapes(tmp_path: Path) -> None:
    pairs = _write_pairs(tmp_path)
    training_dataset = PairedRestorationDataset(
        pairs, transform=create_training_transform(crop_size=64, seed=42)
    )
    validation_dataset = PairedRestorationDataset(pairs)
    training_batch = next(
        iter(create_dataloader(training_dataset, batch_size=4, shuffle=False))
    )
    validation_batch = next(
        iter(create_dataloader(validation_dataset, batch_size=4, shuffle=False))
    )
    assert training_batch["input"].shape == (4, 1, 64, 64)
    assert training_batch["target"].shape == (4, 1, 128, 128)
    assert validation_batch["input"].shape == (4, 1, 128, 128)
    assert validation_batch["target"].shape == (4, 1, 256, 256)


# --- Experiment 6: 96x96 LR / 192x192 GT crop configuration ---


def test_create_training_transform_default_crop_is_still_64() -> None:
    import inspect

    default_crop_size = inspect.signature(create_training_transform).parameters["crop_size"].default
    assert default_crop_size == 64


def test_96_crop_produces_exact_96_lr_and_192_gt_shapes() -> None:
    low_resolution, ground_truth = _coordinate_pair(128, 128)
    cropped_lr, cropped_gt = aligned_paired_crop(
        low_resolution, ground_truth, crop_size=96, y=16, x=16, scale=2
    )
    assert cropped_lr.shape == (1, 96, 96)
    assert cropped_gt.shape == (1, 192, 192)
    assert torch.equal(cropped_lr, low_resolution[:, 16:112, 16:112])
    assert torch.equal(cropped_gt, ground_truth[:, 32:224, 32:224])


def test_96_crop_is_spatially_aligned_at_scale_2() -> None:
    low_resolution, ground_truth = _coordinate_pair(128, 128)
    cropped_lr, cropped_gt = aligned_paired_crop(
        low_resolution, ground_truth, crop_size=96, y=10, x=20, scale=2
    )
    _assert_exact_2x_alignment(cropped_lr, cropped_gt)


def test_96_crop_via_paired_random_crop_and_training_transform() -> None:
    low_resolution, ground_truth = _coordinate_pair(128, 128)
    cropper = PairedRandomCrop(crop_size=96, scale=2, generator=torch.Generator().manual_seed(0))
    cropped_lr, cropped_gt = cropper(low_resolution, ground_truth)
    assert cropped_lr.shape == (1, 96, 96)
    assert cropped_gt.shape == (1, 192, 192)
    _assert_exact_2x_alignment(cropped_lr, cropped_gt)

    training_transform = create_training_transform(crop_size=96, seed=7)
    transformed_lr, transformed_gt = training_transform(low_resolution, ground_truth)
    assert transformed_lr.shape == (1, 96, 96)
    assert transformed_gt.shape == (1, 192, 192)
    _assert_exact_2x_alignment(transformed_lr, transformed_gt)
    assert torch.isfinite(transformed_lr).all()
    assert torch.isfinite(transformed_gt).all()


def test_96_crop_dataset_batch_shapes_while_validation_stays_full_image(
    tmp_path: Path,
) -> None:
    """Crop size only affects the training dataset; validation must be unaffected."""
    pairs = _write_pairs(tmp_path)
    training_dataset = PairedRestorationDataset(
        pairs, transform=create_training_transform(crop_size=96, seed=42)
    )
    validation_dataset = PairedRestorationDataset(pairs)  # no transform, same as always
    training_batch = next(iter(create_dataloader(training_dataset, batch_size=4, shuffle=False)))
    validation_batch = next(
        iter(create_dataloader(validation_dataset, batch_size=4, shuffle=False))
    )
    assert training_batch["input"].shape == (4, 1, 96, 96)
    assert training_batch["target"].shape == (4, 1, 192, 192)
    # Unchanged from the 64-crop test above: validation is always full-image.
    assert validation_batch["input"].shape == (4, 1, 128, 128)
    assert validation_batch["target"].shape == (4, 1, 256, 256)


def test_96_crop_exceeding_source_image_fails_clearly() -> None:
    low_resolution, ground_truth = _coordinate_pair(64, 64)  # smaller than a 96 crop
    with pytest.raises(ValueError, match="exceeds"):
        PairedRandomCrop(crop_size=96)(low_resolution, ground_truth)


@pytest.mark.parametrize("crop_size", [0, -1, -96])
def test_non_positive_crop_size_is_rejected(crop_size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        PairedRandomCrop(crop_size=crop_size)


def test_96_crop_augmentation_alignment_still_correct() -> None:
    low_resolution, ground_truth = _coordinate_pair(128, 128)
    transform = create_training_transform(crop_size=96, seed=3, augment=True)
    transformed_lr, transformed_gt = transform(low_resolution, ground_truth)
    assert transformed_lr.shape == (1, 96, 96)
    assert transformed_gt.shape == (1, 192, 192)
    _assert_exact_2x_alignment(transformed_lr, transformed_gt)
