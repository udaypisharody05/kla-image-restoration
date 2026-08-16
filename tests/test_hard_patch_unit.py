"""Tests for gradient-energy-weighted informative-patch ("hard-patch") sampling.

Covers: LR/GT alignment is preserved exactly like the existing random crop,
deterministic seeded behavior, the hard/random mixture probability actually
controls which code path runs, and crop origins always stay within legal
boundaries (including the flat/oversized edge cases).
"""

import torch

from src.transforms import (
    PairedHardPatchCrop,
    PairedMixedCrop,
    PairedRandomCrop,
    create_training_transform,
    gradient_energy_map,
    sample_informative_crop_origin,
)


def _coordinate_pair(height: int = 128, width: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    low_resolution = (y * 1000.0 + x - 100.0).unsqueeze(0)
    ground_truth = low_resolution.repeat_interleave(2, -2).repeat_interleave(2, -1)
    return low_resolution, ground_truth


def _assert_exact_2x_alignment(low_resolution: torch.Tensor, ground_truth: torch.Tensor) -> None:
    expected = low_resolution.repeat_interleave(2, -2).repeat_interleave(2, -1)
    assert torch.equal(ground_truth, expected)


def _high_energy_patch(height: int = 128, width: int = 128) -> torch.Tensor:
    """Flat image except for a bright 16x16 block in the bottom-right corner --
    the only region with nonzero gradient energy."""
    image = torch.zeros((1, height, width), dtype=torch.float32)
    image[:, height - 16 :, width - 16 :] = 1.0
    return image


# --- gradient_energy_map / sample_informative_crop_origin -------------------


def test_gradient_energy_map_is_zero_for_flat_image() -> None:
    flat = torch.full((1, 32, 32), 0.5)
    energy = gradient_energy_map(flat)
    assert energy.shape == (32, 32)
    assert torch.equal(energy, torch.zeros(32, 32))


def test_gradient_energy_map_nonzero_only_near_edges() -> None:
    image = _high_energy_patch(32, 32)
    energy = gradient_energy_map(image)
    # Everything outside the last 17 rows/cols (block interior + its boundary) is flat.
    assert torch.equal(energy[: 32 - 17, : 32 - 17], torch.zeros(32 - 17, 32 - 17))
    assert energy.sum() > 0


def test_sample_informative_crop_origin_stays_in_bounds_over_many_draws() -> None:
    image = _high_energy_patch()
    for seed in range(30):
        generator = torch.Generator().manual_seed(seed)
        y, x = sample_informative_crop_origin(image, crop_size=64, generator=generator)
        assert 0 <= y <= 64
        assert 0 <= x <= 64


def test_sample_informative_crop_origin_biases_toward_high_energy_region() -> None:
    """Over many draws, crops should overlap the bright corner far more often
    than a uniform-random baseline would (uniform expectation ~= 1/65 of draws
    place the origin at the single corner-covering coordinate; here energy is
    concentrated there so it should dominate)."""
    image = _high_energy_patch()
    hits = 0
    trials = 200
    for seed in range(trials):
        generator = torch.Generator().manual_seed(seed)
        y, x = sample_informative_crop_origin(image, crop_size=64, generator=generator)
        # A 64x64 crop overlaps the bottom-right 16x16 bright block whenever
        # its origin is within the last 64 rows/cols (always true here), but
        # it captures MORE of the block the closer the origin is to (64, 64).
        if y >= 48 and x >= 48:
            hits += 1
    # Uniform-random origins would put ~ (17/65)^2 ~= 6.8% of draws in that
    # same region; weighted sampling toward the bright block should clear it
    # by a wide margin.
    assert hits / trials > 0.30


def test_uniform_image_sampling_does_not_crash_or_produce_nan() -> None:
    flat = torch.full((1, 128, 128), 0.5)
    generator = torch.Generator().manual_seed(0)
    y, x = sample_informative_crop_origin(flat, crop_size=64, generator=generator)
    assert 0 <= y <= 64
    assert 0 <= x <= 64


# --- alignment ---------------------------------------------------------------


def test_hard_patch_crop_preserves_exact_2x_alignment() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    crop = PairedHardPatchCrop(crop_size=64, scale=2, generator=torch.Generator().manual_seed(7))
    cropped_lr, cropped_gt = crop(low_resolution, ground_truth)
    assert cropped_lr.shape == (1, 64, 64)
    assert cropped_gt.shape == (1, 128, 128)
    _assert_exact_2x_alignment(cropped_lr, cropped_gt)


def test_mixed_crop_preserves_exact_2x_alignment_across_both_branches() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    for prob in (0.0, 0.5, 1.0):
        crop = PairedMixedCrop(
            crop_size=64, scale=2, hard_patch_prob=prob, generator=torch.Generator().manual_seed(3)
        )
        for _ in range(10):
            cropped_lr, cropped_gt = crop(low_resolution, ground_truth)
            _assert_exact_2x_alignment(cropped_lr, cropped_gt)


# --- deterministic seeded behavior -------------------------------------------


def test_hard_patch_crop_is_reproducible_given_same_seed() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    first = PairedHardPatchCrop(64, generator=torch.Generator().manual_seed(42))
    second = PairedHardPatchCrop(64, generator=torch.Generator().manual_seed(42))
    first_lr, first_gt = first(low_resolution, ground_truth)
    second_lr, second_gt = second(low_resolution, ground_truth)
    assert torch.equal(first_lr, second_lr)
    assert torch.equal(first_gt, second_gt)


def test_mixed_crop_sequence_is_reproducible_given_same_seed() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    first = PairedMixedCrop(64, hard_patch_prob=0.5, generator=torch.Generator().manual_seed(123))
    second = PairedMixedCrop(64, hard_patch_prob=0.5, generator=torch.Generator().manual_seed(123))
    for _ in range(20):
        first_lr, first_gt = first(low_resolution, ground_truth)
        second_lr, second_gt = second(low_resolution, ground_truth)
        assert torch.equal(first_lr, second_lr)
        assert torch.equal(first_gt, second_gt)


def test_create_training_transform_hard_patch_seed_is_reproducible() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    first = create_training_transform(
        crop_size=64, seed=99, hard_patch_sampling=True, hard_patch_prob=0.5, augment=False
    )
    second = create_training_transform(
        crop_size=64, seed=99, hard_patch_sampling=True, hard_patch_prob=0.5, augment=False
    )
    first_lr, first_gt = first(low_resolution, ground_truth)
    second_lr, second_gt = second(low_resolution, ground_truth)
    assert torch.equal(first_lr, second_lr)
    assert torch.equal(first_gt, second_gt)


# --- probability behavior -----------------------------------------------------


def test_hard_patch_prob_zero_never_calls_informative_sampling(monkeypatch) -> None:
    low_resolution, ground_truth = _coordinate_pair()

    def _boom(*args, **kwargs):
        raise AssertionError("informative sampling should not run when hard_patch_prob=0.0")

    monkeypatch.setattr("src.transforms.sample_informative_crop_origin", _boom)
    crop = PairedMixedCrop(64, hard_patch_prob=0.0, generator=torch.Generator().manual_seed(1))
    for _ in range(25):
        crop(low_resolution, ground_truth)  # must not raise


def test_hard_patch_prob_one_always_uses_informative_sampling(monkeypatch) -> None:
    low_resolution, ground_truth = _coordinate_pair()
    calls = {"count": 0}
    original = sample_informative_crop_origin

    def _counting_wrapper(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("src.transforms.sample_informative_crop_origin", _counting_wrapper)
    crop = PairedMixedCrop(64, hard_patch_prob=1.0, generator=torch.Generator().manual_seed(1))
    for _ in range(10):
        crop(low_resolution, ground_truth)
    assert calls["count"] == 10


def test_hard_patch_prob_intermediate_uses_both_branches(monkeypatch) -> None:
    low_resolution, ground_truth = _coordinate_pair()
    calls = {"hard": 0}
    original = sample_informative_crop_origin

    def _counting_wrapper(*args, **kwargs):
        calls["hard"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("src.transforms.sample_informative_crop_origin", _counting_wrapper)
    crop = PairedMixedCrop(64, hard_patch_prob=0.5, generator=torch.Generator().manual_seed(0))
    trials = 200
    for _ in range(trials):
        crop(low_resolution, ground_truth)
    # With prob=0.5 over 200 draws, expect roughly half to be informative --
    # allow generous slack since this is a real (seeded, not mocked) Bernoulli draw.
    assert 40 < calls["hard"] < 160


# --- crop boundaries -----------------------------------------------------------


def test_hard_patch_crop_matches_full_image_size_is_legal() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    crop = PairedHardPatchCrop(crop_size=128, generator=torch.Generator().manual_seed(0))
    cropped_lr, cropped_gt = crop(low_resolution, ground_truth)
    assert torch.equal(cropped_lr, low_resolution)
    assert torch.equal(cropped_gt, ground_truth)


def test_hard_patch_crop_oversized_raises_value_error() -> None:
    low_resolution, ground_truth = _coordinate_pair()
    crop = PairedHardPatchCrop(crop_size=129)
    try:
        crop(low_resolution, ground_truth)
        raise AssertionError("expected ValueError for oversized crop")
    except ValueError as error:
        assert "exceeds" in str(error)


def test_hard_patch_crop_origin_always_within_legal_range() -> None:
    image = _high_energy_patch()
    ground_truth = image.repeat_interleave(2, -2).repeat_interleave(2, -1)
    for seed in range(50):
        crop = PairedHardPatchCrop(64, generator=torch.Generator().manual_seed(seed))
        cropped_lr, _ = crop(image, ground_truth)
        assert cropped_lr.shape == (1, 64, 64)
