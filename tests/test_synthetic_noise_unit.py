"""Fast, dataset-free tests for signal-dependent synthetic noise (Experiment 24)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.degradation import downsample
from src.scene_groups import count_cross_split_groups, group_aware_split
from src.synthetic_noise import (
    STUDENT_T_DEGREES_OF_FREEDOM,
    VARIANCE_COEFFICIENTS,
    SyntheticNoiseAugmentation,
    bicubic_downsample,
    build_synthetic_noise,
    build_synthetic_noise_config,
    noise_sigma,
    noise_variance,
    sample_epsilon,
)
from src.transforms import create_training_transform
from train import build_datasets, load_checkpoint_for_resume, save_checkpoint


# --- Variance model ---


def test_variance_formula_matches_the_published_coefficients_exactly() -> None:
    c0, c1, c2 = VARIANCE_COEFFICIENTS
    intensity = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    expected = c0 + c1 * intensity + c2 * intensity**2
    assert torch.allclose(noise_variance(intensity), expected, atol=1e-12)


def test_variance_is_clamped_to_be_non_negative() -> None:
    """The fitted constant is negative, so variance would go negative near I=0."""
    assert VARIANCE_COEFFICIENTS[0] < 0
    assert noise_variance(torch.zeros(5)).min().item() == 0.0


def test_variance_floor_raises_the_clamp() -> None:
    floored = noise_variance(torch.zeros(3), variance_floor=1.43e-4)
    assert torch.allclose(floored, torch.full((3,), 1.43e-4))


def test_variance_floor_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        noise_variance(torch.zeros(3), variance_floor=-1.0)


def test_sigma_is_finite_and_non_negative_across_the_full_intensity_range() -> None:
    sigma = noise_sigma(torch.linspace(-0.1, 1.4, 400))
    assert torch.isfinite(sigma).all()
    assert (sigma >= 0).all()


def test_sigma_increases_with_signal_intensity() -> None:
    sigma = noise_sigma(torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert torch.all(sigma[1:] > sigma[:-1])


def test_sigma_matches_measured_magnitude_at_mid_and_high_intensity() -> None:
    """Experiment 22 measured std ~0.0935 at I=0.512 and ~0.1507 at I=0.913."""
    assert noise_sigma(torch.tensor([0.512])).item() == pytest.approx(0.0935, abs=0.005)
    assert noise_sigma(torch.tensor([0.913])).item() == pytest.approx(0.1507, abs=0.005)


# --- Epsilon sampling ---


@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_epsilon_is_zero_mean_unit_variance(distribution: str) -> None:
    rng = np.random.default_rng(0)
    epsilon = sample_epsilon((400_000,), distribution, rng)
    assert abs(float(epsilon.mean())) < 0.02
    assert float(epsilon.std()) == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_epsilon_is_finite(distribution: str) -> None:
    epsilon = sample_epsilon((10_000,), distribution, np.random.default_rng(1))
    assert np.isfinite(epsilon).all()


def test_student_t_is_heavier_tailed_than_gaussian() -> None:
    rng = np.random.default_rng(2)
    gaussian = sample_epsilon((500_000,), "gaussian", rng)
    student = sample_epsilon((500_000,), "student_t", rng)

    def excess_kurtosis(x: np.ndarray) -> float:
        centered = x - x.mean()
        return float(np.mean(centered**4) / x.std() ** 4 - 3.0)

    assert excess_kurtosis(student) > excess_kurtosis(gaussian) + 1.0


def test_student_t_rejects_degrees_of_freedom_without_finite_variance() -> None:
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        sample_epsilon((10,), "student_t", np.random.default_rng(3), degrees_of_freedom=2.0)


def test_sample_epsilon_rejects_unknown_distribution() -> None:
    with pytest.raises(ValueError, match="Unknown distribution"):
        sample_epsilon((4,), "cauchy", np.random.default_rng(4))


# --- Downsampling parity ---


def test_torch_bicubic_downsample_matches_the_degradation_analysis_implementation() -> None:
    """The augmentation must use the exact GT->LR model Experiment 22 selected."""
    gt = np.random.default_rng(5).uniform(0, 1, size=(64, 64)).astype(np.float32)
    reference = downsample(gt, "bicubic")
    result = bicubic_downsample(torch.from_numpy(gt).unsqueeze(0))[0].numpy()
    assert np.allclose(result, reference, atol=1e-6)


def test_bicubic_downsample_halves_spatial_dimensions() -> None:
    assert bicubic_downsample(torch.rand(1, 64, 64)).shape == (1, 32, 32)
    assert bicubic_downsample(torch.rand(4, 1, 64, 64)).shape == (4, 1, 32, 32)


def test_bicubic_downsample_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[C,H,W\]"):
        bicubic_downsample(torch.rand(64, 64))


# --- Synthesis ---


def _augmentation(**overrides) -> SyntheticNoiseAugmentation:
    options = {"probability": 1.0, "seed": 42, "distribution": "gaussian"}
    options.update(overrides)
    return SyntheticNoiseAugmentation(**options)


def test_synthetic_lr_has_the_expected_downsampled_shape() -> None:
    result = _augmentation().synthesize(torch.rand(1, 64, 64), np.random.default_rng(0))
    assert result.shape == (1, 32, 32)


def test_synthetic_lr_is_finite() -> None:
    result = _augmentation().synthesize(torch.rand(1, 64, 64), np.random.default_rng(0))
    assert torch.isfinite(result).all()


def test_synthetic_noise_magnitude_grows_with_signal_intensity() -> None:
    """Dark GT must produce visibly less noise than bright GT."""
    augmentation = _augmentation()
    dark = torch.full((1, 128, 128), 0.05)
    bright = torch.full((1, 128, 128), 0.95)
    dark_noise = (augmentation.synthesize(dark, np.random.default_rng(0)) - 0.05).std()
    bright_noise = (augmentation.synthesize(bright, np.random.default_rng(0)) - 0.95).std()
    assert bright_noise > 4 * dark_noise


def test_synthetic_residual_std_tracks_the_variance_model() -> None:
    augmentation = _augmentation()
    intensity = 0.6
    gt = torch.full((1, 256, 256), intensity)
    residual = augmentation.synthesize(gt, np.random.default_rng(7)) - intensity
    expected = noise_sigma(torch.tensor([intensity])).item()
    assert float(residual.std()) == pytest.approx(expected, rel=0.05)


def test_synthesis_does_not_clip_out_of_range_values() -> None:
    """Real NoisyLR contains values outside [0,1]; clipping would be less faithful."""
    result = _augmentation().synthesize(torch.full((1, 64, 64), 0.999), np.random.default_rng(8))
    assert result.max().item() > 1.0


# --- Probability policy and determinism ---


def test_probability_zero_never_synthesizes() -> None:
    augmentation = _augmentation(probability=0.0)
    assert all(augmentation.maybe_synthesize(torch.rand(1, 32, 32), i) is None for i in range(50))


def test_probability_one_always_synthesizes() -> None:
    augmentation = _augmentation(probability=1.0)
    assert all(
        augmentation.maybe_synthesize(torch.rand(1, 32, 32), i) is not None for i in range(50)
    )


def test_probability_half_selects_roughly_half_the_samples() -> None:
    augmentation = _augmentation(probability=0.5)
    chosen = sum(augmentation.use_synthetic(i) for i in range(4000))
    assert 1800 < chosen < 2200


def test_selection_and_noise_are_deterministic_for_a_fixed_seed_and_epoch() -> None:
    first, second = _augmentation(probability=0.5), _augmentation(probability=0.5)
    gt = torch.rand(1, 32, 32)
    for index in range(20):
        left = first.maybe_synthesize(gt, index)
        right = second.maybe_synthesize(gt, index)
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left, right)


def test_each_epoch_draws_a_different_noise_realization() -> None:
    """Fresh realizations per epoch are the entire point of the augmentation."""
    augmentation = _augmentation(probability=1.0)
    gt = torch.rand(1, 32, 32)
    augmentation.set_epoch(0)
    first = augmentation.maybe_synthesize(gt, 3)
    augmentation.set_epoch(1)
    second = augmentation.maybe_synthesize(gt, 3)
    assert not torch.equal(first, second)


def test_different_samples_get_different_noise_within_an_epoch() -> None:
    augmentation = _augmentation(probability=1.0)
    gt = torch.rand(1, 32, 32)
    assert not torch.equal(
        augmentation.maybe_synthesize(gt, 0), augmentation.maybe_synthesize(gt, 1)
    )


def test_augmentation_rejects_invalid_probability_and_distribution() -> None:
    with pytest.raises(ValueError, match="probability"):
        SyntheticNoiseAugmentation(probability=1.5)
    with pytest.raises(ValueError, match="Unknown distribution"):
        SyntheticNoiseAugmentation(probability=0.5, distribution="cauchy")


# --- Config round trip ---


def test_config_records_everything_needed_to_reconstruct_the_augmentation() -> None:
    config = build_synthetic_noise_config(0.5, seed=42, distribution="gaussian")
    assert config["enabled"] is True
    assert config["probability"] == 0.5
    assert config["distribution"] == "gaussian"
    assert config["variance_coefficients"] == list(VARIANCE_COEFFICIENTS)
    assert config["variance_floor"] == 0.0
    assert config["downsampling"] == "bicubic_align_corners_false"
    assert config["scale"] == 2


def test_disabled_config_is_none_so_historical_commands_are_unchanged() -> None:
    assert build_synthetic_noise_config(0.0, seed=42) is None
    assert build_synthetic_noise(None) is None


def test_config_round_trip_reproduces_identical_noise() -> None:
    config = build_synthetic_noise_config(1.0, seed=7, distribution="student_t")
    original = build_synthetic_noise(config)
    restored = build_synthetic_noise(config)
    gt = torch.rand(1, 32, 32)
    assert torch.equal(original.maybe_synthesize(gt, 0), restored.maybe_synthesize(gt, 0))
    assert restored.degrees_of_freedom == pytest.approx(STUDENT_T_DEGREES_OF_FREEDOM)


# --- Dataset integration and alignment ---


def _write_pair(root: Path, sample_id: str, lr_size: int = 16, scale: int = 2) -> ImagePair:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    np.save(input_dir / f"{sample_id}.npy", rng.uniform(0, 1, (lr_size, lr_size)).astype(np.float32))
    np.save(
        target_dir / f"{sample_id}.npy",
        rng.uniform(0, 1, (lr_size * scale, lr_size * scale)).astype(np.float32),
    )
    return ImagePair(sample_id, input_dir / f"{sample_id}.npy", target_dir / f"{sample_id}.npy")


def test_dataset_without_augmentation_returns_the_real_lr_unchanged(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, "000")]
    plain = PairedRestorationDataset(pairs)
    augmented = PairedRestorationDataset(pairs, synthetic_noise=_augmentation(probability=0.0))
    assert torch.equal(plain[0]["input"], augmented[0]["input"])
    assert plain[0]["synthetic"] is False


def test_dataset_with_probability_one_substitutes_a_synthetic_input(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, "000")]
    plain = PairedRestorationDataset(pairs)
    augmented = PairedRestorationDataset(pairs, synthetic_noise=_augmentation(probability=1.0))
    sample = augmented[0]
    assert sample["synthetic"] is True
    assert not torch.equal(plain[0]["input"], sample["input"])
    assert sample["input"].shape == plain[0]["input"].shape
    # Target must be untouched by the augmentation.
    assert torch.equal(plain[0]["target"], sample["target"])


def test_synthetic_input_stays_aligned_with_gt_through_crop_and_flips(tmp_path: Path) -> None:
    """The strongest alignment check: after the full transform, downsampling the
    returned GT crop must reproduce the returned LR crop up to the noise that
    was added -- which is only true if crop and geometry stayed synchronized."""
    pairs = [_write_pair(tmp_path, "000", lr_size=32)]
    dataset = PairedRestorationDataset(
        pairs,
        transform=create_training_transform(crop_size=16, scale=2, augment=True, seed=1),
        synthetic_noise=_augmentation(probability=1.0),
    )
    sample = dataset[0]
    assert sample["synthetic"] is True
    lr, gt = sample["input"], sample["target"]
    assert gt.shape[-1] == lr.shape[-1] * 2
    implied_clean = bicubic_downsample(gt)
    residual = (lr - implied_clean).abs()
    # Alignment failure would misregister structure and blow this far past the
    # noise scale; the model's sigma tops out around 0.16.
    assert float(residual.mean()) < 0.15


def test_synthetic_residual_matches_the_noise_model_after_transform(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, "000", lr_size=64)]
    dataset = PairedRestorationDataset(
        pairs,
        transform=create_training_transform(crop_size=32, scale=2, augment=True, seed=3),
        synthetic_noise=_augmentation(probability=1.0),
    )
    sample = dataset[0]
    residual = sample["input"] - bicubic_downsample(sample["target"])
    expected = noise_sigma(bicubic_downsample(sample["target"])).mean().item()
    assert float(residual.std()) == pytest.approx(expected, rel=0.4)


def test_set_epoch_changes_the_dataset_sample(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, "000")]
    dataset = PairedRestorationDataset(pairs, synthetic_noise=_augmentation(probability=1.0))
    first = dataset[0]["input"].clone()
    dataset.set_epoch(5)
    assert not torch.equal(first, dataset[0]["input"])


def test_set_epoch_is_a_no_op_without_augmentation(tmp_path: Path) -> None:
    dataset = PairedRestorationDataset([_write_pair(tmp_path, "000")])
    dataset.set_epoch(3)  # must not raise
    assert dataset[0]["synthetic"] is False


def test_build_datasets_never_gives_validation_a_synthetic_stream(tmp_path: Path) -> None:
    """Validation must be 100% real: no synthetic input may reach a reported
    metric, the scheduler, or checkpoint selection."""
    root = tmp_path / "Data-public"
    train_inputs = root / "train" / "train" / "NoisyLR"
    targets = root / "train" / "train" / "GT"
    test_inputs = root / "Test_NoisyLR" / "NoisyLR"
    for directory in (train_inputs, targets, test_inputs):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(10):
        name = f"{index:06d}.npy"
        np.save(train_inputs / name, rng.uniform(0, 1, (16, 16)).astype(np.float32))
        np.save(targets / name, rng.uniform(0, 1, (32, 32)).astype(np.float32))
    np.save(test_inputs / "000100.npy", np.zeros((16, 16), dtype=np.float32))

    train_dataset, validation_dataset, _ = build_datasets(
        root, 0.2, 42, 8, 2, None, None, synthetic_noise=_augmentation(probability=1.0)
    )
    assert train_dataset.synthetic_noise is not None
    assert validation_dataset.synthetic_noise is None
    assert all(validation_dataset[i]["synthetic"] is False for i in range(len(validation_dataset)))


def test_dataloader_batches_carry_the_synthetic_flag(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, f"{i:03d}") for i in range(4)]
    dataset = PairedRestorationDataset(pairs, synthetic_noise=_augmentation(probability=1.0))
    batch = next(iter(create_dataloader(dataset, batch_size=4, shuffle=False)))
    assert bool(batch["synthetic"].all())


# --- Checkpoint / resume ---


def _model_config() -> dict:
    return {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}


def _save(tmp_path: Path, synthetic_noise_config: dict | None) -> Path:
    from src.models import ResidualSRNet

    model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={},
        synthetic_noise_config=synthetic_noise_config,
    )
    return path


def _resume(path: Path, synthetic_noise_config):
    from src.models import ResidualSRNet

    model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return load_checkpoint_for_resume(
        path,
        model,
        optimizer,
        _model_config(),
        torch.device("cpu"),
        synthetic_noise_config=synthetic_noise_config,
    )


def test_checkpoint_stores_the_synthetic_noise_config(tmp_path: Path) -> None:
    config = build_synthetic_noise_config(0.5, seed=42)
    checkpoint = torch.load(_save(tmp_path, config), map_location="cpu", weights_only=False)
    assert checkpoint["synthetic_noise_config"] == config


def test_matching_synthetic_noise_resume_succeeds(tmp_path: Path) -> None:
    config = build_synthetic_noise_config(0.5, seed=42)
    start_epoch, best, _ = _resume(_save(tmp_path, config), config)
    assert start_epoch == 2
    assert best == 10.0


def test_resume_rejects_a_different_probability(tmp_path: Path) -> None:
    path = _save(tmp_path, build_synthetic_noise_config(0.5, seed=42))
    with pytest.raises(ValueError, match="synthetic_noise_config"):
        _resume(path, build_synthetic_noise_config(0.25, seed=42))


def test_resume_rejects_a_different_distribution(tmp_path: Path) -> None:
    path = _save(tmp_path, build_synthetic_noise_config(0.5, seed=42, distribution="gaussian"))
    with pytest.raises(ValueError, match="synthetic_noise_config"):
        _resume(path, build_synthetic_noise_config(0.5, seed=42, distribution="student_t"))


def test_resume_rejects_a_different_variance_floor(tmp_path: Path) -> None:
    path = _save(tmp_path, build_synthetic_noise_config(0.5, seed=42, variance_floor=0.0))
    with pytest.raises(ValueError, match="synthetic_noise_config"):
        _resume(path, build_synthetic_noise_config(0.5, seed=42, variance_floor=1e-4))


def test_resume_rejects_disabling_augmentation_on_a_synthetic_checkpoint(tmp_path: Path) -> None:
    path = _save(tmp_path, build_synthetic_noise_config(0.5, seed=42))
    with pytest.raises(ValueError, match="synthetic_noise_config"):
        _resume(path, None)


def test_resume_rejects_enabling_augmentation_on_a_plain_checkpoint(tmp_path: Path) -> None:
    path = _save(tmp_path, None)
    with pytest.raises(ValueError, match="synthetic_noise_config"):
        _resume(path, build_synthetic_noise_config(0.5, seed=42))


def test_historical_checkpoint_without_the_key_still_resumes(tmp_path: Path) -> None:
    """Every Experiment 1-23 checkpoint predates this key entirely."""
    from src.models import ResidualSRNet

    model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 90,
            "best_val_psnr": 27.99,
            "model_config": _model_config(),
            "training_config": {},
        },
        path,
    )
    start_epoch, best, _ = _resume(path, None)
    assert start_epoch == 91
    assert best == 27.99


def test_resume_without_passing_the_config_skips_the_check(tmp_path: Path) -> None:
    """Callers that omit the parameter entirely (pre-existing tests) are unaffected."""
    from src.models import ResidualSRNet

    path = _save(tmp_path, build_synthetic_noise_config(0.5, seed=42))
    model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch, _, _ = load_checkpoint_for_resume(
        path, model, optimizer, _model_config(), torch.device("cpu")
    )
    assert start_epoch == 2


# --- Group-aware secondary split ---


def test_group_aware_split_keeps_scene_groups_together() -> None:
    pairs = tuple(
        ImagePair(f"{i:03d}", Path(f"lr/{i}.npy"), Path(f"gt/{i}.npy")) for i in range(40)
    )
    groups = [["000", "001"], ["010", "011", "012"]]
    train, validation = group_aware_split(pairs, groups, val_fraction=0.25, seed=42)
    leakage = count_cross_split_groups(groups, train, validation)
    assert leakage["groups_spanning_split"] == 0
    assert len(train) + len(validation) == len(pairs)


def test_group_aware_split_is_deterministic() -> None:
    pairs = tuple(
        ImagePair(f"{i:03d}", Path(f"lr/{i}.npy"), Path(f"gt/{i}.npy")) for i in range(30)
    )
    groups = [["000", "001"]]
    first = group_aware_split(pairs, groups, val_fraction=0.2, seed=7)
    second = group_aware_split(pairs, groups, val_fraction=0.2, seed=7)
    assert [p.pair_id for p in first[1]] == [p.pair_id for p in second[1]]


def test_count_cross_split_groups_detects_leakage() -> None:
    pairs = tuple(ImagePair(f"{i:03d}", Path("a"), Path("b")) for i in range(4))
    leakage = count_cross_split_groups([["000", "001"]], pairs[:1], pairs[1:])
    assert leakage["groups_spanning_split"] == 1
    assert leakage["validation_images_with_train_twin"] == 1


def test_group_aware_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        group_aware_split((), [], val_fraction=0.0)


# --- EMA still works alongside the augmentation ---


def test_ema_updates_normally_while_training_on_synthetic_inputs(tmp_path: Path) -> None:
    from src.ema import ExponentialMovingAverage
    from src.models import ResidualSRNet
    from train import train_one_epoch

    pairs = [_write_pair(tmp_path, f"{i:03d}") for i in range(4)]
    dataset = PairedRestorationDataset(
        pairs,
        transform=create_training_transform(crop_size=8, scale=2, seed=42),
        synthetic_noise=_augmentation(probability=1.0),
    )
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    model = ResidualSRNet(**_model_config())
    ema = ExponentialMovingAverage(model, decay=0.9)
    before = {name: p.clone() for name, p in ema.shadow_model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss = train_one_epoch(model, loader, optimizer, nn.L1Loss(), torch.device("cpu"), ema=ema)

    assert math.isfinite(loss)
    assert any(
        not torch.equal(before[name], p) for name, p in ema.shadow_model.named_parameters()
    )
