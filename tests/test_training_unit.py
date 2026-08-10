"""Fast, dataset-free tests for checkpointing and a single training step.

Uses tiny synthetic NoisyLR/GT pairs written to ``tmp_path`` -- no real dataset
and no GPU are required.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evaluate_checkpoint import load_model
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.losses import loss_label
from src.models import EDSRLite, ResidualSRNet, build_model_config
from src.transforms import create_training_transform
from train import (
    build_datasets,
    load_checkpoint_for_resume,
    save_checkpoint,
    train_one_epoch,
    validate,
    warn_on_resume_config_mismatch,
)


def _write_pair(root: Path, sample_id: str, lr_size: int = 16, scale: int = 2) -> ImagePair:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    degraded = rng.uniform(-0.1, 1.1, size=(lr_size, lr_size)).astype(np.float32)
    target = rng.uniform(0.0, 1.0, size=(lr_size * scale, lr_size * scale)).astype(np.float32)
    input_path = input_dir / f"{sample_id}.npy"
    target_path = target_dir / f"{sample_id}.npy"
    np.save(input_path, degraded)
    np.save(target_path, target)
    return ImagePair(sample_id, input_path, target_path)


def _tiny_loaders(tmp_path: Path) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    pairs = [_write_pair(tmp_path, f"{index:03d}") for index in range(4)]
    training_dataset = PairedRestorationDataset(
        pairs, transform=create_training_transform(crop_size=8, scale=2, seed=42)
    )
    validation_dataset = PairedRestorationDataset(pairs)
    train_loader = create_dataloader(training_dataset, batch_size=2, shuffle=False)
    validation_loader = create_dataloader(validation_dataset, batch_size=2, shuffle=False)
    return train_loader, validation_loader


def test_single_training_step_is_finite_and_updates_parameters(tmp_path: Path) -> None:
    train_loader, _ = _tiny_loaders(tmp_path)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    mean_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, torch.device("cpu"))

    assert math.isfinite(mean_loss)
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_validate_produces_finite_metrics(tmp_path: Path) -> None:
    _, validation_loader = _tiny_loaders(tmp_path)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    metrics = validate(model, validation_loader, nn.L1Loss(), torch.device("cpu"))
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["psnr"])
    assert math.isfinite(metrics["ssim"])


def test_checkpoint_save_and_load_preserves_state(tmp_path: Path) -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 4,
        "num_blocks": 1,
        "scale": 2,
    }
    training_config = {"lr": 1e-3}

    optimizer.zero_grad()
    output = model(torch.randn(1, 1, 8, 8))
    nn.L1Loss()(output, torch.randn(1, 1, 16, 16)).backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_val_psnr=21.5,
        model_config=model_config,
        training_config=training_config,
    )

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_val_psnr"] == 21.5
    assert checkpoint["model_config"] == model_config
    assert checkpoint["training_config"] == training_config

    restored_model = ResidualSRNet(**checkpoint["model_config"])
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    for (name, original), (_, restored) in zip(
        model.named_parameters(), restored_model.named_parameters()
    ):
        assert torch.equal(original, restored), name

    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert (
        restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    )


def test_resume_restores_epoch_best_psnr_model_and_optimizer_state(tmp_path: Path) -> None:
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 4,
        "num_blocks": 1,
        "scale": 2,
    }
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    output = model(torch.randn(1, 1, 8, 8))
    nn.L1Loss()(output, torch.randn(1, 1, 16, 16)).backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=19.25,
        model_config=model_config,
        training_config={"seed": 42, "val_fraction": 0.2},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    start_epoch, best_val_psnr, training_config = load_checkpoint_for_resume(
        checkpoint_path, resumed_model, resumed_optimizer, model_config, torch.device("cpu")
    )

    assert start_epoch == 6
    assert best_val_psnr == 19.25
    assert training_config == {"seed": 42, "val_fraction": 0.2}
    for (name, original), (_, restored) in zip(
        model.named_parameters(), resumed_model.named_parameters()
    ):
        assert torch.equal(original, restored), name
    assert (
        resumed_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    )


def test_resume_rejects_mismatched_model_config(tmp_path: Path) -> None:
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 4,
        "num_blocks": 1,
        "scale": 2,
    }
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
    )

    mismatched_config = dict(model_config, num_features=8)
    other_model = ResidualSRNet(**mismatched_config)
    other_optimizer = torch.optim.Adam(other_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_for_resume(
            checkpoint_path, other_model, other_optimizer, mismatched_config, torch.device("cpu")
        )


def _write_discoverable_pairs(root: Path, count: int, lr_size: int = 8, scale: int = 2) -> Path:
    """Build a minimal directory layout that ``discover_layout`` can find."""
    data_root = root / "Data-public"
    train_inputs = data_root / "train" / "train" / "NoisyLR"
    targets = data_root / "train" / "train" / "GT"
    test_inputs = data_root / "Test_NoisyLR" / "NoisyLR"
    for directory in (train_inputs, targets, test_inputs):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        sample_id = f"{index:06d}"
        # Content equals the sample index so identity is easy to check.
        lr = np.full((lr_size, lr_size), float(index), dtype=np.float32)
        gt = np.full((lr_size * scale, lr_size * scale), float(index), dtype=np.float32)
        np.save(train_inputs / f"{sample_id}.npy", lr)
        np.save(targets / f"{sample_id}.npy", gt)
    np.save(test_inputs / "999999.npy", np.zeros((lr_size, lr_size), dtype=np.float32))
    return data_root


def test_max_samples_only_truncate_the_canonical_split(tmp_path: Path) -> None:
    data_dir = _write_discoverable_pairs(tmp_path, count=20)

    full_train, full_validation, total = build_datasets(
        data_dir,
        val_fraction=0.2,
        seed=42,
        crop_size=4,
        scale=2,
        max_train_samples=None,
        max_val_samples=None,
    )
    assert total == 20
    assert len(full_train) == 16
    assert len(full_validation) == 4

    truncated_train, truncated_validation, total_again = build_datasets(
        data_dir,
        val_fraction=0.2,
        seed=42,
        crop_size=4,
        scale=2,
        max_train_samples=5,
        max_val_samples=2,
    )
    assert total_again == total
    assert len(truncated_train) == 5
    assert len(truncated_validation) == 2

    # Truncation must be a prefix of the same canonical split, not a re-split.
    assert [p.pair_id for p in truncated_train.pairs] == [
        p.pair_id for p in full_train.pairs[:5]
    ]
    assert [p.pair_id for p in truncated_validation.pairs] == [
        p.pair_id for p in full_validation.pairs[:2]
    ]


# --- Experiment 3: checkpointing and resume with the larger 64/8 capacity config ---


def test_checkpoint_stores_experiment_3_model_config(tmp_path: Path) -> None:
    exp3_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 64,
        "num_blocks": 8,
        "scale": 2,
    }
    model = ResidualSRNet(**exp3_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = tmp_path / "exp3_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=exp3_config,
        training_config={},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["num_features"] == 64
    assert checkpoint["model_config"]["num_blocks"] == 8


def test_resume_rejects_experiment_2_checkpoint_for_experiment_3_config(
    tmp_path: Path,
) -> None:
    """Resuming a 64/8 (Experiment 3) run from a 32/4 (Experiment 1/2) checkpoint
    must fail with a clear, actionable error -- not a cryptic tensor-shape mismatch.
    """
    exp2_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 32,
        "num_blocks": 4,
        "scale": 2,
    }
    exp2_model = ResidualSRNet(**exp2_config)
    exp2_optimizer = torch.optim.Adam(exp2_model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp2_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        exp2_model,
        exp2_optimizer,
        epoch=40,
        best_val_psnr=27.2959,
        model_config=exp2_config,
        training_config={},
    )

    exp3_config = dict(exp2_config, num_features=64, num_blocks=8)
    exp3_model = ResidualSRNet(**exp3_config)
    exp3_optimizer = torch.optim.Adam(exp3_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_for_resume(
            checkpoint_path, exp3_model, exp3_optimizer, exp3_config, torch.device("cpu")
        )


# --- Experiment 4: loss selection (Charbonnier vs L1) checkpointing and resume ---


def _exp3_capacity_model_config() -> dict:
    return {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}


def test_checkpoint_stores_correct_charbonnier_loss_config(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_config = {"name": "charbonnier", "epsilon": 1e-3}

    checkpoint_path = tmp_path / "exp4_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
        loss_config=loss_config,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_config"] == {"name": "charbonnier", "epsilon": 1e-3}


def test_checkpoint_without_explicit_loss_config_defaults_to_l1(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
        # loss_config intentionally omitted, like every pre-Experiment-4 call site.
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_config"] == {"name": "l1"}


def test_loss_config_survives_save_and_resume(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_config = {"name": "charbonnier", "epsilon": 1e-3}

    checkpoint_path = tmp_path / "exp4_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=20.0,
        model_config=model_config,
        training_config={},
        loss_config=loss_config,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_config,
        torch.device("cpu"),
        loss_config=loss_config,
    )
    assert start_epoch == 6
    assert best_val_psnr == 20.0


def test_legacy_checkpoint_without_loss_config_is_treated_as_l1(tmp_path: Path) -> None:
    """Simulates a real pre-Experiment-4 checkpoint (Experiments 1-3): no loss_config key at all."""
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    legacy_checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 38,
            "best_val_psnr": 27.6212,
            "model_config": model_config,
            "training_config": {},
            # No scheduler_state_dict/scheduler_config/loss_config keys at all.
        },
        legacy_checkpoint_path,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    # Requesting l1 (the correct historical default) must succeed cleanly.
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        legacy_checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_config,
        torch.device("cpu"),
        loss_config={"name": "l1"},
    )
    assert start_epoch == 39
    assert best_val_psnr == 27.6212


def test_resume_rejects_charbonnier_request_against_legacy_l1_checkpoint(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    legacy_checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 38,
            "best_val_psnr": 27.6212,
            "model_config": model_config,
            "training_config": {},
        },
        legacy_checkpoint_path,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            legacy_checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "charbonnier", "epsilon": 1e-3},
        )


def test_resume_rejects_l1_request_against_charbonnier_checkpoint(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp4_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=10,
        best_val_psnr=15.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "charbonnier", "epsilon": 1e-3},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "l1"},
        )


# --- Experiment 5: L1+SSIM composite loss checkpointing and resume ---


def test_checkpoint_stores_correct_l1_ssim_loss_config(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_config = {"name": "l1_ssim", "ssim_weight": 0.1}

    checkpoint_path = tmp_path / "exp5_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
        loss_config=loss_config,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_config"] == {"name": "l1_ssim", "ssim_weight": 0.1}


def test_l1_ssim_loss_config_survives_save_and_resume(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_config = {"name": "l1_ssim", "ssim_weight": 0.1}

    checkpoint_path = tmp_path / "exp5_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=7,
        best_val_psnr=22.0,
        model_config=model_config,
        training_config={},
        loss_config=loss_config,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_config,
        torch.device("cpu"),
        loss_config=loss_config,
    )
    assert start_epoch == 8
    assert best_val_psnr == 22.0


def test_resume_rejects_l1_ssim_request_against_legacy_l1_checkpoint(tmp_path: Path) -> None:
    """Resuming the real Experiment 3 checkpoint (no loss_config, i.e. L1) as l1_ssim must fail."""
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    legacy_checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 38,
            "best_val_psnr": 27.6212,
            "model_config": model_config,
            "training_config": {},
        },
        legacy_checkpoint_path,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            legacy_checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "l1_ssim", "ssim_weight": 0.1},
        )


def test_resume_rejects_l1_request_against_l1_ssim_checkpoint(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp5_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=18.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "l1_ssim", "ssim_weight": 0.1},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "l1"},
        )


def test_resume_rejects_charbonnier_request_against_l1_ssim_checkpoint(tmp_path: Path) -> None:
    """Distinguishing l1_ssim from charbonnier too, not just from l1."""
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp5_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=18.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "l1_ssim", "ssim_weight": 0.1},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "charbonnier", "epsilon": 1e-3},
        )


# --- Experiment 6: crop-size configuration, logging, and resume-mismatch warning ---


def test_build_datasets_with_96_crop_leaves_validation_full_image(tmp_path: Path) -> None:
    data_dir = _write_discoverable_pairs(tmp_path, count=6, lr_size=128, scale=2)
    train_dataset, validation_dataset, total = build_datasets(
        data_dir,
        val_fraction=0.34,  # -> 2 validation, 4 train out of 6
        seed=42,
        crop_size=96,
        scale=2,
        max_train_samples=None,
        max_val_samples=None,
    )
    assert total == 6
    training_sample = train_dataset[0]
    assert training_sample["input"].shape == (1, 96, 96)
    assert training_sample["target"].shape == (1, 192, 192)
    # Validation is built with no transform regardless of crop_size -- full images.
    validation_sample = validation_dataset[0]
    assert validation_sample["input"].shape == (1, 128, 128)
    assert validation_sample["target"].shape == (1, 256, 256)


def test_warn_on_resume_config_mismatch_flags_crop_size_change(capsys) -> None:
    previous_config = {"seed": 42, "val_fraction": 0.2, "crop_size": 96}
    warn_on_resume_config_mismatch(previous_config, seed=42, val_fraction=0.2, crop_size=64)
    output = capsys.readouterr().out
    assert "--crop-size" in output
    assert "64" in output and "96" in output


def test_warn_on_resume_config_mismatch_silent_when_everything_matches(capsys) -> None:
    previous_config = {"seed": 42, "val_fraction": 0.2, "crop_size": 96}
    warn_on_resume_config_mismatch(previous_config, seed=42, val_fraction=0.2, crop_size=96)
    assert capsys.readouterr().out == ""


def test_warn_on_resume_config_mismatch_still_flags_seed_and_val_fraction(capsys) -> None:
    previous_config = {"seed": 42, "val_fraction": 0.2, "crop_size": 64}
    warn_on_resume_config_mismatch(previous_config, seed=7, val_fraction=0.2, crop_size=64)
    output = capsys.readouterr().out
    assert "--seed/--val-fraction" in output
    assert "--crop-size" not in output


def test_checkpoint_stores_experiment_6_crop_size_in_training_config(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_config = {
        "epochs": 40,
        "batch_size": 16,
        "lr": 1e-4,
        "seed": 42,
        "val_fraction": 0.2,
        "crop_size": 96,
        "data_dir": "data/Data-public",
    }
    checkpoint_path = tmp_path / "exp6_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config=training_config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["crop_size"] == 96
    assert checkpoint["model_config"]["scale"] == 2


# --- Experiment 8: MSE loss checkpointing and resume ---


def test_checkpoint_stores_correct_mse_loss_config(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp8_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "mse"},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_config"] == {"name": "mse"}


def test_mse_checkpoint_resumes_successfully_with_mse(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp8_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=9,
        best_val_psnr=25.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "mse"},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_config,
        torch.device("cpu"),
        loss_config={"name": "mse"},
    )
    assert start_epoch == 10
    assert best_val_psnr == 25.0


def test_mse_checkpoint_rejects_l1_resume(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp8_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=20.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "mse"},
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "l1"},
        )


def test_l1_checkpoint_rejects_mse_resume(tmp_path: Path) -> None:
    """Uses a real Exp3/Exp6-shaped legacy checkpoint (no loss_config key -> treated as L1)."""
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    legacy_checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 38,
            "best_val_psnr": 27.7090,
            "model_config": model_config,
            "training_config": {},
        },
        legacy_checkpoint_path,
    )

    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="loss_config"):
        load_checkpoint_for_resume(
            legacy_checkpoint_path,
            resumed_model,
            resumed_optimizer,
            model_config,
            torch.device("cpu"),
            loss_config={"name": "mse"},
        )


def test_mse_checkpoint_rejects_charbonnier_and_l1_ssim_resume(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp8_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=20.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "mse"},
    )

    for mismatched_loss_config in (
        {"name": "charbonnier", "epsilon": 1e-3},
        {"name": "l1_ssim", "ssim_weight": 0.1},
    ):
        resumed_model = ResidualSRNet(**model_config)
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="loss_config"):
            load_checkpoint_for_resume(
                checkpoint_path,
                resumed_model,
                resumed_optimizer,
                model_config,
                torch.device("cpu"),
                loss_config=mismatched_loss_config,
            )


def test_evaluate_checkpoint_load_model_reads_mse_checkpoint(tmp_path: Path) -> None:
    model_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp8_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=15.0,
        model_config=model_config,
        training_config={},
        loss_config={"name": "mse"},
    )

    loaded_model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded_model, ResidualSRNet)
    assert checkpoint["loss_config"] == {"name": "mse"}
    assert loss_label(checkpoint["loss_config"]["name"]) == "MSE"


# --- Experiment 9: EDSRLite architecture checkpointing and resume compatibility ---


def _tiny_edsr_lite_config() -> dict:
    return build_model_config(
        "edsr_lite", num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )


def test_edsr_lite_checkpoint_stores_architecture_identifier(tmp_path: Path) -> None:
    model_config = _tiny_edsr_lite_config()
    model = EDSRLite(
        in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp9_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=model_config,
        training_config={},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["architecture"] == "edsr_lite"
    assert checkpoint["model_config"]["residual_scale"] == 0.1


def test_matching_edsr_lite_resume_succeeds(tmp_path: Path) -> None:
    model_config = _tiny_edsr_lite_config()
    model = EDSRLite(
        in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp9_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=7,
        best_val_psnr=22.0,
        model_config=model_config,
        training_config={},
    )

    resumed_model = EDSRLite(
        in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path, resumed_model, resumed_optimizer, model_config, torch.device("cpu")
    )
    assert start_epoch == 8
    assert best_val_psnr == 22.0
    for (name, original), (_, restored) in zip(
        model.named_parameters(), resumed_model.named_parameters()
    ):
        assert torch.equal(original, restored), name


def test_residual_sr_checkpoint_rejects_edsr_lite_resume(tmp_path: Path) -> None:
    """Real Exp1-8-shaped checkpoint (ResidualSRNet, no architecture key) must
    reject a --model edsr_lite resume attempt."""
    residual_sr_config = _exp3_capacity_model_config()
    model = ResidualSRNet(**residual_sr_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp6_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=38,
        best_val_psnr=27.7090,
        model_config=residual_sr_config,
        training_config={},
    )

    edsr_lite_config = build_model_config(
        "edsr_lite",
        num_features=residual_sr_config["num_features"],
        num_blocks=residual_sr_config["num_blocks"],
        scale=residual_sr_config["scale"],
        residual_scale=0.1,
    )
    edsr_model = EDSRLite(
        num_features=residual_sr_config["num_features"],
        num_blocks=residual_sr_config["num_blocks"],
        scale=residual_sr_config["scale"],
        residual_scale=0.1,
    )
    edsr_optimizer = torch.optim.Adam(edsr_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_for_resume(
            checkpoint_path, edsr_model, edsr_optimizer, edsr_lite_config, torch.device("cpu")
        )


def test_edsr_lite_checkpoint_rejects_residual_sr_resume(tmp_path: Path) -> None:
    edsr_lite_config = _tiny_edsr_lite_config()
    model = EDSRLite(
        in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp9_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=20.0,
        model_config=edsr_lite_config,
        training_config={},
    )

    residual_sr_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": 4,
        "num_blocks": 1,
        "scale": 2,
    }
    residual_model = ResidualSRNet(**residual_sr_config)
    residual_optimizer = torch.optim.Adam(residual_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            residual_model,
            residual_optimizer,
            residual_sr_config,
            torch.device("cpu"),
        )


def test_evaluate_checkpoint_load_model_reconstructs_edsr_lite(tmp_path: Path) -> None:
    """Covers both evaluate_checkpoint.py and infer_test.py, which reuses load_model()."""
    model_config = _tiny_edsr_lite_config()
    model = EDSRLite(
        in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2, residual_scale=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "exp9_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=15.0,
        model_config=model_config,
        training_config={},
    )

    loaded_model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded_model, EDSRLite)
    assert checkpoint["model_config"]["architecture"] == "edsr_lite"
    output = loaded_model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)
