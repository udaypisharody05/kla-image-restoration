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

from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.models import ResidualSRNet
from src.transforms import create_training_transform
from train import (
    build_datasets,
    load_checkpoint_for_resume,
    save_checkpoint,
    train_one_epoch,
    validate,
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
    assert math.isfinite(metrics["l1"])
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
