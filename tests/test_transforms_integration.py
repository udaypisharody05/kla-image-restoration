"""Real-data train-versus-validation preprocessing checks."""

from pathlib import Path

import pytest
import torch

from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.splits import split_pairs
from src.transforms import create_training_transform


pytestmark = pytest.mark.integration


def test_real_training_crops_and_full_validation_batch(real_dataset_dir: Path) -> None:
    pairs = discover_pairs(discover_layout(real_dataset_dir)).pairs
    train_pairs, validation_pairs = split_pairs(pairs, val_fraction=0.2, seed=42)
    training_dataset = PairedRestorationDataset(
        train_pairs,
        transform=create_training_transform(crop_size=64, scale=2, seed=42),
    )
    validation_dataset = PairedRestorationDataset(validation_pairs)
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
    for batch in (training_batch, validation_batch):
        assert batch["input"].dtype == batch["target"].dtype == torch.float32
        assert torch.isfinite(batch["input"]).all()
        assert torch.isfinite(batch["target"]).all()
