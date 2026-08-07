"""Real-data integration checks for the lazy PyTorch data layer."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    PairedRestorationDataset,
    RestorationTestDataset,
    create_dataloader,
)
from src.dataset_discovery import discover_layout, discover_pairs, image_files
from src.io_utils import load_image_array
from src.splits import split_pairs


pytestmark = pytest.mark.integration


def test_real_paired_datasets_and_batch(real_dataset_dir: Path) -> None:
    layout = discover_layout(real_dataset_dir)
    pairs = discover_pairs(layout).pairs
    train_pairs, validation_pairs = split_pairs(pairs, val_fraction=0.2, seed=42)
    train_dataset = PairedRestorationDataset(train_pairs)
    validation_dataset = PairedRestorationDataset(validation_pairs)

    assert len(train_dataset) == 2560
    assert len(validation_dataset) == 640
    batch = next(
        iter(create_dataloader(validation_dataset, batch_size=4, shuffle=False))
    )
    assert batch["input"].shape == (4, 1, 128, 128)
    assert batch["target"].shape == (4, 1, 256, 256)
    assert batch["input"].dtype == batch["target"].dtype == torch.float32
    assert torch.isfinite(batch["input"]).all()
    assert torch.isfinite(batch["target"]).all()

    raw = load_image_array(pairs[0].input_path)
    sample = PairedRestorationDataset([pairs[0]])[0]
    assert torch.equal(sample["input"][0], torch.from_numpy(raw))
    assert float(sample["input"].min()) < 0.0 or float(sample["input"].max()) > 1.0


def test_real_competition_test_dataset(real_dataset_dir: Path) -> None:
    layout = discover_layout(real_dataset_dir)
    paths = image_files(layout.test_input_dir)[:3]
    dataset = RestorationTestDataset(paths)
    batch = next(iter(create_dataloader(dataset, batch_size=2, shuffle=False)))
    assert len(dataset) == 3
    assert batch["input"].shape == (2, 1, 128, 128)
    assert batch["input"].dtype == torch.float32
    assert torch.isfinite(batch["input"]).all()
    assert batch["filename"] == [path.name for path in paths[:2]]
