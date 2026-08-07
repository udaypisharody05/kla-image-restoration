"""Portable tests for lazy PyTorch datasets and standard batching."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    PairedRestorationDataset,
    RestorationTestDataset,
    create_dataloader,
)
from src.dataset_discovery import ImagePair


def _write_pair(
    root: Path,
    sample_id: str,
    input_array: np.ndarray | None = None,
    target_array: np.ndarray | None = None,
) -> tuple[ImagePair, np.ndarray, np.ndarray]:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    degraded = input_array if input_array is not None else np.linspace(
        -0.25, 1.5, 128 * 128, dtype=np.float32
    ).reshape(128, 128)
    target = target_array if target_array is not None else np.linspace(
        0.0, 1.0, 256 * 256, dtype=np.float32
    ).reshape(256, 256)
    input_path = input_dir / f"{sample_id}.npy"
    target_path = target_dir / f"{sample_id}.npy"
    np.save(input_path, degraded)
    np.save(target_path, target)
    return ImagePair(sample_id, input_path, target_path), degraded, target


def test_paired_dataset_preserves_values_shape_dtype_and_pairing(tmp_path: Path) -> None:
    pair, degraded, target = _write_pair(tmp_path, "000007")
    dataset = PairedRestorationDataset([pair])
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["filename"] == "000007.npy"
    assert isinstance(sample["input"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    assert sample["input"].dtype == sample["target"].dtype == torch.float32
    assert sample["input"].shape == (1, 128, 128)
    assert sample["target"].shape == (1, 256, 256)
    assert torch.equal(sample["input"][0], torch.from_numpy(degraded))
    assert torch.equal(sample["target"][0], torch.from_numpy(target))
    assert float(sample["input"].min()) < 0.0
    assert float(sample["input"].max()) > 1.0
    assert np.array_equal(np.load(pair.input_path), degraded)


def test_deterministic_indexing_and_test_dataset_filename(tmp_path: Path) -> None:
    first, _, _ = _write_pair(tmp_path, "000001")
    second, _, _ = _write_pair(tmp_path, "000002")
    paired = PairedRestorationDataset([first, second])
    first_read = paired[0]
    second_read = paired[0]
    assert first_read["filename"] == second_read["filename"] == "000001.npy"
    assert torch.equal(first_read["input"], second_read["input"])

    test_dataset = RestorationTestDataset([second.input_path, first.input_path])
    test_sample = test_dataset[0]
    assert len(test_dataset) == 2
    assert test_sample["filename"] == "000002.npy"
    assert test_sample["input"].shape == (1, 128, 128)
    assert test_sample["input"].dtype == torch.float32
    assert "target" not in test_sample


@pytest.mark.parametrize(
    ("degraded", "target", "message"),
    [
        (np.zeros((2, 8, 8), np.float32), np.zeros((16, 16), np.float32), "2D"),
        (np.zeros((8, 8), np.float32), np.zeros((15, 16), np.float32), "2x"),
        (np.full((8, 8), np.nan, np.float32), np.zeros((16, 16), np.float32), "non-finite"),
        (np.zeros((8, 8), np.float32), np.full((16, 16), np.inf, np.float32), "non-finite"),
    ],
)
def test_paired_dataset_rejects_malformed_samples(
    tmp_path: Path, degraded: np.ndarray, target: np.ndarray, message: str
) -> None:
    pair, _, _ = _write_pair(tmp_path, "bad", degraded, target)
    with pytest.raises(ValueError, match=message):
        PairedRestorationDataset([pair])[0]


def test_standard_batching_and_filename_collation(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, f"{index:06d}")[0] for index in range(5)]
    dataset = PairedRestorationDataset(pairs)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch["input"].shape == (4, 1, 128, 128)
    assert batch["target"].shape == (4, 1, 256, 256)
    assert batch["input"].dtype == batch["target"].dtype == torch.float32
    assert batch["filename"] == [f"{index:06d}.npy" for index in range(4)]


def test_seeded_shuffle_is_reproducible(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, f"{index:06d}")[0] for index in range(8)]
    dataset = PairedRestorationDataset(pairs)
    first = next(iter(create_dataloader(dataset, 8, shuffle=True, seed=123)))["filename"]
    second = next(iter(create_dataloader(dataset, 8, shuffle=True, seed=123)))["filename"]
    assert first == second
