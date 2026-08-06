"""Validation against the optional full hackathon dataset."""

from pathlib import Path

import numpy as np
import pytest

from inspect_dataset import build_report
from src.dataset_discovery import discover_layout, discover_pairs
from src.io_utils import load_image_array


pytestmark = pytest.mark.integration


def test_load_real_file_and_raw_values_are_not_clipped(
    real_dataset_dir: Path,
) -> None:
    layout = discover_layout(real_dataset_dir)
    path = next(iter(sorted(layout.train_input_dir.glob("*.npy"))))
    expected = np.load(path, allow_pickle=False)
    actual = load_image_array(path)
    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_real_pair_discovery_is_complete_and_consistent(
    real_dataset_dir: Path,
) -> None:
    found = discover_pairs(discover_layout(real_dataset_dir))
    assert found.pairs
    assert [pair.pair_id for pair in found.pairs] == sorted(
        pair.pair_id for pair in found.pairs
    )
    assert all(
        pair.input_path.stem == pair.target_path.stem == pair.pair_id
        for pair in found.pairs
    )
    assert not found.missing_inputs and not found.missing_targets


def test_real_pair_dimensions_are_valid(real_dataset_dir: Path) -> None:
    pair = discover_pairs(discover_layout(real_dataset_dir)).pairs[0]
    source = load_image_array(pair.input_path)
    target = load_image_array(pair.target_path)
    assert min(source.shape[:2]) > 0 and min(target.shape[:2]) > 0
    assert target.shape[0] % source.shape[0] == 0
    assert target.shape[1] % source.shape[1] == 0


def test_real_dataset_report_generation(real_dataset_dir: Path) -> None:
    report = build_report(real_dataset_dir, max_samples=2)
    assert report["counts"]["valid_pairs"] > 0
    assert report["scale_factor_consistent"]
