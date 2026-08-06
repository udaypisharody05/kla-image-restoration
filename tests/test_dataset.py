from pathlib import Path

import numpy as np

from inspect_dataset import build_report, inspect_group, save_report
from src.dataset_discovery import discover_layout, discover_pairs
from src.io_utils import load_image_array

DATA = Path("data")


def test_load_real_file_and_raw_values_are_not_clipped() -> None:
    layout = discover_layout(DATA)
    path = next(iter(sorted(layout.train_input_dir.glob("*.npy"))))
    expected = np.load(path, allow_pickle=False)
    actual = load_image_array(path)
    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_pair_discovery_is_complete_sorted_and_consistent() -> None:
    found = discover_pairs(discover_layout(DATA))
    assert found.pairs
    assert [p.pair_id for p in found.pairs] == sorted(p.pair_id for p in found.pairs)
    assert all(p.input_path.stem == p.target_path.stem == p.pair_id for p in found.pairs)
    assert not found.missing_inputs and not found.missing_targets


def test_real_pair_dimensions_are_valid() -> None:
    pair = discover_pairs(discover_layout(DATA)).pairs[0]
    source, target = load_image_array(pair.input_path), load_image_array(pair.target_path)
    assert min(source.shape[:2]) > 0 and min(target.shape[:2]) > 0
    assert target.shape[0] % source.shape[0] == target.shape[1] % source.shape[1] == 0


def test_nonfinite_detection(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[np.nan, np.inf]], dtype=np.float32))
    stats = inspect_group([path], 0, "test")
    assert stats["nan_count"] == 1 and stats["infinite_count"] == 1


def test_report_generation(tmp_path: Path) -> None:
    report = build_report(DATA, max_samples=2)
    save_report(report, tmp_path)
    assert (tmp_path / "dataset_report.json").is_file()
    assert (tmp_path / "dataset_report.md").is_file()
    assert report["counts"]["valid_pairs"] > 0


def test_loader_does_not_clip_synthetic_float(tmp_path: Path) -> None:
    path = tmp_path / "range.npy"
    expected = np.array([[-0.25, 1.5]], dtype=np.float32)
    np.save(path, expected)
    assert np.array_equal(load_image_array(path), expected)
