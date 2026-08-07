"""Portable tests using only tiny, generated arrays."""

from pathlib import Path
import json

import numpy as np

from inspect_dataset import (
    REPOSITORY_ROOT,
    build_report,
    configured_data_dir,
    inspect_group,
    save_report,
)
from src.dataset_discovery import discover_layout, discover_pairs
from src.io_utils import load_image_array


def test_loader_preserves_synthetic_float_values(tmp_path: Path) -> None:
    path = tmp_path / "range.npy"
    expected = np.array([[-0.25, 1.5]], dtype=np.float32)
    np.save(path, expected)
    actual = load_image_array(path)
    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_pair_discovery_is_complete_sorted_and_consistent(
    synthetic_dataset_dir: Path,
) -> None:
    found = discover_pairs(discover_layout(synthetic_dataset_dir))
    assert [pair.pair_id for pair in found.pairs] == ["000000", "000001"]
    assert all(
        pair.input_path.stem == pair.target_path.stem == pair.pair_id
        for pair in found.pairs
    )
    assert not found.missing_inputs and not found.missing_targets
    assert not found.duplicate_input_ids and not found.duplicate_target_ids


def test_pair_dimensions_and_scale_are_valid(synthetic_dataset_dir: Path) -> None:
    pair = discover_pairs(discover_layout(synthetic_dataset_dir)).pairs[0]
    source = load_image_array(pair.input_path)
    target = load_image_array(pair.target_path)
    assert source.shape == (2, 2)
    assert target.shape == (4, 4)
    assert target.shape[0] // source.shape[0] == 2
    assert target.shape[1] // source.shape[1] == 2


def test_nonfinite_detection(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[np.nan, np.inf]], dtype=np.float32))
    stats = inspect_group([path], 0, "test")
    assert stats["nan_count"] == 1
    assert stats["infinite_count"] == 1


def test_report_generation(synthetic_dataset_dir: Path, tmp_path: Path) -> None:
    report = build_report(synthetic_dataset_dir, max_samples=0)
    save_report(report, tmp_path)
    assert report["counts"] == {
        "training_inputs": 2,
        "ground_truths": 2,
        "valid_pairs": 2,
        "test_inputs": 1,
    }
    assert report["scale_factors"] == {"2x2": 2}
    assert report["input"]["outside_0_1_count"] == 4
    assert (tmp_path / "dataset_report.json").is_file()
    assert (tmp_path / "dataset_report.md").is_file()


def test_report_paths_are_portable(
    synthetic_dataset_dir: Path, tmp_path: Path
) -> None:
    report = build_report(synthetic_dataset_dir, max_samples=1)
    save_report(report, tmp_path)

    assert report["dataset_root"] == "Data-public"
    assert report["requested_data_dir"] == "."
    assert report["directories"] == {
        "train_input": "train/train/NoisyLR",
        "ground_truth": "train/train/GT",
        "test_input": "Test_NoisyLR/NoisyLR",
    }

    json_text = (tmp_path / "dataset_report.json").read_text(encoding="utf-8")
    markdown_text = (tmp_path / "dataset_report.md").read_text(encoding="utf-8")
    serialized = json_text + markdown_text
    assert str(synthetic_dataset_dir.resolve()) not in serialized
    assert str(Path.home()) not in serialized
    assert "C:\\" not in serialized
    assert "train/train/NoisyLR" in serialized
    assert json.loads(json_text)["directories"] == report["directories"]


def test_failed_file_diagnostics_do_not_leak_absolute_paths(tmp_path: Path) -> None:
    broken = tmp_path / "dataset" / "train" / "broken.npy"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a numpy file")

    stats = inspect_group([broken], 0, "test", path_base=tmp_path / "dataset")
    diagnostic = json.dumps(stats["failed_files"])
    assert stats["failed_files"][0]["path"] == "train/broken.npy"
    assert str(tmp_path) not in diagnostic


def test_configured_data_dir_supports_relative_environment_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SEMICON_DATA_DIR", "external/dataset")
    assert configured_data_dir() == (REPOSITORY_ROOT / "external/dataset").resolve()
