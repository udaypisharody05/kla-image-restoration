"""Fast, dataset-free tests for validate_restored_outputs.py's manifest generation."""

from pathlib import Path

import numpy as np
import pytest

from validate_restored_outputs import validate


def _make_pair_dirs(tmp_path: Path, count: int, lr_size: int = 8, scale: int = 2) -> tuple[Path, Path]:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    for index in range(count):
        name = f"{index:06d}.npy"
        np.save(input_dir / name, np.zeros((lr_size, lr_size), dtype=np.float32))
        np.save(output_dir / name, np.zeros((lr_size * scale, lr_size * scale), dtype=np.float32))
    return input_dir, output_dir


def test_validate_passes_for_matched_correctly_shaped_outputs(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=5)
    manifest = validate(input_dir, output_dir, scale=2, model="test-model", tta="none")
    assert manifest["status"] == "PASS"
    assert manifest["num_inputs"] == 5
    assert manifest["num_outputs"] == 5
    assert manifest["num_missing"] == 0
    assert manifest["num_extra"] == 0
    assert manifest["all_finite"] is True
    assert manifest["output_shapes_seen"] == [[16, 16]]
    assert manifest["problems"] == []


def test_validate_detects_missing_output(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=3)
    (output_dir / "000001.npy").unlink()
    manifest = validate(input_dir, output_dir, scale=2, model="m", tta="none")
    assert manifest["status"] == "FAIL"
    assert manifest["num_missing"] == 1
    assert any("no corresponding output" in problem for problem in manifest["problems"])


def test_validate_detects_extra_output(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=3)
    np.save(output_dir / "999999.npy", np.zeros((16, 16), dtype=np.float32))
    manifest = validate(input_dir, output_dir, scale=2, model="m", tta="none")
    assert manifest["status"] == "FAIL"
    assert manifest["num_extra"] == 1
    assert any("no corresponding input" in problem for problem in manifest["problems"])


def test_validate_detects_wrong_output_shape(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=2)
    np.save(output_dir / "000000.npy", np.zeros((10, 10), dtype=np.float32))  # should be 16x16
    manifest = validate(input_dir, output_dir, scale=2, model="m", tta="none")
    assert manifest["status"] == "FAIL"
    assert any("shape" in problem for problem in manifest["problems"])


def test_validate_detects_non_finite_values(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=2)
    bad = np.zeros((16, 16), dtype=np.float32)
    bad[0, 0] = np.nan
    np.save(output_dir / "000000.npy", bad)
    manifest = validate(input_dir, output_dir, scale=2, model="m", tta="none")
    assert manifest["status"] == "FAIL"
    assert manifest["all_finite"] is False
    assert any("non-finite" in problem for problem in manifest["problems"])


def test_validate_detects_wrong_dtype(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=2)
    np.save(output_dir / "000000.npy", np.zeros((16, 16), dtype=np.float64))
    manifest = validate(input_dir, output_dir, scale=2, model="m", tta="none")
    assert manifest["status"] == "FAIL"
    assert any("dtype" in problem for problem in manifest["problems"])


def test_validate_records_value_range_and_metadata(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=1)
    np.save(output_dir / "000000.npy", np.array([[-0.5, 1.5], [0.0, 1.0]], dtype=np.float32))
    manifest = validate(input_dir, output_dir, scale=2, model="residual_sr", tta="x8")
    assert manifest["value_range_overall"] == [-0.5, 1.5]
    assert manifest["model"] == "residual_sr"
    assert manifest["tta"] == "x8"
    assert manifest["no_ground_truth_included"] is True
    assert manifest["output_format"] == ".npy, float32, grayscale [H,W]"


def test_validate_writes_manifest_json_file(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=2)
    import json

    from validate_restored_outputs import main
    import sys

    argv = sys.argv
    sys.argv = [
        "validate_restored_outputs.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]
    try:
        main()
    finally:
        sys.argv = argv

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"


def test_validate_main_exits_nonzero_on_failure(tmp_path: Path) -> None:
    input_dir, output_dir = _make_pair_dirs(tmp_path, count=2)
    (output_dir / "000000.npy").unlink()
    import sys

    from validate_restored_outputs import main

    argv = sys.argv
    sys.argv = [
        "validate_restored_outputs.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
    finally:
        sys.argv = argv
