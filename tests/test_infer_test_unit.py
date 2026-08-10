"""Fast, dataset-free tests for infer_test.py's inference sanity-check helpers."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluate_checkpoint import load_model
from infer_test import (
    clip_for_display,
    run_inference,
    select_test_files,
    validate_prediction_before_saving,
)
from src.models import ResidualSRNet
from train import save_checkpoint


def _tiny_model_config() -> dict:
    return {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}


# --- Checkpoint reconstruction (reused via evaluate_checkpoint.load_model, not duplicated) ---


def test_load_model_reconstructs_checkpoint_config_and_weights(tmp_path: Path) -> None:
    model_config = _tiny_model_config()
    original_model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(original_model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(
        checkpoint_path,
        original_model,
        optimizer,
        epoch=38,
        best_val_psnr=27.7090,
        model_config=model_config,
        training_config={},
    )

    loaded_model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded_model, ResidualSRNet)
    assert checkpoint["model_config"] == model_config
    assert checkpoint["epoch"] == 38
    assert checkpoint["best_val_psnr"] == 27.7090
    for original_param, loaded_param in zip(
        original_model.parameters(), loaded_model.parameters()
    ):
        assert torch.equal(original_param, loaded_param)
    assert not loaded_model.training  # load_model() already calls .eval()


# --- Full-resolution inference and expected 2x output dimensions ---


def test_run_inference_produces_exact_2x_shape_on_full_resolution_input() -> None:
    model = ResidualSRNet(**_tiny_model_config())
    # Full-resolution-like, non-cropped, non-square input (mirrors real 128x128
    # test images without requiring the real dataset).
    input_tensor = torch.rand(1, 20, 24)  # [C,H,W]
    prediction = run_inference(model, input_tensor, torch.device("cpu"))
    assert prediction.shape == (40, 48)
    assert prediction.dtype == np.float32


def test_run_inference_output_is_finite() -> None:
    model = ResidualSRNet(**_tiny_model_config())
    input_tensor = torch.rand(1, 16, 16)
    prediction = run_inference(model, input_tensor, torch.device("cpu"))
    assert np.isfinite(prediction).all()


def test_run_inference_does_not_alter_input_tensor() -> None:
    model = ResidualSRNet(**_tiny_model_config())
    input_tensor = torch.rand(1, 16, 16)
    original = input_tensor.clone()
    run_inference(model, input_tensor, torch.device("cpu"))
    assert torch.equal(input_tensor, original)


# --- Prediction clipping for saving/display ---


def test_clip_for_display_clamps_to_0_1() -> None:
    prediction = np.array([[-0.5, 0.0], [0.7, 1.8]], dtype=np.float32)
    clipped = clip_for_display(prediction)
    expected = np.array([[0.0, 0.0], [0.7, 1.0]], dtype=np.float32)
    assert np.allclose(clipped, expected)


def test_clip_for_display_does_not_mutate_input() -> None:
    prediction = np.array([-0.5, 1.8], dtype=np.float32)
    original = prediction.copy()
    clip_for_display(prediction)
    assert np.array_equal(prediction, original)


def test_clip_for_display_leaves_in_range_values_unchanged() -> None:
    prediction = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    assert np.array_equal(clip_for_display(prediction), prediction)


# --- Finite-value / shape validation before saving ---


def test_validate_prediction_accepts_well_formed_prediction() -> None:
    prediction = np.zeros((32, 32), dtype=np.float32)
    validate_prediction_before_saving(prediction, (32, 32), "000000.npy")  # must not raise


def test_validate_prediction_rejects_wrong_shape() -> None:
    prediction = np.zeros((30, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_prediction_before_saving(prediction, (32, 32), "000000.npy")


def test_validate_prediction_rejects_nan() -> None:
    prediction = np.zeros((4, 4), dtype=np.float32)
    prediction[0, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction_before_saving(prediction, (4, 4), "000000.npy")


def test_validate_prediction_rejects_inf() -> None:
    prediction = np.zeros((4, 4), dtype=np.float32)
    prediction[1, 1] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction_before_saving(prediction, (4, 4), "000000.npy")


# --- Deterministic test-file selection ---


def _write_test_layout(root: Path, count: int) -> Path:
    """Minimal 3-leaf directory layout discover_layout can find (train/target/test)."""
    data_root = root / "Data-public"
    train_inputs = data_root / "train" / "train" / "NoisyLR"
    targets = data_root / "train" / "train" / "GT"
    test_inputs = data_root / "Test_NoisyLR" / "NoisyLR"
    for directory in (train_inputs, targets, test_inputs):
        directory.mkdir(parents=True, exist_ok=True)
    # A couple of real training pairs so discover_layout has enough to work with.
    np.save(train_inputs / "000000.npy", np.zeros((4, 4), dtype=np.float32))
    np.save(targets / "000000.npy", np.zeros((8, 8), dtype=np.float32))
    # Unsorted creation order on purpose -- selection must still come out sorted.
    ids = [f"{index:06d}" for index in range(count)][::-1]
    for sample_id in ids:
        np.save(test_inputs / f"{sample_id}.npy", np.zeros((4, 4), dtype=np.float32))
    return data_root


def test_select_test_files_returns_first_n_in_sorted_order(tmp_path: Path) -> None:
    data_dir = _write_test_layout(tmp_path, count=15)
    selected = select_test_files(data_dir, max_samples=10)
    assert [path.stem for path in selected] == [f"{index:06d}" for index in range(10)]


def test_select_test_files_is_deterministic_across_calls(tmp_path: Path) -> None:
    data_dir = _write_test_layout(tmp_path, count=15)
    first = select_test_files(data_dir, max_samples=10)
    second = select_test_files(data_dir, max_samples=10)
    assert first == second


def test_select_test_files_rejects_non_positive_max_samples(tmp_path: Path) -> None:
    data_dir = _write_test_layout(tmp_path, count=5)
    with pytest.raises(ValueError, match="positive"):
        select_test_files(data_dir, max_samples=0)


def test_select_test_files_returns_fewer_when_fewer_available(tmp_path: Path) -> None:
    data_dir = _write_test_layout(tmp_path, count=3)
    selected = select_test_files(data_dir, max_samples=10)
    assert len(selected) == 3


# --- Experiment 10: x8 TTA CLI defaults and integration ---


def test_infer_test_tta_flag_exists_with_expected_choices() -> None:
    """Exercises the real argparse parser in infer_test.main() via --help,
    rather than a hand-built duplicate parser. That the default is actually
    "none" and behaves identically to omitting the flag is verified directly
    by test_run_inference_default_tta_none_matches_explicit_none above."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "infer_test.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--tta {none,x8}" in result.stdout


def test_run_inference_default_tta_none_matches_explicit_none() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    model.eval()
    input_tensor = torch.rand(1, 8, 8)
    torch.manual_seed(0)
    default_result = run_inference(model, input_tensor, torch.device("cpu"))
    torch.manual_seed(0)
    explicit_none_result = run_inference(model, input_tensor, torch.device("cpu"), tta="none")
    assert np.array_equal(default_result, explicit_none_result)


def test_run_inference_x8_produces_correct_shape_and_finite_values() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    input_tensor = torch.rand(1, 8, 8)
    prediction = run_inference(model, input_tensor, torch.device("cpu"), tta="x8")
    assert prediction.shape == (16, 16)
    assert np.isfinite(prediction).all()


def test_run_inference_x8_differs_from_single_pass_in_general() -> None:
    """Not a mathematical requirement, but confirms x8 is actually doing
    something different from a single forward pass for a typical (non-equivariant)
    trained-style model, i.e. that the tta branch is really wired in."""
    torch.manual_seed(1)
    model = ResidualSRNet(num_features=4, num_blocks=2, scale=2)
    input_tensor = torch.rand(1, 24, 24)
    single = run_inference(model, input_tensor, torch.device("cpu"), tta="none")
    x8 = run_inference(model, input_tensor, torch.device("cpu"), tta="x8")
    assert not np.allclose(single, x8)
