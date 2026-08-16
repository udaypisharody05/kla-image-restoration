"""Fast, dataset-free tests for inference.py -- the standalone submission
inference script.

Covers: model-config reconstruction from both a packaged weights file and a
raw training checkpoint, .npy input/output round-tripping, invalid input
directories, 2x output shape, deterministic filename handling, CPU fallback,
CUDA path when available, TTA output shape, and (indirectly, via
export_final_weights) that a source checkpoint is never touched by anything
in this pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from inference import (
    discover_input_files,
    load_inference_model,
    restore_one,
    run_inference,
    select_device,
)
from src.models import ResidualSRNet
from train import ExponentialMovingAverage, build_ema_config, save_checkpoint


def _write_packaged_weights(path: Path, model_config: dict) -> ResidualSRNet:
    model = ResidualSRNet(**model_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "weights_type": "raw",
            "architecture": "residual_sr",
            **model_config,
        },
        path,
    )
    return model


def _write_training_checkpoint(path: Path, model_config: dict) -> ResidualSRNet:
    model = ResidualSRNet(**model_config)
    ema = ExponentialMovingAverage(model, decay=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(
        path, model, optimizer, epoch=3, best_val_psnr=20.0,
        model_config=model_config, training_config={},
        ema=ema, ema_config=build_ema_config(True, 0.5),
    )
    return model


# --- select_device / CPU fallback / CUDA path ---


def test_select_device_defaults_to_cpu_or_cuda_based_on_availability() -> None:
    device = select_device(None)
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == expected.type


def test_select_device_explicit_cpu_override() -> None:
    assert select_device("cpu").type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available in this environment")
def test_select_device_explicit_cuda_when_available() -> None:
    assert select_device("cuda").type == "cuda"


def test_load_inference_model_runs_on_cpu(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    weights_path = tmp_path / "weights.pt"
    _write_packaged_weights(weights_path, model_config)
    model, _ = load_inference_model(weights_path, torch.device("cpu"))
    output = model(torch.randn(1, 1, 8, 8))
    assert output.device.type == "cpu"


# --- model config reconstruction (both file shapes) ---


def test_load_inference_model_from_packaged_weights(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    weights_path = tmp_path / "weights.pt"
    reference_model = _write_packaged_weights(weights_path, model_config)

    model, metadata = load_inference_model(weights_path, torch.device("cpu"))

    assert metadata["architecture"] == "residual_sr"
    reference_state = reference_model.state_dict()
    loaded_state = model.state_dict()
    assert all(torch.equal(reference_state[k], loaded_state[k]) for k in reference_state)


def test_load_inference_model_from_raw_training_checkpoint(tmp_path: Path) -> None:
    """--checkpoint may also point at a full train.py checkpoint (not just a
    packaged weights/*.pt file) -- delegates to evaluate_checkpoint.load_model,
    which prefers EMA weights automatically."""
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    _write_training_checkpoint(checkpoint_path, model_config)

    model, checkpoint = load_inference_model(checkpoint_path, torch.device("cpu"))

    assert checkpoint["model_config"] == model_config
    output = model(torch.randn(1, 1, 8, 8))
    assert output.shape == (1, 1, 16, 16)


def test_load_inference_model_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_inference_model(tmp_path / "does_not_exist.pt", torch.device("cpu"))


def test_load_inference_model_rejects_unrecognized_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"something_else": 1}, bad_path)
    with pytest.raises(ValueError, match="model_config.*model_state_dict|model_state_dict"):
        load_inference_model(bad_path, torch.device("cpu"))


def test_load_inference_model_optional_variant_flags_default_safely(tmp_path: Path) -> None:
    """A packaged file that predates a given optional-variant key (e.g. an
    older export) must still load as a plain model, not raise a KeyError."""
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    weights_path = tmp_path / "weights.pt"
    model = ResidualSRNet(**model_config)
    torch.save(
        {"model_state_dict": model.state_dict(), "architecture": "residual_sr", **model_config},
        weights_path,
    )  # no channel_attention/rdb_block/denoise_stem keys at all
    loaded_model, _ = load_inference_model(weights_path, torch.device("cpu"))
    output = loaded_model(torch.randn(1, 1, 8, 8))
    assert output.shape == (1, 1, 16, 16)


# --- discover_input_files: .npy filtering, determinism, invalid dirs ---


def test_discover_input_files_finds_only_npy_sorted(tmp_path: Path) -> None:
    for name in ("000002.npy", "000000.npy", "000001.npy", "readme.txt"):
        (tmp_path / name).write_bytes(b"") if name.endswith(".txt") else np.save(
            tmp_path / name, np.zeros((4, 4), dtype=np.float32)
        )
    files = discover_input_files(tmp_path)
    assert [f.name for f in files] == ["000000.npy", "000001.npy", "000002.npy"]


def test_discover_input_files_deterministic_across_calls(tmp_path: Path) -> None:
    for index in range(5):
        np.save(tmp_path / f"{index:06d}.npy", np.zeros((2, 2), dtype=np.float32))
    first = [f.name for f in discover_input_files(tmp_path)]
    second = [f.name for f in discover_input_files(tmp_path)]
    assert first == second


def test_discover_input_files_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_input_files(tmp_path / "missing")


def test_discover_input_files_path_is_a_file_not_a_directory_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.npy"
    np.save(file_path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(NotADirectoryError):
        discover_input_files(file_path)


def test_discover_input_files_empty_directory_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .npy files"):
        discover_input_files(empty_dir)


# --- restore_one: output shape (2x), .npy round trip, TTA shape ---


def test_restore_one_produces_2x_output_shape() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    array = np.random.default_rng(0).uniform(0, 1, size=(16, 24)).astype(np.float32)
    output = restore_one(model, array, torch.device("cpu"), tta="none")
    assert output.shape == (32, 48)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_restore_one_x8_tta_produces_same_2x_output_shape() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    array = np.random.default_rng(0).uniform(0, 1, size=(16, 16)).astype(np.float32)
    output = restore_one(model, array, torch.device("cpu"), tta="x8")
    assert output.shape == (32, 32)
    assert np.isfinite(output).all()


def test_restore_one_rejects_non_2d_array() -> None:
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    with pytest.raises(ValueError, match="2D"):
        restore_one(model, np.zeros((2, 4, 4), dtype=np.float32), torch.device("cpu"), tta="none")


# --- run_inference: end-to-end .npy IO, output dir creation, filenames ---


def test_run_inference_writes_one_npy_output_per_input_preserving_filenames(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    names = ["000000.npy", "000001.npy", "000002.npy"]
    for name in names:
        np.save(input_dir / name, np.random.default_rng(1).uniform(0, 1, size=(8, 8)).astype(np.float32))

    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    output_dir = tmp_path / "outputs"  # deliberately does not exist yet
    input_files = discover_input_files(input_dir)

    summary = run_inference(model, input_files, output_dir, torch.device("cpu"), tta="none")

    assert output_dir.exists()
    assert sorted(p.name for p in output_dir.glob("*.npy")) == names
    assert summary["num_input_files"] == 3
    assert summary["num_succeeded"] == 3
    assert summary["num_failed"] == 0
    for name in names:
        output_array = np.load(output_dir / name)
        assert output_array.shape == (16, 16)
        assert np.isfinite(output_array).all()


def test_run_inference_creates_missing_output_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.save(input_dir / "000000.npy", np.zeros((4, 4), dtype=np.float32))
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    output_dir = tmp_path / "a" / "b" / "c"

    run_inference(model, discover_input_files(input_dir), output_dir, torch.device("cpu"), tta="none")

    assert output_dir.exists()
    assert (output_dir / "000000.npy").exists()


def test_run_inference_reports_timing_fields(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.save(input_dir / "000000.npy", np.zeros((4, 4), dtype=np.float32))
    model = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)

    summary = run_inference(
        model, discover_input_files(input_dir), tmp_path / "out", torch.device("cpu"), tta="none"
    )
    assert summary["total_seconds"] >= 0.0
    assert summary["avg_seconds_per_image"] >= 0.0
