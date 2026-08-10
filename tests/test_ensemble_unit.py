"""Fast, dataset-free tests for src/ensemble.py and evaluate_ensemble.py's ensemble path."""

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from evaluate_checkpoint import load_model
from evaluate_ensemble import validate_ensemble
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.ensemble import weighted_average_predictions
from src.metrics import psnr as psnr_fn
from src.metrics import ssim as ssim_fn
from src.models import ResidualSRNet
from src.tta import predict_x8


# --- weighted_average_predictions ---


def test_50_50_average_is_mathematically_correct() -> None:
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    result = weighted_average_predictions([a, b], [0.5, 0.5])
    assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))


def test_weighted_average_is_mathematically_correct() -> None:
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[4.0, 8.0]])
    result = weighted_average_predictions([a, b], [0.75, 0.25])
    assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))


def test_weights_normalize_correctly_when_not_summing_to_one() -> None:
    a = torch.tensor([[0.0]])
    b = torch.tensor([[4.0]])
    # [3, 1] normalizes to [0.75, 0.25] -- same result as passing that directly.
    result = weighted_average_predictions([a, b], [3.0, 1.0])
    assert torch.allclose(result, torch.tensor([[1.0]]))


def test_default_weights_are_equal() -> None:
    a = torch.tensor([[1.0]])
    b = torch.tensor([[3.0]])
    result = weighted_average_predictions([a, b])
    assert torch.allclose(result, torch.tensor([[2.0]]))


def test_shape_mismatch_is_rejected() -> None:
    a = torch.rand(1, 1, 4, 4)
    b = torch.rand(1, 1, 4, 5)
    with pytest.raises(ValueError, match="[Ss]hape"):
        weighted_average_predictions([a, b])


def test_empty_prediction_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_average_predictions([])


def test_single_prediction_is_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_average_predictions([torch.rand(1, 1, 4, 4)])


def test_prediction_weight_count_mismatch_is_rejected() -> None:
    a = torch.rand(1, 1, 4, 4)
    b = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        weighted_average_predictions([a, b], [0.5, 0.3, 0.2])


def test_non_positive_weights_are_rejected() -> None:
    a = torch.rand(1, 1, 4, 4)
    b = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        weighted_average_predictions([a, b], [1.0, 0.0])


def test_batch_dimension_is_preserved() -> None:
    a = torch.rand(5, 1, 8, 8)
    b = torch.rand(5, 1, 8, 8)
    assert weighted_average_predictions([a, b]).shape[0] == 5


def test_channel_dimension_is_preserved() -> None:
    a = torch.rand(2, 3, 8, 8)
    b = torch.rand(2, 3, 8, 8)
    assert weighted_average_predictions([a, b]).shape[1] == 3


def test_spatial_dimensions_are_preserved() -> None:
    a = torch.rand(1, 1, 16, 24)
    b = torch.rand(1, 1, 16, 24)
    assert weighted_average_predictions([a, b]).shape[-2:] == (16, 24)


def test_raw_values_outside_0_1_remain_unclipped() -> None:
    a = torch.full((1, 1, 4, 4), 2.0)
    b = torch.full((1, 1, 4, 4), 2.0)
    result = weighted_average_predictions([a, b])
    assert torch.allclose(result, torch.full((1, 1, 4, 4), 2.0))
    assert result.max().item() > 1.0


def test_negative_raw_values_remain_unclipped() -> None:
    a = torch.full((1, 1, 4, 4), -1.0)
    b = torch.full((1, 1, 4, 4), -1.0)
    result = weighted_average_predictions([a, b])
    assert result.min().item() < 0.0


def test_finite_output_for_finite_inputs() -> None:
    a = torch.rand(2, 1, 8, 8)
    b = torch.rand(2, 1, 8, 8)
    assert torch.isfinite(weighted_average_predictions([a, b])).all()


def test_identical_predictions_produce_identical_ensemble_output() -> None:
    a = torch.rand(2, 1, 8, 8)
    result = weighted_average_predictions([a, a.clone()], [0.3, 0.7])
    assert torch.allclose(result, a, atol=1e-6)


def test_three_predictions_supported() -> None:
    a = torch.tensor([[3.0]])
    b = torch.tensor([[3.0]])
    c = torch.tensor([[3.0]])
    assert torch.allclose(weighted_average_predictions([a, b, c]), torch.tensor([[3.0]]))


# --- Normal + x8 ensemble evaluation paths (tiny synthetic dataset) ---


def _write_pair(root: Path, sample_id: str, lr_size: int = 8, scale: int = 2) -> ImagePair:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    degraded = rng.uniform(0.0, 1.0, size=(lr_size, lr_size)).astype(np.float32)
    target = rng.uniform(0.0, 1.0, size=(lr_size * scale, lr_size * scale)).astype(np.float32)
    input_path = input_dir / f"{sample_id}.npy"
    target_path = target_dir / f"{sample_id}.npy"
    np.save(input_path, degraded)
    np.save(target_path, target)
    return ImagePair(sample_id, input_path, target_path)


def _tiny_validation_loader(tmp_path: Path) -> torch.utils.data.DataLoader:
    pairs = [_write_pair(tmp_path, f"{index:03d}") for index in range(4)]
    dataset = PairedRestorationDataset(pairs)  # no transform: full images, like real validation
    return create_dataloader(dataset, batch_size=2, shuffle=False)


def test_normal_ensemble_path_produces_finite_metrics(tmp_path: Path) -> None:
    loader = _tiny_validation_loader(tmp_path)
    model_a = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    model_b = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    metrics = validate_ensemble(
        model_a, model_b, loader, [0.5, 0.5], nn.L1Loss(), torch.device("cpu"), tta="none"
    )
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["psnr"])
    assert math.isfinite(metrics["ssim"])


def test_x8_ensemble_path_reuses_predict_x8(tmp_path: Path) -> None:
    """Manually reproduces the x8 ensemble result using predict_x8 + weighted_average_predictions
    directly, and checks validate_ensemble(tta="x8") matches -- proving the CLI path is
    genuinely built on src.tta.predict_x8, not a second TTA implementation."""
    torch.manual_seed(0)
    model_a = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    model_b = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    loader = _tiny_validation_loader(tmp_path)
    batch = next(iter(loader))
    inputs, targets = batch["input"], batch["target"]

    with torch.no_grad():
        prediction_a = predict_x8(model_a, inputs)
        prediction_b = predict_x8(model_b, inputs)
    expected_output = weighted_average_predictions([prediction_a, prediction_b], [0.5, 0.5])

    metrics = validate_ensemble(
        model_a, model_b, [batch], [0.5, 0.5], nn.L1Loss(), torch.device("cpu"), tta="x8"
    )
    assert math.isclose(metrics["psnr"], psnr_fn(expected_output, targets), rel_tol=1e-5)
    assert math.isclose(metrics["ssim"], ssim_fn(expected_output, targets), rel_tol=1e-5)


# --- Real checkpoint loading (Experiments 6 and 9) ---


def test_exp6_checkpoint_loads_correctly() -> None:
    checkpoint_path = Path("checkpoints/exp6_crop96/checkpoint_best.pt")
    if not checkpoint_path.exists():
        pytest.skip("Experiment 6 checkpoint not available in this environment")
    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert checkpoint["model_config"].get("architecture", "residual_sr") == "residual_sr"
    assert not model.training


def test_exp9_checkpoint_loads_correctly() -> None:
    checkpoint_path = Path("checkpoints/exp9_edsr_lite/checkpoint_best.pt")
    if not checkpoint_path.exists():
        pytest.skip("Experiment 9 checkpoint not available in this environment")
    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert checkpoint["model_config"].get("architecture") == "edsr_lite"
    assert not model.training


# --- CLI ---


def test_evaluate_ensemble_cli_exists_with_expected_flags() -> None:
    result = subprocess.run(
        [sys.executable, "evaluate_ensemble.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--checkpoint-a" in result.stdout
    assert "--checkpoint-b" in result.stdout
    assert "--weight-a" in result.stdout
    assert "--weight-b" in result.stdout
    assert "--tta {none,x8}" in result.stdout
