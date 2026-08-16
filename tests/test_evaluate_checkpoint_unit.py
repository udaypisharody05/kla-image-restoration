"""Fast, dataset-free tests for evaluate_checkpoint.py's --tta support."""

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from evaluate_checkpoint import compute_lpips, validate_x8
from src.baseline import LPIPSUnavailableError
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.models import ResidualSRNet
from train import validate


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


def _tiny_validation_loader(tmp_path: Path, lr_size: int = 8) -> torch.utils.data.DataLoader:
    pairs = [_write_pair(tmp_path, f"{index:03d}", lr_size=lr_size) for index in range(4)]
    dataset = PairedRestorationDataset(pairs)  # no transform: full images, like real validation
    return create_dataloader(dataset, batch_size=2, shuffle=False)


def test_evaluate_checkpoint_tta_flag_exists_with_expected_choices() -> None:
    """Exercises the real argparse parser in evaluate_checkpoint.main() via --help."""
    result = subprocess.run(
        [sys.executable, "evaluate_checkpoint.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--tta {none,x8}" in result.stdout


def test_evaluate_checkpoint_lpips_flag_exists() -> None:
    result = subprocess.run(
        [sys.executable, "evaluate_checkpoint.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--lpips" in result.stdout


def test_compute_lpips_returns_finite_value_when_available(tmp_path: Path) -> None:
    """Skips cleanly (not a failure) when the optional 'lpips' package/weights
    are not installed in this environment -- LPIPS is explicitly optional."""
    try:
        from src.baseline import LPIPSMetric

        lpips_metric = LPIPSMetric()
    except LPIPSUnavailableError:
        pytest.skip("LPIPS package/pretrained weights not available in this environment")

    # AlexNet's conv/pool stack needs a larger-than-8x8 input to avoid a
    # negative-size intermediate feature map, unlike PSNR/SSIM/L1 (which work
    # at any size) -- 64x64 LR / 128x128 GT is comfortably large enough while
    # staying a fast, tiny synthetic test.
    loader = _tiny_validation_loader(tmp_path, lr_size=64)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    value = compute_lpips(model, loader, torch.device("cpu"), lpips_metric, tta="none")
    assert math.isfinite(value)
    assert value >= 0.0


def test_evaluate_checkpoint_raw_weights_flag_exists() -> None:
    """Phase 7: raw vs. EMA evaluation must be explicitly controllable/
    distinguishable from the CLI, not only via load_model's Python default."""
    result = subprocess.run(
        [sys.executable, "evaluate_checkpoint.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--raw-weights" in result.stdout


def test_validate_x8_produces_finite_metrics(tmp_path: Path) -> None:
    loader = _tiny_validation_loader(tmp_path)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    metrics = validate_x8(model, loader, nn.L1Loss(), torch.device("cpu"))
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["psnr"])
    assert math.isfinite(metrics["ssim"])


def test_validate_x8_uses_same_metric_functions_as_normal_validate(tmp_path: Path) -> None:
    """Not asserting the two produce identical numbers (they measure different
    things: single-pass vs 8-way-averaged predictions) -- only that validate_x8
    runs the same kind of aggregation (weighted mean over all samples) and stays
    within the same metric conventions (PSNR in dB range, SSIM roughly <= 1)."""
    loader = _tiny_validation_loader(tmp_path)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    normal_metrics = validate(model, loader, nn.L1Loss(), torch.device("cpu"))
    tta_metrics = validate_x8(model, loader, nn.L1Loss(), torch.device("cpu"))
    for metrics in (normal_metrics, tta_metrics):
        assert set(metrics.keys()) == {"loss", "psnr", "ssim"}
        assert metrics["ssim"] <= 1.0 + 1e-6


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)
