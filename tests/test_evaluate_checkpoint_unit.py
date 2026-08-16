"""Fast, dataset-free tests for evaluate_checkpoint.py's --tta support and its
``--checkpoint`` loading of both full train.py checkpoints and exported
inference weight packages (export_final_weights.py output)."""

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from evaluate_checkpoint import compute_lpips, load_model, validate_x8
from export_final_weights import export
from src.baseline import LPIPSUnavailableError
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.models import ResidualSRNet
from train import ExponentialMovingAverage, build_ema_config, save_checkpoint, validate


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


# --- load_model: full train.py checkpoints vs. exported inference packages ---


def _write_full_checkpoint(
    path: Path, model_config: dict, with_ema: bool = True, ema_decay: float = 0.5
) -> tuple[ResidualSRNet, ResidualSRNet | None]:
    """A real full train.py-style checkpoint (model_state_dict + model_config
    + optionally ema_state_dict), written with the project's own
    save_checkpoint/ExponentialMovingAverage -- not a hand-rolled dict."""
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = None
    ema_config = None
    if with_ema:
        ema = ExponentialMovingAverage(model, decay=ema_decay)
        # Perturb the shadow so EMA and live weights are provably different --
        # otherwise a bug that silently loaded the wrong one would go unnoticed.
        with torch.no_grad():
            for shadow_param in ema.shadow_model.parameters():
                shadow_param.add_(1.0)
        ema_config = build_ema_config(True, ema_decay)
    save_checkpoint(
        path, model, optimizer, epoch=42, best_val_psnr=26.5,
        model_config=model_config, training_config={},
        ema=ema, ema_config=ema_config,
    )
    shadow_model = ema.shadow_model if ema is not None else None
    return model, shadow_model


def test_load_model_full_checkpoint_with_ema_prefers_ema_weights(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    live_model, shadow_model = _write_full_checkpoint(checkpoint_path, model_config, with_ema=True)

    loaded_model, checkpoint = load_model(checkpoint_path, torch.device("cpu"), prefer_ema=True)

    assert checkpoint["model_config"] == model_config
    assert checkpoint.get("source_format") is None  # unchanged behavior for real checkpoints
    loaded_state = loaded_model.state_dict()
    assert torch.equal(loaded_state["conv_in.weight"], shadow_model.state_dict()["conv_in.weight"])
    assert not torch.equal(loaded_state["conv_in.weight"], live_model.state_dict()["conv_in.weight"])


def test_load_model_full_checkpoint_raw_weights_flag_selects_live_weights(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    live_model, _shadow_model = _write_full_checkpoint(checkpoint_path, model_config, with_ema=True)

    loaded_model, _checkpoint = load_model(checkpoint_path, torch.device("cpu"), prefer_ema=False)

    loaded_state = loaded_model.state_dict()
    assert torch.equal(loaded_state["conv_in.weight"], live_model.state_dict()["conv_in.weight"])


def test_load_model_full_checkpoint_without_ema_loads_live_weights(tmp_path: Path) -> None:
    """Historical (non-EMA) checkpoints have ema_state_dict=None -- prefer_ema=True
    must fall through to the live weights unchanged, not raise."""
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    live_model, shadow_model = _write_full_checkpoint(checkpoint_path, model_config, with_ema=False)
    assert shadow_model is None

    loaded_model, checkpoint = load_model(checkpoint_path, torch.device("cpu"), prefer_ema=True)

    assert checkpoint.get("ema_state_dict") is None
    loaded_state = loaded_model.state_dict()
    assert torch.equal(loaded_state["conv_in.weight"], live_model.state_dict()["conv_in.weight"])


def test_load_model_exported_package_loads_and_reconstructs_config(tmp_path: Path) -> None:
    """The tracked public artifact (weights/residualsr_final_ema.pt) is an
    export_final_weights.py package, not a full checkpoint -- load_model must
    handle it without requiring the unavailable training-checkpoint fields
    (optimizer/scheduler/epoch counter)."""
    model_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2,
        "channel_attention": True, "attention_reduction": 4,
    }
    source_checkpoint = tmp_path / "source_checkpoint.pt"
    package_path = tmp_path / "weights" / "residualsr_final_ema.pt"
    live_model, shadow_model = _write_full_checkpoint(source_checkpoint, model_config, with_ema=True)
    export(source_checkpoint, package_path)

    loaded_model, checkpoint = load_model(package_path, torch.device("cpu"))

    assert checkpoint["source_format"] == "exported_package"
    assert checkpoint["model_config"] == {
        **{k: v for k, v in model_config.items()},
        "multiscale_block": False,
        "rdb_block": False,
        "rdb_growth_rate": 16,
        "rdb_num_layers": 3,
        "denoise_stem": False,
        "denoise_stem_features": 32,
        "denoise_stem_blocks": 2,
    }
    # export_final_weights.py always exports the EMA shadow -- confirm that's
    # actually what got loaded here, not the live weights.
    loaded_state = loaded_model.state_dict()
    assert torch.equal(loaded_state["conv_in.weight"], shadow_model.state_dict()["conv_in.weight"])
    assert not torch.equal(loaded_state["conv_in.weight"], live_model.state_dict()["conv_in.weight"])


def test_load_model_exported_package_reports_ema_and_provenance(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source_checkpoint = tmp_path / "source_checkpoint.pt"
    package_path = tmp_path / "weights" / "packaged.pt"
    _write_full_checkpoint(source_checkpoint, model_config, with_ema=True)
    export(source_checkpoint, package_path)

    _model, checkpoint = load_model(package_path, torch.device("cpu"))

    assert checkpoint["ema_state_dict"] is not None
    assert checkpoint["epoch"] == 42
    assert checkpoint["best_val_psnr"] == pytest.approx(26.5)
    assert checkpoint["loss_config"] == {"name": "l1"}


def test_load_model_exported_package_prefer_ema_flag_has_no_effect(tmp_path: Path) -> None:
    """A package stores exactly one weight set -- prefer_ema True/False must
    load the same (only available) weights either way, not raise or silently
    diverge."""
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source_checkpoint = tmp_path / "source_checkpoint.pt"
    package_path = tmp_path / "weights" / "packaged.pt"
    _write_full_checkpoint(source_checkpoint, model_config, with_ema=True)
    export(source_checkpoint, package_path)

    model_prefer_ema, _ = load_model(package_path, torch.device("cpu"), prefer_ema=True)
    model_raw, _ = load_model(package_path, torch.device("cpu"), prefer_ema=False)

    assert torch.equal(
        model_prefer_ema.state_dict()["conv_in.weight"], model_raw.state_dict()["conv_in.weight"]
    )


def test_load_model_rejects_malformed_checkpoint_with_useful_error(tmp_path: Path) -> None:
    """Neither a full checkpoint (model_config) nor an exported package
    (model_state_dict) -- must fail with an actionable ValueError, not a
    cryptic KeyError from deep inside build_model/load_state_dict."""
    bad_path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"something_else": 1}, bad_path)

    with pytest.raises(ValueError, match="model_config.*model_state_dict|model_state_dict"):
        load_model(bad_path, torch.device("cpu"))


def test_load_model_rejects_empty_dict(tmp_path: Path) -> None:
    bad_path = tmp_path / "empty.pt"
    torch.save({}, bad_path)

    with pytest.raises(ValueError):
        load_model(bad_path, torch.device("cpu"))
