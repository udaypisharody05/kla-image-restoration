"""Fast, dataset-free tests for export_final_weights.py.

Uses a tiny synthetic EMA checkpoint (not the real ~10 MiB champion
checkpoint) so these tests run instantly and never touch real files on disk.
"""

from pathlib import Path

import pytest
import torch

from export_final_weights import export
from src.models import ResidualSRNet
from train import ExponentialMovingAverage, build_ema_config, save_checkpoint


def _write_synthetic_ema_checkpoint(path: Path, model_config: dict) -> ResidualSRNet:
    model = ResidualSRNet(**model_config)
    ema = ExponentialMovingAverage(model, decay=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Perturb the EMA shadow so it's provably different from the live model --
    # otherwise an export bug that silently used live weights would go unnoticed.
    with torch.no_grad():
        for shadow_param in ema.shadow_model.parameters():
            shadow_param.add_(1.0)
    save_checkpoint(
        path, model, optimizer, epoch=7, best_val_psnr=25.5,
        model_config=model_config, training_config={},
        ema=ema, ema_config=build_ema_config(True, 0.5),
    )
    return model


def test_export_writes_ema_weights_not_live_weights(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "weights" / "packaged.pt"
    live_model = _write_synthetic_ema_checkpoint(source, model_config)

    package = export(source, destination)

    live_state = live_model.state_dict()
    exported_state = package["model_state_dict"]
    # The EMA shadow was deliberately perturbed (+1.0) relative to the live
    # model above, so an exported EMA weight must NOT equal the live weight.
    assert not torch.equal(live_state["conv_in.weight"], exported_state["conv_in.weight"])


def test_export_package_reconstructs_identical_model(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "weights" / "packaged.pt"
    _write_synthetic_ema_checkpoint(source, model_config)

    package = export(source, destination)

    rebuilt = ResidualSRNet(
        in_channels=package["in_channels"], out_channels=package["out_channels"],
        num_features=package["num_features"], num_blocks=package["num_blocks"], scale=package["scale"],
    )
    rebuilt.load_state_dict(package["model_state_dict"])  # must not raise

    output = rebuilt(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()


def test_export_includes_provenance_and_reference_metrics(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "weights" / "packaged.pt"
    _write_synthetic_ema_checkpoint(source, model_config)

    package = export(source, destination)

    assert package["source_checkpoint"] == str(source)
    assert package["source_epoch"] == 7
    assert package["source_best_val_psnr"] == 25.5
    assert package["weights_type"] == "ema"
    assert package["architecture"] == "residual_sr"
    assert "reference_metrics" in package
    assert package["reference_metrics"]["val_psnr_db"] == pytest.approx(27.9893)


def test_export_does_not_modify_source_checkpoint_file(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "weights" / "packaged.pt"
    _write_synthetic_ema_checkpoint(source, model_config)

    original_bytes = source.read_bytes()
    original_mtime = source.stat().st_mtime

    export(source, destination)

    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime == original_mtime


def test_export_writes_output_file_to_destination(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "nested" / "weights" / "packaged.pt"
    _write_synthetic_ema_checkpoint(source, model_config)

    assert not destination.exists()
    export(source, destination)
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_export_rejects_checkpoint_without_ema(tmp_path: Path) -> None:
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    source = tmp_path / "no_ema_checkpoint.pt"
    save_checkpoint(
        source, model, optimizer, epoch=1, best_val_psnr=10.0,
        model_config=model_config, training_config={},
    )
    with pytest.raises(ValueError, match="ema_state_dict"):
        export(source, tmp_path / "weights" / "packaged.pt")


def test_export_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export(tmp_path / "does_not_exist.pt", tmp_path / "weights" / "packaged.pt")


def test_export_preserves_optional_variant_flags(tmp_path: Path) -> None:
    model_config = {
        "in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2,
        "channel_attention": True, "attention_reduction": 4,
    }
    source = tmp_path / "source_checkpoint.pt"
    destination = tmp_path / "weights" / "packaged.pt"
    _write_synthetic_ema_checkpoint(source, model_config)

    package = export(source, destination)

    assert package["channel_attention"] is True
    assert package["attention_reduction"] == 4
    assert package["multiscale_block"] is False  # default preserved for a key never set
