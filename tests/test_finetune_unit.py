"""Fast, dataset-free tests for fine-tuning support (--finetune-from) and the
--global-bicubic-residual CLI mapping.

Fine-tuning (``load_weights_for_finetune``) is deliberately a much smaller
surface than ``load_checkpoint_for_resume``: it loads weights only and
touches nothing else, so a fresh optimizer/scheduler/epoch counter/best-score
naturally results just from *not* restoring them -- there is no separate
"reset" code path to test beyond "the loaded model's weights match the
checkpoint, and nothing else about the caller's fresh objects changed."
"""

import hashlib
from pathlib import Path

import pytest
import torch

from src.ema import ExponentialMovingAverage
from src.models import ResidualSRNet
from train import (
    load_weights_for_finetune,
    resolve_model_architecture,
    save_checkpoint,
    validate_finetune_args,
)


def _model_config() -> dict:
    return {"in_channels": 1, "out_channels": 1, "num_features": 8, "num_blocks": 2, "scale": 2}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- load_weights_for_finetune: weights only, nothing else ---


def test_finetune_loads_matching_model_weights(tmp_path: Path) -> None:
    source_model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    path = tmp_path / "source.pt"
    save_checkpoint(
        path, source_model, optimizer, epoch=50, best_val_psnr=27.5,
        model_config=_model_config(), training_config={},
    )

    target_model = ResidualSRNet(**_model_config())
    load_weights_for_finetune(path, target_model, torch.device("cpu"))

    for source_param, target_param in zip(
        source_model.state_dict().values(), target_model.state_dict().values()
    ):
        assert torch.equal(source_param, target_param)


def test_finetune_prefers_ema_weights_when_present(tmp_path: Path) -> None:
    source_model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    ema = ExponentialMovingAverage(source_model, decay=0.9)
    # Diverge the EMA shadow from the live weights so the two are distinguishable.
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.add_(1.0)
    ema.update(source_model)

    path = tmp_path / "source.pt"
    save_checkpoint(
        path, source_model, optimizer, epoch=50, best_val_psnr=27.5,
        model_config=_model_config(), training_config={}, ema=ema, ema_config={"enabled": True, "decay": 0.9},
    )

    target_model = ResidualSRNet(**_model_config())
    load_weights_for_finetune(path, target_model, torch.device("cpu"))

    for shadow_param, target_param in zip(
        ema.shadow_model.state_dict().values(), target_model.state_dict().values()
    ):
        assert torch.equal(shadow_param, target_param)
    # And NOT equal to the live (post-update) source weights, proving EMA (not live) was loaded.
    live_params = list(source_model.state_dict().values())
    target_params = list(target_model.state_dict().values())
    assert any(not torch.equal(live, target) for live, target in zip(live_params, target_params))


def test_finetune_falls_back_to_live_weights_when_no_ema(tmp_path: Path) -> None:
    source_model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    path = tmp_path / "source.pt"
    save_checkpoint(
        path, source_model, optimizer, epoch=50, best_val_psnr=27.5,
        model_config=_model_config(), training_config={},
    )

    target_model = ResidualSRNet(**_model_config())
    load_weights_for_finetune(path, target_model, torch.device("cpu"), prefer_ema=True)
    for source_param, target_param in zip(
        source_model.state_dict().values(), target_model.state_dict().values()
    ):
        assert torch.equal(source_param, target_param)


def test_finetune_does_not_modify_the_source_checkpoint_file(tmp_path: Path) -> None:
    source_model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    path = tmp_path / "source.pt"
    save_checkpoint(
        path, source_model, optimizer, epoch=50, best_val_psnr=27.5,
        model_config=_model_config(), training_config={},
    )
    before_hash = _file_hash(path)

    target_model = ResidualSRNet(**_model_config())
    load_weights_for_finetune(path, target_model, torch.device("cpu"))

    assert _file_hash(path) == before_hash


def test_finetune_leaves_a_fresh_optimizer_and_epoch_counter_untouched(tmp_path: Path) -> None:
    """Unlike load_checkpoint_for_resume, load_weights_for_finetune never
    touches an optimizer or returns an epoch/best-score -- callers naturally
    keep whatever fresh objects they constructed themselves."""
    source_model = ResidualSRNet(**_model_config())
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4)
    path = tmp_path / "source.pt"
    save_checkpoint(
        path, source_model, source_optimizer, epoch=50, best_val_psnr=27.5,
        model_config=_model_config(), training_config={},
    )

    target_model = ResidualSRNet(**_model_config())
    fresh_optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-5)
    return_value = load_weights_for_finetune(path, target_model, torch.device("cpu"))

    assert return_value is None  # no epoch/best-score/training_config returned
    assert fresh_optimizer.state_dict()["state"] == {}  # never touched -> no momentum state
    assert fresh_optimizer.param_groups[0]["lr"] == 1e-5  # caller's own fresh --lr, untouched


# --- validate_finetune_args ---


def test_validate_finetune_args_allows_neither_flag() -> None:
    validate_finetune_args(None, None, Path("checkpoints/anything"))


def test_validate_finetune_args_allows_resume_alone() -> None:
    validate_finetune_args(Path("checkpoints/exp/checkpoint_latest.pt"), None, Path("checkpoints/exp"))


def test_validate_finetune_args_allows_finetune_alone_with_separate_dir(tmp_path: Path) -> None:
    source = tmp_path / "source" / "checkpoint_best.pt"
    validate_finetune_args(None, source, tmp_path / "finetuned")


def test_validate_finetune_args_rejects_both_resume_and_finetune() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_finetune_args(
            Path("checkpoints/exp/checkpoint_latest.pt"),
            Path("checkpoints/exp/checkpoint_best.pt"),
            Path("checkpoints/new"),
        )


def test_validate_finetune_args_rejects_same_checkpoint_dir_as_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "exp23_ema_extended90"
    source_dir.mkdir()
    source = source_dir / "checkpoint_best.pt"
    with pytest.raises(ValueError, match="separate checkpoint directory"):
        validate_finetune_args(None, source, source_dir)


# --- resolve_model_architecture (--global-bicubic-residual mapping) ---


def test_resolve_model_architecture_off_is_a_pure_passthrough() -> None:
    for name in ("residual_sr", "edsr_lite", "nafnet_sr", "swinir_lite", "residual_sr_bicubic"):
        assert resolve_model_architecture(name, False) == name


def test_resolve_model_architecture_on_maps_residual_sr_to_bicubic_variant() -> None:
    assert resolve_model_architecture("residual_sr", True) == "residual_sr_bicubic"


def test_resolve_model_architecture_on_is_a_no_op_for_already_bicubic() -> None:
    assert resolve_model_architecture("residual_sr_bicubic", True) == "residual_sr_bicubic"


@pytest.mark.parametrize("other_architecture", ["edsr_lite", "nafnet_sr", "swinir_lite"])
def test_resolve_model_architecture_rejects_incompatible_architectures(other_architecture: str) -> None:
    with pytest.raises(ValueError, match="global-bicubic-residual"):
        resolve_model_architecture(other_architecture, True)
