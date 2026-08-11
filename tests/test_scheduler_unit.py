"""Fast, dataset-free tests for the optional ReduceLROnPlateau scheduler.

No real dataset and no GPU are required.
"""

import pytest
import torch
from torch import nn

from src.models import ResidualSRNet
from train import (
    build_scheduler,
    build_scheduler_config,
    current_lr,
    load_checkpoint_for_resume,
    save_checkpoint,
    scheduler_step,
)


def _model_config() -> dict:
    return {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}


def _model_and_optimizer(lr: float = 1e-4) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def test_scheduler_none_preserves_fixed_lr() -> None:
    assert build_scheduler_config("none", factor=0.5, patience=3, min_lr=1e-6) is None
    _, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, None)
    assert scheduler is None
    # With no scheduler, nothing ever touches optimizer LR outside train.py's own code.
    assert current_lr(optimizer) == 1e-4


def test_build_scheduler_constructs_reduce_lr_on_plateau() -> None:
    config = build_scheduler_config("plateau", factor=0.5, patience=3, min_lr=1e-6)
    assert config == {"name": "plateau", "mode": "max", "factor": 0.5, "patience": 3, "min_lr": 1e-6}
    _, optimizer = _model_and_optimizer()
    scheduler = build_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 3
    assert scheduler.min_lrs == [1e-6]


def test_lr_decreases_after_patience_non_improving_epochs() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(
        optimizer, build_scheduler_config("plateau", factor=0.5, patience=2, min_lr=1e-6)
    )
    # First call establishes the baseline (10.0); next two stagnant calls are
    # "bad" epochs 1 and 2 (still within patience=2); the fourth call is the
    # 3rd consecutive bad epoch, exceeding patience, and triggers the drop.
    psnr_sequence = [10.0, 9.0, 9.0, 9.0]
    for value in psnr_sequence[:-1]:
        scheduler.step(value)
    assert current_lr(optimizer) == 1e-4
    scheduler.step(psnr_sequence[-1])
    assert current_lr(optimizer) == 5e-5


def test_lr_does_not_decrease_while_psnr_improves() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(
        optimizer, build_scheduler_config("plateau", factor=0.5, patience=2, min_lr=1e-6)
    )
    for value in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
        scheduler.step(value)
    assert current_lr(optimizer) == 1e-4


def test_lr_never_goes_below_min_lr() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(
        optimizer, build_scheduler_config("plateau", factor=0.1, patience=0, min_lr=1e-6)
    )
    for _ in range(15):
        scheduler.step(1.0)
    assert current_lr(optimizer) == pytest.approx(1e-6)


def test_checkpoint_includes_scheduler_state_when_enabled(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    scheduler_config = build_scheduler_config("plateau", factor=0.5, patience=3, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, scheduler_config)
    scheduler.step(20.0)
    scheduler.step(19.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_config"] == scheduler_config
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["scheduler_state_dict"]["num_bad_epochs"] == 1
    assert checkpoint["scheduler_state_dict"]["best"] == 20.0


def test_checkpoint_omits_scheduler_state_when_disabled(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={},
        scheduler=None,
        scheduler_config=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"] is None
    assert checkpoint["scheduler_config"] is None


def test_scheduler_state_restores_correctly_on_resume(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler_config = build_scheduler_config("plateau", factor=0.5, patience=1, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, scheduler_config)
    # Drive it through one real reduction so state_dict has non-default history.
    for value in [10.0, 9.0, 9.0]:
        scheduler.step(value)
    assert current_lr(optimizer) == 5e-5  # confirm a reduction actually happened

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={"seed": 42, "val_fraction": 0.2},
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    resumed_scheduler = build_scheduler(resumed_optimizer, scheduler_config)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=resumed_scheduler,
    )

    assert start_epoch == 4
    assert best_val_psnr == 10.0
    # Scheduler history (best/num_bad_epochs/cooldown_counter) must carry over,
    # not silently restart.
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_current_lr_after_resume_matches_saved_lr(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler_config = build_scheduler_config("plateau", factor=0.5, patience=1, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, scheduler_config)
    for value in [10.0, 9.0, 9.0]:  # triggers one reduction -> LR becomes 5e-5
        scheduler.step(value)
    saved_lr = current_lr(optimizer)
    assert saved_lr == 5e-5

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    resumed_scheduler = build_scheduler(resumed_optimizer, scheduler_config)
    load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=resumed_scheduler,
    )
    assert current_lr(resumed_optimizer) == saved_lr


def test_resume_handles_checkpoint_without_scheduler_state(tmp_path, capsys) -> None:
    """Simulates a real pre-scheduler Experiment 1 checkpoint (missing keys entirely)."""
    model, optimizer = _model_and_optimizer(lr=1e-4)
    legacy_checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 20,
            "best_val_psnr": 27.087,
            "model_config": _model_config(),
            "training_config": {"seed": 42, "val_fraction": 0.2},
            # Deliberately no scheduler_state_dict / scheduler_config keys.
        },
        legacy_checkpoint_path,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    scheduler_config = build_scheduler_config("plateau", factor=0.5, patience=3, min_lr=1e-6)
    resumed_scheduler = build_scheduler(resumed_optimizer, scheduler_config)
    fresh_state = resumed_scheduler.state_dict().copy()

    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        legacy_checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=resumed_scheduler,
    )

    assert start_epoch == 21
    assert best_val_psnr == 27.087
    # No crash, and the scheduler was left at its freshly constructed state
    # (nothing to restore) rather than raising a KeyError.
    assert resumed_scheduler.state_dict() == fresh_state
    assert "no stored scheduler state" in capsys.readouterr().out


def test_resume_without_requesting_scheduler_warns_state_is_dropped(tmp_path, capsys) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler_config = build_scheduler_config("plateau", factor=0.5, patience=1, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, scheduler_config)
    scheduler.step(10.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=None,  # --scheduler none requested despite checkpoint having state
    )
    assert "will NOT be resumed" in capsys.readouterr().out


# --- Experiment 14: CosineAnnealingLR ---


def test_build_cosine_scheduler_config_stores_exact_fields() -> None:
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    assert config == {"name": "cosine", "t_max": 40, "eta_min": 1e-6}


def test_cosine_scheduler_config_requires_t_max() -> None:
    with pytest.raises(ValueError, match="scheduler-t-max"):
        build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=None)


def test_build_scheduler_constructs_cosine_annealing_lr() -> None:
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    _, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 40
    assert scheduler.eta_min == 1e-6


def test_cosine_lr_decreases_monotonically_across_the_schedule() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    previous_lr = current_lr(optimizer)
    for _ in range(40):
        scheduler_step(scheduler, config, val_psnr=20.0)  # PSNR is irrelevant to cosine
        assert current_lr(optimizer) <= previous_lr
        previous_lr = current_lr(optimizer)


def test_cosine_lr_never_goes_below_eta_min() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(60):  # well past T_max -- cosine holds at eta_min afterward
        scheduler_step(scheduler, config, val_psnr=20.0)
        assert current_lr(optimizer) >= 1e-6 - 1e-12


def test_cosine_lr_reaches_exactly_eta_min_at_t_max() -> None:
    _, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(40):
        scheduler_step(scheduler, config, val_psnr=20.0)
    assert current_lr(optimizer) == pytest.approx(1e-6, abs=1e-12)


def test_cosine_stepping_ignores_validation_psnr_value() -> None:
    """Unlike plateau, the cosine trajectory must be identical regardless of
    what PSNR values are passed in -- it is a fixed function of epoch count."""
    _, optimizer_a = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler_a = build_scheduler(optimizer_a, config)
    for value in [1.0, 5.0, 3.0, 100.0, 0.1]:
        scheduler_step(scheduler_a, config, val_psnr=value)

    _, optimizer_b = _model_and_optimizer(lr=1e-4)
    scheduler_b = build_scheduler(optimizer_b, config)
    for value in [50.0, 50.0, 50.0, 50.0, 50.0]:
        scheduler_step(scheduler_b, config, val_psnr=value)

    assert current_lr(optimizer_a) == current_lr(optimizer_b)


def test_plateau_stepping_still_requires_and_uses_validation_psnr() -> None:
    """scheduler_step dispatches plateau to scheduler.step(psnr) -- reusing the
    exact pre-existing ReduceLROnPlateau behavior, unchanged by cosine's addition."""
    _, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("plateau", factor=0.5, patience=1, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, config)
    for value in [10.0, 9.0, 9.0]:  # triggers one reduction via scheduler_step
        scheduler_step(scheduler, config, val_psnr=value)
    assert current_lr(optimizer) == 5e-5


def test_checkpoint_stores_cosine_scheduler_state(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(5):
        scheduler_step(scheduler, config, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_config"] == {"name": "cosine", "t_max": 40, "eta_min": 1e-6}
    assert checkpoint["scheduler_state_dict"]["last_epoch"] == 5


def test_cosine_resume_restores_lr_and_scheduler_state(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(10):
        scheduler_step(scheduler, config, val_psnr=20.0)
    saved_lr = current_lr(optimizer)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=10,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    resumed_scheduler = build_scheduler(resumed_optimizer, config)
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=resumed_scheduler,
        scheduler_config=config,
    )
    assert start_epoch == 11
    assert best_val_psnr == 20.0
    assert current_lr(resumed_optimizer) == saved_lr
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_resumed_cosine_trajectory_matches_uninterrupted_trajectory() -> None:
    """Train 40 epochs straight vs. train 15, save/resume, then continue to 40
    -- the LR at epoch 40 must be identical either way (fixed T_max=40 horizon,
    not re-derived from the interrupted run's epoch count)."""
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)

    _, uninterrupted_optimizer = _model_and_optimizer(lr=1e-4)
    uninterrupted_scheduler = build_scheduler(uninterrupted_optimizer, config)
    for _ in range(40):
        scheduler_step(uninterrupted_scheduler, config, val_psnr=20.0)
    uninterrupted_final_lr = current_lr(uninterrupted_optimizer)

    model, optimizer = _model_and_optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(15):
        scheduler_step(scheduler, config, val_psnr=20.0)

    import tempfile
    from pathlib import Path as _Path

    checkpoint_path = _Path(tempfile.mkdtemp()) / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=15,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    resumed_scheduler = build_scheduler(resumed_optimizer, config)
    start_epoch, _, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
        scheduler=resumed_scheduler,
        scheduler_config=config,
    )
    for _ in range(start_epoch, 41):
        scheduler_step(resumed_scheduler, config, val_psnr=20.0)

    assert current_lr(resumed_optimizer) == pytest.approx(uninterrupted_final_lr)


def test_resume_rejects_cosine_request_against_plateau_checkpoint(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    plateau_config = build_scheduler_config("plateau", factor=0.5, patience=3, min_lr=1e-6)
    scheduler = build_scheduler(optimizer, plateau_config)
    scheduler_step(scheduler, plateau_config, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=plateau_config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    cosine_config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    resumed_scheduler = build_scheduler(resumed_optimizer, cosine_config)
    with pytest.raises(ValueError, match="scheduler_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            _model_config(),
            torch.device("cpu"),
            scheduler=resumed_scheduler,
            scheduler_config=cosine_config,
        )


def test_resume_rejects_plateau_request_against_cosine_checkpoint(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    cosine_config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, cosine_config)
    scheduler_step(scheduler, cosine_config, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=cosine_config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    plateau_config = build_scheduler_config("plateau", factor=0.5, patience=3, min_lr=1e-6)
    resumed_scheduler = build_scheduler(resumed_optimizer, plateau_config)
    with pytest.raises(ValueError, match="scheduler_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            _model_config(),
            torch.device("cpu"),
            scheduler=resumed_scheduler,
            scheduler_config=plateau_config,
        )


def test_resume_rejects_different_t_max_against_cosine_checkpoint(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    config_40 = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config_40)
    scheduler_step(scheduler, config_40, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config_40,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    config_20 = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=20)
    resumed_scheduler = build_scheduler(resumed_optimizer, config_20)
    with pytest.raises(ValueError, match="scheduler_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            _model_config(),
            torch.device("cpu"),
            scheduler=resumed_scheduler,
            scheduler_config=config_20,
        )


def test_resume_rejects_different_eta_min_against_cosine_checkpoint(tmp_path) -> None:
    model, optimizer = _model_and_optimizer(lr=1e-4)
    config_eta6 = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config_eta6)
    scheduler_step(scheduler, config_eta6, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config_eta6,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    config_eta5 = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-5, t_max=40)
    resumed_scheduler = build_scheduler(resumed_optimizer, config_eta5)
    with pytest.raises(ValueError, match="scheduler_config"):
        load_checkpoint_for_resume(
            checkpoint_path,
            resumed_model,
            resumed_optimizer,
            _model_config(),
            torch.device("cpu"),
            scheduler=resumed_scheduler,
            scheduler_config=config_eta5,
        )


def test_scheduler_config_none_skips_mismatch_check_for_legacy_resume(tmp_path) -> None:
    """A caller that does not pass scheduler_config (e.g. --scheduler none, or
    a test that doesn't care) must not be affected by the new strict check --
    this is what keeps historical no-scheduler checkpoints resumable."""
    model, optimizer = _model_and_optimizer(lr=1e-4)
    config = build_scheduler_config("cosine", factor=0.5, patience=3, min_lr=1e-6, t_max=40)
    scheduler = build_scheduler(optimizer, config)
    scheduler_step(scheduler, config, val_psnr=20.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=20.0,
        model_config=_model_config(),
        training_config={},
        scheduler=scheduler,
        scheduler_config=config,
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(lr=1e-4)
    # No scheduler_config passed -- must not raise, even though the checkpoint
    # actually has a (different, unrequested) cosine config stored.
    start_epoch, best_val_psnr, _ = load_checkpoint_for_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        _model_config(),
        torch.device("cpu"),
    )
    assert start_epoch == 2
    assert best_val_psnr == 20.0
