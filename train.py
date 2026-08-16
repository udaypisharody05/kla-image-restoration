"""Train the small residual-CNN restoration baseline.

Reuses the repository's existing dataset discovery, canonical train/validation
split, lazy PyTorch datasets, and aligned training preprocessing. Trains a
lightweight residual CNN (see ``src/models/residual_sr.py``) with L1 loss and
Adam, validates on the full deterministic 640-image validation split every
epoch, and checkpoints the latest and best (by validation PSNR) models.

An optional learning-rate scheduler can be enabled with ``--scheduler``:
``plateau`` (``ReduceLROnPlateau``, monitoring validation PSNR, ``mode="max"``)
or ``cosine`` (``CosineAnnealingLR`` over a fixed, explicitly-configured
``--scheduler-t-max`` horizon -- never implicitly derived from ``--epochs``,
so a smoke test run with a small ``--epochs`` does not distort the intended
schedule). The default ``--scheduler none`` reproduces Experiment 1's
fixed-LR behavior exactly.

An optional exponential moving average of the model weights (see
``src/ema.py``) can be enabled with ``--ema`` (``--ema-decay``, default
``0.999``). When enabled, validation (and therefore scheduler decisions and
best-checkpoint selection) uses the EMA shadow weights instead of the live
training weights, while the live weights keep training normally underneath.
Default is off, reproducing every prior experiment's behavior exactly.

``--synthetic-noise-prob`` (default ``0.0``, off) mixes in extra *training*
inputs synthesized from GT via the Experiment 22 signal-dependent degradation
model (see ``src/synthetic_noise.py``). Validation always stays 100% real.

``--noise-conditioning`` (default off) instead trains exclusively on real
NoisyLR/GT pairs, but gives the model an explicit second input channel: a
per-pixel estimate of the same Experiment 22 noise model's sigma, computed
from the real LR itself (see ``src/noise_conditioning.py``). Mutually
independent of ``--synthetic-noise-prob`` -- Experiment 25 uses this instead
of, not alongside, Experiment 24's augmentation.

``--loss mixed`` (``--mixed-loss-alpha``, default ``0.5``) trains on
``alpha*L1 + (1-alpha)*MSE`` -- see ``src/losses.py::MixedL1MSELoss``.

``--finetune-from <checkpoint>`` loads a checkpoint's model weights only
(preferring its EMA weights when present) and starts a brand-new,
independent experiment: fresh optimizer/``--lr``/scheduler, epoch counter
reset to 1, its own best-score tracking, and its own ``--checkpoint-dir``
(which must differ from the source checkpoint's directory, so the source is
only ever read, never overwritten). This is distinct from ``--resume``
(which restores full training state -- optimizer, scheduler, epoch,
best-score -- to continue the *same* interrupted run); the two are mutually
exclusive.

``--global-bicubic-residual`` (default off) switches ``--model residual_sr``
to Experiment 17's ``ResidualSRBicubic`` (``prediction = bicubic_upsample(LR)
+ learned_residual(LR)``) -- see ``src/models/residual_sr_bicubic.py``. Same
learned-branch topology and parameter count as ``ResidualSRNet``; only the
final forward step differs.

``--channel-attention`` (``--attention-reduction``, default ``8``) and
``--multiscale-block`` (both default off, residual_sr only) each optionally
swap in a variant residual block -- see
``src/models/attention.py::ChannelAttention`` and
``src/models/residual_sr.py::MultiScaleBlock``. Independently selectable,
never auto-combined, and structured so a plain ``--model residual_sr`` run
with neither flag reconstructs the exact historical architecture and
checkpoint format.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from inspect_dataset import configured_data_dir
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.ema import ExponentialMovingAverage
from src.losses import build_loss, build_loss_config, loss_label
from src.metrics import psnr, ssim
from src.models import build_model, build_model_config
from src.noise_conditioning import build_noise_conditioning_config, wrap_for_conditioning
from src.splits import split_pairs
from src.synthetic_noise import (
    DISTRIBUTIONS,
    STUDENT_T_DEGREES_OF_FREEDOM,
    SyntheticNoiseAugmentation,
    build_synthetic_noise,
    build_synthetic_noise_config,
)
from src.thermal import GpuTemperatureGuard
from src.transforms import create_training_transform


# Established bicubic validation baseline (see results/bicubic_baseline.json).
BICUBIC_PSNR_DB = 23.1413
BICUBIC_SSIM = 0.550604


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(
    data_dir: Path,
    val_fraction: float,
    seed: int,
    crop_size: int,
    scale: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    synthetic_noise: SyntheticNoiseAugmentation | None = None,
    hard_patch_sampling: bool = False,
    hard_patch_prob: float = 0.5,
) -> tuple[PairedRestorationDataset, PairedRestorationDataset, int]:
    """Rebuild the canonical split and wrap it with the project's dataset classes."""
    layout = discover_layout(data_dir)
    pairs = discover_pairs(layout).pairs
    train_pairs, validation_pairs = split_pairs(pairs, val_fraction=val_fraction, seed=seed)
    if max_train_samples is not None:
        train_pairs = train_pairs[:max_train_samples]
    if max_val_samples is not None:
        validation_pairs = validation_pairs[:max_val_samples]

    training_transform = create_training_transform(
        crop_size=crop_size,
        scale=scale,
        augment=True,
        hard_patch_sampling=hard_patch_sampling,
        hard_patch_prob=hard_patch_prob,
    )
    train_dataset = PairedRestorationDataset(
        train_pairs, scale=scale, transform=training_transform, synthetic_noise=synthetic_noise
    )
    # Validation stays deterministic, full-resolution, and 100% REAL -- no
    # synthetic_noise is ever passed here, so no reported metric, scheduler
    # decision, or checkpoint selection can ever see a synthesized input.
    validation_dataset = PairedRestorationDataset(validation_pairs, scale=scale)
    return train_dataset, validation_dataset, len(pairs)


Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau | torch.optim.lr_scheduler.CosineAnnealingLR


def build_scheduler_config(
    scheduler_name: str, factor: float, patience: int, min_lr: float, t_max: int | None = None
) -> dict | None:
    """Turn CLI scheduler options into a plain, checkpoint-serializable dict.

    *t_max* (cosine only) is the intended full-experiment epoch horizon --
    callers must pass it explicitly (e.g. ``--scheduler-t-max 40``) rather
    than deriving it from ``--epochs``, so a short smoke-test run does not
    silently produce a different (compressed) schedule than the real run.
    """
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return {
            "name": "plateau",
            "mode": "max",
            "factor": factor,
            "patience": patience,
            "min_lr": min_lr,
        }
    if scheduler_name == "cosine":
        if t_max is None or t_max < 1:
            raise ValueError("--scheduler-t-max must be a positive integer for --scheduler cosine")
        return {
            "name": "cosine",
            "t_max": t_max,
            "eta_min": min_lr,
        }
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict | None) -> Scheduler | None:
    """Construct the scheduler described by *scheduler_config*, or None."""
    if scheduler_config is None:
        return None
    if scheduler_config["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
        )
    if scheduler_config["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["t_max"],
            eta_min=scheduler_config["eta_min"],
        )
    raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")


def scheduler_step(scheduler: Scheduler, scheduler_config: dict, val_psnr: float) -> None:
    """Advance *scheduler* by exactly one epoch, using the right call signature
    for its type: ``ReduceLROnPlateau.step(metric)`` needs this epoch's
    validation PSNR to decide whether to reduce; ``CosineAnnealingLR.step()``
    takes no argument -- its schedule is a fixed function of epoch count,
    independent of validation performance.
    """
    if scheduler_config["name"] == "plateau":
        scheduler.step(val_psnr)
    else:
        scheduler.step()


def build_ema_config(enabled: bool, decay: float) -> dict | None:
    """Turn CLI EMA options into a plain, checkpoint-serializable dict.

    ``enabled=False`` (the default) returns ``None`` -- mirroring
    ``build_scheduler_config``'s "off" convention -- so every historical
    command that never mentions ``--ema`` produces an identical ``None``
    both here and in the checkpoints it saves.
    """
    if not enabled:
        return None
    return {"enabled": True, "decay": decay}


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def warn_on_resume_config_mismatch(
    previous_config: dict,
    seed: int,
    val_fraction: float,
    crop_size: int,
    hard_patch_sampling: bool = False,
    hard_patch_prob: float = 0.5,
) -> None:
    """Print explicit (never silent) warnings when resume args differ from what
    the checkpoint's stored ``training_config`` actually used.

    Unlike ``model_config``/``loss_config`` (hard-rejected in
    ``load_checkpoint_for_resume`` since they change the architecture or the
    objective being optimized), these fields change the data pipeline/split
    without breaking anything structurally, so a warning -- not a rejection --
    is the established, less disruptive policy for this class of setting.
    """
    if previous_config.get("seed") != seed or previous_config.get("val_fraction") != val_fraction:
        print(
            "WARNING: --seed/--val-fraction differ from the checkpoint's stored "
            f"training_config ({previous_config}); this changes which pairs are "
            "in the train/validation split and makes best_val_psnr incomparable."
        )
    if previous_config.get("crop_size") != crop_size:
        print(
            f"WARNING: --crop-size ({crop_size}) differs from the checkpoint's "
            f"stored training crop_size ({previous_config.get('crop_size')}); training "
            "will continue using the new crop size immediately, which is a different "
            "training regime than produced this checkpoint's saved metrics."
        )
    if previous_config.get("hard_patch_sampling") is not None and (
        previous_config.get("hard_patch_sampling") != hard_patch_sampling
        or (hard_patch_sampling and previous_config.get("hard_patch_prob") != hard_patch_prob)
    ):
        print(
            f"WARNING: --hard-patch-sampling/--hard-patch-prob ({hard_patch_sampling}/"
            f"{hard_patch_prob}) differ from the checkpoint's stored training_config "
            f"(hard_patch_sampling={previous_config.get('hard_patch_sampling')}, "
            f"hard_patch_prob={previous_config.get('hard_patch_prob')}); training will "
            "continue using the new sampling policy immediately."
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_psnr: float,
    model_config: dict,
    training_config: dict,
    scheduler: Scheduler | None = None,
    scheduler_config: dict | None = None,
    loss_config: dict | None = None,
    ema: ExponentialMovingAverage | None = None,
    ema_config: dict | None = None,
    synthetic_noise_config: dict | None = None,
    noise_conditioning_config: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Always the live/raw training weights, regardless of EMA -- this
            # key's meaning never changes, so every historical loader (and
            # resume) keeps working unmodified. The EMA shadow (when enabled)
            # is stored separately below; it is what "checkpoint_best.pt"
            # actually evaluated to reach its recorded PSNR (see
            # evaluate_checkpoint.load_model's prefer_ema handling).
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_psnr": best_val_psnr,
            "model_config": model_config,
            "training_config": training_config,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scheduler_config": scheduler_config,
            # Every run has some reconstruction loss; a caller that doesn't pass
            # one explicitly gets the same default all historical checkpoints
            # implicitly used.
            "loss_config": loss_config if loss_config is not None else {"name": "l1"},
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "ema_config": ema_config,
            # Fully reconstructs the augmentation (probability, distribution,
            # variance coefficients, floor, downsampling, seed) -- see
            # SyntheticNoiseAugmentation.config(). None means "not used".
            "synthetic_noise_config": synthetic_noise_config,
            # Fully reconstructs the [lr, sigma] input-preparation step -- see
            # NoiseConditionedModel / build_noise_conditioning_config. None
            # means the model takes plain single-channel LR, as historically.
            "noise_conditioning_config": noise_conditioning_config,
        },
        path,
    )


_UNSET = object()  # distinguishes "caller didn't pass ema_config" from "caller passed None"


def load_checkpoint_for_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    device: torch.device,
    scheduler: Scheduler | None = None,
    loss_config: dict | None = None,
    scheduler_config: dict | None = None,
    ema: ExponentialMovingAverage | None = None,
    ema_config: dict | None = _UNSET,
    synthetic_noise_config: dict | None = _UNSET,
    noise_conditioning_config: dict | None = _UNSET,
) -> tuple[int, float, dict]:
    """Restore model/optimizer/(optional) scheduler/(optional) EMA state from *path*.

    Returns (next_epoch, best_val_psnr, training_config) from the checkpoint.
    Older checkpoints saved before scheduler/EMA support existed simply lack
    the corresponding ``*_state_dict``/``*_config`` keys; ``.get()`` treats
    that the same as an explicit ``None`` rather than raising.

    *loss_config*, when given, must match the checkpoint's stored loss config
    exactly (mirroring the strict ``model_config`` check below) -- a resume
    must never silently switch the reconstruction loss a run is being trained
    against. Pass ``None`` to skip this check (used by tests that don't care).

    *scheduler_config*, when given, must match the checkpoint's stored
    scheduler config exactly -- same rationale: switching between
    ``plateau``/``cosine``, or resuming a cosine schedule with a different
    ``t_max``/``eta_min``, silently changes the learning-rate trajectory a
    run is being trained against. Pass ``None`` to skip this check (used by
    tests that don't care, and by callers resuming with ``--scheduler none``
    where there is nothing to compare).

    *ema_config*, unlike the two checks above, defaults to the private
    ``_UNSET`` sentinel rather than ``None`` -- because ``None`` is EMA's own
    legitimate "disabled" value (``build_ema_config(False, ...)`` returns
    ``None``), a plain ``None`` default could not distinguish "caller doesn't
    want this checked" from "caller explicitly disabled EMA and wants that
    enforced". ``train.py``'s real resume path always passes the actual
    computed ``ema_config`` (never omits it), so in real usage this check is
    always active and strict: an EMA checkpoint resumed with ``--ema`` off,
    resumed with a different ``--ema-decay``, or a non-EMA checkpoint resumed
    with ``--ema`` on, are all rejected. Only callers that omit the parameter
    entirely (tests that don't care) skip the check.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_config"] != model_config:
        raise ValueError(
            f"Checkpoint model_config {checkpoint['model_config']} does not match "
            f"the requested model_config {model_config}; pass matching "
            "--model/--num-features/--num-blocks/--scale/--residual-scale to resume."
        )
    if loss_config is not None:
        # Checkpoints saved before loss selection existed have no stored
        # loss_config; every historical run used plain L1, so that is the
        # only sensible default to compare against.
        checkpoint_loss_config = checkpoint.get("loss_config", {"name": "l1"})
        if checkpoint_loss_config != loss_config:
            raise ValueError(
                f"Checkpoint loss_config {checkpoint_loss_config} does not match "
                f"the requested loss_config {loss_config}; pass matching "
                "--loss/--charbonnier-eps/--ssim-weight to resume."
            )
    if scheduler_config is not None:
        # Checkpoints saved before scheduler support existed have no stored
        # scheduler_config; every historical run used no scheduler at all, so
        # None is the only sensible default to compare against.
        checkpoint_scheduler_config = checkpoint.get("scheduler_config")
        if checkpoint_scheduler_config != scheduler_config:
            raise ValueError(
                f"Checkpoint scheduler_config {checkpoint_scheduler_config} does not match "
                f"the requested scheduler_config {scheduler_config}; pass matching "
                "--scheduler/--scheduler-factor/--scheduler-patience/--scheduler-t-max/--min-lr to resume."
            )
    if ema_config is not _UNSET:
        # Checkpoints saved before EMA support existed have no stored
        # ema_config; every historical run had no EMA at all, so None is the
        # only sensible default to compare against -- same as scheduler_config.
        checkpoint_ema_config = checkpoint.get("ema_config")
        if checkpoint_ema_config != ema_config:
            raise ValueError(
                f"Checkpoint ema_config {checkpoint_ema_config} does not match "
                f"the requested ema_config {ema_config}; pass matching "
                "--ema/--ema-decay to resume."
            )
    if synthetic_noise_config is not _UNSET:
        # Same _UNSET-sentinel rationale as ema_config: None is the augmentation's
        # own legitimate "disabled" value, so it cannot double as "skip the check".
        # Changing the augmentation mid-run silently changes what the model is
        # being trained on, so every difference is rejected.
        checkpoint_synthetic_config = checkpoint.get("synthetic_noise_config")
        if checkpoint_synthetic_config != synthetic_noise_config:
            raise ValueError(
                f"Checkpoint synthetic_noise_config {checkpoint_synthetic_config} does not "
                f"match the requested synthetic_noise_config {synthetic_noise_config}; pass "
                "matching --synthetic-noise-prob/--synthetic-noise-distribution/"
                "--synthetic-noise-nu/--synthetic-noise-variance-floor to resume."
            )
    if noise_conditioning_config is not _UNSET:
        # Same _UNSET-sentinel rationale as ema_config/synthetic_noise_config:
        # None is conditioning's own legitimate "disabled" value. Changing
        # whether/how the model's input is conditioned mid-run would silently
        # feed it a differently-shaped or differently-computed input than what
        # produced its saved weights, so every difference is rejected.
        checkpoint_noise_conditioning_config = checkpoint.get("noise_conditioning_config")
        if checkpoint_noise_conditioning_config != noise_conditioning_config:
            raise ValueError(
                f"Checkpoint noise_conditioning_config {checkpoint_noise_conditioning_config} "
                f"does not match the requested noise_conditioning_config "
                f"{noise_conditioning_config}; pass matching "
                "--noise-conditioning/--noise-conditioning-variance-floor to resume."
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None:
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            print("Restored scheduler state from checkpoint.")
        else:
            print(
                "WARNING: --scheduler was requested but this checkpoint has no "
                "stored scheduler state (an older or no-scheduler checkpoint); "
                "the scheduler is starting fresh from the resumed learning rate."
            )
    elif scheduler_state is not None:
        print(
            "WARNING: this checkpoint contains scheduler state but --scheduler "
            "none was requested; the learning-rate schedule will NOT be resumed."
        )

    ema_state = checkpoint.get("ema_state_dict")
    if ema is not None:
        if ema_state is not None:
            ema.load_state_dict(ema_state)
            print("Restored EMA state from checkpoint.")
        else:
            print(
                "WARNING: --ema was requested but this checkpoint has no stored "
                "EMA state (an older or non-EMA checkpoint); EMA is starting "
                "fresh from the resumed live weights."
            )
    elif ema_state is not None:
        print(
            "WARNING: this checkpoint contains EMA state but --ema was not "
            "requested; the EMA trajectory will NOT be resumed."
        )

    return checkpoint["epoch"] + 1, checkpoint["best_val_psnr"], checkpoint["training_config"]


def load_weights_for_finetune(
    path: Path, model: nn.Module, device: torch.device, prefer_ema: bool = True
) -> None:
    """Load only *model*'s weights from *path* -- no optimizer, scheduler,
    epoch, or best-score state.

    Used to start an independent fine-tuning experiment (fresh
    optimizer/LR/scheduler, epoch counter reset, its own checkpoint
    directory) from an existing checkpoint's weights, unlike
    ``load_checkpoint_for_resume`` (which restores full training state to
    continue the *same* interrupted run). *path* is only ever read here,
    never written to.

    Prefers the checkpoint's EMA shadow weights when present (mirrors
    ``evaluate_checkpoint.load_model``'s default): for an EMA-trained
    checkpoint, the EMA weights are what actually produced its recorded
    validation PSNR, so that is the more sensible fine-tuning starting point
    than the noisier live weights.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ema_state_dict = checkpoint.get("ema_state_dict")
    if prefer_ema and ema_state_dict is not None:
        model.load_state_dict(ema_state_dict)
        print(f"Fine-tuning from {path}: loaded EMA weights (optimizer/scheduler/epoch are fresh).")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Fine-tuning from {path}: loaded live weights (optimizer/scheduler/epoch are fresh).")


def validate_finetune_args(
    resume: Path | None, finetune_from: Path | None, checkpoint_dir: Path
) -> None:
    """Fail fast, before any dataset/model work, on invalid ``--resume``/
    ``--finetune-from`` combinations.

    The two express mutually exclusive intents -- continue an interrupted
    run with its exact saved state, vs. start a brand-new experiment from
    existing weights -- so requesting both is always a mistake. A
    fine-tuning run must also never share its ``--checkpoint-dir`` with the
    source checkpoint's own directory, so it can never overwrite the
    checkpoint it started from.
    """
    if resume is not None and finetune_from is not None:
        raise ValueError("--resume and --finetune-from are mutually exclusive: pick one.")
    if finetune_from is not None:
        source_dir = finetune_from.resolve().parent
        if checkpoint_dir.resolve() == source_dir:
            raise ValueError(
                f"--checkpoint-dir ({checkpoint_dir}) must not be the same directory as "
                f"the --finetune-from checkpoint's directory ({source_dir}); a fine-tuning "
                "run must save into a separate checkpoint directory so it can never "
                "overwrite the source checkpoint."
            )


def resolve_model_architecture(model_name: str, global_bicubic_residual: bool) -> str:
    """Map ``--model``/``--global-bicubic-residual`` to the actual architecture
    name passed to ``build_model_config``.

    Off (the default) returns *model_name* completely unchanged, so every
    historical command that never mentions the flag is unaffected -- this is
    never silently turned on. On, only ``residual_sr`` (which becomes
    ``residual_sr_bicubic``, reusing Experiment 17's already-implemented,
    already-tested ``ResidualSRBicubic``) and ``residual_sr_bicubic`` itself
    (already the target -- a harmless no-op) are accepted; every other
    architecture raises rather than silently ignoring the flag.
    """
    if not global_bicubic_residual:
        return model_name
    if model_name in ("residual_sr", "residual_sr_bicubic"):
        return "residual_sr_bicubic"
    raise ValueError(
        "--global-bicubic-residual is only supported for --model residual_sr "
        f"(reuses Experiment 17's ResidualSRBicubic implementation), not --model {model_name}."
    )


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    thermal_guard: GpuTemperatureGuard | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # EMA updates once per optimizer step, immediately after it -- so the
        # very first update happens after this epoch's first batch (or, on a
        # resumed run, after the first batch following the resumed epoch;
        # the shadow's prior trajectory was already restored before this
        # loop started). The live model above is never modified by this.
        if ema is not None:
            ema.update(model)
        batch_size = inputs.shape[0]
        # loss.item() below blocks until this batch's GPU work has completed
        # (a CUDA tensor .item() call synchronizes implicitly), so the
        # thermal check that follows never needs its own explicit
        # torch.cuda.synchronize() -- see src/thermal.py for details.
        total_loss += loss.item() * batch_size
        total_count += batch_size
        # Batch is fully complete (optimizer.step() already applied); safe to
        # pause here without redoing, skipping, or partially processing work.
        if thermal_guard is not None:
            thermal_guard.on_batch_complete()
    return total_loss / total_count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    thermal_guard: GpuTemperatureGuard | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs)
        # The reconstruction loss (whichever one was selected) is left
        # unclamped -- it reports the raw model error. PSNR/SSIM clip the
        # prediction to [0,1] by default (see src/metrics.py), matching the
        # bicubic baseline's convention so the printed numbers stay comparable.
        loss = loss_fn(outputs, targets)
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += psnr(outputs, targets) * batch_size
        total_ssim += ssim(outputs, targets) * batch_size
        total_count += batch_size
        # Reuses the same guard/abstraction as training; never touches model
        # state or metrics, only pauses between already-completed batches.
        if thermal_guard is not None:
            thermal_guard.on_batch_complete()
    return {
        "loss": total_loss / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument(
        "--hard-patch-sampling",
        action="store_true",
        help=(
            "Bias training-crop origins toward high-gradient-energy ('hard'/informative) "
            "regions instead of always sampling uniformly at random. Default: off (plain "
            "PairedRandomCrop, byte-for-byte historical behavior)."
        ),
    )
    parser.add_argument(
        "--hard-patch-prob",
        type=float,
        default=0.5,
        help=(
            "Probability of using a gradient-energy-weighted crop origin per training sample "
            "(ignored unless --hard-patch-sampling); the rest fall back to a plain uniform-"
            "random crop, so training is never restricted to only high-gradient regions."
        ),
    )
    parser.add_argument("--num-features", type=int, default=32, help="Residual block channel width")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument(
        "--model",
        type=str,
        choices=["residual_sr", "edsr_lite", "nafnet_sr", "swinir_lite", "residual_sr_bicubic"],
        default="residual_sr",
        help="Architecture. 'residual_sr' (default) reproduces Experiments 1-8 exactly.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="EDSR-style fixed residual-block scale (ignored unless --model edsr_lite)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=48,
        help="SwinIR-lite token embedding width (ignored unless --model swinir_lite)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="SwinIR-lite number of Swin Transformer blocks (ignored unless --model swinir_lite)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
        help="SwinIR-lite attention heads; must divide --embed-dim (ignored unless --model swinir_lite)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="SwinIR-lite attention window size (ignored unless --model swinir_lite)",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=2.0,
        help="SwinIR-lite MLP hidden-dim expansion ratio (ignored unless --model swinir_lite)",
    )
    parser.add_argument(
        "--global-bicubic-residual",
        action="store_true",
        help=(
            "prediction = bicubic_upsample(LR) + learned_residual(LR) instead of the learned "
            "branch alone (switches --model residual_sr to Experiment 17's ResidualSRBicubic; "
            "same topology/parameter count). Default: off. Only valid with --model residual_sr "
            "or --model residual_sr_bicubic."
        ),
    )
    parser.add_argument(
        "--channel-attention",
        action="store_true",
        help=(
            "Insert a lightweight squeeze-and-excitation channel-attention gate into each "
            "ResidualSR residual block (ignored unless --model residual_sr). Default: off."
        ),
    )
    parser.add_argument(
        "--attention-reduction",
        type=int,
        default=8,
        help="Channel-attention squeeze reduction ratio (ignored unless --channel-attention)",
    )
    parser.add_argument(
        "--multiscale-block",
        action="store_true",
        help=(
            "Replace ResidualSR's residual blocks with a local-3x3 + dilated-3x3(dilation=2) "
            "multi-scale block for a larger receptive field (ignored unless --model "
            "residual_sr). Independent of --channel-attention -- never auto-combined. "
            "Mutually exclusive with --rdb-block (both replace the residual block type). "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--rdb-block",
        action="store_true",
        help=(
            "Replace ResidualSR's residual blocks with a lightweight Residual Dense Block "
            "(dense feature reuse within each block, inspired by RDN but not a full RDN; "
            "ignored unless --model residual_sr). Mutually exclusive with --multiscale-block. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--rdb-growth-rate",
        type=int,
        default=16,
        help="Channels added per dense layer inside each RDB (ignored unless --rdb-block)",
    )
    parser.add_argument(
        "--rdb-num-layers",
        type=int,
        default=3,
        help="Number of densely-connected conv layers per RDB (ignored unless --rdb-block)",
    )
    parser.add_argument(
        "--denoise-stem",
        action="store_true",
        help=(
            "Insert an optional lightweight pre-trunk denoising stem (Conv3x3 -> gated "
            "restoration blocks -> Conv3x3 residual correction) before ResidualSR's existing "
            "feature trunk (ignored unless --model residual_sr); see "
            "src/models/denoise_stem.py. Default: off."
        ),
    )
    parser.add_argument(
        "--denoise-stem-features",
        type=int,
        default=32,
        help="Channel width inside the denoise stem (ignored unless --denoise-stem)",
    )
    parser.add_argument(
        "--denoise-stem-blocks",
        type=int,
        default=2,
        help="Number of gated restoration blocks inside the denoise stem (2-4 recommended; "
        "ignored unless --denoise-stem)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto-detects")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Optional smoke/subset limit"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None, help="Optional smoke/subset limit"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Checkpoint path to resume training from"
    )
    parser.add_argument(
        "--finetune-from",
        type=Path,
        default=None,
        help=(
            "Load model weights only (preferring EMA weights if present) from this checkpoint "
            "and start a brand-new experiment: fresh optimizer/--lr/scheduler, epoch counter "
            "reset, its own best-score tracking. Unlike --resume (restores full training state "
            "to continue the same run), the source checkpoint is only read, never written to -- "
            "--checkpoint-dir must be a different directory. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "plateau", "cosine"],
        default="none",
        help="LR scheduler. 'none' (default) reproduces Experiment 1's fixed LR.",
    )
    parser.add_argument(
        "--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau LR multiplier"
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="Epochs with no Val PSNR improvement before reducing LR",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Lower bound the scheduler will not cross (plateau) / eta_min (cosine)",
    )
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=40,
        help=(
            "CosineAnnealingLR T_max: the intended full-experiment epoch horizon "
            "(ignored unless --scheduler cosine). Set explicitly -- never derived from "
            "--epochs, so a short smoke-test run does not compress the real schedule."
        ),
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help=(
            "Maintain an exponential moving average of the model weights; validation "
            "(and therefore scheduler decisions and best-checkpoint selection) uses the "
            "EMA weights instead of the live training weights. Default: off."
        ),
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay rate (ignored unless --ema)"
    )
    parser.add_argument(
        "--synthetic-noise-prob",
        type=float,
        default=0.0,
        help=(
            "Per-sample probability of replacing the real NoisyLR with one synthesized from "
            "GT using the Experiment 22 signal-dependent degradation model. 0.0 (default) "
            "disables the augmentation and reproduces historical behavior exactly. "
            "Validation is always 100%% real regardless of this setting."
        ),
    )
    parser.add_argument(
        "--synthetic-noise-distribution",
        type=str,
        choices=list(DISTRIBUTIONS),
        default="gaussian",
        help="Noise shape (ignored unless --synthetic-noise-prob > 0)",
    )
    parser.add_argument(
        "--synthetic-noise-nu",
        type=float,
        default=STUDENT_T_DEGREES_OF_FREEDOM,
        help="Student-t degrees of freedom (ignored unless distribution is student_t)",
    )
    parser.add_argument(
        "--synthetic-noise-variance-floor",
        type=float,
        default=0.0,
        help=(
            "Lower bound on modelled noise variance. 0.0 (default) is the Experiment 22 "
            "model verbatim; a small positive floor compensates for it under-predicting "
            "sigma in the darkest intensity bin."
        ),
    )
    parser.add_argument(
        "--noise-conditioning",
        action="store_true",
        help=(
            "Give the model a second input channel: a per-pixel Experiment 22 "
            "signal-dependent sigma estimate computed from the real NoisyLR. Sets "
            "model in_channels=2. Trains exclusively on real data -- independent of "
            "--synthetic-noise-prob. Default: off (single-channel LR, as historically)."
        ),
    )
    parser.add_argument(
        "--noise-conditioning-variance-floor",
        type=float,
        default=0.0,
        help=(
            "Lower bound on the conditioning sigma's variance (ignored unless "
            "--noise-conditioning). 0.0 (default) is the Experiment 22 model verbatim."
        ),
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["l1", "mse", "charbonnier", "l1_ssim", "mixed", "weighted_l1"],
        default="l1",
        help="Reconstruction loss. 'l1' (default) reproduces Experiments 1-3 exactly.",
    )
    parser.add_argument(
        "--charbonnier-eps",
        type=float,
        default=1e-3,
        help="Charbonnier loss epsilon (ignored unless --loss charbonnier)",
    )
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=0.1,
        help="Weight on (1 - differentiable SSIM) in L1 + weight*(1-SSIM) (ignored unless --loss l1_ssim)",
    )
    parser.add_argument(
        "--mixed-loss-alpha",
        type=float,
        default=0.5,
        help="Weight on L1 in alpha*L1 + (1-alpha)*MSE (ignored unless --loss mixed)",
    )
    parser.add_argument(
        "--weighted-l1-eps",
        type=float,
        default=1e-2,
        help=(
            "Stabilizer added to the fitted noise std before inverting for per-pixel weights "
            "(ignored unless --loss weighted_l1); see src/losses.py::VarianceWeightedL1Loss."
        ),
    )
    parser.add_argument(
        "--weighted-l1-variance-floor",
        type=float,
        default=0.0,
        help=(
            "Floor on the fitted Experiment 22 variance model (ignored unless --loss "
            "weighted_l1); 0.0 (default) reproduces the model exactly as fitted."
        ),
    )
    parser.add_argument(
        "--gpu-temp-limit",
        type=float,
        default=0.0,
        help="GPU temp (C) at which to enter a thermal pause between batches; 0 disables the guard (default)",
    )
    parser.add_argument(
        "--gpu-temp-resume",
        type=float,
        default=78.0,
        help="GPU temp (C) at/below which to resume after a thermal pause (ignored when the guard is disabled)",
    )
    parser.add_argument(
        "--gpu-temp-check-interval",
        type=int,
        default=5,
        help="Completed batches between GPU temperature checks (ignored when the guard is disabled)",
    )
    parser.add_argument(
        "--gpu-temp-poll-seconds",
        type=float,
        default=3.0,
        help="Seconds to sleep between temperature checks while paused (ignored when the guard is disabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Fail fast on invalid --resume/--finetune-from combinations, before any
    # dataset/model work -- same "validate before doing anything expensive"
    # policy as the thermal guard below.
    validate_finetune_args(args.resume, args.finetune_from, args.checkpoint_dir)

    # Constructed (and validated) early, before any dataset work, so invalid
    # thermal args fail fast. Disabled by default (--gpu-temp-limit 0); when
    # disabled this never calls nvidia-smi or sleeps, so CPU/CUDA training
    # behaves exactly as before this feature existed.
    thermal_guard = GpuTemperatureGuard(
        limit=args.gpu_temp_limit,
        resume_threshold=args.gpu_temp_resume,
        check_interval=args.gpu_temp_check_interval,
        poll_seconds=args.gpu_temp_poll_seconds,
    )
    if thermal_guard.enabled:
        print(
            "GPU temperature guard:\n"
            f"  limit: {thermal_guard.limit:.0f}°C\n"
            f"  resume: {thermal_guard.resume_threshold:.0f}°C\n"
            f"  check interval: every {thermal_guard.check_interval} batches\n"
            f"  polling interval while paused: {thermal_guard.poll_seconds} seconds"
        )
        thermal_guard.verify_monitoring()  # fail fast if nvidia-smi is broken/missing
    else:
        print("GPU temperature guard: disabled")

    synthetic_noise_config = build_synthetic_noise_config(
        args.synthetic_noise_prob,
        seed=args.seed,
        distribution=args.synthetic_noise_distribution,
        degrees_of_freedom=args.synthetic_noise_nu,
        variance_floor=args.synthetic_noise_variance_floor,
    )
    synthetic_noise = build_synthetic_noise(synthetic_noise_config)
    if synthetic_noise is not None:
        print(f"Synthetic noise augmentation: {synthetic_noise_config}")
    else:
        print("Synthetic noise augmentation: disabled")

    train_dataset, validation_dataset, total_pairs = build_datasets(
        args.data_dir,
        args.val_fraction,
        args.seed,
        args.crop_size,
        args.scale,
        args.max_train_samples,
        args.max_val_samples,
        synthetic_noise=synthetic_noise,
        hard_patch_sampling=args.hard_patch_sampling,
        hard_patch_prob=args.hard_patch_prob,
    )
    if args.hard_patch_sampling:
        print(f"Hard-patch sampling: enabled (prob={args.hard_patch_prob})")
    else:
        print("Hard-patch sampling: disabled")
    print(
        f"Discovered {total_pairs} pairs -> "
        f"train={len(train_dataset)} val={len(validation_dataset)}"
    )
    print(
        f"Training crop: LR = {args.crop_size}x{args.crop_size} "
        f"GT = {args.crop_size * args.scale}x{args.crop_size * args.scale} "
        "(validation always uses full images)"
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    noise_conditioning_config = build_noise_conditioning_config(
        args.noise_conditioning, variance_floor=args.noise_conditioning_variance_floor
    )
    # Off (the default) returns args.model unchanged; only ever raises/switches
    # when --global-bicubic-residual is explicitly passed -- never silent.
    model_name = resolve_model_architecture(args.model, args.global_bicubic_residual)
    # in_channels=2 only because conv_in must accept the extra sigma channel --
    # every other architectural detail (blocks, upsampling head) is untouched.
    model_config = build_model_config(
        model_name,
        in_channels=2 if noise_conditioning_config else 1,
        out_channels=1,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        scale=args.scale,
        residual_scale=args.residual_scale,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        channel_attention=args.channel_attention,
        attention_reduction=args.attention_reduction,
        multiscale_block=args.multiscale_block,
        rdb_block=args.rdb_block,
        rdb_growth_rate=args.rdb_growth_rate,
        rdb_num_layers=args.rdb_num_layers,
        denoise_stem=args.denoise_stem,
        denoise_stem_features=args.denoise_stem_features,
        denoise_stem_blocks=args.denoise_stem_blocks,
    )
    base_model = build_model(model_config).to(device)
    # wrap_for_conditioning returns base_model itself (no wrapper at all) when
    # conditioning is off, so every historical command's model object is
    # byte-for-byte the same as before this feature existed.
    model = wrap_for_conditioning(base_model, noise_conditioning_config)
    if args.finetune_from is not None:
        # Loads weights only, before the optimizer/EMA are constructed below --
        # so EMA's initial deep-copy (if --ema is also requested) captures
        # these fine-tuned starting weights, not a fresh random init.
        load_weights_for_finetune(args.finetune_from, model, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} ({param_count:,} trainable parameters)")
    print(f"Model config: {model_config}")
    print(f"Noise conditioning: {noise_conditioning_config or 'disabled'}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_config = build_loss_config(
        args.loss,
        args.charbonnier_eps,
        args.ssim_weight,
        args.mixed_loss_alpha,
        args.weighted_l1_eps,
        args.weighted_l1_variance_floor,
    )
    loss_fn = build_loss(loss_config)
    label = loss_label(loss_config["name"])
    print(f"Loss: {loss_config}")

    scheduler_config = build_scheduler_config(
        args.scheduler, args.scheduler_factor, args.scheduler_patience, args.min_lr,
        t_max=args.scheduler_t_max,
    )
    scheduler = build_scheduler(optimizer, scheduler_config)
    if scheduler is not None:
        print(f"Scheduler: {scheduler_config}")

    ema_config = build_ema_config(args.ema, args.ema_decay)
    # Deep-copies model's current weights (freshly initialized here, pre-resume) --
    # never zeros. If resuming into an EMA run, load_checkpoint_for_resume() below
    # overwrites this initial copy with the checkpoint's actual saved EMA state.
    ema = ExponentialMovingAverage(model, ema_config["decay"]).to(device) if ema_config else None
    if ema is not None:
        print(f"EMA: {ema_config}")

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "crop_size": args.crop_size,
        "hard_patch_sampling": args.hard_patch_sampling,
        "hard_patch_prob": args.hard_patch_prob,
        "data_dir": str(args.data_dir),
        # Informational only -- a runtime/wall-clock setting, not part of the
        # ML computation. warn_on_resume_config_mismatch() and
        # load_checkpoint_for_resume() deliberately never compare these
        # fields, so resuming is always legal regardless of thermal settings.
        "gpu_temp_limit": args.gpu_temp_limit,
        "gpu_temp_resume": args.gpu_temp_resume,
        "gpu_temp_check_interval": args.gpu_temp_check_interval,
        "gpu_temp_poll_seconds": args.gpu_temp_poll_seconds,
        # Informational only, like the gpu_temp_* fields above -- records
        # provenance for reproducibility but is never compared during resume.
        "finetuned_from": str(args.finetune_from) if args.finetune_from is not None else None,
    }

    start_epoch = 1
    best_val_psnr = float("-inf")
    if args.resume is not None:
        start_epoch, best_val_psnr, previous_config = load_checkpoint_for_resume(
            args.resume,
            model,
            optimizer,
            model_config,
            device,
            scheduler=scheduler,
            loss_config=loss_config,
            scheduler_config=scheduler_config,
            ema=ema,
            ema_config=ema_config,
            synthetic_noise_config=synthetic_noise_config,
            noise_conditioning_config=noise_conditioning_config,
        )
        print(
            f"Resumed from {args.resume}: continuing at epoch {start_epoch} "
            f"(best Val PSNR so far: {best_val_psnr:.4f} dB, "
            f"current LR: {current_lr(optimizer):.6e})"
        )
        warn_on_resume_config_mismatch(
            previous_config,
            args.seed,
            args.val_fraction,
            args.crop_size,
            args.hard_patch_sampling,
            args.hard_patch_prob,
        )
        if start_epoch > args.epochs:
            print(f"Nothing to do: resumed epoch {start_epoch} is beyond --epochs {args.epochs}.")
            return

    latest_path = args.checkpoint_dir / "checkpoint_latest.pt"
    best_path = args.checkpoint_dir / "checkpoint_best.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        epoch_lr = current_lr(optimizer)
        # Advances the synthetic-noise stream so each epoch draws fresh
        # realizations while staying a pure function of (seed, epoch, index).
        # No-op when the augmentation is disabled.
        train_dataset.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, thermal_guard, ema=ema
        )
        # Validation (and therefore the scheduler decision and best-checkpoint
        # selection below, both of which only ever see val_metrics) uses the EMA
        # shadow weights when EMA is enabled, instead of the live training
        # weights -- this is the entire point of Experiment 19. The live model
        # keeps training normally regardless; only which weights get *scored*
        # changes.
        eval_model = ema.shadow_model if ema is not None else model
        val_metrics = validate(eval_model, validation_loader, loss_fn, device, thermal_guard)
        # Wall-clock only: includes any thermal-pause sleep time when the
        # guard is enabled and triggered. Not compensated for -- "Epoch time"
        # is elapsed wall-clock time, same as before this feature existed.
        elapsed = time.time() - started

        print(f"Epoch {epoch}")
        print(f"Learning rate: {epoch_lr:.6e}")
        print(f"Train {label}: {train_loss:.6f}")
        print(f"Val {label}: {val_metrics['loss']:.6f}")
        print(f"Val PSNR: {val_metrics['psnr']:.4f} dB")
        print(f"Val SSIM: {val_metrics['ssim']:.6f}")
        print(f"Bicubic PSNR: {BICUBIC_PSNR_DB:.4f} dB")
        print(f"Bicubic SSIM: {BICUBIC_SSIM:.6f}")
        print(f"Epoch time: {elapsed:.1f}s")

        # scheduler.step() happens after validation PSNR is known and before
        # checkpointing, so the saved optimizer LR and scheduler state both
        # already reflect this epoch's result -- a resume then just continues,
        # with no need to replay this epoch's step() call.
        if scheduler is not None:
            previous_lr = current_lr(optimizer)
            scheduler_step(scheduler, scheduler_config, val_metrics["psnr"])
            new_lr = current_lr(optimizer)
            if new_lr < previous_lr:
                print("Learning rate reduced:")
                print(f"{previous_lr:.6e} -> {new_lr:.6e}")

        # Update best_val_psnr before saving checkpoint_latest.pt so a resume
        # picks up the true best-so-far, including this epoch's own result --
        # not a stale value from before this epoch ran.
        is_new_best = val_metrics["psnr"] > best_val_psnr
        if is_new_best:
            best_val_psnr = val_metrics["psnr"]

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            best_val_psnr,
            model_config,
            training_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            ema=ema,
            ema_config=ema_config,
            synthetic_noise_config=synthetic_noise_config,
            noise_conditioning_config=noise_conditioning_config,
        )
        if is_new_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_psnr,
                model_config,
                training_config,
                scheduler=scheduler,
                scheduler_config=scheduler_config,
                loss_config=loss_config,
                ema=ema,
                ema_config=ema_config,
                synthetic_noise_config=synthetic_noise_config,
                noise_conditioning_config=noise_conditioning_config,
            )
            print(f"New best checkpoint saved ({best_path}); Val PSNR={best_val_psnr:.4f} dB")

    print("Training complete.")


if __name__ == "__main__":
    main()
