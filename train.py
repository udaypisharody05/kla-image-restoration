"""Train the small residual-CNN restoration baseline.

Reuses the repository's existing dataset discovery, canonical train/validation
split, lazy PyTorch datasets, and aligned training preprocessing. Trains a
lightweight residual CNN (see ``src/models/residual_sr.py``) with L1 loss and
Adam, validates on the full deterministic 640-image validation split every
epoch, and checkpoints the latest and best (by validation PSNR) models.

An optional ``ReduceLROnPlateau`` learning-rate scheduler (monitoring
validation PSNR, ``mode="max"``) can be enabled with ``--scheduler plateau``.
The default ``--scheduler none`` reproduces Experiment 1's fixed-LR behavior
exactly.
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
from src.losses import build_loss, build_loss_config, loss_label
from src.metrics import psnr, ssim
from src.models import ResidualSRNet
from src.splits import split_pairs
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
) -> tuple[PairedRestorationDataset, PairedRestorationDataset, int]:
    """Rebuild the canonical split and wrap it with the project's dataset classes."""
    layout = discover_layout(data_dir)
    pairs = discover_pairs(layout).pairs
    train_pairs, validation_pairs = split_pairs(pairs, val_fraction=val_fraction, seed=seed)
    if max_train_samples is not None:
        train_pairs = train_pairs[:max_train_samples]
    if max_val_samples is not None:
        validation_pairs = validation_pairs[:max_val_samples]

    training_transform = create_training_transform(crop_size=crop_size, scale=scale, augment=True)
    train_dataset = PairedRestorationDataset(train_pairs, scale=scale, transform=training_transform)
    # Validation stays deterministic and full-resolution, directly comparable to bicubic.
    validation_dataset = PairedRestorationDataset(validation_pairs, scale=scale)
    return train_dataset, validation_dataset, len(pairs)


def build_scheduler_config(
    scheduler_name: str, factor: float, patience: int, min_lr: float
) -> dict | None:
    """Turn CLI scheduler options into a plain, checkpoint-serializable dict."""
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
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_config: dict | None
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
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
    raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def warn_on_resume_config_mismatch(
    previous_config: dict, seed: int, val_fraction: float, crop_size: int
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_psnr: float,
    model_config: dict,
    training_config: dict,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scheduler_config: dict | None = None,
    loss_config: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
        },
        path,
    )


def load_checkpoint_for_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    loss_config: dict | None = None,
) -> tuple[int, float, dict]:
    """Restore model/optimizer/(optional) scheduler state from *path*.

    Returns (next_epoch, best_val_psnr, training_config) from the checkpoint.
    Older checkpoints saved before scheduler support existed simply lack the
    ``scheduler_state_dict``/``scheduler_config`` keys; ``.get()`` treats that
    the same as an explicit ``None`` rather than raising.

    *loss_config*, when given, must match the checkpoint's stored loss config
    exactly (mirroring the strict ``model_config`` check below) -- a resume
    must never silently switch the reconstruction loss a run is being trained
    against. Pass ``None`` to skip this check (used by tests that don't care).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_config"] != model_config:
        raise ValueError(
            f"Checkpoint model_config {checkpoint['model_config']} does not match "
            f"the requested model_config {model_config}; pass matching "
            "--num-features/--num-blocks/--scale to resume."
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

    return checkpoint["epoch"] + 1, checkpoint["best_val_psnr"], checkpoint["training_config"]


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    thermal_guard: GpuTemperatureGuard | None = None,
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
    parser.add_argument("--num-features", type=int, default=32, help="Residual block channel width")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual blocks")
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
        "--scheduler",
        type=str,
        choices=["none", "plateau"],
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
        "--min-lr", type=float, default=1e-6, help="Lower bound the scheduler will not cross"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["l1", "charbonnier", "l1_ssim"],
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

    train_dataset, validation_dataset, total_pairs = build_datasets(
        args.data_dir,
        args.val_fraction,
        args.seed,
        args.crop_size,
        args.scale,
        args.max_train_samples,
        args.max_val_samples,
    )
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

    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_features": args.num_features,
        "num_blocks": args.num_blocks,
        "scale": args.scale,
    }
    model = ResidualSRNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_config = build_loss_config(args.loss, args.charbonnier_eps, args.ssim_weight)
    loss_fn = build_loss(loss_config)
    label = loss_label(loss_config["name"])
    print(f"Loss: {loss_config}")

    scheduler_config = build_scheduler_config(
        args.scheduler, args.scheduler_factor, args.scheduler_patience, args.min_lr
    )
    scheduler = build_scheduler(optimizer, scheduler_config)
    if scheduler is not None:
        print(f"Scheduler: {scheduler_config}")

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "crop_size": args.crop_size,
        "data_dir": str(args.data_dir),
        # Informational only -- a runtime/wall-clock setting, not part of the
        # ML computation. warn_on_resume_config_mismatch() and
        # load_checkpoint_for_resume() deliberately never compare these
        # fields, so resuming is always legal regardless of thermal settings.
        "gpu_temp_limit": args.gpu_temp_limit,
        "gpu_temp_resume": args.gpu_temp_resume,
        "gpu_temp_check_interval": args.gpu_temp_check_interval,
        "gpu_temp_poll_seconds": args.gpu_temp_poll_seconds,
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
        )
        print(
            f"Resumed from {args.resume}: continuing at epoch {start_epoch} "
            f"(best Val PSNR so far: {best_val_psnr:.4f} dB, "
            f"current LR: {current_lr(optimizer):.6e})"
        )
        warn_on_resume_config_mismatch(
            previous_config, args.seed, args.val_fraction, args.crop_size
        )
        if start_epoch > args.epochs:
            print(f"Nothing to do: resumed epoch {start_epoch} is beyond --epochs {args.epochs}.")
            return

    latest_path = args.checkpoint_dir / "checkpoint_latest.pt"
    best_path = args.checkpoint_dir / "checkpoint_best.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        epoch_lr = current_lr(optimizer)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, thermal_guard
        )
        val_metrics = validate(model, validation_loader, loss_fn, device, thermal_guard)
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
            scheduler.step(val_metrics["psnr"])
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
            )
            print(f"New best checkpoint saved ({best_path}); Val PSNR={best_val_psnr:.4f} dB")

    print("Training complete.")


if __name__ == "__main__":
    main()
