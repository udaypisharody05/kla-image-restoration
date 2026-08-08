"""Evaluate a saved neural restoration checkpoint on the fixed validation split.

Loads a checkpoint saved by ``train.py``, reconstructs the matching
``ResidualSRNet``, evaluates the deterministic validation split, and prints
L1/PSNR/SSIM alongside the established bicubic baseline for comparison. Does
not run inference on the competition test set.

Regardless of which reconstruction loss a checkpoint was *trained* with (see
``src/losses.py`` and ``--loss`` in ``train.py``), evaluation always reports
actual L1 as its "Val L1" diagnostic -- this keeps that number directly
comparable across every experiment, including ones trained with Charbonnier
loss. The checkpoint's training loss is reported separately and explicitly.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from inspect_dataset import configured_data_dir
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.losses import loss_label
from src.models import ResidualSRNet
from src.splits import split_pairs
from train import BICUBIC_PSNR_DB, BICUBIC_SSIM, select_device, validate


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResidualSRNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional subset limit")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device)
    print(
        f"Loaded checkpoint {args.checkpoint} (epoch={checkpoint.get('epoch')}, "
        f"best_val_psnr={checkpoint.get('best_val_psnr')})"
    )
    # Checkpoints saved before loss selection existed have no stored
    # loss_config; every historical run (Experiments 1-3) used plain L1.
    loss_config = checkpoint.get("loss_config", {"name": "l1"})
    print(f"Training loss: {loss_label(loss_config['name'])} ({loss_config})")

    pairs = discover_pairs(discover_layout(args.data_dir)).pairs
    _, validation_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if args.max_val_samples is not None:
        validation_pairs = validation_pairs[: args.max_val_samples]
    validation_dataset = PairedRestorationDataset(
        validation_pairs, scale=checkpoint["model_config"]["scale"]
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Always score with real L1 here (independent of loss_config above), so
    # this number stays directly comparable across every experiment even when
    # training losses differ.
    metrics = validate(model, validation_loader, nn.L1Loss(), device)

    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Val L1 (diagnostic, always L1 regardless of training loss): {metrics['loss']:.6f}")
    print(f"Val PSNR: {metrics['psnr']:.4f} dB")
    print(f"Val SSIM: {metrics['ssim']:.6f}")
    print(f"Bicubic PSNR: {BICUBIC_PSNR_DB:.4f} dB")
    print(f"Bicubic SSIM: {BICUBIC_SSIM:.6f}")
    psnr_delta = metrics["psnr"] - BICUBIC_PSNR_DB
    ssim_delta = metrics["ssim"] - BICUBIC_SSIM
    print(f"PSNR vs bicubic: {psnr_delta:+.4f} dB")
    print(f"SSIM vs bicubic: {ssim_delta:+.6f}")


if __name__ == "__main__":
    main()
