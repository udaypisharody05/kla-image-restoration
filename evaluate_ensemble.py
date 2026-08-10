"""Evaluate a weighted ensemble of two checkpoints on the canonical validation split.

Inference-only: loads two already-trained checkpoints (e.g. Experiment 6 and
Experiment 9), runs each independently, weighted-averages their raw predictions
(``src.ensemble.weighted_average_predictions``), and scores the ensemble with the
exact same canonical PSNR/SSIM used everywhere else in this project. No
retraining, no checkpoint modification.

``--tta {none,x8}`` optionally routes *each* model through the existing x8
geometric self-ensemble (``src/tta.py::predict_x8``) before the two raw
predictions are combined -- the same TTA implementation ``evaluate_checkpoint.py``
uses, not a second copy of it.
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from evaluate_checkpoint import load_model
from inspect_dataset import configured_data_dir
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.ensemble import weighted_average_predictions
from src.metrics import psnr, ssim
from src.splits import split_pairs
from src.tta import predict_x8
from train import BICUBIC_PSNR_DB, BICUBIC_SSIM, select_device


@torch.no_grad()
def validate_ensemble(
    model_a: nn.Module,
    model_b: nn.Module,
    loader,
    weights: list[float],
    loss_fn: nn.Module,
    device: torch.device,
    tta: str = "none",
) -> dict[str, float]:
    """Same aggregation convention as ``train.validate``/``evaluate_checkpoint.validate_x8``,
    scoring the weighted average of the two models' raw predictions instead of a
    single model's output. Both models always run in eval mode; TTA (if any) is
    delegated entirely to ``predict_x8``.
    """
    model_a.eval()
    model_b.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        if tta == "x8":
            prediction_a = predict_x8(model_a, inputs)
            prediction_b = predict_x8(model_b, inputs)
        else:
            prediction_a = model_a(inputs)
            prediction_b = model_b(inputs)
        outputs = weighted_average_predictions([prediction_a, prediction_b], weights)
        loss = loss_fn(outputs, targets)
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += psnr(outputs, targets) * batch_size
        total_ssim += ssim(outputs, targets) * batch_size
        total_count += batch_size
    return {
        "loss": total_loss / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-a", type=Path, default=Path("checkpoints/exp6_crop96/checkpoint_best.pt")
    )
    parser.add_argument(
        "--checkpoint-b", type=Path, default=Path("checkpoints/exp9_edsr_lite/checkpoint_best.pt")
    )
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional subset limit")
    parser.add_argument(
        "--tta",
        type=str,
        choices=["none", "x8"],
        default="none",
        help="Apply x8 geometric self-ensemble to each model before combining.",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    model_a, checkpoint_a = load_model(args.checkpoint_a, device)
    model_b, checkpoint_b = load_model(args.checkpoint_b, device)
    architecture_a = checkpoint_a["model_config"].get("architecture", "residual_sr")
    architecture_b = checkpoint_b["model_config"].get("architecture", "residual_sr")
    print(f"Checkpoint A: {args.checkpoint_a} (architecture={architecture_a}, epoch={checkpoint_a.get('epoch')})")
    print(f"Checkpoint B: {args.checkpoint_b} (architecture={architecture_b}, epoch={checkpoint_b.get('epoch')})")
    print(f"Weights: A={args.weight_a}, B={args.weight_b}")
    print(f"TTA: {'x8 geometric self-ensemble per model' if args.tta == 'x8' else 'disabled'}")

    scale_a = checkpoint_a["model_config"]["scale"]
    scale_b = checkpoint_b["model_config"]["scale"]
    if scale_a != scale_b:
        raise ValueError(f"Scale mismatch between checkpoints: A={scale_a} vs B={scale_b}")

    pairs = discover_pairs(discover_layout(args.data_dir)).pairs
    _, validation_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if args.max_val_samples is not None:
        validation_pairs = validation_pairs[: args.max_val_samples]
    validation_dataset = PairedRestorationDataset(validation_pairs, scale=scale_a)
    validation_loader = create_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    start_time = time.perf_counter()
    metrics = validate_ensemble(
        model_a,
        model_b,
        validation_loader,
        [args.weight_a, args.weight_b],
        nn.L1Loss(),
        device,
        tta=args.tta,
    )
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Val L1: {metrics['loss']:.6f}")
    print(f"Val PSNR: {metrics['psnr']:.4f} dB")
    print(f"Val SSIM: {metrics['ssim']:.6f}")
    print(f"Bicubic PSNR: {BICUBIC_PSNR_DB:.4f} dB")
    print(f"Bicubic SSIM: {BICUBIC_SSIM:.6f}")
    print(f"PSNR vs bicubic: {metrics['psnr'] - BICUBIC_PSNR_DB:+.4f} dB")
    print(f"SSIM vs bicubic: {metrics['ssim'] - BICUBIC_SSIM:+.6f}")
    print(f"Elapsed: {elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
