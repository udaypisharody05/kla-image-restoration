"""Secondary, leakage-aware validation metric (diagnostic only).

Experiment 22 showed 38 of the 640 canonical validation images have a
near-identical scene in train, so canonical numbers are slightly optimistic.
This utility re-evaluates a checkpoint on a **group-aware** split where no
near-duplicate scene group straddles the boundary, giving a cleaner read on
generalization.

**This metric is diagnostic only.** It does not drive the scheduler, does not
select checkpoints, does not replace canonical validation, and does not
retroactively change any historical experiment number. It lives outside
``train.py`` on purpose -- training is untouched by it.

Usage::

    python evaluate_group_aware.py --checkpoint checkpoints/exp23_ema_extended90/checkpoint_best.pt
    python evaluate_group_aware.py --checkpoint ... --tta x8
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from evaluate_checkpoint import load_model, validate_x8
from inspect_dataset import configured_data_dir
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.scene_groups import count_cross_split_groups, find_scene_groups, group_aware_split
from src.splits import split_pairs
from train import BICUBIC_PSNR_DB, BICUBIC_SSIM, select_device, validate


def evaluate(model, pairs, scale, device, batch_size, num_workers, tta):
    dataset = PairedRestorationDataset(pairs, scale=scale)  # never synthetic
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    runner = validate_x8 if tta == "x8" else validate
    return runner(model, loader, nn.L1Loss(), device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta", type=str, choices=["none", "x8"], default="none")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")
    model, checkpoint = load_model(args.checkpoint, device)
    scale = checkpoint["model_config"]["scale"]
    print(f"Loaded {args.checkpoint} (epoch={checkpoint.get('epoch')})")
    print(f"TTA: {args.tta}")

    pairs = discover_pairs(discover_layout(args.data_dir)).pairs
    print(f"Finding near-duplicate scene groups across {len(pairs):,} pairs...")
    groups = find_scene_groups(pairs)
    print(f"  {len(groups)} groups covering {sum(len(g) for g in groups)} images")

    canonical_train, canonical_validation = split_pairs(
        pairs, val_fraction=args.val_fraction, seed=args.seed
    )
    grouped_train, grouped_validation = group_aware_split(
        pairs, groups, val_fraction=args.val_fraction, seed=args.seed
    )
    canonical_leak = count_cross_split_groups(groups, canonical_train, canonical_validation)
    grouped_leak = count_cross_split_groups(groups, grouped_train, grouped_validation)

    print("\nEvaluating canonical validation split (the historical metric)...")
    canonical_metrics = evaluate(
        model, canonical_validation, scale, device, args.batch_size, args.num_workers, args.tta
    )
    print("Evaluating group-aware validation split (diagnostic)...")
    grouped_metrics = evaluate(
        model, grouped_validation, scale, device, args.batch_size, args.num_workers, args.tta
    )

    results = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "tta": args.tta,
        "note": "Group-aware metric is DIAGNOSTIC ONLY; canonical remains authoritative.",
        "scene_groups": len(groups),
        "canonical": {
            "count": len(canonical_validation),
            **canonical_metrics,
            **canonical_leak,
        },
        "group_aware": {
            "count": len(grouped_validation),
            **grouped_metrics,
            **grouped_leak,
        },
        "delta": {
            key: float(grouped_metrics[key] - canonical_metrics[key])
            for key in ("loss", "psnr", "ssim")
        },
    }

    print(f"\n{'split':<14}{'n':>6}{'L1':>12}{'PSNR':>11}{'SSIM':>11}{'leaked imgs':>13}")
    for name in ("canonical", "group_aware"):
        row = results[name]
        print(
            f"{name:<14}{row['count']:>6}{row['loss']:>12.6f}{row['psnr']:>11.4f}"
            f"{row['ssim']:>11.6f}{row['validation_images_with_train_twin']:>13}"
        )
    print(
        f"{'delta':<14}{'':>6}{results['delta']['loss']:>12.6f}"
        f"{results['delta']['psnr']:>11.4f}{results['delta']['ssim']:>11.6f}"
    )
    print(f"\nBicubic reference: {BICUBIC_PSNR_DB:.4f} dB / {BICUBIC_SSIM:.6f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
