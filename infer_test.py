"""Inference sanity check on the official (ground-truth-free) competition test set.

Loads a trained checkpoint (default: the Experiment 6 champion,
``checkpoints/exp6_crop96/checkpoint_best.pt``), runs it on a small,
deterministic slice of the official ``Test_NoisyLR`` images, and saves raw
predictions plus bicubic/neural comparison visualizations for manual
inspection.

The official test set has no locally available ground truth, so this script
deliberately does NOT compute PSNR, SSIM, or any other quality metric --
doing so would require inventing a target. This is a visual/numerical sanity
check only, not model evaluation or selection.

Reuses existing project logic rather than duplicating it:
``evaluate_checkpoint.load_model`` for checkpoint/model reconstruction,
``train.select_device`` for device selection, ``src.dataset_discovery`` +
``src.dataset.RestorationTestDataset`` for deterministic, un-cropped,
un-normalized test loading (the same raw-value convention validation
already uses), and ``src.baseline.bicubic_upscale`` for the classical
comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_checkpoint import load_model
from inspect_dataset import configured_data_dir
from src.baseline import bicubic_upscale
from src.dataset import RestorationTestDataset
from src.dataset_discovery import discover_layout, image_files
from src.tta import predict_x8
from train import select_device


def select_test_files(data_dir: Path, max_samples: int) -> list[Path]:
    """Deterministically select the first *max_samples* official test files.

    ``image_files`` already returns paths sorted by POSIX path, so this is a
    plain prefix of that sorted list -- no random sampling anywhere.
    """
    if max_samples < 1:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    layout = discover_layout(data_dir)
    files = image_files(layout.test_input_dir)
    if not files:
        raise RuntimeError(f"No official test images found under {layout.test_input_dir}")
    return files[:max_samples]


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    tta: str = "none",
) -> np.ndarray:
    """Run the model on one full-resolution ``[1,H,W]`` LR tensor.

    Returns the raw (unclipped) float32 ``[2H,2W]`` prediction as a numpy
    array. Model output is not altered in any way before this point (no
    clipping, no sigmoid/tanh, no cropping) -- only converted to numpy.
    ``tta="none"`` (the default) is byte-for-byte the original single-pass
    behavior; ``tta="x8"`` averages the 8-way geometric self-ensemble instead
    (see ``src/tta.py``), same raw-prediction convention either way.
    """
    model.eval()
    batch = input_tensor.unsqueeze(0).to(device)  # [1,1,H,W]
    if tta == "x8":
        output = predict_x8(model, batch)
    else:
        with torch.inference_mode():
            output = model(batch)
    return output[0, 0].detach().cpu().numpy().astype(np.float32)


def validate_prediction_before_saving(
    prediction: np.ndarray, expected_shape: tuple[int, int], filename: str
) -> None:
    """Stop clearly (raise) before saving anything if the prediction looks wrong."""
    if prediction.shape != expected_shape:
        raise ValueError(
            f"{filename}: prediction shape {prediction.shape} != expected "
            f"{expected_shape} (2x the LR input size)"
        )
    if not np.isfinite(prediction).all():
        nan_count = int(np.isnan(prediction).sum())
        inf_count = int(np.isinf(prediction).sum())
        raise ValueError(
            f"{filename}: prediction contains non-finite values "
            f"(NaN={nan_count}, Inf={inf_count})"
        )


def clip_for_display(prediction: np.ndarray) -> np.ndarray:
    """Clip to [0,1] for saved/displayed prediction images only.

    Matches the established evaluation-metric convention
    (``src.baseline.metric_arrays``, ``clip_prediction=True``): the raw
    ``.npy`` prediction saved to disk stays unclipped (preserves full
    numerical information for eventual submission use), but any PNG a human
    looks at is clipped the same way validation/bicubic metrics already are.
    """
    return np.clip(prediction, 0.0, 1.0)


def save_comparison_png(
    path: Path,
    filename: str,
    noisy_lr: np.ndarray,
    bicubic: np.ndarray,
    prediction_for_display: np.ndarray,
) -> None:
    """Noisy LR | Bicubic 2x | Neural prediction, side by side. No GT panel."""
    finite_lr = noisy_lr[np.isfinite(noisy_lr)]
    lr_vmin, lr_vmax = float(finite_lr.min()), float(finite_lr.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    panels = (
        ("Noisy LR", noisy_lr, lr_vmin, lr_vmax),
        ("Bicubic 2x", bicubic, 0.0, 1.0),
        ("Exp6 Neural Prediction", prediction_for_display, 0.0, 1.0),
    )
    for ax, (title, array, vmin, vmax) in zip(axes, panels):
        ax.imshow(array, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}\n{array.shape[0]}x{array.shape[1]}")
        ax.axis("off")
    fig.suptitle(f"{filename}  (official test image -- no local GT)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(
    path: Path,
    filenames: list[str],
    noisy_lr_list: list[np.ndarray],
    bicubic_list: list[np.ndarray],
    prediction_list: list[np.ndarray],
) -> None:
    """One row per sample, columns Noisy LR | Bicubic | Exp6 Prediction. Visual only, no metrics."""
    rows = len(filenames)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    column_titles = ("Noisy LR", "Bicubic 2x", "Exp6 Prediction")
    for row in range(rows):
        finite_lr = noisy_lr_list[row][np.isfinite(noisy_lr_list[row])]
        lr_vmin, lr_vmax = float(finite_lr.min()), float(finite_lr.max())
        panels = (
            (noisy_lr_list[row], lr_vmin, lr_vmax),
            (bicubic_list[row], 0.0, 1.0),
            (prediction_list[row], 0.0, 1.0),
        )
        for col, (array, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(array, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(column_titles[col])
            if col == 0:
                ax.set_ylabel(filenames[row], fontsize=8)
    fig.suptitle("Experiment 6 official-test inference sanity check (no GT available; visual inspection only)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/exp6_crop96/checkpoint_best.pt")
    )
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("results/test_sanity_exp6"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--tta",
        type=str,
        choices=["none", "x8"],
        default="none",
        help="Test-time augmentation. 'none' (default) is byte-for-byte the original behavior.",
    )
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")

    device = select_device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device)
    print(
        f"Loaded checkpoint {args.checkpoint} "
        f"(epoch={checkpoint.get('epoch')}, best_val_psnr={checkpoint.get('best_val_psnr')})"
    )
    print(f"Model config: {checkpoint['model_config']}")
    print(f"TTA: {'x8 geometric self-ensemble' if args.tta == 'x8' else 'disabled'}")
    scale = checkpoint["model_config"]["scale"]

    test_files = select_test_files(args.data_dir, args.max_samples)
    print(f"Selected {len(test_files)} deterministic test images (sorted filename order):")
    for path in test_files:
        print(f"  {path.name}")

    dataset = RestorationTestDataset(test_files)

    predictions_dir = args.output_dir / "predictions"
    bicubic_dir = args.output_dir / "bicubic"
    comparisons_dir = args.output_dir / "comparisons"
    for directory in (predictions_dir, bicubic_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    lr_panels, bicubic_panels, prediction_panels, panel_names = [], [], [], []
    per_sample_stats = []

    for index in range(len(dataset)):
        sample = dataset[index]
        input_tensor = sample["input"]  # [1,H,W], raw unclipped values -- same convention as validation
        filename = sample["filename"]
        stem = Path(filename).stem
        input_height, input_width = input_tensor.shape[-2:]
        expected_shape = (input_height * scale, input_width * scale)

        raw_prediction = run_inference(model, input_tensor, device, args.tta)
        validate_prediction_before_saving(raw_prediction, expected_shape, filename)

        noisy_lr = input_tensor[0].numpy()
        bicubic_prediction = bicubic_upscale(noisy_lr, scale=scale)

        # Raw .npy stays unclipped -- preserves full numeric information,
        # matching the project's existing "clip only for metrics/display,
        # keep raw for storage" convention (see src/baseline.py).
        np.save(predictions_dir / filename, raw_prediction)
        np.save(bicubic_dir / filename, bicubic_prediction)

        prediction_for_display = clip_for_display(raw_prediction)
        save_comparison_png(
            comparisons_dir / f"{stem}.png",
            filename,
            noisy_lr,
            bicubic_prediction,
            prediction_for_display,
        )

        stats = {
            "filename": filename,
            "input_shape": list(input_tensor.shape[-2:]),
            "output_shape": list(raw_prediction.shape),
            "prediction_min": float(raw_prediction.min()),
            "prediction_max": float(raw_prediction.max()),
            "prediction_mean": float(raw_prediction.mean()),
            "finite": bool(np.isfinite(raw_prediction).all()),
        }
        per_sample_stats.append(stats)
        print(
            f"{filename}: input={stats['input_shape']} output={stats['output_shape']} "
            f"min={stats['prediction_min']:.4f} max={stats['prediction_max']:.4f} "
            f"mean={stats['prediction_mean']:.4f} finite={stats['finite']}"
        )

        lr_panels.append(noisy_lr)
        bicubic_panels.append(bicubic_prediction)
        prediction_panels.append(prediction_for_display)
        panel_names.append(filename)

    contact_sheet_path = args.output_dir / "contact_sheet.png"
    save_contact_sheet(contact_sheet_path, panel_names, lr_panels, bicubic_panels, prediction_panels)

    all_finite = all(sample["finite"] for sample in per_sample_stats)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_psnr": checkpoint.get("best_val_psnr"),
        "model_config": checkpoint["model_config"],
        "num_samples": len(per_sample_stats),
        "selected_filenames": [sample["filename"] for sample in per_sample_stats],
        "all_finite": all_finite,
        "prediction_min_overall": min(sample["prediction_min"] for sample in per_sample_stats),
        "prediction_max_overall": max(sample["prediction_max"] for sample in per_sample_stats),
        "prediction_mean_overall": float(
            np.mean([sample["prediction_mean"] for sample in per_sample_stats])
        ),
        "samples": per_sample_stats,
        "note": (
            "Official test set has no local ground truth; PSNR/SSIM were NOT "
            "computed and must not be inferred from this file."
        ),
    }
    stats_path = args.output_dir / "sanity_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"All {len(per_sample_stats)} predictions finite: {all_finite}")
    print(
        "Prediction value range across samples: "
        f"[{summary['prediction_min_overall']:.4f}, {summary['prediction_max_overall']:.4f}]"
    )
    print(f"Saved raw predictions to {predictions_dir}")
    print(f"Saved bicubic baselines to {bicubic_dir}")
    print(f"Saved comparison PNGs to {comparisons_dir}")
    print(f"Saved contact sheet to {contact_sheet_path}")
    print(f"Saved sanity stats to {stats_path}")
    print()
    print("No test PSNR/SSIM computed: the official test set has no local ground truth.")


if __name__ == "__main__":
    main()
