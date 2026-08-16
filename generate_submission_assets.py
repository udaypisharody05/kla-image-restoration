"""Generate submission-ready figures under submission/assets/.

Three deliverables, each reusing already-measured/implemented project state
(no invented numbers, no fabricated modules):

1. ``pipeline.png`` -- the actual inference pipeline
   (Noisy LR -> aligned preprocessing -> ResidualSR -> 2x reconstruction ->
   optional x8 self-ensemble -> restored HR).
2. ``metrics.png`` -- measured Bicubic / ResidualSR raw / ResidualSR + x8
   PSNR/SSIM/LPIPS, read from results/final_metrics.json (not recomputed
   here, so this can never silently drift from the authoritative numbers).
3. ``results/sample_XXXXXX.png`` -- 3-5 representative
   "Noisy LR | Bicubic | Restored | GT" panels, deterministically selected
   (evenly spaced indices through the canonical 640-image validation split,
   documented below -- not cherry-picked).
"""

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
from src.dataset_discovery import discover_layout, discover_pairs
from src.splits import split_pairs
from src.io_utils import load_image_array
from train import select_device

ASSETS_DIR = Path("submission/assets")
RESULTS_DIR = ASSETS_DIR / "results"
METRICS_PATH = Path("results/final_metrics.json")
CHECKPOINT_PATH = Path("checkpoints/exp23_ema_extended90/checkpoint_best.pt")

# Deterministic, documented selection: 5 indices evenly spaced through the
# canonical 640-sample validation split (seed=42, val_fraction=0.2) in its
# fixed sorted order -- not a random or hand-picked "best-looking" subset.
SAMPLE_INDICES = [0, 128, 256, 384, 511]


def draw_pipeline(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages = [
        ("Noisy LR\n128x128\n(1ch, .npy)", "#cfe8ff"),
        ("Aligned crop +\npaired flip/rotate\n(training only)", "#e6e6e6"),
        ("ResidualSR\n64F / 8B\nEMA weights", "#ffe3b3"),
        ("PixelShuffle x2\nreconstruction", "#d9f2d9"),
        ("Optional x8\ngeometric\nself-ensemble", "#f3d9f7"),
        ("Restored HR\n256x256\n(1ch, .npy)", "#cfe8ff"),
    ]
    box_width, box_height, gap = 1.8, 1.6, 0.55
    x = 0.3
    centers = []
    for label, color in stages:
        rect = plt.Rectangle((x, 0.8), box_width, box_height, facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + box_width / 2, 0.8 + box_height / 2, label, ha="center", va="center", fontsize=9.5)
        centers.append(x + box_width)
        x += box_width + gap

    for i in range(len(stages) - 1):
        ax.annotate(
            "", xy=(centers[i] + gap - 0.05, 0.8 + box_height / 2), xytext=(centers[i], 0.8 + box_height / 2),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
        )

    ax.text(
        6.5, 2.85,
        "Inference pipeline -- ResidualSR (630,724 parameters), champion checkpoint exp23_ema_extended90",
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
    ax.text(
        6.5, 0.35,
        "No normalization/clipping applied to model input or output; x8 self-ensemble is optional (--tta x8), off by default.",
        ha="center", va="center", fontsize=8, style="italic", color="#444444",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_metrics(output_path: Path, metrics: dict) -> None:
    labels = ["Bicubic", "ResidualSR\n(raw)", "ResidualSR\n+ x8 TTA"]
    keys = ["bicubic", "residualsr_raw", "residualsr_x8_tta"]
    psnr_values = [metrics[k]["psnr_db"] for k in keys]
    ssim_values = [metrics[k]["ssim"] for k in keys]
    lpips_values = [metrics[k]["lpips"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    colors = ["#a8c8f0", "#f5b971", "#8fd18f"]

    for ax, values, title, ylabel, higher_is_better in (
        (axes[0], psnr_values, "PSNR (dB)", "dB", True),
        (axes[1], ssim_values, "SSIM", "SSIM", True),
        (axes[2], lpips_values, "LPIPS", "LPIPS", False),
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.8)
        ax.set_title(f"{title} ({'higher is better' if higher_is_better else 'lower is better'})", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:.4f}" if title != "PSNR (dB)" else f"{value:.2f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylim(0, max(values) * 1.2)

    fig.suptitle(
        "Measured validation metrics (640-sample canonical split, seed=42) -- see results/final_metrics.json",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_sample_panels(output_dir: Path) -> list[str]:
    device = select_device(None)
    model, checkpoint = load_model(CHECKPOINT_PATH, device)
    scale = checkpoint["model_config"]["scale"]

    pairs = discover_pairs(discover_layout(configured_data_dir())).pairs
    _, validation_pairs = split_pairs(pairs, val_fraction=0.2, seed=42)

    saved = []
    for sample_index in SAMPLE_INDICES:
        pair = validation_pairs[sample_index]
        lr = load_image_array(pair.input_path)
        gt = load_image_array(pair.target_path)
        bicubic = bicubic_upscale(lr, scale=scale)

        with torch.no_grad():
            tensor = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            restored = model(tensor)[0, 0].cpu().numpy()

        restored_display = np.clip(restored, 0.0, 1.0)
        bicubic_display = np.clip(bicubic, 0.0, 1.0)
        gt_display = np.clip(gt, 0.0, 1.0)
        lr_finite = lr[np.isfinite(lr)]
        lr_vmin, lr_vmax = float(lr_finite.min()), float(lr_finite.max())

        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        panels = (
            ("Noisy LR (128x128)", lr, lr_vmin, lr_vmax),
            ("Bicubic 2x (256x256)", bicubic_display, 0.0, 1.0),
            ("Restored (ResidualSR, 256x256)", restored_display, 0.0, 1.0),
            ("Ground Truth (256x256)", gt_display, 0.0, 1.0),
        )
        for ax, (title, array, vmin, vmax) in zip(axes, panels):
            ax.imshow(array, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9.5)
            ax.axis("off")
        fig.suptitle(
            f"Validation sample #{sample_index} ({pair.pair_id}) -- canonical split, seed=42",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        output_path = output_dir / f"sample_{pair.pair_id}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(output_path))
        print(f"Saved {output_path}")
    return saved


def main() -> None:
    draw_pipeline(ASSETS_DIR / "pipeline.png")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    draw_metrics(ASSETS_DIR / "metrics.png", metrics)

    draw_sample_panels(RESULTS_DIR)


if __name__ == "__main__":
    main()
