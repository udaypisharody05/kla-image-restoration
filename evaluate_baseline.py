"""Evaluate the deterministic 2x bicubic validation baseline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from inspect_dataset import configured_data_dir
from src.baseline import (
    LPIPSMetric,
    LPIPSUnavailableError,
    bicubic_upscale,
    evaluate_metrics,
)
from src.dataset_discovery import discover_layout, discover_pairs
from src.io_utils import load_image_array
from src.splits import split_pairs


def evaluate_validation_baseline(
    data_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    clip_prediction: bool = True,
    enable_lpips: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run bicubic interpolation and metrics on the deterministic validation set."""
    discovered = discover_pairs(discover_layout(data_dir))
    train_pairs, validation_pairs = split_pairs(
        discovered.pairs, val_fraction=val_fraction, seed=seed
    )
    evaluated_pairs = validation_pairs[:max_samples] if max_samples else validation_pairs
    if not evaluated_pairs:
        raise RuntimeError("Validation split contains no samples to evaluate")

    lpips_metric = None
    lpips_status = "not requested"
    if enable_lpips:
        try:
            lpips_metric = LPIPSMetric()
            lpips_status = "measured"
        except LPIPSUnavailableError as exc:
            lpips_status = f"unavailable: {exc}"

    metric_values: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    interpolation_seconds = 0.0
    for pair in evaluated_pairs:
        degraded = load_image_array(pair.input_path)
        target = load_image_array(pair.target_path)
        started = perf_counter()
        prediction = bicubic_upscale(degraded, output_shape=target.shape[:2])
        interpolation_seconds += perf_counter() - started
        metrics = evaluate_metrics(
            prediction,
            target,
            clip_prediction=clip_prediction,
            lpips_metric=lpips_metric,
        )
        for name, value in metrics.items():
            if value is not None:
                metric_values[name].append(value)

    evaluated_count = len(evaluated_pairs)
    mean_seconds = interpolation_seconds / evaluated_count
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Pillow float32 bicubic",
        "scale": 2,
        "seed": seed,
        "validation_fraction": val_fraction,
        "train_samples": len(train_pairs),
        "validation_samples": len(validation_pairs),
        "evaluated_samples": evaluated_count,
        "clip_prediction_for_metrics": clip_prediction,
        "metric_data_range": 1.0,
        "psnr_db": float(np.mean(metric_values["psnr"])),
        "ssim": float(np.mean(metric_values["ssim"])),
        "lpips": (
            float(np.mean(metric_values["lpips"])) if metric_values["lpips"] else None
        ),
        "lpips_status": lpips_status,
        "total_interpolation_seconds": interpolation_seconds,
        "mean_interpolation_ms": mean_seconds * 1000.0,
        "interpolation_images_per_second": 1.0 / mean_seconds,
        "timing_scope": "bicubic interpolation only; excludes loading and metrics",
    }


def format_summary(result: dict[str, Any]) -> str:
    lpips_value = "N/A" if result["lpips"] is None else f"{result['lpips']:.6f}"
    return "\n".join(
        [
            "Bicubic 2x Validation Baseline",
            "-------------------------------",
            f"Seed:                 {result['seed']}",
            f"Validation split:     {result['validation_fraction']:.1%}",
            f"Train samples:        {result['train_samples']}",
            f"Validation samples:   {result['validation_samples']}",
            f"Evaluated samples:    {result['evaluated_samples']}",
            f"PSNR:                 {result['psnr_db']:.4f} dB",
            f"SSIM:                 {result['ssim']:.6f}",
            f"LPIPS:                {lpips_value} ({result['lpips_status']})",
            f"Interpolation total:  {result['total_interpolation_seconds']:.4f} s",
            f"Mean interpolation:   {result['mean_interpolation_ms']:.3f} ms/image",
            f"Throughput:           {result['interpolation_images_per_second']:.2f} images/s",
            f"Prediction clipping:  {result['clip_prediction_for_metrics']} (metrics only)",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/bicubic_baseline.json"))
    parser.add_argument("--max-samples", type=int, help="Evaluate only the first N validation samples")
    parser.add_argument("--no-clip-prediction", action="store_true")
    parser.add_argument("--lpips", action="store_true", help="Attempt optional LPIPS evaluation")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    result = evaluate_validation_baseline(
        args.data_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        clip_prediction=not args.no_clip_prediction,
        enable_lpips=args.lpips,
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_summary(result))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
