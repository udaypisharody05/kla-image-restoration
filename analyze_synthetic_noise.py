"""Experiment 24 gate: does the synthetic degradation actually reproduce the real one?

Analysis only -- trains and evaluates nothing. Two jobs:

1. **Choose the epsilon distribution** (Gaussian vs Student-t) by measuring the
   *standardized* real residual ``(NoisyLR - bicubic(GT)) / sigma(I)`` and
   comparing each candidate's moments and percentiles against it.
2. **Gate the experiment**: synthesize noise for real GT images and check the
   synthetic residuals reproduce Experiment 22's key properties (variance curve,
   moments, percentiles, spatial whiteness). A bad mismatch means Experiment 24
   should not be trained until the model is revised.

Usage::

    python analyze_synthetic_noise.py --max-pairs 400
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from inspect_dataset import configured_data_dir
from src.dataset_discovery import discover_layout, discover_pairs
from src.degradation import autocorrelation_at_offsets, downsample
from src.io_utils import load_image_array
from src.synthetic_noise import (
    STUDENT_T_DEGREES_OF_FREEDOM,
    VARIANCE_COEFFICIENTS,
    noise_sigma,
    sample_epsilon,
)


AUTOCORRELATION_OFFSETS = [(0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]
# Below this intensity the fitted quadratic is clamped/unreliable, so standardized
# statistics there are dominated by division by a near-zero sigma rather than by
# the true shape of epsilon.
STANDARDIZATION_MIN_INTENSITY = 0.05


def moments(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    centered = flat - mean
    return {
        "count": int(flat.size),
        "mean": mean,
        "std": std,
        "skewness": float(np.mean(centered**3) / std**3) if std > 0 else 0.0,
        "excess_kurtosis": float(np.mean(centered**4) / std**4 - 3.0) if std > 0 else 0.0,
        "percentiles": {
            f"p{q}": float(np.percentile(flat, q))
            for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
        },
    }


def binned_std(intensity: np.ndarray, residual: np.ndarray, edges: np.ndarray) -> list[dict]:
    index = np.clip(np.digitize(intensity.ravel(), edges) - 1, 0, len(edges) - 2)
    flat = residual.ravel()
    rows = []
    for i in range(len(edges) - 1):
        mask = index == i
        count = int(mask.sum())
        if count < 1000:
            continue
        rows.append(
            {
                "bin_center": float((edges[i] + edges[i + 1]) / 2),
                "count": count,
                "std": float(flat[mask].std()),
            }
        )
    return rows


def collect(pairs, variance_floor: float, seed: int) -> dict:
    """Gather real residuals and matched synthetic residuals over *pairs*."""
    edges = np.linspace(0.0, 1.0, 41)
    real_parts, standardized_parts = [], []
    synthetic_parts = {"gaussian": [], "student_t": []}
    intensity_parts = []
    real_autocorrelation, synthetic_autocorrelation = [], []
    rng = np.random.default_rng(seed)

    for pair in pairs:
        lr = np.asarray(load_image_array(pair.input_path), dtype=np.float64)
        gt = np.asarray(load_image_array(pair.target_path), dtype=np.float32)
        clean = downsample(gt, "bicubic")
        real_residual = lr - clean

        sigma = noise_sigma(
            torch.from_numpy(clean), VARIANCE_COEFFICIENTS, variance_floor
        ).numpy()

        real_parts.append(real_residual.ravel())
        intensity_parts.append(clean.ravel())
        usable = clean >= STANDARDIZATION_MIN_INTENSITY
        standardized_parts.append(real_residual[usable] / sigma[usable])
        real_autocorrelation.append(autocorrelation_at_offsets(real_residual, AUTOCORRELATION_OFFSETS))

        for distribution in synthetic_parts:
            epsilon = sample_epsilon(clean.shape, distribution, rng, STUDENT_T_DEGREES_OF_FREEDOM)
            synthetic_residual = sigma * epsilon
            synthetic_parts[distribution].append(synthetic_residual.ravel())
            if distribution == "student_t":
                synthetic_autocorrelation.append(
                    autocorrelation_at_offsets(synthetic_residual, AUTOCORRELATION_OFFSETS)
                )

    intensity = np.concatenate(intensity_parts)
    real = np.concatenate(real_parts)
    result = {
        "real": moments(real),
        "real_standardized": moments(np.concatenate(standardized_parts)),
        "real_binned_std": binned_std(intensity, real, edges),
        "real_autocorrelation": {
            key: float(np.mean([row[key] for row in real_autocorrelation]))
            for key in real_autocorrelation[0]
        },
        "synthetic_autocorrelation": {
            key: float(np.mean([row[key] for row in synthetic_autocorrelation]))
            for key in synthetic_autocorrelation[0]
        },
    }
    for distribution, parts in synthetic_parts.items():
        synthetic = np.concatenate(parts)
        result[f"synthetic_{distribution}"] = moments(synthetic)
        result[f"synthetic_{distribution}_binned_std"] = binned_std(intensity, synthetic, edges)
    return result


def distribution_score(target: dict, candidate: dict) -> dict[str, float]:
    """Absolute errors of *candidate* against *target* on the moments that matter."""
    percentile_error = float(
        np.mean(
            [
                abs(candidate["percentiles"][k] - target["percentiles"][k])
                for k in target["percentiles"]
            ]
        )
    )
    return {
        "std_error": abs(candidate["std"] - target["std"]),
        "skewness_error": abs(candidate["skewness"] - target["skewness"]),
        "excess_kurtosis_error": abs(candidate["excess_kurtosis"] - target["excess_kurtosis"]),
        "mean_abs_percentile_error": percentile_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_noise_analysis"))
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--variance-floor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = list(discover_pairs(discover_layout(args.data_dir)).pairs[: args.max_pairs])
    print(f"Comparing real vs synthetic degradation over {len(pairs):,} pairs...")
    measurements = collect(pairs, args.variance_floor, args.seed)

    # Reference epsilon shapes, sampled independently of any image -- reported as
    # supporting evidence for how each candidate compares to the *standardized*
    # residual.
    rng = np.random.default_rng(args.seed + 1)
    reference = {
        name: moments(sample_epsilon((4_000_000,), name, rng, STUDENT_T_DEGREES_OF_FREEDOM))
        for name in ("gaussian", "student_t")
    }
    epsilon_scores = {
        name: distribution_score(measurements["real_standardized"], stats)
        for name, stats in reference.items()
    }

    # THE operative criterion: how closely does each candidate's *synthetic
    # residual* reproduce the *real residual*? That end-to-end comparison is what
    # the augmentation actually has to get right, and it differs from the
    # standardized view because heteroscedastic mixing contributes excess
    # kurtosis on its own, before epsilon's own tails are considered.
    scores = {
        name: distribution_score(measurements["real"], measurements[f"synthetic_{name}"])
        for name in ("gaussian", "student_t")
    }
    chosen = min(
        scores,
        key=lambda name: (
            scores[name]["excess_kurtosis_error"] + 100.0 * scores[name]["mean_abs_percentile_error"]
        ),
    )

    variance_curve = []
    for real_row, synthetic_row in zip(
        measurements["real_binned_std"], measurements[f"synthetic_{chosen}_binned_std"]
    ):
        ratio = synthetic_row["std"] / real_row["std"] if real_row["std"] > 0 else float("nan")
        variance_curve.append(
            {
                "bin_center": real_row["bin_center"],
                "real_std": real_row["std"],
                "synthetic_std": synthetic_row["std"],
                "ratio": float(ratio),
            }
        )
    ratios = np.array([row["ratio"] for row in variance_curve])
    worst_index = int(np.argmax(np.abs(ratios - 1.0)))

    results = {
        "experiment": "Experiment 24 -- synthetic noise gate",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Analysis only. No model was trained, loaded, or evaluated.",
        "pairs_analyzed": len(pairs),
        "variance_floor": args.variance_floor,
        "variance_coefficients": list(VARIANCE_COEFFICIENTS),
        "student_t_degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
        "standardization_min_intensity": STANDARDIZATION_MIN_INTENSITY,
        "measurements": measurements,
        "epsilon_reference": reference,
        "epsilon_scores_vs_standardized_residual": epsilon_scores,
        "distribution_scores_vs_real_residual": scores,
        "chosen_distribution": chosen,
        "selection_criterion": (
            "minimize |excess kurtosis error| + 100 * mean |percentile error| of the "
            "synthetic residual against the real residual"
        ),
        "variance_curve": variance_curve,
        "variance_curve_summary": {
            "median_ratio": float(np.median(ratios)),
            "min_ratio": float(ratios.min()),
            "max_ratio": float(ratios.max()),
            "worst_bin_center": variance_curve[worst_index]["bin_center"],
            "worst_ratio": variance_curve[worst_index]["ratio"],
            "bins_within_10_percent": int(np.sum(np.abs(ratios - 1.0) <= 0.10)),
            "bins_total": int(len(ratios)),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "synthetic_noise_report.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    summary = results["variance_curve_summary"]
    standardized = measurements["real_standardized"]
    print("\nSupporting view -- measured STANDARDIZED residual vs raw epsilon shapes:")
    print(
        f"  real standardized  std {standardized['std']:.4f}  "
        f"skew {standardized['skewness']:+.4f}  exkurt {standardized['excess_kurtosis']:+.4f}"
    )
    for name, stats in reference.items():
        print(
            f"  {name:10s}         std {stats['std']:.4f}  skew {stats['skewness']:+.4f}  "
            f"exkurt {stats['excess_kurtosis']:+.4f}  "
            f"(|kurt err| {epsilon_scores[name]['excess_kurtosis_error']:.4f})"
        )
    print("\nOperative criterion -- SYNTHETIC residual vs REAL residual:")
    real = measurements["real"]
    print(f"  {'quantity':18s} {'real':>10} {'gaussian':>10} {'student_t':>10}")
    for key in ("std", "skewness", "excess_kurtosis"):
        print(
            f"  {key:18s} {real[key]:>10.4f} "
            f"{measurements['synthetic_gaussian'][key]:>10.4f} "
            f"{measurements['synthetic_student_t'][key]:>10.4f}"
        )
    for name in ("gaussian", "student_t"):
        print(
            f"  {name:10s} -> |kurt err| {scores[name]['excess_kurtosis_error']:.4f}, "
            f"mean |pct err| {scores[name]['mean_abs_percentile_error']:.5f}"
        )
    print(f"  CHOSEN: {chosen}")
    print(
        f"\nVariance curve: median ratio {summary['median_ratio']:.4f}, "
        f"{summary['bins_within_10_percent']}/{summary['bins_total']} bins within 10%, "
        f"worst {summary['worst_ratio']:.3f} at I={summary['worst_bin_center']:.3f}"
    )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
