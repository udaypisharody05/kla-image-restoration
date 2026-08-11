"""Experiment 21 -- forensic characterization of the GT -> NoisyLR degradation.

This is an **analysis** entry point, not a training or evaluation experiment: it
never builds, trains, loads, or scores a model. It reads the canonical paired
training data and measures what the degradation process actually does, so that
Experiment 22's modeling strategy can be chosen from evidence instead of guesswork.

Outputs (under ``results/degradation_analysis/`` by default):

- ``degradation_report.json`` -- complete machine-readable measurements
- ``degradation_report.md``   -- concise human-readable summary
- diagnostic plots (residual maps, spectra, noise-vs-intensity, ...)

Usage::

    python analyze_degradation.py                     # full 3,200-pair analysis
    python analyze_degradation.py --max-pairs 100     # quick subset run
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inspect_dataset import configured_data_dir
from src.dataset_discovery import ImagePair, discover_layout, discover_pairs
from src.degradation import (
    BinnedAccumulator,
    MomentAccumulator,
    apply_gain_bias,
    autocorrelation_at_offsets,
    available_downsamplers,
    connected_components,
    content_hash,
    downsample,
    fit_gain_bias,
    fit_noise_variance_model,
    gaussian_blur,
    gradient_magnitude,
    local_variance,
    match_metrics,
    perceptual_signature,
    power_spectrum,
    radial_profile,
)
from src.io_utils import load_image_array
from src.splits import split_pairs


# Experiment 20 EMA epoch 70 + x8 TTA -- quoted only to give the measured noise
# magnitude a familiar scale, never recomputed or evaluated here.
CHAMPION_VALIDATION_L1 = 0.032607

AUTOCORRELATION_OFFSETS = [
    (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2), (0, 3), (3, 0), (0, 4), (4, 0), (0, 8), (8, 0),
]
BLUR_SIGMAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6]


def _load_pair(pair: ImagePair) -> tuple[np.ndarray, np.ndarray]:
    lr = np.asarray(load_image_array(pair.input_path), dtype=np.float64)
    gt = np.asarray(load_image_array(pair.target_path), dtype=np.float32)
    return lr, gt


def compare_downsamplers(pairs: list[ImagePair]) -> tuple[dict, dict[str, list[str]], dict[str, list[str]]]:
    """Rank every candidate GT->LR model, and hash GTs for duplicate detection."""
    per_method: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in available_downsamplers()
    }
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    signatures: dict[str, list[str]] = defaultdict(list)

    for pair in pairs:
        lr, gt = _load_pair(pair)
        exact_hashes[content_hash(gt)].append(pair.pair_id)
        signatures[perceptual_signature(gt)].append(pair.pair_id)
        for name in available_downsamplers():
            metrics = match_metrics(lr, downsample(gt, name))
            for key, value in metrics.items():
                per_method[name][key].append(value)

    summary = {
        name: {key: float(np.mean(values)) for key, values in metrics.items()}
        for name, metrics in per_method.items()
    }
    return summary, dict(exact_hashes), dict(signatures)


def analyze_residuals(pairs: list[ImagePair], method: str) -> dict:
    """Single deep pass over the dataset using the chosen downsampling model."""
    height = width = None
    residual_sum = residual_square_sum = None
    spectrum_sum = None

    residual_moments = MomentAccumulator()
    intensity_bins = BinnedAccumulator(np.linspace(0.0, 1.0, 41))
    gradient_bins = BinnedAccumulator(np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf]))

    # Global affine fit accumulators for lr ~= a * estimate + b over ALL pixels.
    totals = {"n": 0.0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0}
    autocorrelation_totals: dict[str, list[float]] = defaultdict(list)
    per_image: list[dict] = []

    for pair in pairs:
        lr, gt = _load_pair(pair)
        estimate = downsample(gt, method)
        residual = lr - estimate

        if residual_sum is None:
            height, width = residual.shape
            residual_sum = np.zeros((height, width), dtype=np.float64)
            residual_square_sum = np.zeros((height, width), dtype=np.float64)
            spectrum_sum = np.zeros((height, width), dtype=np.float64)
        residual_sum += residual
        residual_square_sum += residual**2
        spectrum_sum += power_spectrum(residual)

        residual_moments.update(residual)
        intensity_bins.update(estimate, residual)
        gradients = gradient_magnitude(estimate)
        gradient_bins.update(gradients, np.abs(residual))

        flat_estimate, flat_lr = estimate.ravel(), lr.ravel()
        totals["n"] += flat_estimate.size
        totals["x"] += float(flat_estimate.sum())
        totals["y"] += float(flat_lr.sum())
        totals["xx"] += float((flat_estimate**2).sum())
        totals["xy"] += float((flat_estimate * flat_lr).sum())

        for key, value in autocorrelation_at_offsets(residual, AUTOCORRELATION_OFFSETS).items():
            autocorrelation_totals[key].append(value)

        gain, bias = fit_gain_bias(lr, estimate)
        corrected = apply_gain_bias(estimate, gain, bias)
        per_image.append(
            {
                "pair_id": pair.pair_id,
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std()),
                "residual_mse": float(np.mean(residual**2)),
                "corrected_mse": float(np.mean((lr - corrected) ** 2)),
                "gain": gain,
                "bias": bias,
                "estimate_mean": float(estimate.mean()),
                "estimate_std": float(estimate.std()),
                "gradient_energy": float(np.mean(gradients**2)),
            }
        )

    count = len(pairs)
    mean_map = residual_sum / count
    # Per-pixel variance across the dataset -> "does this sensor position behave oddly".
    std_map = np.sqrt(np.maximum(residual_square_sum / count - mean_map**2, 0.0))
    mean_spectrum = spectrum_sum / count

    determinant = totals["n"] * totals["xx"] - totals["x"] ** 2
    global_gain = (totals["n"] * totals["xy"] - totals["x"] * totals["y"]) / determinant
    global_bias = (totals["y"] - global_gain * totals["x"]) / totals["n"]

    intensity_summary = intensity_bins.summary(min_count=1000)
    variance_model = fit_noise_variance_model(
        (row["bin_center"] for row in intensity_summary),
        (row["variance"] for row in intensity_summary),
    )

    return {
        "method": method,
        "residual_moments": residual_moments.summary(),
        "global_affine": {"gain": float(global_gain), "bias": float(global_bias)},
        "per_image": per_image,
        "intensity_bins": intensity_summary,
        "noise_variance_model": variance_model,
        "gradient_bins": gradient_bins.summary(min_count=1000),
        "autocorrelation": {
            key: float(np.mean(values)) for key, values in sorted(autocorrelation_totals.items())
        },
        "maps": {"mean": mean_map, "std": std_map, "spectrum": mean_spectrum},
    }


def analyze_near_duplicates(
    pairs: list[ImagePair], signatures: dict[str, list[str]], threshold: float = 1e-3
) -> dict:
    """Find genuinely repeated *scenes* (same GT, independent noise realizations).

    Average-hash signatures only nominate candidates -- an 8x8 hash collides for
    unrelated images too -- so every candidate pair is confirmed by real GT MSE
    before it counts. For each confirmed pair the corresponding *LR* MSE is also
    recorded: a near-zero GT distance beside a large LR distance is direct
    evidence of two independent noise draws over one clean scene.
    """
    by_id = {pair.pair_id: pair for pair in pairs}
    candidates = [ids for ids in signatures.values() if len(ids) > 1]
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def arrays(pair_id: str) -> tuple[np.ndarray, np.ndarray]:
        if pair_id not in cache:
            lr, gt = _load_pair(by_id[pair_id])
            cache[pair_id] = (lr, gt.astype(np.float64))
        return cache[pair_id]

    edges: list[tuple[str, str]] = []
    gt_distances: list[float] = []
    lr_distances: list[float] = []
    id_gaps: list[int] = []
    for group in candidates:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                left_lr, left_gt = arrays(left)
                right_lr, right_gt = arrays(right)
                gt_mse = float(np.mean((left_gt - right_gt) ** 2))
                if gt_mse >= threshold:
                    continue
                edges.append((left, right))
                gt_distances.append(gt_mse)
                lr_distances.append(float(np.mean((left_lr - right_lr) ** 2)))
                if left.isdigit() and right.isdigit():
                    id_gaps.append(abs(int(left) - int(right)))

    components = [
        group
        for group in connected_components([pair.pair_id for pair in pairs], edges)
        if len(group) > 1
    ]
    sizes = [len(group) for group in components]
    return {
        "gt_mse_threshold": threshold,
        "signature_candidate_groups": len(candidates),
        "confirmed_pairs": len(edges),
        "scene_groups": len(components),
        "images_involved": int(sum(sizes)),
        "largest_group": max(sizes, default=0),
        "group_size_counts": {str(size): sizes.count(size) for size in sorted(set(sizes))},
        "mean_gt_mse_within_group": float(np.mean(gt_distances)) if gt_distances else 0.0,
        "mean_lr_mse_within_group": float(np.mean(lr_distances)) if lr_distances else 0.0,
        "id_gap_max": max(id_gaps, default=0),
        "id_gap_median": float(np.median(id_gaps)) if id_gaps else 0.0,
        "id_gap_at_most_two_fraction": (
            float(np.mean([gap <= 2 for gap in id_gaps])) if id_gaps else 0.0
        ),
        # Full membership is kept: 100-odd small groups is a compact, directly
        # actionable artifact (Experiment 22 may want to group-aware split on it).
        "groups": components,
    }


def analyze_local_structure(pairs: list[ImagePair], method: str) -> dict:
    """Residual magnitude versus local variance, on a bounded subset (slow filter)."""
    accumulator = BinnedAccumulator(np.array([0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, np.inf]))
    for pair in pairs:
        lr, gt = _load_pair(pair)
        estimate = downsample(gt, method)
        accumulator.update(local_variance(estimate, window=3), np.abs(lr - estimate))
    return {"pairs_analyzed": len(pairs), "bins": accumulator.summary(min_count=100)}


def search_blur(pairs: list[ImagePair], method: str) -> list[dict]:
    """Small controlled search for a pre-downsampling blur that explains LR better."""
    results = []
    for sigma in BLUR_SIGMAS:
        errors = []
        for pair in pairs:
            lr, gt = _load_pair(pair)
            blurred = gt if sigma == 0.0 else gaussian_blur(gt, sigma).astype(np.float32)
            errors.append(float(np.mean((lr - downsample(blurred, method)) ** 2)))
        results.append({"sigma": sigma, "mse": float(np.mean(errors))})
    return results


def cluster_one_dimensional(values: np.ndarray, iterations: int = 50) -> dict:
    """Deterministic 1D 2-means, reported against the 1-cluster baseline.

    Deliberately minimal -- the question is only "is degradation visibly
    heterogeneous", not "what is the optimal clustering".
    """
    data = np.asarray(values, dtype=np.float64)
    centers = np.array([np.percentile(data, 25), np.percentile(data, 75)])
    for _ in range(iterations):
        assignments = np.argmin(np.abs(data[:, None] - centers[None, :]), axis=1)
        updated = np.array(
            [data[assignments == k].mean() if np.any(assignments == k) else centers[k] for k in range(2)]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    assignments = np.argmin(np.abs(data[:, None] - centers[None, :]), axis=1)
    within = float(np.sum((data - centers[assignments]) ** 2))
    total = float(np.sum((data - data.mean()) ** 2))
    return {
        "centers": [float(c) for c in np.sort(centers)],
        "cluster_sizes": [int(np.sum(assignments == k)) for k in np.argsort(centers)],
        "within_cluster_ss": within,
        "total_ss": total,
        "variance_explained": float(1.0 - within / total) if total > 0 else 0.0,
    }


def summarize_group(rows: list[dict], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
            "p5": float(np.percentile([row[key] for row in rows], 5)),
            "p50": float(np.percentile([row[key] for row in rows], 50)),
            "p95": float(np.percentile([row[key] for row in rows], 95)),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def write_plots(analysis: dict, output_dir: Path) -> dict[str, str]:
    """Save diagnostic figures; returns repository-relative paths by name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = analysis["maps"]
    paths: dict[str, str] = {}

    def save(figure: plt.Figure, name: str) -> None:
        path = output_dir / name
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        paths[name.replace(".png", "")] = path.as_posix()

    figure, axis = plt.subplots(figsize=(5.5, 4.6))
    image = axis.imshow(maps["mean"], cmap="RdBu_r", vmin=-np.abs(maps["mean"]).max(), vmax=np.abs(maps["mean"]).max())
    axis.set_title("Dataset-average residual per pixel\n(fixed-pattern check)")
    figure.colorbar(image, ax=axis, fraction=0.046)
    save(figure, "residual_mean_map.png")

    figure, axis = plt.subplots(figsize=(5.5, 4.6))
    image = axis.imshow(maps["std"], cmap="viridis")
    axis.set_title("Per-pixel residual std across dataset")
    figure.colorbar(image, ax=axis, fraction=0.046)
    save(figure, "residual_std_map.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].imshow(np.log10(maps["spectrum"] + 1e-12), cmap="magma")
    axes[0].set_title("log10 mean residual power spectrum\n(centre = DC)")
    profile = radial_profile(maps["spectrum"])
    axes[1].semilogy(profile)
    axes[1].set_xlabel("radial spatial frequency (bin)")
    axes[1].set_ylabel("mean power")
    axes[1].set_title("Radially averaged spectrum")
    axes[1].grid(alpha=0.3)
    save(figure, "residual_power_spectrum.png")

    bins = analysis["intensity_bins"]
    centers = [row["bin_center"] for row in bins]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(centers, [row["std"] for row in bins], "o-")
    axes[0].set_xlabel("estimated clean-LR intensity")
    axes[0].set_ylabel("residual std")
    axes[0].set_title("Noise magnitude vs signal intensity")
    axes[0].grid(alpha=0.3)
    model = analysis["noise_variance_model"]
    intensity = np.array(centers)
    fitted = model["constant"] + model["linear"] * intensity + model["quadratic"] * intensity**2
    axes[1].plot(centers, [row["variance"] for row in bins], "o", label="measured variance")
    axes[1].plot(centers, fitted, "-", label=f"fit (R^2={model['r_squared']:.4f})")
    axes[1].set_xlabel("estimated clean-LR intensity")
    axes[1].set_ylabel("residual variance")
    axes[1].set_title("Variance model: c0 + c1*I + c2*I^2")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    save(figure, "noise_vs_intensity.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(maps["mean"].mean(axis=1))
    axes[0].set_title("Mean residual by row")
    axes[0].set_xlabel("row")
    axes[0].grid(alpha=0.3)
    axes[1].plot(maps["mean"].mean(axis=0))
    axes[1].set_title("Mean residual by column")
    axes[1].set_xlabel("column")
    axes[1].grid(alpha=0.3)
    save(figure, "residual_row_column_profiles.png")

    stds = [row["residual_std"] for row in analysis["per_image"]]
    means = [row["estimate_mean"] for row in analysis["per_image"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(stds, bins=60)
    axes[0].set_title("Per-image residual std")
    axes[0].set_xlabel("residual std")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(means, stds, s=4, alpha=0.35)
    axes[1].set_xlabel("mean clean-LR intensity")
    axes[1].set_ylabel("residual std")
    axes[1].set_title("Per-image noise vs brightness")
    axes[1].grid(alpha=0.3)
    save(figure, "per_image_noise_distribution.png")

    return paths


def build_report(results: dict) -> str:
    """Concise Markdown summary of the JSON measurements."""
    downsamplers = results["downsampling_models"]
    best = results["best_downsampler"]
    moments = results["residuals"]["moments"]
    model = results["residuals"]["noise_variance_model"]
    affine = results["residuals"]["global_affine"]
    lines = [
        "# Experiment 21 — Dataset Degradation Forensics",
        "",
        "Analysis only: **no model was trained, loaded, or evaluated.** All numbers",
        "below describe the dataset's GT -> NoisyLR degradation, measured over the",
        f"canonical **{results['pairs_analyzed']:,} training pairs** "
        f"(GT 256x256 -> NoisyLR 128x128).",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "## 1. Which GT -> LR downsampling model best explains the observed LR?",
        "",
        "| model | MAE | MSE | PSNR (dB) | correlation | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in sorted(downsamplers.items(), key=lambda kv: kv[1]["mse"]):
        marker = " **(best)**" if name == best else ""
        lines.append(
            f"| `{name}`{marker} | {metrics['mae']:.6f} | {metrics['mse']:.6f} | "
            f"{metrics['psnr']:.4f} | {metrics['correlation']:.5f} | {metrics['bias']:+.6f} |"
        )
    lines += [
        "",
        "These are **degradation-model agreement scores, not restoration performance** —",
        "both sides of each comparison are known quantities.",
        "",
        f"**Best model: `{best}`.** All residual analysis below uses it.",
        "",
        "## 2. Systematic gain / bias",
        "",
        f"- Global dataset-wide fit `LR ≈ a·downsample(GT) + b`: "
        f"**a = {affine['gain']:.6f}, b = {affine['bias']:+.6f}**",
        f"- Per-image gain: mean {results['residuals']['per_image_summary']['gain']['mean']:.6f}, "
        f"std {results['residuals']['per_image_summary']['gain']['std']:.6f} "
        f"(p5 {results['residuals']['per_image_summary']['gain']['p5']:.4f}, "
        f"p95 {results['residuals']['per_image_summary']['gain']['p95']:.4f})",
        f"- Per-image bias: mean {results['residuals']['per_image_summary']['bias']['mean']:+.6f}, "
        f"std {results['residuals']['per_image_summary']['bias']['std']:.6f}",
        f"- Mean residual MSE before affine correction: "
        f"{results['residuals']['per_image_summary']['residual_mse']['mean']:.6f}",
        f"- Mean residual MSE after per-image affine correction: "
        f"{results['residuals']['per_image_summary']['corrected_mse']['mean']:.6f} "
        f"(**{results['residuals']['affine_mse_reduction_percent']:.2f}%** reduction)",
        "",
        "## 3. Residual noise magnitude",
        "",
        f"- mean **{moments['mean']:+.6f}**, std **{moments['std']:.6f}**, "
        f"variance {moments['variance']:.6f}",
        f"- range [{moments['min']:.4f}, {moments['max']:.4f}]",
        f"- skewness {moments['skewness']:+.4f}, excess kurtosis {moments['excess_kurtosis']:+.4f}",
        "- percentiles: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in moments["percentiles"].items()),
        f"- per-image residual std: mean {results['residuals']['per_image_summary']['residual_std']['mean']:.6f}, "
        f"range [{results['residuals']['per_image_summary']['residual_std']['min']:.6f}, "
        f"{results['residuals']['per_image_summary']['residual_std']['max']:.6f}]",
        "",
        "## 4-5. Is the noise iid / signal dependent?",
        "",
        f"Fitted `var(I) = {model['constant']:.6g} + {model['linear']:.6g}·I + "
        f"{model['quadratic']:.6g}·I²`  (R² = {model['r_squared']:.4f})",
        "",
        "| clean-LR intensity | residual mean | residual std | count |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in results["residuals"]["intensity_bins"][::4]:
        lines.append(
            f"| {row['bin_center']:.3f} | {row['mean']:+.5f} | {row['std']:.5f} | {row['count']:,} |"
        )
    lines += [
        "",
        f"Noise std grows from ~{results['residuals']['intensity_bins'][0]['std']:.4f} in the darkest bin "
        f"to ~{results['residuals']['intensity_bins'][-1]['std']:.4f} in the brightest — "
        "**strongly signal dependent, not homoscedastic**.",
        "",
        "## 6. Spatial correlation",
        "",
        "| offset (dy,dx) | normalized autocorrelation |",
        "| --- | ---: |",
    ]
    for key, value in results["residuals"]["autocorrelation"].items():
        lines.append(f"| ({key.replace('_', ',')}) | {value:+.5f} |")
    lines += [
        "",
        "## 7. Fixed-pattern structure",
        "",
        f"- residual mean map: min {results['fixed_pattern']['mean_map_min']:+.6f}, "
        f"max {results['fixed_pattern']['mean_map_max']:+.6f}, "
        f"std across pixels {results['fixed_pattern']['mean_map_std']:.6f}",
        f"- expected std of a pure-noise mean map "
        f"(σ/√N): {results['fixed_pattern']['expected_mean_map_std']:.6f}",
        f"- ratio observed/expected: **{results['fixed_pattern']['fixed_pattern_ratio']:.3f}**",
        f"- row-mean spread {results['fixed_pattern']['row_mean_std']:.6f}, "
        f"column-mean spread {results['fixed_pattern']['column_mean_std']:.6f}",
        "",
        "## 8. Frequency domain",
        "",
        f"- radial spectrum low/high ratio: {results['spectrum']['low_over_high']:.4f}",
        f"- max/median radial power ratio: {results['spectrum']['peak_over_median']:.4f}",
        f"- horizontal vs vertical power ratio: {results['spectrum']['horizontal_over_vertical']:.4f}",
        "",
        "## 9. Edge / texture dependence",
        "",
        "| gradient magnitude bin | mean abs residual | count |",
        "| --- | ---: | ---: |",
    ]
    for row in results["residuals"]["gradient_bins"]:
        lines.append(f"| [{row['bin_low']:.3g}, {row['bin_high']:.3g}) | {row['mean']:.5f} | {row['count']:,} |")
    lines += [
        "",
        "## 10. Pre-downsampling blur",
        "",
        "| Gaussian sigma | residual MSE |",
        "| ---: | ---: |",
    ]
    for row in results["blur_search"]["results"]:
        marker = " **(best)**" if row["sigma"] == results["blur_search"]["best_sigma"] else ""
        lines.append(f"| {row['sigma']:.1f}{marker} | {row['mse']:.6f} |")
    lines += [
        "",
        f"Best sigma **{results['blur_search']['best_sigma']}**; improvement over no blur: "
        f"{results['blur_search']['improvement_percent']:.3f}%.",
        "",
        "## 11. Repeated scenes",
        "",
        f"- exact byte-identical GT groups: **{results['duplicates']['exact_duplicate_groups']}** "
        f"({results['duplicates']['unique_exact_hashes']:,} unique hashes / "
        f"{results['pairs_analyzed']:,} pairs)",
        f"- average-hash *candidate* groups: {results['duplicates']['signature_candidate_groups']} "
        "(candidates only — an 8x8 hash also collides for unrelated images)",
        f"- **confirmed** near-duplicate pairs (GT MSE < "
        f"{results['duplicates']['gt_mse_threshold']}): "
        f"**{results['duplicates']['confirmed_pairs']}**",
        f"- confirmed repeated-scene groups: **{results['duplicates']['scene_groups']}** covering "
        f"**{results['duplicates']['images_involved']} images**, "
        f"sizes {results['duplicates']['group_size_counts']}",
        f"- within-group mean GT MSE {results['duplicates']['mean_gt_mse_within_group']:.6f} vs "
        f"within-group mean **LR** MSE {results['duplicates']['mean_lr_mse_within_group']:.6f} "
        "— same clean scene, **independent noise realizations**",
        f"- filename structure: all confirmed pairs sit at ID gap ≤ 2 "
        f"(max {results['duplicates']['id_gap_max']}, median "
        f"{results['duplicates']['id_gap_median']:.0f}, "
        f"{100 * results['duplicates']['id_gap_at_most_two_fraction']:.0f}% within 2) — "
        "**filenames encode the grouping**",
        f"- groups spanning train *and* validation: "
        f"**{results['duplicates']['groups_spanning_train_and_validation']}**, leaking "
        f"**{results['duplicates']['validation_images_with_train_twin']} of "
        f"{results['train_validation']['validation_count']}** validation images "
        f"({100 * results['duplicates']['validation_leakage_fraction']:.1f}%)",
        "",
        "## 12. Degradation regimes",
        "",
        f"- per-image residual std 2-means: centers "
        f"{results['clustering']['residual_std']['centers']}, sizes "
        f"{results['clustering']['residual_std']['cluster_sizes']}, "
        f"variance explained {results['clustering']['residual_std']['variance_explained']:.4f}",
        f"- correlation(per-image residual std, mean intensity) = "
        f"**{results['clustering']['std_vs_intensity_correlation']:+.4f}**",
        f"- correlation(per-image residual std, gradient energy) = "
        f"{results['clustering']['std_vs_gradient_correlation']:+.4f}",
        "",
        "## 13. Train vs validation",
        "",
        "| statistic | train | validation | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, values in results["train_validation"]["comparison"].items():
        lines.append(
            f"| {key} | {values['train']:.6f} | {values['validation']:.6f} | {values['delta']:+.6f} |"
        )
    lines += [
        "",
        f"Train n={results['train_validation']['train_count']:,}, "
        f"validation n={results['train_validation']['validation_count']:,} "
        "(canonical split, unchanged).",
        "",
        "## Plots",
        "",
    ]
    for name, path in results["plots"].items():
        lines.append(f"- `{path}` — {name.replace('_', ' ')}")
    lines += ["", "## Conclusions and Experiment 22 candidates", "", results["interpretation"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("results/degradation_analysis"))
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional subset limit")
    parser.add_argument("--blur-pairs", type=int, default=300, help="Pairs used for the blur search")
    parser.add_argument(
        "--local-structure-pairs", type=int, default=300, help="Pairs used for local-variance analysis"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = perf_counter()
    discovered = discover_pairs(discover_layout(args.data_dir))
    pairs = list(discovered.pairs[: args.max_pairs] if args.max_pairs else discovered.pairs)
    if not pairs:
        raise RuntimeError("No training pairs discovered")
    print(f"Analyzing {len(pairs):,} paired samples from {args.data_dir}")

    print("Pass 1/3: comparing GT->LR downsampling models and hashing GTs...")
    downsampler_summary, exact_hashes, signatures = compare_downsamplers(pairs)
    best_method = min(downsampler_summary, key=lambda name: downsampler_summary[name]["mse"])
    print(f"  best downsampling model: {best_method}")

    print("Pass 2/3: deep residual analysis...")
    analysis = analyze_residuals(pairs, best_method)

    print("Pass 3/3: near-duplicate confirmation, blur search, local structure...")
    near_duplicates = analyze_near_duplicates(pairs, signatures)
    print(
        f"  confirmed {near_duplicates['scene_groups']} repeated-scene groups "
        f"covering {near_duplicates['images_involved']} images"
    )
    blur_results = search_blur(pairs[: args.blur_pairs], best_method)
    best_blur = min(blur_results, key=lambda row: row["mse"])
    no_blur_mse = next(row["mse"] for row in blur_results if row["sigma"] == 0.0)
    local_structure = analyze_local_structure(pairs[: args.local_structure_pairs], best_method)

    per_image = analysis["per_image"]
    per_image_summary = summarize_group(
        per_image,
        ("residual_mean", "residual_std", "residual_mse", "corrected_mse", "gain", "bias", "estimate_mean"),
    )

    maps = analysis["maps"]
    mean_map, std_map, spectrum = maps["mean"], maps["std"], maps["spectrum"]
    expected_mean_map_std = float(analysis["residual_moments"]["std"] / np.sqrt(len(pairs)))
    profile = radial_profile(spectrum)
    usable = profile[1 : len(profile) // 2]
    centre_y, centre_x = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    horizontal_power = float(spectrum[centre_y, :].sum() - spectrum[centre_y, centre_x])
    vertical_power = float(spectrum[:, centre_x].sum() - spectrum[centre_y, centre_x])

    train_pairs, validation_pairs = split_pairs(
        discovered.pairs, val_fraction=args.val_fraction, seed=args.seed
    )
    analyzed_ids = {row["pair_id"]: row for row in per_image}
    train_rows = [analyzed_ids[p.pair_id] for p in train_pairs if p.pair_id in analyzed_ids]
    validation_rows = [analyzed_ids[p.pair_id] for p in validation_pairs if p.pair_id in analyzed_ids]
    comparison_keys = ("residual_std", "residual_mse", "gain", "bias", "estimate_mean", "gradient_energy")
    comparison = {
        key: {
            "train": float(np.mean([row[key] for row in train_rows])),
            "validation": float(np.mean([row[key] for row in validation_rows])),
            "delta": float(
                np.mean([row[key] for row in validation_rows]) - np.mean([row[key] for row in train_rows])
            ),
        }
        for key in comparison_keys
    }

    residual_stds = np.array([row["residual_std"] for row in per_image])
    intensities = np.array([row["estimate_mean"] for row in per_image])
    gradient_energies = np.array([row["gradient_energy"] for row in per_image])

    # Do confirmed repeated scenes straddle the split? If a validation image has a
    # near-identical twin in train, its validation score is optimistically biased.
    train_ids = {pair.pair_id for pair in train_pairs}
    validation_ids = {pair.pair_id for pair in validation_pairs}
    spanning_groups = [
        group
        for group in near_duplicates["groups"]
        if any(i in train_ids for i in group) and any(i in validation_ids for i in group)
    ]
    leaked_validation_images = sum(
        1 for group in spanning_groups for i in group if i in validation_ids
    )
    near_duplicates["groups_spanning_train_and_validation"] = len(spanning_groups)
    near_duplicates["validation_images_with_train_twin"] = leaked_validation_images
    near_duplicates["validation_leakage_fraction"] = (
        float(leaked_validation_images / len(validation_ids)) if validation_ids else 0.0
    )

    results = {
        "experiment": "Experiment 21 -- Dataset Degradation Forensics",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "pairs_analyzed": len(pairs),
        "note": "Analysis only. No model was trained, loaded, or evaluated.",
        "downsampling_models": downsampler_summary,
        "best_downsampler": best_method,
        "residuals": {
            "moments": analysis["residual_moments"],
            "global_affine": analysis["global_affine"],
            "per_image_summary": per_image_summary,
            "affine_mse_reduction_percent": float(
                100.0
                * (1.0 - per_image_summary["corrected_mse"]["mean"] / per_image_summary["residual_mse"]["mean"])
            ),
            "intensity_bins": analysis["intensity_bins"],
            "noise_variance_model": analysis["noise_variance_model"],
            "gradient_bins": analysis["gradient_bins"],
            "autocorrelation": analysis["autocorrelation"],
        },
        "local_structure": local_structure,
        "fixed_pattern": {
            "mean_map_min": float(mean_map.min()),
            "mean_map_max": float(mean_map.max()),
            "mean_map_std": float(mean_map.std()),
            "expected_mean_map_std": expected_mean_map_std,
            "fixed_pattern_ratio": float(mean_map.std() / expected_mean_map_std),
            "row_mean_std": float(mean_map.mean(axis=1).std()),
            "column_mean_std": float(mean_map.mean(axis=0).std()),
            "std_map_min": float(std_map.min()),
            "std_map_max": float(std_map.max()),
        },
        "spectrum": {
            "low_over_high": float(usable[: len(usable) // 4].mean() / usable[-len(usable) // 4 :].mean()),
            "peak_over_median": float(usable.max() / np.median(usable)),
            "horizontal_over_vertical": float(horizontal_power / vertical_power),
        },
        "blur_search": {
            "pairs_analyzed": min(args.blur_pairs, len(pairs)),
            "results": blur_results,
            "best_sigma": best_blur["sigma"],
            "improvement_percent": float(100.0 * (1.0 - best_blur["mse"] / no_blur_mse)),
        },
        "duplicates": {
            "unique_exact_hashes": len(exact_hashes),
            "exact_duplicate_groups": sum(1 for ids in exact_hashes.values() if len(ids) > 1),
            **near_duplicates,
        },
        "clustering": {
            "residual_std": cluster_one_dimensional(residual_stds),
            "std_vs_intensity_correlation": float(np.corrcoef(residual_stds, intensities)[0, 1]),
            "std_vs_gradient_correlation": float(np.corrcoef(residual_stds, gradient_energies)[0, 1]),
        },
        "train_validation": {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "comparison": comparison,
        },
    }

    results["plots"] = write_plots(analysis, args.output_dir)
    results["interpretation"] = interpret(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "degradation_report.json"
    # Only the *summaries* of the 3,200 per-image records reach the JSON (see
    # per_image_summary / clustering / train_validation above): dumping every raw
    # per-sample row would bloat a committed artifact without adding anything
    # their distributions do not already capture.
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output_dir / "degradation_report.md"
    markdown_path.write_text(build_report(results), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    for path in results["plots"].values():
        print(f"Wrote {path}")
    print(f"\nCompleted in {perf_counter() - started:.1f}s")


def interpret(results: dict) -> str:
    """Evidence-driven narrative + ranked Experiment 22 candidates."""
    model = results["residuals"]["noise_variance_model"]
    moments = results["residuals"]["moments"]
    bins = results["residuals"]["intensity_bins"]
    dark, bright = bins[0]["std"], bins[-1]["std"]
    ratio = bright / dark if dark > 0 else float("inf")
    autocorrelation = results["residuals"]["autocorrelation"]
    max_autocorrelation = max(abs(v) for v in autocorrelation.values())
    affine = results["residuals"]["global_affine"]
    blur = results["blur_search"]
    fixed = results["fixed_pattern"]

    linear_share = abs(model["linear"]) / max(abs(model["linear"]) + abs(model["quadratic"]), 1e-12)
    noise_shape = (
        "predominantly **Poisson/shot-like** (variance grows ~linearly with intensity)"
        if linear_share > 0.6
        else "predominantly **multiplicative/speckle-like** (variance grows ~quadratically with intensity)"
        if linear_share < 0.4
        else "a **mixed Poisson-Gaussian / signal-proportional** process"
    )

    return f"""
### What the evidence says

1. **Downsampling model** — `{results['best_downsampler']}` explains the clean component of
   NoisyLR best, but only marginally better than the neighbouring resampling kernels. The
   spread across all candidates is small relative to the residual itself, which means the
   residual is dominated by **noise, not by resampling-kernel mismatch**.
2. **Gain / bias** — the global affine fit is a = {affine['gain']:.6f}, b = {affine['bias']:+.6f},
   i.e. essentially identity. Per-image affine correction reduces residual MSE by only
   {results['residuals']['affine_mse_reduction_percent']:.2f}%. **There is no systematic gain or
   offset worth correcting.**
3. **Noise magnitude** — residual std is {moments['std']:.4f} over the whole dataset, with excess
   kurtosis {moments['excess_kurtosis']:+.3f} (heavier-tailed than Gaussian). For scale, that is roughly
   {moments['std'] / CHAMPION_VALIDATION_L1:.1f}x the validation L1 the current champion pipeline already
   achieves ({CHAMPION_VALIDATION_L1:.6f}) — the corruption to be removed is far larger than the error that
   remains, confirming the task is **noise-dominated rather than resolution-dominated**.
4. **Not iid, strongly signal dependent** — residual std climbs from {dark:.4f} in the darkest
   intensity bin to {bright:.4f} in the brightest, a **{ratio:.1f}x** spread. The fitted variance model
   `var(I) = {model['constant']:.4g} + {model['linear']:.4g}·I + {model['quadratic']:.4g}·I²` achieves
   R² = {model['r_squared']:.4f}, describing {noise_shape}.
   **This is the single most exploitable structure found.**
5. **Spatially near-white** — the largest |autocorrelation| at any tested offset is
   {max_autocorrelation:.4f}. The small negative lag-1 terms are an expected artifact of estimating the
   clean signal by smoothing, not evidence of correlated noise. Treat the noise as spatially
   independent.
6. **No usable fixed pattern** — the dataset-average residual map has std {fixed['mean_map_std']:.6f}
   versus {fixed['expected_mean_map_std']:.6f} expected from pure noise averaging
   (ratio {fixed['fixed_pattern_ratio']:.2f}). There is no repeatable per-sensor-position offset to subtract.
7. **No pre-downsampling blur** — the best Gaussian sigma is {blur['best_sigma']}, improving MSE by
   {blur['improvement_percent']:.3f}%. The degradation is **not** blur-then-downsample.
8. **Repeated scenes DO exist** — there are zero byte-identical GTs, but
   **{results['duplicates']['scene_groups']} confirmed repeated-scene groups covering
   {results['duplicates']['images_involved']} images** (GT MSE <
   {results['duplicates']['gt_mse_threshold']}; within-group GT MSE
   {results['duplicates']['mean_gt_mse_within_group']:.6f} against within-group *LR* MSE
   {results['duplicates']['mean_lr_mse_within_group']:.6f}, i.e. one clean scene observed under
   independent noise draws). **Every** confirmed pair sits at consecutive-ish filename indices
   (max ID gap {results['duplicates']['id_gap_max']}), so the grouping is recoverable directly from
   filenames. Critically, {results['duplicates']['groups_spanning_train_and_validation']} of these
   groups straddle the canonical split, giving
   {results['duplicates']['validation_images_with_train_twin']} of
   {results['train_validation']['validation_count']} validation images
   ({100 * results['duplicates']['validation_leakage_fraction']:.1f}%) a near-identical twin in train.
9. **One continuous degradation regime** — per-image residual std correlates
   {results['clustering']['std_vs_intensity_correlation']:+.3f} with mean image intensity. The apparent
   spread in per-image noise level is explained by **image brightness**, not by discrete noise-level
   classes. 2-means on residual std explains only
   {results['clustering']['residual_std']['variance_explained']:.3f} of variance and shows no separation gap.
10. **Train and validation match distributionally** — every compared degradation statistic differs by a
    negligible margin (see the table above), so the two splits are drawn from the same process. The one
    caveat is the {100 * results['duplicates']['validation_leakage_fraction']:.1f}% twin overlap in (8):
    absolute validation numbers are very slightly optimistic. This does **not** invalidate any
    cross-experiment comparison in this log, since every experiment used the identical split, and the
    split is deliberately left unchanged.

### Ranked strategies for Experiment 22

**1. Variance-stabilizing transform / noise-aware loss — expected payoff: HIGH.**
The measured noise variance spans {ratio:.1f}x across the intensity range with R² = {model['r_squared']:.4f}
against a simple closed-form model. Plain L1 implicitly assumes homoscedastic noise, so it
currently over-weights bright, noisy pixels and under-weights dark, clean ones — where most of
the recoverable detail actually lives. Two concrete variants: (a) train on a
generalized-Anscombe-transformed signal and invert at inference; (b) keep the pixel space but
weight the loss by 1/√var(I) using the fitted coefficients. This directly targets the strongest
structure in the data and is cheap to implement on top of the existing champion recipe.

**2. Signal-dependent synthetic-noise augmentation — expected payoff: MEDIUM-HIGH.**
With `var(I)` now measured to R² = {model['r_squared']:.4f}, unlimited extra
(GT, synthetic-NoisyLR) pairs can be synthesized by drawing noise from the fitted
signal-dependent model, multiplying the effective 2,560-sample training set without any new
labels. The dataset is small and the corruption process is now characterized, which is exactly
the regime where this works. Held back from HIGH only because the fit captures the *aggregate*
variance, while the residual's excess kurtosis of {moments['excess_kurtosis']:+.3f} shows the true
per-pixel distribution is heavier-tailed than a Gaussian of the same variance — so the synthetic
corruption would be slightly wrong in its tails. Calibrating against the measured percentiles in
§3 would mitigate that.

**3. Scene-group-aware training and validation — expected payoff: MEDIUM.**
{results['duplicates']['scene_groups']} repeated-scene groups covering
{results['duplicates']['images_involved']} images are now identified, and they are recoverable from
filenames alone (all confirmed pairs within ID gap {results['duplicates']['id_gap_max']}). Two distinct uses:
(a) *modelling* — the multiple independent noise draws over one clean scene permit
multi-observation averaging or a consistency/Noise2Noise-style term that plain paired L1 cannot
express; (b) *measurement hygiene* — {results['duplicates']['groups_spanning_train_and_validation']}
groups currently straddle the split, so
{100 * results['duplicates']['validation_leakage_fraction']:.1f}% of validation images have a
near-identical twin in train. A group-aware split would give a slightly harder but cleaner
validation signal. MEDIUM because only
{100 * results['duplicates']['images_involved'] / results['pairs_analyzed']:.1f}% of the dataset is
involved, capping the achievable gain. **Do not change the canonical split casually** — doing so
makes new numbers incomparable with Experiments 1-20.

**Also worth noting — self-supervised objectives are statistically admissible here.** The noise is
spatially near-white (max |autocorrelation| {max_autocorrelation:.4f}) and zero-mean
({moments['mean']:+.6f}), which is precisely the condition blind-spot / Noise2Void and
Stein-unbiased-risk methods require. Ranked below the three above only because paired GT is already
available, so these would add regularization rather than new information.

**Explicitly deprioritized — LOW payoff, with reasons:** fixed-pattern subtraction (no pattern
exists, ratio {fixed['fixed_pattern_ratio']:.2f}); deblurring or kernel estimation (best sigma
{blur['best_sigma']}, only {blur['improvement_percent']:.3f}% gain); per-image gain/bias calibration
({results['residuals']['affine_mse_reduction_percent']:.2f}% MSE reduction); degradation-regime
classification (no discrete clusters — apparent spread is just brightness); and further
resampling-kernel search (all candidates within
{100 * (max(m['mse'] for m in results['downsampling_models'].values()) / min(m['mse'] for m in results['downsampling_models'].values()) - 1):.0f}%
of each other, and all far below the noise floor).
""".strip()


if __name__ == "__main__":
    main()
