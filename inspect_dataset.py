"""Inspect a paired restoration dataset and write JSON and Markdown reports."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_discovery import discover_layout, discover_pairs, image_files
from src.io_utils import load_image_array


REPOSITORY_ROOT = Path(__file__).resolve().parent


def configured_data_dir() -> Path:
    """Resolve SEMICON_DATA_DIR or the repository-local data directory."""
    configured = os.environ.get("SEMICON_DATA_DIR")
    candidate = Path(configured) if configured else REPOSITORY_ROOT / "data"
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _sample(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    indexes = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[int(i)] for i in indexes]


def _channels(array: np.ndarray) -> int:
    return 1 if array.ndim == 2 else int(array.shape[-1])


def _relative_posix(path: Path, base: Path) -> str:
    """Represent a dataset path relative to *base* with portable separators."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix() or "."
    except ValueError:
        return path.name


def inspect_group(
    paths: list[Path],
    max_samples: int,
    range_role: str,
    path_base: Path | None = None,
) -> dict[str, Any]:
    chosen = _sample(paths, max_samples)
    shapes, dtypes, channels = Counter(), Counter(), Counter()
    failures: list[dict[str, str]] = []
    constant: list[str] = []
    total = below = above = nan_count = inf_count = 0
    finite_sum = finite_sq_sum = finite_count = 0
    minimum, maximum = np.inf, -np.inf
    for path in chosen:
        try:
            array = load_image_array(path)
            shapes[str(list(array.shape))] += 1
            dtypes[str(array.dtype)] += 1
            channels[str(_channels(array))] += 1
            values = np.asarray(array)
            total += values.size
            nan_count += int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
            inf_count += int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
            finite = values[np.isfinite(values)]
            if finite.size:
                f64 = finite.astype(np.float64, copy=False)
                minimum, maximum = min(minimum, float(f64.min())), max(maximum, float(f64.max()))
                finite_sum += float(f64.sum())
                finite_sq_sum += float(np.square(f64).sum())
                finite_count += f64.size
                below += int((f64 < 0).sum())
                above += int((f64 > 1).sum())
                if float(f64.max() - f64.min()) <= 1e-8:
                    constant.append(
                        _relative_posix(path, path_base) if path_base else path.as_posix()
                    )
        except Exception as exc:  # report the affected file; inspection must continue
            display_path = (
                _relative_posix(path, path_base) if path_base else path.as_posix()
            )
            error = str(exc).replace(str(path), display_path).replace(
                path.as_posix(), display_path
            )
            failures.append({
                "path": display_path,
                "error": error,
            })
    mean = finite_sum / finite_count if finite_count else None
    variance = max(0.0, finite_sq_sum / finite_count - mean * mean) if mean is not None else None
    outside = below + above
    return {
        "file_count": len(paths), "inspected_file_count": len(chosen),
        "shapes": dict(sorted(shapes.items())), "dtypes": dict(sorted(dtypes.items())),
        "channels": dict(sorted(channels.items())), "all_grayscale": set(channels) <= {"1"},
        "minimum": None if not finite_count else minimum, "maximum": None if not finite_count else maximum,
        "mean": mean, "standard_deviation": None if variance is None else variance ** 0.5,
        "value_count": total, "below_zero_count": below,
        "below_zero_percent": 100 * below / total if total else 0,
        "above_one_count": above, "above_one_percent": 100 * above / total if total else 0,
        "outside_0_1_count": outside, "outside_0_1_percent": 100 * outside / total if total else 0,
        "nan_count": nan_count, "infinite_count": inf_count,
        "constant_or_nearly_constant_files": constant, "failed_files": failures,
        "range_validation": range_role,
    }


def _hierarchy(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() + "/" for p in sorted((x for x in root.rglob("*") if x.is_dir()))]


def _portable_roots(configured_data_dir: Path, dataset_root: Path) -> tuple[str, str]:
    """Return non-machine-specific labels for the configured and resolved roots."""
    configured = configured_data_dir.resolve()
    resolved_dataset = dataset_root.resolve()
    try:
        configured_label = configured.relative_to(REPOSITORY_ROOT).as_posix() or "."
        dataset_label = resolved_dataset.relative_to(REPOSITORY_ROOT).as_posix() or "."
    except ValueError:
        configured_label = "."
        dataset_label = _relative_posix(resolved_dataset, configured)
    return configured_label, dataset_label


def build_report(data_dir: Path, max_samples: int = 100) -> dict[str, Any]:
    layout = discover_layout(data_dir)
    configured_data_dir = Path(data_dir).resolve()
    configured_label, dataset_label = _portable_roots(configured_data_dir, layout.root)
    pairing = discover_pairs(layout)
    inputs, targets, tests = image_files(layout.train_input_dir), image_files(layout.target_dir), image_files(layout.test_input_dir)
    physical = [p for p in configured_data_dir.rglob("*") if p.is_file()]
    scales, invalid_dimensions = Counter(), []
    for pair in _sample(list(pairing.pairs), max_samples):
        try:
            a, b = load_image_array(pair.input_path), load_image_array(pair.target_path)
            if a.shape[0] <= 0 or a.shape[1] <= 0 or b.shape[0] <= 0 or b.shape[1] <= 0:
                invalid_dimensions.append(pair.pair_id)
            else:
                scales[f"{b.shape[0] / a.shape[0]:g}x{b.shape[1] / a.shape[1]:g}"] += 1
        except Exception:
            pass
    warnings = []
    if pairing.missing_targets or pairing.missing_inputs: warnings.append("Unmatched training identifiers detected.")
    if pairing.duplicate_input_ids or pairing.duplicate_target_ids: warnings.append("Duplicate pair identifiers detected.")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": dataset_label, "requested_data_dir": configured_label,
        "folder_hierarchy": _hierarchy(configured_data_dir),
        "total_physical_files": len(physical),
        "file_counts_by_extension": dict(sorted(Counter((p.suffix.lower() or "<none>") for p in physical).items())),
        "approximate_extracted_size_bytes": sum(p.stat().st_size for p in physical),
        "directories": {
            "train_input": _relative_posix(layout.train_input_dir, layout.root),
            "ground_truth": _relative_posix(layout.target_dir, layout.root),
            "test_input": _relative_posix(layout.test_input_dir, layout.root),
        },
        "counts": {"training_inputs": len(inputs), "ground_truths": len(targets), "valid_pairs": len(pairing.pairs), "test_inputs": len(tests)},
        "pairing": {
            "convention": "exact complete filename stem match",
            "missing_targets": list(pairing.missing_targets),
            "missing_inputs": list(pairing.missing_inputs),
            "duplicate_input_ids": {
                key: [_relative_posix(Path(path), layout.root) for path in paths]
                for key, paths in pairing.duplicate_input_ids.items()
            },
            "duplicate_target_ids": {
                key: [_relative_posix(Path(path), layout.root) for path in paths]
                for key, paths in pairing.duplicate_target_ids.items()
            },
        },
        "inspection_limit_per_group": max_samples,
        "input": inspect_group(inputs, max_samples, "degraded values may be outside [0,1]", layout.root),
        "ground_truth": inspect_group(targets, max_samples, "ground truth expected within [0,1]", layout.root),
        "test_input": inspect_group(tests, max_samples, "degraded values may be outside [0,1]", layout.root),
        "scale_factors": dict(sorted(scales.items())), "scale_factor_consistent": len(scales) == 1,
        "invalid_dimension_pair_ids": invalid_dimensions, "warnings": warnings,
    }
    return report


def report_markdown(report: dict[str, Any]) -> str:
    c, p = report["counts"], report["pairing"]
    lines = ["# Dataset inspection report", "", f"Generated: {report['generated_at_utc']}", "", "## Layout", "",
        f"- Dataset root: `{report['dataset_root']}`", f"- Training inputs: `{report['directories']['train_input']}`",
        f"- Ground truths: `{report['directories']['ground_truth']}`", f"- Test inputs: `{report['directories']['test_input']}`",
        f"- Physical files: {report['total_physical_files']}", f"- Extracted size: {report['approximate_extracted_size_bytes'] / 2**20:.2f} MiB", "", "## Pairing", "",
        f"- Valid pairs: {c['valid_pairs']} ({c['training_inputs']} inputs, {c['ground_truths']} targets)", f"- Test images: {c['test_inputs']}",
        f"- Convention: {p['convention']}", f"- Missing targets: {len(p['missing_targets'])}", f"- Missing inputs: {len(p['missing_inputs'])}",
        f"- Duplicate input IDs: {len(p['duplicate_input_ids'])}", f"- Duplicate target IDs: {len(p['duplicate_target_ids'])}", "", "## Array findings", ""]
    for key, label in (("input", "Training degraded"), ("ground_truth", "Ground truth"), ("test_input", "Test degraded")):
        s = report[key]
        lines += [f"### {label}", "", f"Inspected {s['inspected_file_count']} of {s['file_count']} files.", "",
            f"- Shapes: `{s['shapes']}`; dtypes: `{s['dtypes']}`; channels: `{s['channels']}`; grayscale: {s['all_grayscale']}",
            f"- Min/max: {s['minimum']:.9g} / {s['maximum']:.9g}; mean/std: {s['mean']:.9g} / {s['standard_deviation']:.9g}",
            f"- Below 0: {s['below_zero_count']} ({s['below_zero_percent']:.6f}%); above 1: {s['above_one_count']} ({s['above_one_percent']:.6f}%)",
            f"- Outside [0,1]: {s['outside_0_1_count']} ({s['outside_0_1_percent']:.6f}%)", f"- NaN/Inf: {s['nan_count']} / {s['infinite_count']}",
            f"- Constant/nearly constant: {len(s['constant_or_nearly_constant_files'])}; load failures: {len(s['failed_files'])}", ""]
    lines += ["## Geometry", "", f"- Scale factors: `{report['scale_factors']}`", f"- Consistent: {report['scale_factor_consistent']}", "", "## Warnings", ""]
    lines += [f"- {w}" for w in report["warnings"]] or ["- None"]
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "dataset_report.md").write_text(report_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--max-samples", type=int, default=100, help="Per group; 0 inspects every file")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = build_report(args.data_dir, args.max_samples)
    save_report(report, args.results_dir)
    print(report_markdown(report))


if __name__ == "__main__":
    main()
