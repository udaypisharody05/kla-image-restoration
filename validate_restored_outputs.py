"""Validate a directory of restored ``.npy`` outputs and write a manifest.

Checks that a directory produced by ``inference.py`` has: exactly one output
per input (no missing/duplicate/extra files), every output at exactly 2x the
corresponding input's resolution, float32 dtype, finite values, and a
sensible numerical range. Writes ``manifest.json`` into the output directory
summarizing the run. Does not require or use any ground truth.

Example::

    python validate_restored_outputs.py \\
        --input-dir data/Data-public/Test_NoisyLR/NoisyLR \\
        --output-dir restored_test_outputs
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from src.dataset_discovery import image_files


def validate(input_dir: Path, output_dir: Path, scale: int, model: str, tta: str) -> dict:
    input_files = {path.name: path for path in image_files(input_dir) if path.suffix.lower() == ".npy"}
    output_files = {path.name: path for path in image_files(output_dir) if path.suffix.lower() == ".npy"}

    missing = sorted(set(input_files) - set(output_files))
    extra = sorted(set(output_files) - set(input_files))

    problems: list[str] = []
    if missing:
        problems.append(f"{len(missing)} input(s) have no corresponding output: {missing[:10]}")
    if extra:
        problems.append(f"{len(extra)} output(s) have no corresponding input: {extra[:10]}")

    per_file = []
    shapes_seen = set()
    dtypes_seen = set()
    value_min, value_max = float("inf"), float("-inf")
    for name in sorted(set(input_files) & set(output_files)):
        input_array = np.load(input_files[name])
        output_array = np.load(output_files[name])
        expected_shape = (input_array.shape[0] * scale, input_array.shape[1] * scale)

        entry = {
            "filename": name,
            "input_shape": list(input_array.shape),
            "output_shape": list(output_array.shape),
            "dtype": str(output_array.dtype),
            "finite": bool(np.isfinite(output_array).all()),
            "min": float(output_array.min()),
            "max": float(output_array.max()),
        }
        per_file.append(entry)
        shapes_seen.add(output_array.shape)
        dtypes_seen.add(str(output_array.dtype))
        value_min = min(value_min, entry["min"])
        value_max = max(value_max, entry["max"])

        if output_array.shape != expected_shape:
            problems.append(f"{name}: shape {output_array.shape} != expected {expected_shape}")
        if not entry["finite"]:
            problems.append(f"{name}: contains non-finite values")
        if output_array.dtype != np.float32:
            problems.append(f"{name}: dtype {output_array.dtype} != float32")

    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_inputs": len(input_files),
        "num_outputs": len(output_files),
        "num_validated_pairs": len(per_file),
        "num_missing": len(missing),
        "num_extra": len(extra),
        "output_format": ".npy, float32, grayscale [H,W]",
        "output_dtypes_seen": sorted(dtypes_seen),
        "output_shapes_seen": [list(shape) for shape in sorted(shapes_seen)],
        "value_range_overall": [value_min, value_max] if per_file else None,
        "all_finite": all(entry["finite"] for entry in per_file) if per_file else False,
        "scale_factor": scale,
        "model": model,
        "tta": tta,
        "no_ground_truth_included": True,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--model", type=str, default="residual_sr (weights/residualsr_final_ema.pt)")
    parser.add_argument("--tta", type=str, default="none")
    parser.add_argument(
        "--manifest-path", type=Path, default=None, help="Default: <output-dir>/manifest.json"
    )
    args = parser.parse_args()

    manifest = validate(args.input_dir, args.output_dir, args.scale, args.model, args.tta)

    manifest_path = args.manifest_path or (args.output_dir / "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Inputs: {manifest['num_inputs']}  Outputs: {manifest['num_outputs']}")
    print(f"Missing: {manifest['num_missing']}  Extra: {manifest['num_extra']}")
    print(f"All finite: {manifest['all_finite']}")
    print(f"Output shapes seen: {manifest['output_shapes_seen']}")
    print(f"Value range: {manifest['value_range_overall']}")
    print(f"Status: {manifest['status']}")
    if manifest["problems"]:
        print("Problems:")
        for problem in manifest["problems"]:
            print(f"  - {problem}")
    print(f"Saved manifest: {manifest_path}")

    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
