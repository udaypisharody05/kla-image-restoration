"""Standalone compliance check for a directory produced by ``run.py``.

Self-contained (no dependency on the parent repository) -- verifies, for a
given ``<input-dir>``/``<output-dir>`` pair, every invariant required of the
submission's outputs: exact filename-set match, one output per input,
correct 2x spatial shape, grayscale, float32, finite, and values in [0, 1].

Usage::

    python validate_outputs.py <input-dir> <output-dir>

Exits 0 and prints "PASS" if every invariant holds, else exits 1 and lists
every problem found.
"""

import sys
from pathlib import Path

import numpy as np


def validate(input_dir: Path, output_dir: Path, scale: int = 2) -> list[str]:
    inputs = {p.name: p for p in sorted(Path(input_dir).glob("*.npy"))}
    outputs = {p.name: p for p in sorted(Path(output_dir).glob("*.npy"))}

    problems: list[str] = []
    missing = sorted(set(inputs) - set(outputs))
    extra = sorted(set(outputs) - set(inputs))
    if missing:
        problems.append(f"{len(missing)} input(s) with no output: {missing[:10]}")
    if extra:
        problems.append(f"{len(extra)} output(s) with no input: {extra[:10]}")

    for name in sorted(set(inputs) & set(outputs)):
        input_array = np.load(inputs[name])
        output_array = np.load(outputs[name])

        input_h, input_w = input_array.shape[0], input_array.shape[1]
        expected_2d = (input_h * scale, input_w * scale)
        expected_3d = expected_2d + (1,)
        if output_array.shape not in (expected_2d, expected_3d):
            problems.append(f"{name}: shape {output_array.shape} != expected {expected_2d} or {expected_3d}")
        if output_array.dtype != np.float32:
            problems.append(f"{name}: dtype {output_array.dtype} != float32")
        if not np.isfinite(output_array).all():
            problems.append(f"{name}: contains NaN/Inf")
        elif output_array.min() < 0.0 or output_array.max() > 1.0:
            problems.append(f"{name}: value range [{output_array.min()}, {output_array.max()}] outside [0, 1]")

    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python validate_outputs.py <input-dir> <output-dir>", file=sys.stderr)
        return 2

    problems = validate(Path(argv[0]), Path(argv[1]))
    if problems:
        print("FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
