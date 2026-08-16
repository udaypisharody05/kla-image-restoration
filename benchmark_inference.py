"""Benchmark the packaged final model's inference speed and memory footprint.

Measures model load time, per-image forward-pass latency (mean/median),
throughput, and peak CUDA memory, for both ``--tta none`` (single pass) and
``--tta x8`` (8-way geometric self-ensemble), so the accuracy/latency
tradeoff documented in README.md is backed by an actual measurement rather
than an assumption.

Uses synthetic ``[1,1,128,128]`` inputs (the real NoisyLR resolution) by
default, so the benchmark does not require the training/test dataset to be
present -- only the packaged weights. Pass ``--real-sample`` to instead time
one real official test image if the dataset is available locally.

Results are measured on **whatever GPU this script is run on** -- this
project's development hardware is an NVIDIA GeForce RTX 4060 **Laptop** GPU
(~8 GB VRAM), which this script's JSON output records explicitly via
``torch.cuda.get_device_name()``. It does NOT predict performance on other
hardware (e.g. the H100 KLA will benchmark on); re-run this script on any
target machine to get numbers for that machine.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from inference import DEFAULT_CHECKPOINT, load_inference_model, select_device
from src.tta import predict_x8

RESULTS_PATH = Path("results/final_benchmark.json")
RESULTS_MD_PATH = Path("results/final_benchmark.md")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_real_sample(data_dir: Path) -> np.ndarray:
    from src.dataset_discovery import discover_layout, image_files
    from src.io_utils import load_image_array

    layout = discover_layout(data_dir)
    files = image_files(layout.test_input_dir)
    if not files:
        raise FileNotFoundError(f"No test images found under {layout.test_input_dir}")
    return load_image_array(files[0])


@torch.no_grad()
def time_forward(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    tta: str,
    iterations: int,
    warmup: int,
) -> dict:
    for _ in range(warmup):
        _ = predict_x8(model, input_tensor) if tta == "x8" else model(input_tensor)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    for _ in range(iterations):
        _sync(device)
        start = time.perf_counter()
        _ = predict_x8(model, input_tensor) if tta == "x8" else model(input_tensor)
        _sync(device)
        durations.append(time.perf_counter() - start)

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
    )
    mean_s = statistics.mean(durations)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "mean_seconds": mean_s,
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "stdev_seconds": statistics.stdev(durations) if iterations > 1 else 0.0,
        "throughput_images_per_second": 1.0 / mean_s if mean_s > 0 else float("inf"),
        "peak_cuda_memory_mib": peak_memory_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=128, help="Synthetic LR square input size")
    parser.add_argument(
        "--real-sample",
        type=Path,
        default=None,
        help="Optional: time one real official test image instead of synthetic input "
        "(e.g. data/Data-public/Test_NoisyLR/NoisyLR/000000.npy)",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    device = select_device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))

    load_start = time.perf_counter()
    model, metadata = load_inference_model(args.checkpoint, device)
    _sync(device)
    load_seconds = time.perf_counter() - load_start
    print(f"Model load time: {load_seconds:.3f}s")

    if args.real_sample is not None:
        array = np.load(args.real_sample) if args.real_sample.suffix == ".npy" else None
        if array is None:
            from src.io_utils import load_image_array

            array = load_image_array(args.real_sample)
        input_tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        input_source = str(args.real_sample)
    else:
        rng = np.random.default_rng(42)
        array = rng.uniform(0.0, 1.0, size=(args.input_size, args.input_size)).astype(np.float32)
        input_tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
        input_source = f"synthetic {args.input_size}x{args.input_size} (seed=42)"
    print(f"Input: {input_source}, shape={tuple(input_tensor.shape)}")

    results = {
        "device": str(device),
        "gpu_name": gpu_name,
        "note": (
            "Measured locally on the hardware named above -- NOT a prediction of "
            "H100 or any other GPU's performance. Re-run this script on the target "
            "machine for its own numbers."
        ),
        "checkpoint": str(args.checkpoint),
        "input_source": input_source,
        "input_shape": list(input_tensor.shape),
        "model_load_seconds": load_seconds,
        "modes": {},
    }

    for tta in ("none", "x8"):
        print(f"\nBenchmarking --tta {tta} ({args.iterations} iterations, {args.warmup} warmup)...")
        mode_result = time_forward(model, input_tensor, device, tta, args.iterations, args.warmup)
        results["modes"][tta] = mode_result
        print(
            f"  mean={mode_result['mean_seconds'] * 1000:.2f}ms  "
            f"median={mode_result['median_seconds'] * 1000:.2f}ms  "
            f"throughput={mode_result['throughput_images_per_second']:.2f} img/s"
            + (
                f"  peak_mem={mode_result['peak_cuda_memory_mib']:.1f}MiB"
                if mode_result["peak_cuda_memory_mib"] is not None
                else ""
            )
        )

    none_mean = results["modes"]["none"]["mean_seconds"]
    x8_mean = results["modes"]["x8"]["mean_seconds"]
    results["x8_slowdown_factor"] = x8_mean / none_mean if none_mean > 0 else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {args.output}")

    md_lines = [
        "# Final Inference Benchmark",
        "",
        f"Device: **{gpu_name or device}** (measured locally; not an H100 prediction)",
        f"Checkpoint: `{args.checkpoint}`",
        f"Input: {input_source}",
        f"Model load time: {load_seconds:.3f}s",
        "",
        "| Mode | Mean (ms) | Median (ms) | Throughput (img/s) | Peak CUDA mem (MiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tta in ("none", "x8"):
        r = results["modes"][tta]
        mem = f"{r['peak_cuda_memory_mib']:.1f}" if r["peak_cuda_memory_mib"] is not None else "n/a"
        md_lines.append(
            f"| {tta} | {r['mean_seconds'] * 1000:.2f} | {r['median_seconds'] * 1000:.2f} | "
            f"{r['throughput_images_per_second']:.2f} | {mem} |"
        )
    if results["x8_slowdown_factor"] is not None:
        md_lines.append("")
        md_lines.append(f"x8 TTA is **{results['x8_slowdown_factor']:.2f}x** slower than a single pass.")
    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()
