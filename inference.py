"""Standalone restoration inference for the KLA image-restoration submission.

Loads the packaged final model (``weights/residualsr_final_ema.pt`` by
default -- the EMA weights of the verified champion,
``checkpoints/exp23_ema_extended90/checkpoint_best.pt``, exported by
``export_final_weights.py``) and restores every ``.npy`` NoisyLR image found
under ``--input-dir`` to its 2x resolution, writing one ``.npy`` output per
input under ``--output-dir``.

Example (official test set)::

    python inference.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR \\
        --output-dir restored_test_outputs

No source-code edits are required: the model architecture is reconstructed
automatically from metadata stored in the checkpoint/weight file, CUDA is
used automatically when available (falling back to CPU otherwise), and no
ground-truth or training-dataset path is ever read.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.dataset_discovery import image_files
from src.io_utils import load_image_array
from src.models import build_model
from src.noise_conditioning import wrap_for_conditioning
from src.tta import predict_x8

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "residualsr_final_ema.pt"

# Set by the benchmark in results/final_benchmark.json: x8 TTA gains +0.0462 dB
# PSNR (27.9893 -> 28.0355) at roughly an 8x inference-time cost. "none" is the
# submission default; --tta x8 remains available for evaluators who prefer the
# small accuracy gain over throughput. See README.md "TTA" section.
DEFAULT_TTA = "none"


def select_device(requested: str | None) -> torch.device:
    """``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` unless
    *requested* pins a specific device -- no hard-coded GPU index, so this
    behaves identically on a laptop RTX 4060 or a datacenter H100."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_config_from_package(package: dict) -> dict:
    """Extract a ``build_model``-ready config from a packaged inference file
    (``export_final_weights.py`` output). Optional-variant keys fall back to
    the same defaults ``src.models.build_model`` itself uses, so a package
    that predates a given variant still loads as a plain ResidualSR."""
    return {
        "in_channels": package["in_channels"],
        "out_channels": package["out_channels"],
        "num_features": package["num_features"],
        "num_blocks": package["num_blocks"],
        "scale": package["scale"],
        "channel_attention": package.get("channel_attention", False),
        "attention_reduction": package.get("attention_reduction", 8),
        "multiscale_block": package.get("multiscale_block", False),
        "rdb_block": package.get("rdb_block", False),
        "rdb_growth_rate": package.get("rdb_growth_rate", 16),
        "rdb_num_layers": package.get("rdb_num_layers", 3),
        "denoise_stem": package.get("denoise_stem", False),
        "denoise_stem_features": package.get("denoise_stem_features", 32),
        "denoise_stem_blocks": package.get("denoise_stem_blocks", 2),
    }


def load_inference_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Reconstruct the restoration model with zero required CLI configuration.

    Accepts two file shapes so ``--checkpoint`` can point at either:

    - a packaged inference-only file from ``export_final_weights.py`` (the
      default), or
    - a full ``train.py`` training checkpoint (e.g. a raw
      ``checkpoints/<exp>/checkpoint_best.pt``) -- delegated to
      ``evaluate_checkpoint.load_model``, which already prefers EMA weights
      and handles every architecture/optional-variant this project supports,
      so that logic is not duplicated here.

    Either way, the evaluator never needs to pass ``--num-features``,
    ``--num-blocks``, or any other architecture flag.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint/weights file not found: {checkpoint_path}\n"
            "Run 'python export_final_weights.py' to create the default packaged "
            "weights, or pass --checkpoint <path> to use a different file."
        )
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" in raw:
        from evaluate_checkpoint import load_model as _load_training_checkpoint

        return _load_training_checkpoint(checkpoint_path, device)

    if "model_state_dict" not in raw:
        raise ValueError(
            f"{checkpoint_path} is neither a packaged inference weight file "
            "(export_final_weights.py output) nor a train.py training checkpoint "
            "-- missing both 'model_config' and 'model_state_dict' keys."
        )
    model_config = _model_config_from_package(raw)
    base_model = build_model(model_config).to(device)
    model = wrap_for_conditioning(base_model, raw.get("noise_conditioning_config"))
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, raw


def discover_input_files(input_dir: Path) -> list[Path]:
    """Every ``.npy`` file under *input_dir*, sorted for deterministic
    ordering. Reuses ``src.dataset_discovery.image_files`` (already excludes
    ``__MACOSX``/``.DS_Store``/``._*`` archive junk) rather than a second
    listing implementation, then narrows to ``.npy`` specifically -- the
    official dataset's actual format."""
    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
    files = [path for path in image_files(input_dir) if path.suffix.lower() == ".npy"]
    if not files:
        raise FileNotFoundError(f"No .npy files found under --input-dir: {input_dir}")
    return files


@torch.no_grad()
def restore_one(model: torch.nn.Module, array: np.ndarray, device: torch.device, tta: str) -> np.ndarray:
    """Run one ``[H,W]`` NoisyLR array through the model. Raw preprocessing
    only (float32 cast, add batch/channel dims) -- no normalization, no
    clipping, matching the project's established convention (raw values,
    including outside [0,1], are fed to the model unchanged; see
    ``docs/dataset_notes.md`` and ``weights/residualsr_final_ema.pt``'s
    stored ``input_range`` metadata). Output is likewise raw/unclipped."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)
    if tta == "x8":
        output = predict_x8(model, tensor)
    else:
        output = model(tensor)
    return output[0, 0].detach().cpu().numpy().astype(np.float32)


def run_inference(
    model: torch.nn.Module,
    input_files: list[Path],
    output_dir: Path,
    device: torch.device,
    tta: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_seconds: list[float] = []
    failures: list[tuple[str, str]] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batch_start = time.perf_counter()

    for index, input_path in enumerate(input_files, start=1):
        try:
            array = load_image_array(input_path)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = restore_one(model, array, device, tta)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            per_image_seconds.append(time.perf_counter() - start)

            if not np.isfinite(prediction).all():
                raise ValueError("model produced non-finite (NaN/Inf) output")

            np.save(output_dir / input_path.name, prediction)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the batch
            failures.append((input_path.name, str(exc)))
            print(f"  [FAILED] {input_path.name}: {exc}", file=sys.stderr)
            continue

        if index % 25 == 0 or index == len(input_files):
            print(f"  [{index}/{len(input_files)}] {input_path.name}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - batch_start

    return {
        "num_input_files": len(input_files),
        "num_succeeded": len(per_image_seconds),
        "num_failed": len(failures),
        "failures": failures,
        "total_seconds": total_seconds,
        "avg_seconds_per_image": (total_seconds / len(input_files)) if input_files else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of .npy NoisyLR inputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write restored .npy outputs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model weights to use (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tta",
        type=str,
        choices=["none", "x8"],
        default=DEFAULT_TTA,
        help=(
            "'none' (default): single forward pass, fastest. 'x8': 8-way geometric "
            "self-ensemble, +0.0462 dB PSNR measured on validation at ~8x the inference "
            "cost -- see results/final_benchmark.json and README.md."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto-detects")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    try:
        input_files = discover_input_files(args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(input_files)} input file(s) under {args.input_dir}")

    load_start = time.perf_counter()
    try:
        model, metadata = load_inference_model(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    load_seconds = time.perf_counter() - load_start
    architecture = metadata.get("architecture") or metadata.get("model_config", {}).get(
        "architecture", "residual_sr"
    )
    print(f"Loaded model from {args.checkpoint} ({architecture}) in {load_seconds:.2f}s")
    print(f"TTA: {'x8 geometric self-ensemble' if args.tta == 'x8' else 'disabled'}")
    print(f"Writing outputs to {args.output_dir}")

    summary = run_inference(model, input_files, args.output_dir, device, args.tta)

    print()
    print(f"Total images processed: {summary['num_input_files']}")
    print(f"Succeeded: {summary['num_succeeded']}  Failed: {summary['num_failed']}")
    print(f"Total inference time: {summary['total_seconds']:.3f}s")
    print(f"Average inference time/image: {summary['avg_seconds_per_image'] * 1000:.2f} ms")
    print(f"Device used: {device}")

    if summary["num_failed"] > 0:
        print(f"\n{summary['num_failed']} file(s) failed -- see stderr above.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
