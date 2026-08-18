"""Evaluator entrypoint for the SuperconductorSemistars KLA hackathon submission.

Usage::

    python run.py <input-dir> <output-dir>

Reads every ``.npy`` grayscale NoisyLR image from ``<input-dir>``, restores
it to 2x spatial resolution with the packaged champion ResidualSR model
(``models/residualsr_final_ema.pt``), and writes one restored ``.npy`` per
input into ``<output-dir>`` (created automatically) under the identical
filename.

No CLI flags, checkpoint path, config file, or device selection are
required -- everything needed to run is resolved relative to this script's
own location (``Path(__file__).resolve().parent``), so the command above
works unchanged regardless of the caller's current working directory.
CUDA is used automatically when available, with an automatic CPU fallback.
No internet access, API keys, or external model downloads are involved.
"""

import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.residual_sr import ResidualSRNet  # noqa: E402

MODEL_PATH = BASE_DIR / "models" / "residualsr_final_ema.pt"


def select_device() -> torch.device:
    """CUDA automatically when available, CPU otherwise -- no user input needed."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load the packaged champion ResidualSR model, fully configured from the
    checkpoint's own stored metadata (no hard-coded architecture flags)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    package = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    scale = int(package["scale"])
    model = ResidualSRNet(
        in_channels=package["in_channels"],
        out_channels=package["out_channels"],
        num_features=package["num_features"],
        num_blocks=package["num_blocks"],
        scale=scale,
    ).to(device)
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()
    return model, scale


def normalize_grayscale(array: np.ndarray, name: str) -> np.ndarray:
    """Accept (H,W), (H,W,1), or (1,H,W) and return a 2D (H,W) array.

    Any other shape is rejected outright rather than guessed at, so a
    malformed input fails loudly instead of being silently misinterpreted.
    """
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        f"{name}: unsupported array shape {array.shape}; expected grayscale "
        "(H,W), (H,W,1), or (1,H,W)"
    )


@torch.inference_mode()
def restore(model: torch.nn.Module, array: np.ndarray, device: torch.device, scale: int) -> np.ndarray:
    """Run one (H,W) array through the model and return a validated,
    saveable (H*scale, W*scale) float32 prediction clipped to [0,1]."""
    height, width = array.shape
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    output = model(tensor)
    prediction = output[0, 0].detach().to("cpu").numpy()

    expected_shape = (height * scale, width * scale)
    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"Model produced shape {prediction.shape}, expected {expected_shape} "
            f"for a {array.shape} input at {scale}x scale"
        )

    prediction = np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced non-finite (NaN/Inf) values")
    prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32, copy=False)

    if not (np.isfinite(prediction).all() and prediction.min() >= 0.0 and prediction.max() <= 1.0):
        raise RuntimeError("Post-clip validation failed unexpectedly")  # defensive; should be unreachable
    return prediction


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python run.py <input-dir> <output-dir>", file=sys.stderr)
        return 2

    input_dir = Path(argv[0]).resolve()
    output_dir = Path(argv[1]).resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if input_dir == output_dir:
        print(
            "Error: input-dir and output-dir must be different directories "
            "(restoring in place would overwrite the source .npy files).",
            file=sys.stderr,
        )
        return 2

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        print(f"Error: no .npy files found in {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Device: {device}")
    model, scale = load_model(device)
    print(f"Loaded model: {MODEL_PATH.name} (scale x{scale})")
    print(f"Found {len(input_files)} input file(s) in {input_dir}")

    succeeded = 0
    for index, input_path in enumerate(input_files, start=1):
        array = np.load(input_path, allow_pickle=False)
        array = normalize_grayscale(array, input_path.name)
        if array.ndim != 2:
            raise ValueError(
                f"{input_path.name}: could not normalize to 2D grayscale, got shape {array.shape}"
            )
        prediction = restore(model, array, device, scale)
        np.save(output_dir / input_path.name, prediction)
        succeeded += 1
        if index % 50 == 0 or index == len(input_files):
            print(f"  [{index}/{len(input_files)}] {input_path.name}")

    print(f"Done: {succeeded}/{len(input_files)} restored -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
