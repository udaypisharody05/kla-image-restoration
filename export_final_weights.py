"""Export a clean, inference-only weight artifact from the verified champion.

Reads the EMA weights out of the full training checkpoint
(``checkpoints/exp23_ema_extended90/checkpoint_best.pt`` -- optimizer state,
scheduler state, EMA shadow, epoch counter, etc.) and writes a small,
self-contained file containing only what ``inference.py`` needs to
reconstruct the exact same model deterministically:

- ``model_state_dict``: the EMA weights (what actually produced the
  champion's recorded validation PSNR/SSIM -- see
  ``evaluate_checkpoint.py::load_model``'s ``prefer_ema`` default).
- Enough architecture metadata (``architecture``, ``in_channels``,
  ``out_channels``, ``num_features``, ``num_blocks``, ``scale``, and every
  optional-variant flag) to call ``src.models.build_model`` with no
  additional CLI flags required.
- Provenance: source checkpoint path, source epoch, source
  ``best_val_psnr``, and the independently re-measured reference metrics
  recorded in ``EXPERIMENT_LOG.md`` (not re-derived here -- this script does
  not run evaluation).

Does not modify, move, or overwrite the source checkpoint in any way -- it
is only ever opened read-only. Run this script once; the output
(``weights/residualsr_final_ema.pt``) is what ``inference.py`` loads by
default.
"""

import argparse
import datetime
from pathlib import Path

import torch

CHAMPION_CHECKPOINT = Path("checkpoints/exp23_ema_extended90/checkpoint_best.pt")
DEFAULT_OUTPUT = Path("weights/residualsr_final_ema.pt")

# Independently re-measured and recorded in EXPERIMENT_LOG.md / this
# session's report -- not computed by this script, and must not be edited
# to make the packaged weights "look better". If these ever need updating,
# update EXPERIMENT_LOG.md first and copy the numbers from there.
REFERENCE_METRICS = {
    "val_psnr_db": 27.9893,
    "val_ssim": 0.756916,
    "val_psnr_db_x8_tta": 28.0355,
    "val_ssim_x8_tta": 0.758519,
    "bicubic_psnr_db": 23.1413,
    "bicubic_ssim": 0.550604,
    "note": (
        "Measured with evaluate_checkpoint.py on the canonical 640-image "
        "validation split (seed=42, val_fraction=0.2). x8 = geometric "
        "self-ensemble TTA (src/tta.py::predict_x8)."
    ),
}


def export(source: Path, destination: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    ema_state_dict = checkpoint.get("ema_state_dict")
    if ema_state_dict is None:
        raise ValueError(
            f"{source} has no ema_state_dict -- this export script is written for the "
            "EMA-trained champion checkpoint specifically; pass a different source "
            "checkpoint explicitly if this is intentional."
        )
    model_config = checkpoint["model_config"]
    if model_config.get("architecture", "residual_sr") != "residual_sr":
        raise ValueError(
            f"Expected a residual_sr checkpoint, got model_config={model_config}"
        )

    package = {
        "model_state_dict": ema_state_dict,
        "weights_type": "ema",
        "architecture": "residual_sr",
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
        "num_features": model_config["num_features"],
        "num_blocks": model_config["num_blocks"],
        "scale": model_config["scale"],
        "channel_attention": model_config.get("channel_attention", False),
        "attention_reduction": model_config.get("attention_reduction", 8),
        "multiscale_block": model_config.get("multiscale_block", False),
        "rdb_block": model_config.get("rdb_block", False),
        "rdb_growth_rate": model_config.get("rdb_growth_rate", 16),
        "rdb_num_layers": model_config.get("rdb_num_layers", 3),
        "denoise_stem": model_config.get("denoise_stem", False),
        "denoise_stem_features": model_config.get("denoise_stem_features", 32),
        "denoise_stem_blocks": model_config.get("denoise_stem_blocks", 2),
        "noise_conditioning_config": checkpoint.get("noise_conditioning_config"),
        # No input normalization is applied anywhere in this project's
        # pipeline (see docs/dataset_notes.md / README "Input format"):
        # raw float32 NoisyLR values are fed to the model unchanged,
        # including values outside [0,1]. Output is likewise raw/unclipped
        # float32, matching the GT convention -- clip to [0,1] only if a
        # display/metric consumer requires it (src.baseline.metric_arrays
        # does this for evaluation; inference.py does not clip by default).
        "input_range": "raw float32, unnormalized, values may fall outside [0,1]",
        "output_range": "raw float32, unclipped, matching GT's expected [0,1] convention",
        "data_format": ".npy, float32, grayscale [H,W] (single channel)",
        "scale_factor": 2,
        "source_checkpoint": str(source),
        "source_epoch": checkpoint.get("epoch"),
        "source_best_val_psnr": checkpoint.get("best_val_psnr"),
        "reference_metrics": REFERENCE_METRICS,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, destination)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CHAMPION_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    package = export(args.source, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Exported inference-only weights: {args.output} ({size_mb:.2f} MiB)")
    print(f"Source checkpoint (untouched): {args.source}")
    print(
        f"Architecture: residual_sr, {package['num_features']}F/{package['num_blocks']}B, "
        f"scale={package['scale']}"
    )
    print(f"Reference metrics: {package['reference_metrics']}")


if __name__ == "__main__":
    main()
