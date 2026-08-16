"""Evaluate a saved neural restoration checkpoint on the fixed validation split.

Loads either a full ``train.py`` training checkpoint (e.g.
``checkpoints/<exp>/checkpoint_best.pt`` -- includes optimizer/EMA/scheduler
state, not tracked in Git) or an exported inference-only weights package
(``export_final_weights.py`` output, e.g. ``weights/residualsr_final_ema.pt``
-- the tracked public artifact), reconstructs the matching model, evaluates
the deterministic validation split, and prints L1/PSNR/SSIM alongside the
established bicubic baseline for comparison. Does not run inference on the
competition test set.

Regardless of which reconstruction loss a checkpoint was *trained* with (see
``src/losses.py`` and ``--loss`` in ``train.py``), evaluation always reports
actual L1 as its "Val L1" diagnostic -- this keeps that number directly
comparable across every experiment, including ones trained with Charbonnier
loss. The checkpoint's training loss is reported separately and explicitly.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from inference import _model_config_from_package
from inspect_dataset import configured_data_dir
from src.baseline import LPIPSMetric, LPIPSUnavailableError
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.losses import loss_label
from src.metrics import psnr, ssim
from src.models import build_model
from src.noise_conditioning import wrap_for_conditioning
from src.splits import split_pairs
from src.tta import predict_x8
from train import BICUBIC_PSNR_DB, BICUBIC_SSIM, select_device, validate


@torch.no_grad()
def validate_x8(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Same metric conventions/aggregation as ``train.validate``, but scores the
    x8 geometric self-ensemble average instead of a single forward pass.

    Uses the exact same ``src.metrics.psnr``/``ssim`` (which clip only the
    prediction, by default) -- no second metric implementation. The averaged
    prediction from ``predict_x8`` is raw/unclipped; clipping (if any) happens
    inside ``psnr``/``ssim`` themselves, identically to the no-TTA path.
    """
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = predict_x8(model, inputs)
        loss = loss_fn(outputs, targets)
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += psnr(outputs, targets) * batch_size
        total_ssim += ssim(outputs, targets) * batch_size
        total_count += batch_size
    return {
        "loss": total_loss / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
    }


@torch.no_grad()
def compute_lpips(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lpips_metric: LPIPSMetric,
    tta: str = "none",
) -> float:
    """Mean LPIPS over *loader*, on the same validation set/predictions PSNR/SSIM
    already scored -- a deliberately separate pass (not fused into
    ``train.validate``/``validate_x8``) so LPIPS is strictly additive and can
    never change existing PSNR/SSIM/L1 numbers or aggregation. Predictions are
    unclipped/raw exactly like the PSNR/SSIM path; ``LPIPSMetric`` clips
    internally (see ``src.baseline.metric_arrays``), matching the project's
    established "clip only at metric time" convention.
    """
    model.eval()
    total_lpips = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = predict_x8(model, inputs) if tta == "x8" else model(inputs)
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        for index in range(outputs_np.shape[0]):
            total_lpips += lpips_metric(outputs_np[index, 0], targets_np[index, 0])
            total_count += 1
    return total_lpips / total_count


def _load_full_checkpoint(
    checkpoint: dict, device: torch.device, prefer_ema: bool
) -> tuple[nn.Module, dict]:
    """Reconstruct the model described by a full ``train.py`` checkpoint's
    ``model_config`` (see ``load_model`` for the EMA/raw-weights selection
    and noise-conditioning wrapping this shares with the exported-package
    path below)."""
    base_model = build_model(checkpoint["model_config"]).to(device)
    model = wrap_for_conditioning(base_model, checkpoint.get("noise_conditioning_config"))
    ema_state_dict = checkpoint.get("ema_state_dict")
    if prefer_ema and ema_state_dict is not None:
        model.load_state_dict(ema_state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _load_exported_package(package: dict, device: torch.device) -> tuple[nn.Module, dict]:
    """Reconstruct the model described by an ``export_final_weights.py``
    package (e.g. ``weights/residualsr_final_ema.pt``) -- the tracked, public
    artifact a fresh clone actually has, unlike ``checkpoints/`` which is
    gitignored.

    Reuses ``inference.py``'s ``_model_config_from_package`` (the same
    architecture-metadata extraction ``inference.py`` uses to load this file
    for standalone restoration) rather than a second, potentially-diverging
    reconstruction of the same config -- one interpretation of the package
    format, not two. The package stores exactly one weight set (labelled by
    ``weights_type``, ``"ema"`` for the submitted champion), so unlike a full
    training checkpoint there is no separate raw-vs-EMA choice to make here.

    Returns a ``checkpoint``-shaped dict carrying a synthesized
    ``model_config`` (plus ``epoch``/``best_val_psnr``/``loss_config``/
    ``ema_state_dict`` derived from the package's own fields) so every
    existing caller of ``load_model`` -- this module's ``main()``,
    ``evaluate_ensemble.py``, ``evaluate_group_aware.py``, ``infer_test.py``
    -- keeps working against exported packages with no changes of their own.
    """
    model_config = _model_config_from_package(package)
    base_model = build_model(model_config).to(device)
    model = wrap_for_conditioning(base_model, package.get("noise_conditioning_config"))
    model.load_state_dict(package["model_state_dict"])
    model.eval()

    normalized = dict(package)
    normalized["model_config"] = model_config
    normalized["epoch"] = package.get("source_epoch")
    normalized["best_val_psnr"] = package.get("source_best_val_psnr")
    normalized.setdefault("loss_config", {"name": "l1"})
    normalized["ema_state_dict"] = (
        package["model_state_dict"] if package.get("weights_type") == "ema" else None
    )
    normalized["source_format"] = "exported_package"
    return model, normalized


def load_model(
    checkpoint_path: Path, device: torch.device, prefer_ema: bool = True
) -> tuple[nn.Module, dict]:
    """Reconstruct the model described by *checkpoint_path*, whichever of the
    two supported file shapes it is:

    - a full ``train.py`` training checkpoint (has a ``model_config`` key,
      e.g. ``checkpoints/exp23_ema_extended90/checkpoint_best.pt`` --
      gitignored, present only where training was actually run), or
    - an exported inference-only weights package (has a ``model_state_dict``
      key but no ``model_config``, e.g.
      ``weights/residualsr_final_ema.pt`` -- the tracked public artifact a
      fresh clone actually has).

    ``build_model()`` reads ``model_config["architecture"]`` (missing ->
    ResidualSRNet, the only interpretation every Experiment 1-8 checkpoint
    ever used), so this one call reconstructs every architecture -- infer_test.py
    reuses this same function and needs no changes either.

    When a full checkpoint has EMA state (``ema_state_dict``, saved by an
    Experiment 19-style ``--ema`` run) and *prefer_ema* is true (the default),
    those EMA weights are loaded instead of the live/raw ``model_state_dict``
    -- the EMA weights are what validation actually scored to produce the
    checkpoint's recorded PSNR, so this is what "the checkpoint" should mean
    for evaluation/inference by default. Historical and non-EMA checkpoints
    have no ``ema_state_dict`` (``None``), so they fall through to the exact
    prior behavior unchanged. Pass ``prefer_ema=False`` to force loading the
    live/raw weights instead, e.g. for diagnostics -- this never changes
    which checkpoint was selected as "best" during training, only which
    weights get loaded from it afterward. An exported package stores only one
    weight set, so *prefer_ema* has no effect on that path (see
    ``_load_exported_package``).

    When the checkpoint/package has a ``noise_conditioning_config``
    (Experiment 25), the reconstructed model is wrapped with
    ``NoiseConditionedModel`` so callers can keep passing plain
    single-channel LR tensors -- the [lr, sigma] expansion happens
    automatically inside the model, identically to how training built it.
    ``infer_test.py``, ``evaluate_group_aware.py``, and x8 TTA all reuse this
    one function and therefore need no changes of their own.

    Raises ``ValueError`` (not a bare ``KeyError``) for a file that is
    neither shape.
    """
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" in raw:
        return _load_full_checkpoint(raw, device, prefer_ema)
    if "model_state_dict" in raw:
        return _load_exported_package(raw, device)
    raise ValueError(
        f"{checkpoint_path} is neither a full train.py training checkpoint "
        "(missing 'model_config') nor an exported inference weights package from "
        "export_final_weights.py (missing 'model_state_dict') -- cannot reconstruct "
        "a model from it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Either a full train.py training checkpoint (e.g. "
            "checkpoints/<exp>/checkpoint_best.pt -- gitignored, present only where "
            "training was run) or an exported inference weights package from "
            "export_final_weights.py (e.g. weights/residualsr_final_ema.pt -- the "
            "tracked public artifact). Both reconstruct the model automatically."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=configured_data_dir())
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional subset limit")
    parser.add_argument(
        "--tta",
        type=str,
        choices=["none", "x8"],
        default="none",
        help="Test-time augmentation. 'none' (default) is byte-for-byte the original evaluation.",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help=(
            "Attempt optional LPIPS evaluation (requires the 'lpips' package and its "
            "pretrained weights). Runs as a separate pass and never changes the "
            "PSNR/SSIM/L1 numbers above. Reported as 'unavailable' rather than failing "
            "the run if the optional dependency/weights are missing."
        ),
    )
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help=(
            "Force loading the live/raw model_state_dict even if the checkpoint has EMA "
            "weights (default: prefer EMA weights when present, matching what training's "
            "own validation scored). Lets the EMA vs. raw comparison be made explicit and "
            "unambiguous instead of only ever seeing whichever the checkpoint happens to "
            "prefer by default."
        ),
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device, prefer_ema=not args.raw_weights)
    has_ema = checkpoint.get("ema_state_dict") is not None
    is_exported_package = checkpoint.get("source_format") == "exported_package"
    if is_exported_package:
        weights_used = f"{checkpoint.get('weights_type', 'unknown')} (exported package -- only one weight set is packaged)"
        if args.raw_weights and has_ema:
            print(
                "Note: --raw-weights requested, but the exported weights package only "
                "contains EMA weights (no separate raw/live weights are packaged) -- "
                "evaluating the packaged EMA weights."
            )
    elif has_ema:
        weights_used = "raw/live" if args.raw_weights else "EMA"
    else:
        weights_used = "raw/live (checkpoint has no EMA weights)"
    print(f"Weights evaluated: {weights_used}")
    print(
        f"Loaded checkpoint {args.checkpoint} (epoch={checkpoint.get('epoch')}, "
        f"best_val_psnr={checkpoint.get('best_val_psnr')})"
    )
    # Checkpoints saved before loss selection existed have no stored
    # loss_config; every historical run (Experiments 1-3) used plain L1.
    loss_config = checkpoint.get("loss_config", {"name": "l1"})
    print(f"Training loss: {loss_label(loss_config['name'])} ({loss_config})")

    pairs = discover_pairs(discover_layout(args.data_dir)).pairs
    _, validation_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if args.max_val_samples is not None:
        validation_pairs = validation_pairs[: args.max_val_samples]
    validation_dataset = PairedRestorationDataset(
        validation_pairs, scale=checkpoint["model_config"]["scale"]
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.tta == "x8":
        print("TTA: x8 geometric self-ensemble enabled (8 D4 transforms, raw-prediction averaging)")
    else:
        print("TTA: disabled")

    # Always score with real L1 here (independent of loss_config above), so
    # this number stays directly comparable across every experiment even when
    # training losses differ. --tta none takes the exact original code path
    # (train.validate), unchanged, so default behavior stays byte-for-byte
    # identical to before TTA support existed.
    if args.tta == "x8":
        metrics = validate_x8(model, validation_loader, nn.L1Loss(), device)
    else:
        metrics = validate(model, validation_loader, nn.L1Loss(), device)

    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Val L1 (diagnostic, always L1 regardless of training loss): {metrics['loss']:.6f}")
    print(f"Val PSNR: {metrics['psnr']:.4f} dB")
    print(f"Val SSIM: {metrics['ssim']:.6f}")
    print(f"Bicubic PSNR: {BICUBIC_PSNR_DB:.4f} dB")
    print(f"Bicubic SSIM: {BICUBIC_SSIM:.6f}")
    psnr_delta = metrics["psnr"] - BICUBIC_PSNR_DB
    ssim_delta = metrics["ssim"] - BICUBIC_SSIM
    print(f"PSNR vs bicubic: {psnr_delta:+.4f} dB")
    print(f"SSIM vs bicubic: {ssim_delta:+.6f}")

    if args.lpips:
        try:
            lpips_metric = LPIPSMetric()
        except LPIPSUnavailableError as exc:
            print(f"LPIPS: unavailable ({exc})")
        else:
            lpips_value = compute_lpips(model, validation_loader, device, lpips_metric, tta=args.tta)
            print(f"Val LPIPS (lower is better): {lpips_value:.6f}")


if __name__ == "__main__":
    main()
