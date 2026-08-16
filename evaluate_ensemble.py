"""Evaluate a weighted ensemble of two OR MORE checkpoints on the canonical
validation split.

Inference-only: loads already-trained checkpoints (e.g. Experiment 6 and
Experiment 9, or several checkpoints from the *same* run for Phase 7
prediction averaging), runs each independently, weighted-averages their raw
predictions (``src.ensemble.weighted_average_predictions``), and scores the
ensemble with the exact same canonical PSNR/SSIM used everywhere else in this
project. No retraining, no checkpoint modification.

Two checkpoint-selection interfaces are supported, and both produce identical
results when given the same two checkpoints/weights:

- ``--checkpoint-a``/``--checkpoint-b``/``--weight-a``/``--weight-b`` -- the
  original two-checkpoint interface (Experiments 11/18 used this exact CLI;
  kept byte-for-byte unchanged for reproducibility).
- ``--checkpoints ckpt1 ckpt2 [ckpt3 ...]`` with optional ``--weights w1 w2
  [w3 ...]`` (defaults to equal weighting) -- generalizes to any number of
  checkpoints, e.g. several epochs of the same run (Phase 7) or more than two
  competitive models (Phase 8).

``--alpha-search`` sweeps a two-model blend weight over ``0.0, 0.05, ...,
1.0`` (applies to the first two checkpoints given via either interface),
reports each raw model's own PSNR, the best alpha, and rejects the ensemble
if it does not beat the stronger individual model -- per the project's
ensemble acceptance rule.

``--tta {none,x8}`` optionally routes *each* model through the existing x8
geometric self-ensemble (``src/tta.py::predict_x8``) before predictions are
combined -- the same TTA implementation ``evaluate_checkpoint.py`` uses, not
a second copy of it.
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from evaluate_checkpoint import load_model
from inspect_dataset import configured_data_dir
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.ensemble import weighted_average_predictions
from src.metrics import psnr, ssim
from src.splits import split_pairs
from src.tta import predict_x8
from train import BICUBIC_PSNR_DB, BICUBIC_SSIM, select_device


@torch.no_grad()
def validate_single_model(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device, tta: str = "none"
) -> dict[str, float]:
    """Raw (non-ensembled) validation of one model -- used to report each
    individual checkpoint's own PSNR alongside the ensemble result, so the
    ensemble can be judged against the stronger of its inputs."""
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = predict_x8(model, inputs) if tta == "x8" else model(inputs)
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
def validate_ensemble(
    model_a: nn.Module,
    model_b: nn.Module,
    loader,
    weights: list[float],
    loss_fn: nn.Module,
    device: torch.device,
    tta: str = "none",
) -> dict[str, float]:
    """Same aggregation convention as ``train.validate``/``evaluate_checkpoint.validate_x8``,
    scoring the weighted average of the two models' raw predictions instead of a
    single model's output. Both models always run in eval mode; TTA (if any) is
    delegated entirely to ``predict_x8``.

    Two-model signature kept byte-for-byte unchanged (Experiments 11/18 and
    ``tests/test_ensemble_unit.py`` call this positionally); see
    ``validate_ensemble_n`` for the general N-model version this delegates to.
    """
    metrics, _ = _validate_ensemble_n([model_a, model_b], loader, weights, loss_fn, device, tta)
    return metrics


@torch.no_grad()
def validate_ensemble_n(
    models: list[nn.Module],
    loader,
    weights: list[float],
    loss_fn: nn.Module,
    device: torch.device,
    tta: str = "none",
) -> dict[str, float]:
    """General N-model (N >= 2) version of ``validate_ensemble``, for Phase 7
    prediction averaging across more than two checkpoints (e.g. several
    epochs of the same run) or Phase 8 ensembles of more than two competitive
    models."""
    metrics, _ = _validate_ensemble_n(models, loader, weights, loss_fn, device, tta)
    return metrics


def _validate_ensemble_n(
    models: list[nn.Module],
    loader,
    weights: list[float],
    loss_fn: nn.Module,
    device: torch.device,
    tta: str,
) -> tuple[dict[str, float], int]:
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models to ensemble, got {len(models)}")
    for model in models:
        model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = [
            predict_x8(model, inputs) if tta == "x8" else model(inputs) for model in models
        ]
        outputs = weighted_average_predictions(predictions, weights)
        loss = loss_fn(outputs, targets)
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += psnr(outputs, targets) * batch_size
        total_ssim += ssim(outputs, targets) * batch_size
        total_count += batch_size
    return (
        {
            "loss": total_loss / total_count,
            "psnr": total_psnr / total_count,
            "ssim": total_ssim / total_count,
        },
        total_count,
    )


def alpha_grid(step: float = 0.05) -> list[float]:
    """``[0.0, 0.05, 0.10, ..., 1.0]`` by default -- rounded to avoid float
    accumulation artifacts (e.g. ``0.15000000000000002``)."""
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    steps = round(1.0 / step)
    return [round(index * step, 10) for index in range(steps + 1)]


def run_alpha_search(
    model_a: nn.Module,
    model_b: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    tta: str,
    step: float = 0.05,
) -> dict:
    """Sweep ``prediction = alpha*A + (1-alpha)*B`` over the standard grid,
    report both raw models' own PSNR, the best ensemble alpha, and whether it
    beats the stronger individual model -- the project's ensemble acceptance
    rule (reject if the ensemble does not beat the stronger raw model).
    """
    raw_a = validate_single_model(model_a, loader, loss_fn, device, tta)
    raw_b = validate_single_model(model_b, loader, loss_fn, device, tta)
    stronger_psnr = max(raw_a["psnr"], raw_b["psnr"])
    stronger_name = "A" if raw_a["psnr"] >= raw_b["psnr"] else "B"

    results = []
    for alpha in alpha_grid(step):
        # weighted_average_predictions requires strictly positive weights
        # (src/ensemble.py), so the pure-model endpoints reuse the raw
        # single-model metrics already computed above instead of passing a
        # zero weight through the ensemble path.
        if alpha == 0.0:
            metrics = raw_b
        elif alpha == 1.0:
            metrics = raw_a
        else:
            metrics = validate_ensemble(
                model_a, model_b, loader, [alpha, 1.0 - alpha], loss_fn, device, tta
            )
        results.append({"alpha": alpha, **metrics})

    best = max(results, key=lambda entry: entry["psnr"])
    accepted = best["psnr"] > stronger_psnr
    return {
        "raw_a": raw_a,
        "raw_b": raw_b,
        "stronger_name": stronger_name,
        "stronger_psnr": stronger_psnr,
        "grid": results,
        "best": best,
        "accepted": accepted,
    }


def _resolve_checkpoints_and_weights(args: argparse.Namespace) -> tuple[list[Path], list[float]]:
    """Merge the two supported CLI interfaces into one checkpoint/weight list.

    ``--checkpoints``/``--weights`` take precedence when given; otherwise
    falls back to the original ``--checkpoint-a``/``--checkpoint-b`` pair, so
    every historical Experiment 11/18 command still works unmodified.
    """
    if args.checkpoints:
        checkpoints = args.checkpoints
        weights = args.weights if args.weights else [1.0] * len(checkpoints)
        if len(weights) != len(checkpoints):
            raise ValueError(
                f"Got {len(checkpoints)} --checkpoints but {len(weights)} --weights -- "
                "counts must match"
            )
        return checkpoints, weights
    return [args.checkpoint_a, args.checkpoint_b], [args.weight_a, args.weight_b]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-a", type=Path, default=Path("checkpoints/exp6_crop96/checkpoint_best.pt")
    )
    parser.add_argument(
        "--checkpoint-b", type=Path, default=Path("checkpoints/exp9_edsr_lite/checkpoint_best.pt")
    )
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Two or more checkpoint paths to ensemble (overrides --checkpoint-a/-b). "
            "E.g. several epochs of the same run for prediction averaging, or more than "
            "two competitive models."
        ),
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights matching --checkpoints, in order (defaults to equal weighting).",
    )
    parser.add_argument(
        "--alpha-search",
        action="store_true",
        help=(
            "Sweep a two-model blend weight over 0.0..1.0 (step --alpha-step) instead of "
            "using a single fixed weight; applies to the first two checkpoints. Reports "
            "each raw model's PSNR, the best alpha, and rejects the ensemble unless it "
            "beats the stronger individual model."
        ),
    )
    parser.add_argument("--alpha-step", type=float, default=0.05, help="Grid step for --alpha-search")
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
        help="Apply x8 geometric self-ensemble to each model before combining.",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    checkpoint_paths, weights = _resolve_checkpoints_and_weights(args)
    if len(checkpoint_paths) < 2:
        raise ValueError(f"Need at least 2 checkpoints to ensemble, got {len(checkpoint_paths)}")

    models, checkpoints = [], []
    for path in checkpoint_paths:
        model, checkpoint = load_model(path, device)
        models.append(model)
        checkpoints.append(checkpoint)

    scales = {checkpoint["model_config"]["scale"] for checkpoint in checkpoints}
    if len(scales) != 1:
        raise ValueError(f"Scale mismatch between checkpoints: {scales}")
    scale = scales.pop()

    for index, (path, checkpoint) in enumerate(zip(checkpoint_paths, checkpoints)):
        architecture = checkpoint["model_config"].get("architecture", "residual_sr")
        print(
            f"Checkpoint {index} ({path}): architecture={architecture}, "
            f"epoch={checkpoint.get('epoch')}, weight={weights[index]}"
        )
    print(f"TTA: {'x8 geometric self-ensemble per model' if args.tta == 'x8' else 'disabled'}")

    pairs = discover_pairs(discover_layout(args.data_dir)).pairs
    _, validation_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if args.max_val_samples is not None:
        validation_pairs = validation_pairs[: args.max_val_samples]
    validation_dataset = PairedRestorationDataset(validation_pairs, scale=scale)
    validation_loader = create_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Validation samples: {len(validation_dataset)}")

    loss_fn = nn.L1Loss()
    start_time = time.perf_counter()

    if args.alpha_search:
        if len(models) != 2:
            raise ValueError("--alpha-search requires exactly 2 checkpoints")
        result = run_alpha_search(
            models[0], models[1], validation_loader, loss_fn, device, args.tta, args.alpha_step
        )
        print(f"Model A raw: PSNR={result['raw_a']['psnr']:.4f} dB, SSIM={result['raw_a']['ssim']:.6f}")
        print(f"Model B raw: PSNR={result['raw_b']['psnr']:.4f} dB, SSIM={result['raw_b']['ssim']:.6f}")
        print(f"Stronger individual model: {result['stronger_name']} ({result['stronger_psnr']:.4f} dB)")
        for entry in result["grid"]:
            print(
                f"  alpha={entry['alpha']:.2f}: PSNR={entry['psnr']:.4f} dB, "
                f"SSIM={entry['ssim']:.6f}, L1={entry['loss']:.6f}"
            )
        best = result["best"]
        print(f"Best alpha: {best['alpha']:.2f} -> PSNR={best['psnr']:.4f} dB, SSIM={best['ssim']:.6f}")
        gain = best["psnr"] - result["stronger_psnr"]
        print(f"Gain vs stronger individual model: {gain:+.4f} dB")
        print("Verdict: ACCEPTED (ensemble beats the stronger model)" if result["accepted"]
              else "Verdict: REJECTED (ensemble does not beat the stronger model)")
    else:
        metrics = validate_ensemble_n(models, validation_loader, weights, loss_fn, device, tta=args.tta)
        elapsed_seconds = time.perf_counter() - start_time
        print(f"Val L1: {metrics['loss']:.6f}")
        print(f"Val PSNR: {metrics['psnr']:.4f} dB")
        print(f"Val SSIM: {metrics['ssim']:.6f}")
        print(f"Bicubic PSNR: {BICUBIC_PSNR_DB:.4f} dB")
        print(f"Bicubic SSIM: {BICUBIC_SSIM:.6f}")
        print(f"PSNR vs bicubic: {metrics['psnr'] - BICUBIC_PSNR_DB:+.4f} dB")
        print(f"SSIM vs bicubic: {metrics['ssim'] - BICUBIC_SSIM:+.6f}")
        print(f"Elapsed: {elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
