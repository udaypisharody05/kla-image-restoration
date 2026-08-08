"""Torch-tensor PSNR/SSIM built on the project's established baseline metrics.

Reuses ``src.baseline.peak_signal_noise_ratio`` and
``src.baseline.structural_similarity_index`` per grayscale image so neural
validation numbers stay directly comparable with the bicubic baseline.

Predictions are clipped to ``[0, data_range]`` before scoring by default,
matching ``src.baseline.metric_arrays``/``evaluate_baseline.py`` (which clip
the bicubic prediction the same way). Targets are never clipped, again
matching that convention. A freshly initialized or early-training model can
output values far outside ``[0,1]``; scoring those raw would make PSNR/SSIM
incomparable to the bicubic baseline's numbers, not just numerically ugly.
"""

import numpy as np
import torch

from .baseline import peak_signal_noise_ratio, structural_similarity_index


def _grayscale_images(
    tensor: torch.Tensor, clip: bool = False, data_range: float = 1.0
) -> list[np.ndarray]:
    """Yield one float64 [H,W] array per (batch, channel) image."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a [B,C,H,W] or [C,H,W] tensor, got shape {tuple(tensor.shape)}")
    array = tensor.detach().to(torch.float32).cpu().numpy().astype(np.float64)
    if clip:
        array = np.clip(array, 0.0, data_range)
    return [array[b, c] for b in range(array.shape[0]) for c in range(array.shape[1])]


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    clip_prediction: bool = True,
) -> float:
    """Mean PSNR (dB) over a batch of grayscale images in ``[0,1]`` range.

    Identical prediction/target images contribute ``inf``, which then makes the
    batch mean ``inf`` as well -- the perfect-match case is surfaced, not hidden.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    predicted_images = _grayscale_images(prediction, clip=clip_prediction, data_range=data_range)
    target_images = _grayscale_images(target)
    values = [
        peak_signal_noise_ratio(p, t, data_range=data_range)
        for p, t in zip(predicted_images, target_images)
    ]
    return float(np.mean(values))


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    clip_prediction: bool = True,
) -> float:
    """Mean grayscale SSIM over a batch of images in ``[0,1]`` range (~1.0 for identical inputs)."""
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    predicted_images = _grayscale_images(prediction, clip=clip_prediction, data_range=data_range)
    target_images = _grayscale_images(target)
    values = [
        structural_similarity_index(p, t, data_range=data_range)
        for p, t in zip(predicted_images, target_images)
    ]
    return float(np.mean(values))
