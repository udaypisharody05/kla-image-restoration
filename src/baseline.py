"""Classical bicubic super-resolution and evaluation metrics."""

from collections.abc import Callable
import math
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


class LPIPSUnavailableError(RuntimeError):
    """Raised when the optional LPIPS stack or pretrained weights are unavailable."""


def bicubic_upscale(
    image: np.ndarray,
    scale: int = 2,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Upscale a 2D image with float-mode bicubic interpolation.

    No clipping, normalization, or mutation of the input is performed.
    """
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {source.shape}")
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    target_shape = output_shape or (source.shape[0] * scale, source.shape[1] * scale)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"Invalid output shape: {target_shape}")

    float_image = Image.fromarray(source.astype(np.float32, copy=False), mode="F")
    resized = float_image.resize(
        (target_shape[1], target_shape[0]), resample=Image.Resampling.BICUBIC
    )
    return np.asarray(resized, dtype=np.float32).copy()


def metric_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    clip_prediction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create float64 metric views, clipping only the prediction when requested."""
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction_array.shape} vs "
            f"{target_array.shape}"
        )
    if prediction_array.ndim != 2:
        raise ValueError("Metrics expect 2D grayscale arrays")
    evaluated_prediction = prediction_array.astype(np.float64, copy=True)
    if clip_prediction:
        np.clip(evaluated_prediction, 0.0, 1.0, out=evaluated_prediction)
    return evaluated_prediction, target_array.astype(np.float64, copy=False)


def peak_signal_noise_ratio(
    prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    """Compute PSNR in decibels, returning infinity for identical arrays."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    mean_squared_error = float(np.mean(np.square(difference)))
    if mean_squared_error == 0.0:
        return math.inf
    return 10.0 * math.log10((data_range * data_range) / mean_squared_error)


def structural_similarity_index(
    prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    """Compute grayscale SSIM using scikit-image and an explicit data range."""
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 2:
        raise ValueError("SSIM expects equally shaped 2D arrays")
    minimum_dimension = min(prediction_array.shape)
    window_size = min(7, minimum_dimension)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires image dimensions of at least 3 pixels")
    return float(
        structural_similarity(
            target_array,
            prediction_array,
            data_range=data_range,
            win_size=window_size,
        )
    )


class LPIPSMetric:
    """Optional LPIPS metric with transformations isolated from raw preprocessing."""

    def __init__(self, network: str = "alex") -> None:
        try:
            import lpips  # type: ignore[import-not-found]
            import torch
        except ImportError as exc:
            raise LPIPSUnavailableError(
                "LPIPS requires optional 'torch' and 'lpips' packages"
            ) from exc
        try:
            self._torch = torch
            self._model = lpips.LPIPS(net=network).eval()
        except Exception as exc:
            raise LPIPSUnavailableError(
                f"LPIPS model or pretrained weights are unavailable: {exc}"
            ) from exc

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction_array, target_array = metric_arrays(
            prediction, target, clip_prediction=True
        )

        def tensor(array: np.ndarray) -> Any:
            clipped = np.clip(array, 0.0, 1.0).astype(np.float32)
            rgb = np.repeat(clipped[None, None, :, :], 3, axis=1)
            return self._torch.from_numpy(rgb * 2.0 - 1.0)

        with self._torch.no_grad():
            value = self._model(tensor(prediction_array), tensor(target_array))
        return float(value.item())


def evaluate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    clip_prediction: bool = True,
    lpips_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, float | None]:
    """Evaluate a raw prediction without modifying it or its target."""
    evaluated_prediction, evaluated_target = metric_arrays(
        prediction, target, clip_prediction=clip_prediction
    )
    return {
        "psnr": peak_signal_noise_ratio(evaluated_prediction, evaluated_target),
        "ssim": structural_similarity_index(evaluated_prediction, evaluated_target),
        "lpips": (
            lpips_metric(evaluated_prediction, evaluated_target)
            if lpips_metric is not None
            else None
        ),
    }
