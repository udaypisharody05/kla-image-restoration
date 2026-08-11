"""Analysis utilities for characterizing the GT -> NoisyLR degradation process.

Pure, deterministic helpers used by ``analyze_degradation.py``. Nothing here
trains, evaluates, or modifies a model -- this module exists to *measure* the
dataset's degradation, so its outputs are diagnostic statistics rather than
restoration performance.

The central quantity is the **residual**::

    r = NoisyLR - downsample(GT)

where ``downsample`` is one of several candidate 2x GT->LR models. ``r``
therefore bundles together everything the chosen downsampling model fails to
explain: sensor/shot noise, any pre-downsampling blur mismatch, resampling
phase error, and quantization. Characterizing ``r`` is how we infer which of
those actually dominate.
"""

from collections.abc import Callable, Iterable, Sequence
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


# --- Candidate GT -> LR downsampling models -----------------------------------


def _interpolate(gt: np.ndarray, mode: str, antialias: bool = False) -> np.ndarray:
    """Deterministic ``torch`` 2x downsample of a 2D float array."""
    tensor = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))[None, None]
    if mode in {"nearest-exact", "area"}:
        result = F.interpolate(tensor, scale_factor=0.5, mode=mode)
    else:
        result = F.interpolate(
            tensor, scale_factor=0.5, mode=mode, align_corners=False, antialias=antialias
        )
    return result[0, 0].numpy()


DOWNSAMPLERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Plain decimation of the two possible even/odd phases -- included to detect
    # whether the LR grid is a subsample of GT rather than an average of it.
    "subsample_even": lambda gt: np.ascontiguousarray(gt[0::2, 0::2]),
    "subsample_odd": lambda gt: np.ascontiguousarray(gt[1::2, 1::2]),
    "nearest": lambda gt: _interpolate(gt, "nearest-exact"),
    "bilinear": lambda gt: _interpolate(gt, "bilinear"),
    "bicubic": lambda gt: _interpolate(gt, "bicubic"),
    # At exactly 2x, "area" is the 2x2 box/mean filter.
    "area": lambda gt: _interpolate(gt, "area"),
    "bilinear_antialias": lambda gt: _interpolate(gt, "bilinear", antialias=True),
    "bicubic_antialias": lambda gt: _interpolate(gt, "bicubic", antialias=True),
}


def available_downsamplers() -> tuple[str, ...]:
    return tuple(DOWNSAMPLERS)


def downsample(gt: np.ndarray, method: str) -> np.ndarray:
    """Apply the named 2x downsampling model to a 2D GT array."""
    array = np.asarray(gt)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D GT array, got shape {array.shape}")
    if min(array.shape) < 2 or array.shape[0] % 2 or array.shape[1] % 2:
        raise ValueError(f"GT dimensions must be even and >= 2, got {array.shape}")
    if method not in DOWNSAMPLERS:
        raise ValueError(f"Unknown downsampling method {method!r}; expected one of {available_downsamplers()}")
    return DOWNSAMPLERS[method](array.astype(np.float32, copy=False)).astype(np.float64)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding; ``sigma <= 0`` is a no-op.

    Implemented locally (rather than pulled from SciPy) so the exact kernel
    truncation and padding used by the blur search are pinned here and stay
    reproducible.
    """
    array = np.asarray(image, dtype=np.float64)
    if sigma <= 0:
        return array.copy()
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(array, radius, mode="reflect")
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, horizontal)


# --- Match quality between an LR observation and a downsampled GT estimate -----


# A constant float64 image does not produce exactly zero variance -- e.g. a
# uniform 0.1 array measures ~1.9e-34 because 0.1 is not exactly representable.
# Anything below this threshold is numerically indistinguishable from constant
# for [0,1]-range image data, and must not be used as a least-squares divisor.
_DEGENERATE_VARIANCE = 1e-12


def _is_degenerate(values: np.ndarray) -> bool:
    """True when *values* carries no usable variation for a regression/correlation."""
    variance = float(np.var(values))
    scale = max(float(np.mean(np.asarray(values, dtype=np.float64) ** 2)), 1.0)
    return variance <= _DEGENERATE_VARIANCE * scale


def match_metrics(lr: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    """Diagnostic agreement between an observed LR image and a clean-LR estimate.

    NOT restoration performance: both inputs are known quantities and the point
    is to rank candidate degradation models, not to score a prediction. No
    clipping is applied -- raw values are compared exactly as stored.

    Correlation is undefined when either side is constant; it is reported as
    0.0 in that case rather than NaN, so the result always serializes to valid
    JSON.
    """
    observed = np.asarray(lr, dtype=np.float64)
    predicted = np.asarray(estimate, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {observed.shape} vs {predicted.shape}")
    difference = observed - predicted
    mse = float(np.mean(difference**2))
    if _is_degenerate(observed) or _is_degenerate(predicted):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(observed.ravel(), predicted.ravel())[0, 1])
    return {
        "mae": float(np.mean(np.abs(difference))),
        "mse": mse,
        "psnr": float("inf") if mse == 0 else float(10.0 * np.log10(data_range**2 / mse)),
        "correlation": correlation,
        "bias": float(np.mean(difference)),
    }


def fit_gain_bias(lr: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    """Least-squares ``a, b`` for ``lr ~= a * estimate + b``.

    A degenerate (effectively constant) estimate yields ``a = 1`` with ``b`` set
    to the mean difference, so the caller always gets a usable affine correction
    instead of a gain fitted against floating-point round-off.
    """
    observed = np.asarray(lr, dtype=np.float64).ravel()
    predicted = np.asarray(estimate, dtype=np.float64).ravel()
    if observed.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {observed.shape} vs {predicted.shape}")
    if _is_degenerate(predicted):
        return 1.0, float(np.mean(observed - predicted))
    covariance = float(np.mean((predicted - predicted.mean()) * (observed - observed.mean())))
    gain = covariance / float(np.var(predicted))
    return float(gain), float(observed.mean() - gain * predicted.mean())


def apply_gain_bias(estimate: np.ndarray, gain: float, bias: float) -> np.ndarray:
    return np.asarray(estimate, dtype=np.float64) * gain + bias


# --- Residual characterization --------------------------------------------------


def residual_moments(residual: np.ndarray) -> dict[str, float]:
    """Mean/std/variance/min/max/skewness/excess-kurtosis of a residual array."""
    values = np.asarray(residual, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Residual array is empty")
    mean = float(values.mean())
    std = float(values.std())
    centered = values - mean
    skewness = float(np.mean(centered**3) / std**3) if std > 0 else 0.0
    kurtosis = float(np.mean(centered**4) / std**4 - 3.0) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "variance": float(std**2),
        "min": float(values.min()),
        "max": float(values.max()),
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
    }


def autocorrelation_at_offsets(
    residual: np.ndarray, offsets: Sequence[tuple[int, int]]
) -> dict[str, float]:
    """Normalized spatial autocorrelation of *residual* at ``(dy, dx)`` offsets.

    1.0 would mean perfectly correlated neighbours; 0.0 means white/iid noise.
    """
    values = np.asarray(residual, dtype=np.float64)
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    results: dict[str, float] = {}
    for dy, dx in offsets:
        if dy < 0 or dx < 0:
            raise ValueError(f"Offsets must be non-negative, got {(dy, dx)}")
        height, width = centered.shape
        if dy >= height or dx >= width:
            raise ValueError(f"Offset {(dy, dx)} exceeds residual shape {centered.shape}")
        shifted = centered[dy:, dx:]
        base = centered[: height - dy, : width - dx]
        results[f"{dy}_{dx}"] = 0.0 if variance <= 0 else float(np.mean(base * shifted) / variance)
    return results


def power_spectrum(residual: np.ndarray) -> np.ndarray:
    """Zero-frequency-centred 2D power spectrum ``|FFT(residual)|^2``."""
    values = np.asarray(residual, dtype=np.float64)
    return np.abs(np.fft.fftshift(np.fft.fft2(values - values.mean()))) ** 2


def radial_profile(spectrum: np.ndarray) -> np.ndarray:
    """Azimuthally averaged profile of a centred 2D spectrum, indexed by radius."""
    values = np.asarray(spectrum, dtype=np.float64)
    height, width = values.shape
    y_grid, x_grid = np.indices((height, width))
    radius = np.hypot(y_grid - height // 2, x_grid - width // 2).astype(int)
    totals = np.bincount(radius.ravel(), weights=values.ravel())
    counts = np.bincount(radius.ravel())
    return totals / np.maximum(counts, 1)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, same shape as *image*."""
    values = np.asarray(image, dtype=np.float64)
    dy, dx = np.gradient(values)
    return np.hypot(dy, dx)


def local_variance(image: np.ndarray, window: int = 3) -> np.ndarray:
    """Sliding-window variance via integral-image-free box means (reflect padded)."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    values = np.asarray(image, dtype=np.float64)
    radius = window // 2
    padded = np.pad(values, radius, mode="reflect")
    padded_squared = padded**2
    kernel = np.ones(window) / window

    def box_mean(array: np.ndarray) -> np.ndarray:
        horizontal = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, array)
        return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, horizontal)

    mean = box_mean(padded)
    mean_of_squares = box_mean(padded_squared)
    return np.maximum(mean_of_squares - mean**2, 0.0)


def content_hash(array: np.ndarray) -> str:
    """Stable SHA-256 of an array's float32 bytes, for exact-duplicate detection."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def perceptual_signature(image: np.ndarray, size: int = 8) -> str:
    """Coarse thumbnail signature for *near*-duplicate scene detection.

    Downsamples to ``size x size`` and records which cells exceed the mean --
    a classic average-hash. Two different noise realizations of the same clean
    scene collide here even though their exact bytes never would.

    This is a **candidate generator only**: an 8x8 average-hash also collides
    for genuinely unrelated images, so every candidate must be confirmed by a
    real pixel-space comparison before being called a duplicate (see
    ``connected_components``' use in ``analyze_degradation.py``).
    """
    values = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(values))[None, None]
    thumbnail = F.interpolate(tensor, size=(size, size), mode="area")[0, 0].numpy()
    bits = (thumbnail > thumbnail.mean()).ravel()
    return "".join("1" if bit else "0" for bit in bits)


def connected_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Group *nodes* into connected components given undirected *edges*.

    Used to merge pairwise near-duplicate confirmations into whole scene groups
    (if A matches B and B matches C, all three are one scene). Each component is
    returned sorted, and components are ordered by their smallest member so the
    output is deterministic.
    """
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left not in parent or right not in parent:
            raise KeyError(f"Edge ({left!r}, {right!r}) references an unknown node")
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    grouped: dict[str, list[str]] = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(members) for members in grouped.values()), key=lambda group: group[0])


# --- Streaming accumulators -----------------------------------------------------


class MomentAccumulator:
    """Exact streaming mean/std/skewness/kurtosis via raw power sums.

    Percentiles come from a fixed-width histogram instead of retained values, so
    memory stays constant across the full ~52M-pixel dataset.
    """

    def __init__(self, low: float = -1.5, high: float = 1.5, bins: int = 6000) -> None:
        if high <= low:
            raise ValueError("high must exceed low")
        self.low = low
        self.high = high
        self.edges = np.linspace(low, high, bins + 1)
        self.histogram = np.zeros(bins, dtype=np.int64)
        self.count = 0
        self.power_sums = np.zeros(4, dtype=np.float64)
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        self.count += flat.size
        for power in range(1, 5):
            self.power_sums[power - 1] += float(np.sum(flat**power))
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.histogram += np.histogram(flat, bins=self.edges)[0]

    def percentile(self, fraction: float) -> float:
        """Histogram-interpolated percentile; resolution is one bin width."""
        if self.histogram.sum() == 0:
            return float("nan")
        cumulative = np.cumsum(self.histogram)
        target = fraction * cumulative[-1]
        index = int(np.searchsorted(cumulative, target))
        index = min(index, len(self.histogram) - 1)
        return float((self.edges[index] + self.edges[index + 1]) / 2)

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No values accumulated")
        n = float(self.count)
        m1, m2, m3, m4 = (self.power_sums / n).tolist()
        mean = m1
        variance = max(m2 - mean**2, 0.0)
        std = float(np.sqrt(variance))
        central3 = m3 - 3 * mean * m2 + 2 * mean**3
        central4 = m4 - 4 * mean * m3 + 6 * mean**2 * m2 - 3 * mean**4
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": std,
            "variance": float(variance),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "skewness": float(central3 / std**3) if std > 0 else 0.0,
            "excess_kurtosis": float(central4 / std**4 - 3.0) if std > 0 else 0.0,
            "percentiles": {
                name: self.percentile(fraction)
                for name, fraction in (
                    ("p0.1", 0.001), ("p1", 0.01), ("p5", 0.05), ("p25", 0.25),
                    ("p50", 0.50), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99), ("p99.9", 0.999),
                )
            },
        }


class BinnedAccumulator:
    """Streaming mean/variance of a value stratified by a binning variable."""

    def __init__(self, edges: np.ndarray) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)
        size = len(self.edges) - 1
        if size < 1:
            raise ValueError("edges must define at least one bin")
        self.counts = np.zeros(size, dtype=np.int64)
        self.sums = np.zeros(size, dtype=np.float64)
        self.squares = np.zeros(size, dtype=np.float64)

    def update(self, binning_values: np.ndarray, values: np.ndarray) -> None:
        keys = np.asarray(binning_values, dtype=np.float64).ravel()
        observations = np.asarray(values, dtype=np.float64).ravel()
        if keys.shape != observations.shape:
            raise ValueError("Binning values and observations must have equal size")
        index = np.clip(np.digitize(keys, self.edges) - 1, 0, len(self.counts) - 1)
        np.add.at(self.counts, index, 1)
        np.add.at(self.sums, index, observations)
        np.add.at(self.squares, index, observations**2)

    def summary(self, min_count: int = 1) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for i, count in enumerate(self.counts):
            if count < min_count:
                continue
            mean = self.sums[i] / count
            variance = max(self.squares[i] / count - mean**2, 0.0)
            rows.append(
                {
                    "bin_low": float(self.edges[i]),
                    "bin_high": float(self.edges[i + 1]),
                    "bin_center": float((self.edges[i] + self.edges[i + 1]) / 2),
                    "count": int(count),
                    "mean": float(mean),
                    "std": float(np.sqrt(variance)),
                    "variance": float(variance),
                }
            )
        return rows


def fit_noise_variance_model(centers: Iterable[float], variances: Iterable[float]) -> dict[str, float]:
    """Least-squares fit of ``var(I) = c0 + c1*I + c2*I^2`` to binned noise variance.

    Distinguishes the classic regimes: ``c1``-dominated is Poisson/shot-like,
    ``c2``-dominated is multiplicative/speckle-like, ``c0``-dominated is
    additive/homoscedastic.
    """
    intensity = np.asarray(list(centers), dtype=np.float64)
    variance = np.asarray(list(variances), dtype=np.float64)
    if intensity.size < 3:
        raise ValueError("Need at least 3 bins to fit a quadratic variance model")
    design = np.stack([np.ones_like(intensity), intensity, intensity**2], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, variance, rcond=None)
    predicted = design @ coefficients
    residual_ss = float(np.sum((variance - predicted) ** 2))
    total_ss = float(np.sum((variance - variance.mean()) ** 2))
    return {
        "constant": float(coefficients[0]),
        "linear": float(coefficients[1]),
        "quadratic": float(coefficients[2]),
        "r_squared": float(1.0 - residual_ss / total_ss) if total_ss > 0 else 1.0,
    }
