"""Signal-dependent synthetic noise augmentation (Experiment 24).

Synthesizes additional *noisy LR* realizations from clean GT images using the
degradation model measured in Experiment 22::

    clean_lr        = bicubic_downsample_2x(GT)
    variance(I)     = max(c0 + c1*I + c2*I^2, variance_floor)
    synthetic_lr    = clean_lr + sqrt(variance(clean_lr)) * epsilon

``epsilon`` is a zero-mean, unit-variance draw. Both a Gaussian and a Student-t
option exist; **Gaussian is the default**, chosen on measured evidence by
``analyze_synthetic_noise.py``. That is deliberately counter-intuitive and worth
stating plainly: the real *standardized* residual does look heavy-tailed (excess
kurtosis ~+3.46), which argues for Student-t, but what the augmentation actually
has to reproduce is the real *residual*, and heteroscedastic mixing alone already
contributes ~+2.45 excess kurtosis. Gaussian epsilon therefore lands at +2.45
against the real +3.52, while Student-t overshoots to +8.54. Gaussian also wins
on percentiles (mean absolute error 0.0144 vs 0.0198). Part of the standardized
residual's apparent tail weight is spread injected by imperfections in
``sigma(I)`` itself, not genuine tail weight in epsilon.

Nothing here clips: real ``NoisyLR`` files themselves contain values outside
``[0,1]`` (measured range roughly ``[-0.003, 1.33]``), so clipping the synthetic
stream would make it *less* like the real data, not more.

Randomness is drawn from an explicitly seeded ``numpy.random.Generator`` derived
per sample from ``(seed, epoch, index)`` via ``numpy.random.SeedSequence`` --
matching the repository's existing ``numpy.random.default_rng(seed)`` convention
in ``src/splits.py``. No global/process RNG is touched, so results are identical
regardless of DataLoader worker count while still giving a *different* noise
realization each epoch (which is the entire point of the augmentation).
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# Fitted on all 3,200 training pairs in Experiment 22 (R^2 = 0.9995 against the
# binned residual variance). See results/degradation_analysis/degradation_report.md.
VARIANCE_COEFFICIENTS: tuple[float, float, float] = (-6.19e-05, 0.00653, 0.0201)

# Student-t degrees of freedom implied by the measured standardized excess
# kurtosis k via the identity k = 6/(nu-4)  ->  nu = 4 + 6/3.459 = 5.735.
STUDENT_T_DEGREES_OF_FREEDOM: float = 5.735

DISTRIBUTIONS = ("gaussian", "student_t")


def bicubic_downsample(gt: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Deterministic 2x bicubic downsample of a ``[C,H,W]`` or ``[N,C,H,W]`` tensor.

    Numerically identical to ``src.degradation.downsample(gt, "bicubic")`` (the
    model Experiment 22 identified as the best GT->LR approximation); kept
    torch-native here so the training hot path avoids a numpy round trip.
    """
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    batched = gt.ndim == 4
    if gt.ndim not in (3, 4):
        raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got shape {tuple(gt.shape)}")
    tensor = gt if batched else gt.unsqueeze(0)
    result = F.interpolate(
        tensor.to(torch.float32), scale_factor=1.0 / scale, mode="bicubic", align_corners=False
    )
    return result if batched else result[0]


def noise_variance(
    intensity: torch.Tensor,
    coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS,
    variance_floor: float = 0.0,
) -> torch.Tensor:
    """Signal-dependent noise variance ``c0 + c1*I + c2*I^2``, floored.

    The fitted ``c0`` is slightly negative, so the floor is what keeps variance
    physical. ``variance_floor=0.0`` reproduces the Experiment 22 model exactly;
    a small positive floor compensates for the fit under-predicting sigma in the
    darkest intensity bin (see ``analyze_synthetic_noise.py``).
    """
    if variance_floor < 0:
        raise ValueError(f"variance_floor must be non-negative, got {variance_floor}")
    c0, c1, c2 = coefficients
    variance = c0 + c1 * intensity + c2 * intensity**2
    return torch.clamp(variance, min=variance_floor)


def noise_sigma(
    intensity: torch.Tensor,
    coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS,
    variance_floor: float = 0.0,
) -> torch.Tensor:
    """Per-pixel noise standard deviation; always finite and non-negative."""
    return torch.sqrt(noise_variance(intensity, coefficients, variance_floor))


def sample_epsilon(
    shape: tuple[int, ...],
    distribution: str,
    rng: np.random.Generator,
    degrees_of_freedom: float = STUDENT_T_DEGREES_OF_FREEDOM,
) -> np.ndarray:
    """Zero-mean, **unit-variance** noise draw of the requested distribution.

    A raw Student-t has variance ``nu/(nu-2)``, so it is rescaled by
    ``sqrt((nu-2)/nu)`` -- without that the synthetic noise would be
    systematically too strong and the variance model would no longer hold.
    """
    if distribution == "gaussian":
        return rng.standard_normal(shape)
    if distribution == "student_t":
        if degrees_of_freedom <= 2:
            raise ValueError(
                f"Student-t needs degrees_of_freedom > 2 for finite variance, got {degrees_of_freedom}"
            )
        scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        return rng.standard_t(degrees_of_freedom, size=shape) * scale
    raise ValueError(f"Unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}")


@dataclass
class SyntheticNoiseAugmentation:
    """Decides per sample whether to substitute a synthesized noisy LR image.

    ``probability`` 0.0 disables the augmentation entirely (the historical
    behavior); 1.0 always synthesizes. The real GT is never modified, and the
    real ``NoisyLR`` files on disk are never touched -- substitution happens in
    memory, before the paired crop/geometry transform, so the synthetic tensor
    is a drop-in replacement carrying the identical spatial layout as the real
    LR it replaces.
    """

    probability: float = 0.0
    seed: int = 42
    distribution: str = "gaussian"
    degrees_of_freedom: float = STUDENT_T_DEGREES_OF_FREEDOM
    coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS
    variance_floor: float = 0.0
    scale: int = 2
    _epoch: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0,1], got {self.probability}")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {self.distribution!r}; expected one of {DISTRIBUTIONS}"
            )
        if self.variance_floor < 0:
            raise ValueError("variance_floor must be non-negative")
        self.coefficients = tuple(float(c) for c in self.coefficients)

    def set_epoch(self, epoch: int) -> None:
        """Advance the noise stream so each epoch draws fresh realizations."""
        self._epoch = int(epoch)

    def generator(self, index: int) -> np.random.Generator:
        """Deterministic per-(seed, epoch, sample) generator -- worker-count independent."""
        return np.random.default_rng(
            np.random.SeedSequence([int(self.seed), int(self._epoch), int(index)])
        )

    def use_synthetic(self, index: int, rng: np.random.Generator | None = None) -> bool:
        """Whether sample *index* uses a synthetic LR this epoch (deterministic)."""
        if self.probability <= 0.0:
            return False
        if self.probability >= 1.0:
            return True
        draw = (rng if rng is not None else self.generator(index)).random()
        return bool(draw < self.probability)

    def synthesize(self, gt: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Build a synthetic noisy LR from a clean ``[C,H,W]`` GT tensor."""
        clean = bicubic_downsample(gt, self.scale)
        sigma = noise_sigma(clean, self.coefficients, self.variance_floor)
        epsilon = torch.from_numpy(
            sample_epsilon(tuple(clean.shape), self.distribution, rng, self.degrees_of_freedom)
        ).to(clean.dtype)
        return clean + sigma * epsilon

    def maybe_synthesize(self, gt: torch.Tensor, index: int) -> torch.Tensor | None:
        """Return a synthetic LR for this sample, or ``None`` to keep the real one.

        Both the decision and the noise come from one per-sample generator, so
        the whole augmentation is a pure function of ``(seed, epoch, index)``.
        """
        if self.probability <= 0.0:
            return None
        rng = self.generator(index)
        if not self.use_synthetic(index, rng):
            return None
        return self.synthesize(gt, rng)

    def config(self) -> dict:
        """Checkpoint-serializable description; must fully reconstruct the augmentation."""
        return {
            "enabled": self.probability > 0.0,
            "probability": float(self.probability),
            "seed": int(self.seed),
            "distribution": self.distribution,
            "degrees_of_freedom": (
                float(self.degrees_of_freedom) if self.distribution == "student_t" else None
            ),
            "variance_coefficients": [float(c) for c in self.coefficients],
            "variance_floor": float(self.variance_floor),
            "downsampling": "bicubic_align_corners_false",
            "scale": int(self.scale),
        }


def build_synthetic_noise_config(
    probability: float,
    seed: int,
    distribution: str = "gaussian",
    degrees_of_freedom: float = STUDENT_T_DEGREES_OF_FREEDOM,
    variance_floor: float = 0.0,
) -> dict | None:
    """CLI options -> checkpoint-serializable config, or ``None`` when disabled.

    Mirrors ``build_scheduler_config``/``build_ema_config``: "off" is ``None`` so
    every historical command that never mentions the flag keeps producing an
    identical ``None`` here and in the checkpoints it writes.
    """
    if probability <= 0.0:
        return None
    return SyntheticNoiseAugmentation(
        probability=probability,
        seed=seed,
        distribution=distribution,
        degrees_of_freedom=degrees_of_freedom,
        variance_floor=variance_floor,
    ).config()


def build_synthetic_noise(config: dict | None) -> SyntheticNoiseAugmentation | None:
    """Reconstruct the augmentation described by *config*, or ``None``."""
    if config is None:
        return None
    return SyntheticNoiseAugmentation(
        probability=config["probability"],
        seed=config["seed"],
        distribution=config["distribution"],
        degrees_of_freedom=config.get("degrees_of_freedom") or STUDENT_T_DEGREES_OF_FREEDOM,
        coefficients=tuple(config["variance_coefficients"]),
        variance_floor=config["variance_floor"],
        scale=config.get("scale", 2),
    )
