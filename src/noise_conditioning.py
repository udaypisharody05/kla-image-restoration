"""Explicit signal-dependent noise conditioning for ResidualSRNet (Experiment 25).

Concatenates a per-pixel estimate of the measured signal-dependent noise level
as a second input channel, alongside the real NoisyLR -- trained exclusively on
real NoisyLR/GT pairs, with **no synthetic input substitution** (contrast with
Experiment 24's ``src/synthetic_noise.py``, which this experiment does not use)::

    channel 0 = NoisyLR                       (unmodified -- never clamped)
    channel 1 = sigma(clamp(NoisyLR, 0, 1))    (Experiment 22's noise model)

Reuses ``src.synthetic_noise``'s variance/sigma formulas rather than
duplicating them -- one formula implementation, not two.
"""

import torch
from torch import nn

from .synthetic_noise import VARIANCE_COEFFICIENTS, noise_sigma

NOISE_CONDITIONING_METHOD = "signal_dependent_sigma"


def conditioning_sigma_map(
    lr: torch.Tensor,
    coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS,
    variance_floor: float = 0.0,
    intensity_clamp: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Per-pixel sigma estimate for conditioning -- never modifies *lr* itself.

    The clamp applies only to the intensity fed into the variance formula, not
    to the returned sigma map or to *lr*: a raw NoisyLR value of, say, 1.3 is
    treated as intensity 1.0 for sigma estimation, but ``lr`` itself is passed
    through to the model completely unchanged as channel 0 (see
    ``prepare_model_input``).
    """
    low, high = intensity_clamp
    intensity = torch.clamp(lr, min=low, max=high)
    return noise_sigma(intensity, coefficients, variance_floor)


def prepare_model_input(lr: torch.Tensor, config: dict | None) -> torch.Tensor:
    """Historical model: returns *lr* unchanged. Noise-conditioned: ``[lr, sigma]``.

    One code path reused identically for training, validation, x8 TTA (via
    ``NoiseConditionedModel`` below -- ``src/tta.py`` needs no changes),
    ``evaluate_checkpoint.py``, ``infer_test.py``, and the group-aware
    diagnostic, so every consumer stays consistent by construction: there is
    nowhere else a second, subtly-different conditioning computation could
    creep in.
    """
    if config is None or not config.get("enabled", False):
        return lr
    sigma = conditioning_sigma_map(
        lr,
        tuple(config["variance_coefficients"]),
        config["variance_floor"],
        tuple(config["input_intensity_clamp"]),
    )
    return torch.cat([lr, sigma], dim=-3)


class NoiseConditionedModel(nn.Module):
    """Wraps a base model so every caller's ``model(lr)`` transparently becomes
    ``base_model(prepare_model_input(lr, config))``.

    Standard ``nn.Module`` composition (``self.model = model``) is what makes
    this work everywhere with zero other changes: ``.parameters()``,
    ``.to(device)``, ``.train()``/``.eval()``, and ``copy.deepcopy`` (used by
    ``src.ema.ExponentialMovingAverage``) all transparently delegate to the
    wrapped model, so EMA "just works". ``src/tta.py::predict_x8``
    geometrically transforms the raw 1-channel LR *before* calling whatever
    model it was given, so wrapping means sigma is always computed fresh from
    the already-transformed LR -- option A from the design spec, achieved
    without touching ``src/tta.py`` at all. Because sigma is an exactly
    pointwise function of LR, this is mathematically identical to
    concatenating first and geometrically transforming both channels together
    (option B); ``test_x8_conditioning_matches_concatenate_then_transform``
    proves the equivalence numerically.
    """

    def __init__(self, model: nn.Module, config: dict) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(prepare_model_input(lr, self.config))


def wrap_for_conditioning(model: nn.Module, config: dict | None) -> nn.Module:
    """Return *model* unchanged when conditioning is off, else wrapped.

    Returning the identical object (not even a trivial pass-through wrapper)
    when *config* is ``None`` is what keeps every historical command's model
    byte-for-byte identical -- no indirection is introduced unless
    conditioning is actually requested.
    """
    if config is None:
        return model
    return NoiseConditionedModel(model, config)


def build_noise_conditioning_config(
    enabled: bool,
    coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS,
    variance_floor: float = 0.0,
    intensity_clamp: tuple[float, float] = (0.0, 1.0),
) -> dict | None:
    """CLI options -> checkpoint-serializable config, or ``None`` when disabled.

    Mirrors ``build_ema_config``/``build_synthetic_noise_config``: "off" is
    ``None`` so every historical command that never mentions the flag produces
    an identical ``None`` here and in the checkpoints it writes.
    """
    if not enabled:
        return None
    return {
        "enabled": True,
        "method": NOISE_CONDITIONING_METHOD,
        "variance_coefficients": [float(c) for c in coefficients],
        "input_intensity_clamp": [float(intensity_clamp[0]), float(intensity_clamp[1])],
        "variance_floor": float(variance_floor),
        "sigma_normalization": "none",
    }
