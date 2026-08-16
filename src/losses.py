"""Reconstruction losses for training the restoration model.

Selection is centralized here (``build_loss_config``/``build_loss``) so
``train.py`` never scatters loss-specific conditionals through the training
loop, and so the choice is a plain, checkpoint-serializable dict just like
``build_scheduler_config``/``build_scheduler``.
"""

import torch
import torch.nn.functional as functional
from torch import nn

from .synthetic_noise import VARIANCE_COEFFICIENTS, noise_sigma


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss: ``mean(sqrt((prediction - target)**2 + eps**2))``.

    Adding ``eps`` before the square root keeps the gradient finite (and zero,
    not undefined) exactly where prediction equals target, unlike plain L1's
    subgradient at zero.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = prediction - target
        return torch.sqrt(difference * difference + self.eps * self.eps).mean()


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Depth-wise Gaussian conv2d kernel, shape [channels, 1, window_size, window_size]."""
    coordinates = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gaussian_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    return gaussian_2d.expand(channels, 1, window_size, window_size).contiguous()


def differentiable_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Fully differentiable local-window (Gaussian) SSIM -- Wang et al. 2004.

    Operates directly on ``[B,C,H,W]`` torch tensors and preserves gradients,
    unlike ``src.metrics.ssim``, which converts to NumPy via scikit-image and
    is only suitable for (non-differentiable) evaluation logging. Returns a
    scalar: the mean SSIM over the whole local-window map, batch, and channels.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 4:
        raise ValueError(f"Expected a [B,C,H,W] tensor, got shape {tuple(prediction.shape)}")

    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2

    def local_mean(x: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(x, window, padding=padding, groups=channels)

    mu_prediction = local_mean(prediction)
    mu_target = local_mean(target)
    mu_prediction_sq = mu_prediction * mu_prediction
    mu_target_sq = mu_target * mu_target
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = local_mean(prediction * prediction) - mu_prediction_sq
    sigma_target_sq = local_mean(target * target) - mu_target_sq
    sigma_prediction_target = local_mean(prediction * target) - mu_prediction_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_prediction_target + c1) * (2 * sigma_prediction_target + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (sigma_prediction_sq + sigma_target_sq + c2)
    return (numerator / denominator).mean()


class SSIMLoss(nn.Module):
    """``1 - differentiable_ssim(...)`` -- a training objective, not an evaluation metric."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - differentiable_ssim(
            prediction, target, self.data_range, self.window_size, self.sigma
        )


class L1SSIMLoss(nn.Module):
    """``L1 + ssim_weight * (1 - differentiable SSIM)``."""

    def __init__(
        self,
        ssim_weight: float = 0.1,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative")
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(data_range=data_range, window_size=window_size, sigma=sigma)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(prediction, target) + self.ssim_weight * self.ssim_loss(prediction, target)


class MixedL1MSELoss(nn.Module):
    """``alpha * L1 + (1 - alpha) * MSE``.

    L1 is robust to outliers; MSE is what PSNR is directly defined in terms
    of (``PSNR = 10*log10(peak**2 / MSE)``). Blending the two lets a
    fine-tuning run lean toward the PSNR objective without abandoning L1's
    robustness outright -- see ``--loss mixed --mixed-loss-alpha`` in
    ``train.py``.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.l1(prediction, target) + (1.0 - self.alpha) * self.mse(
            prediction, target
        )


class VarianceWeightedL1Loss(nn.Module):
    """L1 reweighted by the inverse of the Experiment 22 signal-dependent noise
    model, so pixels the measured degradation makes easier to recover (low
    intensity, low noise variance) contribute more gradient than pixels it
    makes intrinsically harder (bright, high noise variance) -- unlike plain
    L1, which weights every pixel equally.

    This is the direct implementation of the Experiment 22 forensics report's
    HIGHEST-ranked, previously-untried recommendation (see
    ``results/degradation_analysis/degradation_report.md``, "Ranked strategies
    for Experiment 22", item 1): "weight the existing L1 by 1/sqrt(var(I))
    using the fitted coefficients." Reuses
    ``src.synthetic_noise.noise_variance``'s exact fitted coefficients rather
    than re-deriving them, so this is the same measured law Experiments 24/25
    already use, applied a third way.

    **Documented approximation:** the variance model
    ``var(I) = c0 + c1*I + c2*I**2`` was fit against *LR-space* intensity vs.
    *LR-space* residual noise (Experiment 22 measured the GT->NoisyLR
    degradation directly). This loss instead estimates ``I`` from the *target*
    HR image (clamped to ``[0,1]``, the same clamp convention
    ``src/noise_conditioning.py`` uses) because that is the space this loss
    operates in -- the assumption is that HR brightness is a reasonable proxy
    for the LR brightness at the corresponding location, since bicubic
    downsampling (Experiment 22's best-fit kernel) does not change local mean
    intensity much. This is an extrapolation of a measured law, not itself a
    directly measured quantity, and should be validated empirically (see
    ``--loss weighted_l1`` in ``train.py``) rather than assumed.

    Per-pixel weights are normalized to a batch mean of 1 (``weight /
    weight.mean()``) so the loss stays on the same numeric scale as plain L1,
    keeping existing ``--lr`` values sensible starting points instead of
    silently rescaling the effective learning rate.
    """

    def __init__(
        self,
        eps: float = 1e-2,
        coefficients: tuple[float, float, float] = VARIANCE_COEFFICIENTS,
        variance_floor: float = 0.0,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.coefficients = coefficients
        self.variance_floor = variance_floor

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intensity = torch.clamp(target, 0.0, 1.0)
        sigma = noise_sigma(intensity, self.coefficients, self.variance_floor)
        weight = 1.0 / (sigma + self.eps)
        weight = weight / weight.mean()
        return (weight * torch.abs(prediction - target)).mean()


def build_loss_config(
    name: str,
    charbonnier_eps: float = 1e-3,
    ssim_weight: float = 0.1,
    mixed_loss_alpha: float = 0.5,
    weighted_l1_eps: float = 1e-2,
    weighted_l1_variance_floor: float = 0.0,
) -> dict:
    """Turn CLI loss options into a plain, checkpoint-serializable dict."""
    if name == "l1":
        return {"name": "l1"}
    if name == "mse":
        return {"name": "mse"}
    if name == "charbonnier":
        return {"name": "charbonnier", "epsilon": charbonnier_eps}
    if name == "l1_ssim":
        return {"name": "l1_ssim", "ssim_weight": ssim_weight}
    if name == "mixed":
        if not 0.0 <= mixed_loss_alpha <= 1.0:
            raise ValueError("mixed_loss_alpha must be between 0 and 1")
        return {"name": "mixed", "alpha": mixed_loss_alpha}
    if name == "weighted_l1":
        if weighted_l1_eps <= 0:
            raise ValueError("weighted_l1_eps must be positive")
        return {
            "name": "weighted_l1",
            "eps": weighted_l1_eps,
            "variance_coefficients": list(VARIANCE_COEFFICIENTS),
            "variance_floor": weighted_l1_variance_floor,
        }
    raise ValueError(f"Unknown loss: {name}")


def build_loss(loss_config: dict) -> nn.Module:
    """Construct the reconstruction loss described by *loss_config*."""
    name = loss_config["name"]
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "charbonnier":
        return CharbonnierLoss(eps=loss_config["epsilon"])
    if name == "l1_ssim":
        return L1SSIMLoss(ssim_weight=loss_config["ssim_weight"])
    if name == "mixed":
        return MixedL1MSELoss(alpha=loss_config["alpha"])
    if name == "weighted_l1":
        return VarianceWeightedL1Loss(
            eps=loss_config["eps"],
            coefficients=tuple(loss_config["variance_coefficients"]),
            variance_floor=loss_config["variance_floor"],
        )
    raise ValueError(f"Unknown loss: {name}")


def loss_label(name: str) -> str:
    """Human-readable label for logging (e.g. "Train L1" / "Train Charbonnier")."""
    if name == "l1":
        return "L1"
    if name == "mse":
        return "MSE"
    if name == "charbonnier":
        return "Charbonnier"
    if name == "l1_ssim":
        return "L1+SSIM"
    if name == "mixed":
        return "Mixed(L1+MSE)"
    if name == "weighted_l1":
        return "VarianceWeightedL1"
    raise ValueError(f"Unknown loss: {name}")
