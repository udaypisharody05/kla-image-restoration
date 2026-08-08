"""Reconstruction losses for training the restoration model.

Selection is centralized here (``build_loss_config``/``build_loss``) so
``train.py`` never scatters loss-specific conditionals through the training
loop, and so the choice is a plain, checkpoint-serializable dict just like
``build_scheduler_config``/``build_scheduler``.
"""

import torch
import torch.nn.functional as functional
from torch import nn


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


def build_loss_config(name: str, charbonnier_eps: float = 1e-3, ssim_weight: float = 0.1) -> dict:
    """Turn CLI loss options into a plain, checkpoint-serializable dict."""
    if name == "l1":
        return {"name": "l1"}
    if name == "charbonnier":
        return {"name": "charbonnier", "epsilon": charbonnier_eps}
    if name == "l1_ssim":
        return {"name": "l1_ssim", "ssim_weight": ssim_weight}
    raise ValueError(f"Unknown loss: {name}")


def build_loss(loss_config: dict) -> nn.Module:
    """Construct the reconstruction loss described by *loss_config*."""
    name = loss_config["name"]
    if name == "l1":
        return nn.L1Loss()
    if name == "charbonnier":
        return CharbonnierLoss(eps=loss_config["epsilon"])
    if name == "l1_ssim":
        return L1SSIMLoss(ssim_weight=loss_config["ssim_weight"])
    raise ValueError(f"Unknown loss: {name}")


def loss_label(name: str) -> str:
    """Human-readable label for logging (e.g. "Train L1" / "Train Charbonnier")."""
    if name == "l1":
        return "L1"
    if name == "charbonnier":
        return "Charbonnier"
    if name == "l1_ssim":
        return "L1+SSIM"
    raise ValueError(f"Unknown loss: {name}")
