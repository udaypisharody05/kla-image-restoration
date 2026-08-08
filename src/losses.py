"""Reconstruction losses for training the restoration model.

Selection is centralized here (``build_loss_config``/``build_loss``) so
``train.py`` never scatters loss-specific conditionals through the training
loop, and so the choice is a plain, checkpoint-serializable dict just like
``build_scheduler_config``/``build_scheduler``.
"""

import torch
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


def build_loss_config(name: str, charbonnier_eps: float = 1e-3) -> dict:
    """Turn CLI loss options into a plain, checkpoint-serializable dict."""
    if name == "l1":
        return {"name": "l1"}
    if name == "charbonnier":
        return {"name": "charbonnier", "epsilon": charbonnier_eps}
    raise ValueError(f"Unknown loss: {name}")


def build_loss(loss_config: dict) -> nn.Module:
    """Construct the reconstruction loss described by *loss_config*."""
    name = loss_config["name"]
    if name == "l1":
        return nn.L1Loss()
    if name == "charbonnier":
        return CharbonnierLoss(eps=loss_config["epsilon"])
    raise ValueError(f"Unknown loss: {name}")


def loss_label(name: str) -> str:
    """Human-readable label for logging (e.g. "Train L1" / "Train Charbonnier")."""
    if name == "l1":
        return "L1"
    if name == "charbonnier":
        return "Charbonnier"
    raise ValueError(f"Unknown loss: {name}")
