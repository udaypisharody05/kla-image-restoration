"""Trainable restoration model architectures.

``build_model_config``/``build_model`` centralize architecture selection so
``train.py``/``evaluate_checkpoint.py``/``infer_test.py`` share one
reconstruction path instead of duplicating ``ResidualSRNet(**model_config)``
(or an EDSRLite equivalent) in each place -- mirrors the existing
``src.losses.build_loss_config``/``build_loss`` pattern.
"""

from torch import nn

from .edsr_lite import EDSRLite, EDSRResidualBlock
from .residual_sr import ResidualBlock, ResidualSRNet

__all__ = [
    "ResidualBlock",
    "ResidualSRNet",
    "EDSRResidualBlock",
    "EDSRLite",
    "build_model_config",
    "build_model",
]


def build_model_config(
    architecture: str,
    in_channels: int = 1,
    out_channels: int = 1,
    num_features: int = 64,
    num_blocks: int = 8,
    scale: int = 2,
    residual_scale: float = 0.1,
) -> dict:
    """Turn CLI/config values into a plain, checkpoint-serializable dict.

    ``architecture="residual_sr"`` deliberately omits an ``"architecture"``
    key, so it stays byte-identical to every historical checkpoint's
    ``model_config`` (Experiments 1-8). That is what lets the existing
    dict-equality resume check in ``train.py::load_checkpoint_for_resume``
    keep working unmodified: an EDSRLite config (which *does* carry the key)
    can never equal a ResidualSRNet config, so architecture mismatches are
    already rejected with zero new comparison logic.
    """
    if architecture == "residual_sr":
        return {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_features": num_features,
            "num_blocks": num_blocks,
            "scale": scale,
        }
    if architecture == "edsr_lite":
        return {
            "architecture": "edsr_lite",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_features": num_features,
            "num_blocks": num_blocks,
            "scale": scale,
            "residual_scale": residual_scale,
        }
    raise ValueError(f"Unknown architecture: {architecture}")


def build_model(model_config: dict) -> nn.Module:
    """Reconstruct the model described by *model_config*.

    A missing ``"architecture"`` key means the checkpoint was saved before
    architecture selection existed (Experiments 1-8) -- always
    ``ResidualSRNet``, never any other interpretation. Keys are read
    explicitly (never ``**model_config``) so an extra ``"architecture"`` key
    can never trip up ``ResidualSRNet``'s constructor.
    """
    architecture = model_config.get("architecture", "residual_sr")
    if architecture == "residual_sr":
        return ResidualSRNet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            num_features=model_config["num_features"],
            num_blocks=model_config["num_blocks"],
            scale=model_config["scale"],
        )
    if architecture == "edsr_lite":
        return EDSRLite(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            num_features=model_config["num_features"],
            num_blocks=model_config["num_blocks"],
            scale=model_config["scale"],
            residual_scale=model_config["residual_scale"],
        )
    raise ValueError(f"Unknown architecture: {architecture}")
