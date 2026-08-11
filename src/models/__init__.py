"""Trainable restoration model architectures.

``build_model_config``/``build_model`` centralize architecture selection so
``train.py``/``evaluate_checkpoint.py``/``infer_test.py`` share one
reconstruction path instead of duplicating ``ResidualSRNet(**model_config)``
(or an EDSRLite equivalent) in each place -- mirrors the existing
``src.losses.build_loss_config``/``build_loss`` pattern.
"""

from torch import nn

from .edsr_lite import EDSRLite, EDSRResidualBlock
from .nafnet_sr import NAFBlock, NAFNetSR
from .residual_sr import ResidualBlock, ResidualSRNet
from .swinir_lite import SwinIRLite, SwinTransformerBlock

__all__ = [
    "ResidualBlock",
    "ResidualSRNet",
    "EDSRResidualBlock",
    "EDSRLite",
    "NAFBlock",
    "NAFNetSR",
    "SwinTransformerBlock",
    "SwinIRLite",
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
    dw_expand: int = 2,
    ffn_expand: int = 2,
    embed_dim: int = 48,
    depth: int = 6,
    num_heads: int = 6,
    window_size: int = 8,
    mlp_ratio: float = 2.0,
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
    if architecture == "nafnet_sr":
        return {
            "architecture": "nafnet_sr",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_features": num_features,
            "num_blocks": num_blocks,
            "scale": scale,
            "dw_expand": dw_expand,
            "ffn_expand": ffn_expand,
        }
    if architecture == "swinir_lite":
        return {
            "architecture": "swinir_lite",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "scale": scale,
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
    if architecture == "nafnet_sr":
        return NAFNetSR(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            num_features=model_config["num_features"],
            num_blocks=model_config["num_blocks"],
            scale=model_config["scale"],
            dw_expand=model_config["dw_expand"],
            ffn_expand=model_config["ffn_expand"],
        )
    if architecture == "swinir_lite":
        return SwinIRLite(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            window_size=model_config["window_size"],
            mlp_ratio=model_config["mlp_ratio"],
            scale=model_config["scale"],
        )
    raise ValueError(f"Unknown architecture: {architecture}")
