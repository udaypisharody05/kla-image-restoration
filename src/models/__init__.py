"""Trainable restoration model architectures.

``build_model_config``/``build_model`` centralize architecture selection so
``train.py``/``evaluate_checkpoint.py``/``infer_test.py`` share one
reconstruction path instead of duplicating ``ResidualSRNet(**model_config)``
(or an EDSRLite equivalent) in each place -- mirrors the existing
``src.losses.build_loss_config``/``build_loss`` pattern.
"""

from torch import nn

from .attention import ChannelAttention
from .denoise_stem import DenoiseStem, SimpleGateBlock
from .edsr_lite import EDSRLite, EDSRResidualBlock
from .nafnet_sr import NAFBlock, NAFNetSR
from .residual_sr import MultiScaleBlock, ResidualBlock, ResidualDenseBlock, ResidualSRNet
from .residual_sr_bicubic import ResidualSRBicubic, fixed_bicubic_upsample
from .swinir_lite import SwinIRLite, SwinTransformerBlock

__all__ = [
    "ResidualBlock",
    "ResidualSRNet",
    "ChannelAttention",
    "MultiScaleBlock",
    "ResidualDenseBlock",
    "DenoiseStem",
    "SimpleGateBlock",
    "EDSRResidualBlock",
    "EDSRLite",
    "NAFBlock",
    "NAFNetSR",
    "SwinTransformerBlock",
    "SwinIRLite",
    "ResidualSRBicubic",
    "fixed_bicubic_upsample",
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
    channel_attention: bool = False,
    attention_reduction: int = 8,
    multiscale_block: bool = False,
    rdb_block: bool = False,
    rdb_growth_rate: int = 16,
    rdb_num_layers: int = 3,
    denoise_stem: bool = False,
    denoise_stem_features: int = 32,
    denoise_stem_blocks: int = 2,
) -> dict:
    """Turn CLI/config values into a plain, checkpoint-serializable dict.

    ``architecture="residual_sr"`` deliberately omits an ``"architecture"``
    key, so it stays byte-identical to every historical checkpoint's
    ``model_config`` (Experiments 1-8). That is what lets the existing
    dict-equality resume check in ``train.py::load_checkpoint_for_resume``
    keep working unmodified: an EDSRLite config (which *does* carry the key)
    can never equal a ResidualSRNet config, so architecture mismatches are
    already rejected with zero new comparison logic.

    ``channel_attention``/``multiscale_block``/``rdb_block``/``denoise_stem``
    (all default ``False``, residual_sr only) follow the exact same
    convention: their keys are only added to the dict when actually
    requested, so a plain ``--model residual_sr`` run (no ablation flags)
    still produces the identical dict every historical checkpoint has --
    resuming an old checkpoint is unaffected by these variants existing.
    """
    if architecture == "residual_sr":
        config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_features": num_features,
            "num_blocks": num_blocks,
            "scale": scale,
        }
        if channel_attention:
            config["channel_attention"] = True
            config["attention_reduction"] = attention_reduction
        if multiscale_block:
            config["multiscale_block"] = True
        if rdb_block:
            config["rdb_block"] = True
            config["rdb_growth_rate"] = rdb_growth_rate
            config["rdb_num_layers"] = rdb_num_layers
        if denoise_stem:
            config["denoise_stem"] = True
            config["denoise_stem_features"] = denoise_stem_features
            config["denoise_stem_blocks"] = denoise_stem_blocks
        return config
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
    if architecture == "residual_sr_bicubic":
        return {
            "architecture": "residual_sr_bicubic",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_features": num_features,
            "num_blocks": num_blocks,
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
            # .get() with the original defaults: every historical model_config
            # (which never had these keys) reconstructs the exact same plain
            # ResidualSRNet as before either variant existed.
            channel_attention=model_config.get("channel_attention", False),
            attention_reduction=model_config.get("attention_reduction", 8),
            multiscale_block=model_config.get("multiscale_block", False),
            rdb_block=model_config.get("rdb_block", False),
            rdb_growth_rate=model_config.get("rdb_growth_rate", 16),
            rdb_num_layers=model_config.get("rdb_num_layers", 3),
            denoise_stem=model_config.get("denoise_stem", False),
            denoise_stem_features=model_config.get("denoise_stem_features", 32),
            denoise_stem_blocks=model_config.get("denoise_stem_blocks", 2),
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
    if architecture == "residual_sr_bicubic":
        return ResidualSRBicubic(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            num_features=model_config["num_features"],
            num_blocks=model_config["num_blocks"],
            scale=model_config["scale"],
        )
    raise ValueError(f"Unknown architecture: {architecture}")
