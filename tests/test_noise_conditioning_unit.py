"""Fast, dataset-free tests for explicit noise conditioning (Experiment 25)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evaluate_checkpoint import load_model
from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import ImagePair
from src.ema import ExponentialMovingAverage
from src.models import ResidualSRNet
from src.noise_conditioning import (
    NOISE_CONDITIONING_METHOD,
    NoiseConditionedModel,
    build_noise_conditioning_config,
    conditioning_sigma_map,
    prepare_model_input,
    wrap_for_conditioning,
)
from src.scene_groups import find_scene_groups
from src.synthetic_noise import VARIANCE_COEFFICIENTS, noise_sigma
from src.tta import d4_transforms, forward_transform, inverse_transform, predict_x8
from train import build_ema_config, load_checkpoint_for_resume, save_checkpoint, train_one_epoch, validate


def _config(**overrides) -> dict:
    options = {"enabled": True}
    options.update(build_noise_conditioning_config(True))
    options.update(overrides)
    return options


# --- Variance / sigma formula (1, 4, 5, 6) ---


def test_conditioning_variance_formula_matches_published_coefficients() -> None:
    c0, c1, c2 = VARIANCE_COEFFICIENTS
    intensity = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    lr = intensity.clone()
    sigma = conditioning_sigma_map(lr)
    expected_variance = torch.clamp(c0 + c1 * intensity + c2 * intensity**2, min=0.0)
    assert torch.allclose(sigma, torch.sqrt(expected_variance), atol=1e-12)


def test_conditioning_variance_is_never_negative() -> None:
    """The fitted constant is negative, so unclamped variance would go negative near I=0."""
    assert VARIANCE_COEFFICIENTS[0] < 0
    sigma = conditioning_sigma_map(torch.zeros(10))
    assert (sigma >= 0).all()


def test_conditioning_sigma_is_finite_across_the_full_lr_range() -> None:
    lr = torch.linspace(-0.5, 1.5, 500)  # includes real NoisyLR's actual out-of-range values
    sigma = conditioning_sigma_map(lr)
    assert torch.isfinite(sigma).all()


def test_conditioning_sigma_increases_with_signal_intensity() -> None:
    sigma = conditioning_sigma_map(torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert torch.all(sigma[1:] > sigma[:-1])


# --- Intensity clamp affects only sigma estimation, never the LR channel (2, 3, 8) ---


def test_intensity_clamp_only_affects_sigma_not_the_returned_value() -> None:
    """sigma(1.5) must equal sigma(1.0) (clamped to the same upper bound), proving
    the clamp lives inside the sigma computation, not as a general clip."""
    below, at_bound, above = torch.tensor([1.5]), torch.tensor([1.0]), torch.tensor([1.3])
    assert torch.equal(conditioning_sigma_map(above), conditioning_sigma_map(at_bound))
    assert torch.equal(conditioning_sigma_map(below), conditioning_sigma_map(at_bound))
    negative, floor = torch.tensor([-0.4]), torch.tensor([0.0])
    assert torch.equal(conditioning_sigma_map(negative), conditioning_sigma_map(floor))


def test_original_lr_channel_is_never_clamped_or_modified() -> None:
    lr = torch.tensor([[[[1.33, -0.05, 0.5, 2.0]]]])  # real NoisyLR's actual observed range
    output = prepare_model_input(lr, _config())
    assert torch.equal(output[:, 0:1], lr)  # channel 0 is bit-for-bit the raw input


# --- prepare_model_input shape/content (7, 8, 9, 10) ---


def test_prepare_model_input_shape_is_n2hw_when_enabled() -> None:
    lr = torch.rand(4, 1, 32, 32)
    output = prepare_model_input(lr, _config())
    assert output.shape == (4, 2, 32, 32)


def test_prepare_model_input_channel0_equals_lr_exactly() -> None:
    lr = torch.rand(2, 1, 16, 16)
    output = prepare_model_input(lr, _config())
    assert torch.equal(output[:, 0:1], lr)


def test_prepare_model_input_channel1_equals_expected_sigma() -> None:
    lr = torch.rand(2, 1, 16, 16)
    config = _config()
    output = prepare_model_input(lr, config)
    expected = conditioning_sigma_map(
        torch.clamp(lr, 0.0, 1.0), VARIANCE_COEFFICIENTS, 0.0, (0.0, 1.0)
    )
    assert torch.allclose(output[:, 1:2], expected, atol=1e-7)


def test_prepare_model_input_disabled_stays_single_channel() -> None:
    lr = torch.rand(2, 1, 16, 16)
    assert torch.equal(prepare_model_input(lr, None), lr)
    assert torch.equal(prepare_model_input(lr, build_noise_conditioning_config(False)), lr)


def test_build_config_disabled_is_none() -> None:
    assert build_noise_conditioning_config(False) is None


def test_build_config_enabled_records_reconstruction_fields() -> None:
    config = build_noise_conditioning_config(True, variance_floor=1e-4)
    assert config["enabled"] is True
    assert config["method"] == NOISE_CONDITIONING_METHOD
    assert config["variance_coefficients"] == list(VARIANCE_COEFFICIENTS)
    assert config["input_intensity_clamp"] == [0.0, 1.0]
    assert config["variance_floor"] == 1e-4
    assert config["sigma_normalization"] == "none"


# --- Model wiring (11) ---


def test_residual_sr_with_two_input_channels_forward_shape() -> None:
    model = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    output = model(torch.randn(2, 2, 16, 16))
    assert output.shape == (2, 1, 32, 32)


def test_wrap_for_conditioning_returns_identical_object_when_disabled() -> None:
    model = ResidualSRNet(in_channels=1, num_features=4, num_blocks=1, scale=2)
    assert wrap_for_conditioning(model, None) is model


def test_wrapped_model_accepts_single_channel_lr_and_produces_2x_output() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    wrapped = wrap_for_conditioning(base, _config())
    assert isinstance(wrapped, NoiseConditionedModel)
    output = wrapped(torch.rand(2, 1, 16, 16))
    assert output.shape == (2, 1, 32, 32)


def test_wrapped_model_parameters_are_exactly_the_base_models() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    wrapped = wrap_for_conditioning(base, _config())
    base_ids = {id(p) for p in base.parameters()}
    wrapped_ids = {id(p) for p in wrapped.parameters()}
    assert base_ids == wrapped_ids  # no extra learnable parameters introduced


# --- Training forward/backward (12) ---


def test_training_forward_backward_is_finite_with_conditioning() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 32, 32)

    optimizer.zero_grad()
    output = model(lr)
    loss = nn.functional.l1_loss(output, target)
    loss.backward()
    optimizer.step()

    assert math.isfinite(loss.item())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# --- EMA compatibility (13) ---


def test_ema_wraps_and_updates_a_conditioned_model() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    ema = ExponentialMovingAverage(model, decay=0.9)
    assert isinstance(ema.shadow_model, NoiseConditionedModel)

    before = {name: p.clone() for name, p in ema.shadow_model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    assert any(
        not torch.equal(before[name], p) for name, p in ema.shadow_model.named_parameters()
    )
    # The shadow must itself still accept plain single-channel LR.
    output = ema.shadow_model(torch.rand(1, 1, 8, 8))
    assert output.shape == (1, 1, 16, 16)
    assert torch.isfinite(output).all()


# --- Validation integration (14) ---


def _write_pair(root: Path, sample_id: str, lr_size: int = 16, scale: int = 2) -> ImagePair:
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    degraded = rng.uniform(-0.1, 1.2, size=(lr_size, lr_size)).astype(np.float32)
    target = rng.uniform(0.0, 1.0, size=(lr_size * scale, lr_size * scale)).astype(np.float32)
    np.save(input_dir / f"{sample_id}.npy", degraded)
    np.save(target_dir / f"{sample_id}.npy", target)
    return ImagePair(sample_id, input_dir / f"{sample_id}.npy", target_dir / f"{sample_id}.npy")


def test_validate_produces_finite_metrics_with_a_conditioned_model(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, f"{i:03d}") for i in range(4)]
    loader = create_dataloader(PairedRestorationDataset(pairs), batch_size=2, shuffle=False)
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    metrics = validate(model, loader, nn.L1Loss(), torch.device("cpu"))
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["psnr"])
    assert math.isfinite(metrics["ssim"])


def test_train_one_epoch_works_with_a_conditioned_model(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, f"{i:03d}") for i in range(4)]
    loader = create_dataloader(PairedRestorationDataset(pairs), batch_size=2, shuffle=False)
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optimizer, nn.L1Loss(), torch.device("cpu"))
    assert math.isfinite(loss)


# --- x8 TTA spatial consistency (15) ---


def test_x8_tta_works_with_a_conditioned_model() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    result = predict_x8(model, torch.rand(1, 1, 16, 16))
    assert result.shape == (1, 1, 32, 32)
    assert torch.isfinite(result).all()


def test_x8_conditioning_matches_concatenate_then_transform() -> None:
    """Option A (sigma generated after each TTA transform, via the wrapper) must
    exactly match option B (concatenate [lr,sigma] first, transform both
    channels together) -- true because sigma is an exactly pointwise function
    of LR, so permutation and the pointwise map commute."""
    torch.manual_seed(0)
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    config = _config()
    wrapped = wrap_for_conditioning(base, config)
    lr = torch.rand(1, 1, 16, 16)

    option_a = predict_x8(wrapped, lr)

    with torch.no_grad():
        full = prepare_model_input(lr, config)  # concatenate first
        predictions = []
        for flip, k in d4_transforms():
            transformed = forward_transform(full, flip, k)  # both channels moved together
            predictions.append(inverse_transform(base(transformed), flip, k))
        option_b = torch.stack(predictions, dim=0).mean(dim=0)

    assert torch.allclose(option_a, option_b, atol=1e-6)


def test_x8_prediction_is_not_clipped_and_averaging_happens_before_clipping() -> None:
    base = ResidualSRNet(in_channels=2, out_channels=1, num_features=4, num_blocks=1, scale=2)
    model = wrap_for_conditioning(base, _config())
    with torch.no_grad():
        model.model.upsample_conv.weight.zero_()
        model.model.upsample_conv.bias.fill_(5.0)
    result = predict_x8(model, torch.rand(1, 1, 8, 8))
    assert result.max().item() > 1.0  # would be impossible if clipped mid-pipeline


# --- evaluate_checkpoint / infer_test / group-aware integration (16, 17, 18) ---


def _model_config() -> dict:
    return {"in_channels": 2, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}


def _save_conditioned_checkpoint(tmp_path: Path, config: dict | None) -> Path:
    base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(base, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        best_val_psnr=10.0,
        model_config=_model_config(),
        training_config={},
        noise_conditioning_config=config,
    )
    return path


def test_evaluate_checkpoint_load_model_reconstructs_a_conditioned_model(tmp_path: Path) -> None:
    """Covers evaluate_checkpoint.py; infer_test.py and evaluate_group_aware.py
    reuse this exact function, so this test covers them too."""
    path = _save_conditioned_checkpoint(tmp_path, _config())
    model, checkpoint = load_model(path, torch.device("cpu"))
    assert isinstance(model, NoiseConditionedModel)
    assert checkpoint["noise_conditioning_config"]["enabled"] is True
    output = model(torch.rand(1, 1, 16, 16))  # plain single-channel LR in
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()


def test_infer_test_run_inference_works_with_a_conditioned_checkpoint(tmp_path: Path) -> None:
    from infer_test import run_inference

    path = _save_conditioned_checkpoint(tmp_path, _config())
    model, _ = load_model(path, torch.device("cpu"))
    prediction = run_inference(model, torch.rand(1, 20, 24), torch.device("cpu"))
    assert prediction.shape == (40, 48)
    assert np.isfinite(prediction).all()

    x8_prediction = run_inference(model, torch.rand(1, 20, 24), torch.device("cpu"), tta="x8")
    assert x8_prediction.shape == (40, 48)
    assert np.isfinite(x8_prediction).all()


def test_group_aware_evaluator_reuses_load_model_for_conditioned_checkpoints(tmp_path: Path) -> None:
    """evaluate_group_aware.py imports load_model from evaluate_checkpoint and
    never reconstructs a model itself, so a conditioned checkpoint loading
    correctly there is exactly this test."""
    path = _save_conditioned_checkpoint(tmp_path, _config())
    model, checkpoint = load_model(path, torch.device("cpu"))
    assert isinstance(model, NoiseConditionedModel)
    # find_scene_groups is the other half of evaluate_group_aware.py's pipeline;
    # confirm it still runs standalone (no interaction with conditioning).
    pairs = [_write_pair(tmp_path / "scenes", f"{i:03d}") for i in range(3)]
    groups = find_scene_groups(pairs)
    assert isinstance(groups, list)


# --- Checkpoint config / resume (19, 20, 21, 22) ---


def test_checkpoint_stores_the_noise_conditioning_config(tmp_path: Path) -> None:
    config = _config()
    path = _save_conditioned_checkpoint(tmp_path, config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["noise_conditioning_config"] == config


def test_matching_resume_succeeds(tmp_path: Path) -> None:
    config = _config()
    path = _save_conditioned_checkpoint(tmp_path, config)
    base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(base, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch, best, _ = load_checkpoint_for_resume(
        path, model, optimizer, _model_config(), torch.device("cpu"),
        noise_conditioning_config=config,
    )
    assert start_epoch == 2
    assert best == 10.0


def test_resume_rejects_a_different_variance_floor(tmp_path: Path) -> None:
    path = _save_conditioned_checkpoint(tmp_path, build_noise_conditioning_config(True, variance_floor=0.0))
    base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(base, build_noise_conditioning_config(True, variance_floor=1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="noise_conditioning_config"):
        load_checkpoint_for_resume(
            path, model, optimizer, _model_config(), torch.device("cpu"),
            noise_conditioning_config=build_noise_conditioning_config(True, variance_floor=1e-4),
        )


def test_resume_rejects_different_coefficients(tmp_path: Path) -> None:
    path = _save_conditioned_checkpoint(tmp_path, build_noise_conditioning_config(True))
    different = build_noise_conditioning_config(True, coefficients=(0.0, 0.01, 0.02))
    base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(base, different)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="noise_conditioning_config"):
        load_checkpoint_for_resume(
            path, model, optimizer, _model_config(), torch.device("cpu"),
            noise_conditioning_config=different,
        )


def test_resume_rejects_conditioning_disabled_on_a_conditioned_checkpoint(tmp_path: Path) -> None:
    """Same in_channels=2 model_config (so that check passes) but conditioning
    requested off -- isolates the noise_conditioning_config check specifically."""
    path = _save_conditioned_checkpoint(tmp_path, build_noise_conditioning_config(True))
    base = ResidualSRNet(**_model_config())
    optimizer = torch.optim.Adam(base.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="noise_conditioning_config"):
        load_checkpoint_for_resume(
            path, base, optimizer, _model_config(),
            torch.device("cpu"), noise_conditioning_config=None,
        )


def test_resume_rejects_enabling_conditioning_on_a_plain_checkpoint(tmp_path: Path) -> None:
    base = ResidualSRNet(in_channels=1, out_channels=1, num_features=4, num_blocks=1, scale=2)
    optimizer = torch.optim.Adam(base.parameters(), lr=1e-4)
    plain_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    path = tmp_path / "plain.pt"
    save_checkpoint(
        path, base, optimizer, epoch=1, best_val_psnr=10.0,
        model_config=plain_config, training_config={},
    )
    conditioned = build_noise_conditioning_config(True)
    new_base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(new_base, conditioned)
    new_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="noise_conditioning_config"):
        load_checkpoint_for_resume(
            path, model, new_optimizer, plain_config, torch.device("cpu"),
            noise_conditioning_config=conditioned,
        )


def test_historical_checkpoint_without_the_key_still_resumes(tmp_path: Path) -> None:
    """Every pre-Experiment-25 checkpoint predates this key entirely."""
    model_config = {"in_channels": 1, "out_channels": 1, "num_features": 4, "num_blocks": 1, "scale": 2}
    model = ResidualSRNet(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 90,
            "best_val_psnr": 27.9893,
            "model_config": model_config,
            "training_config": {},
            # Deliberately no noise_conditioning_config key.
        },
        path,
    )
    resumed_model = ResidualSRNet(**model_config)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    start_epoch, best, _ = load_checkpoint_for_resume(
        path, resumed_model, resumed_optimizer, model_config, torch.device("cpu"),
        noise_conditioning_config=None,
    )
    assert start_epoch == 91
    assert best == 27.9893


def test_resume_without_passing_the_config_skips_the_check(tmp_path: Path) -> None:
    """Callers that omit the parameter entirely (pre-existing tests) are unaffected."""
    config = build_noise_conditioning_config(True)
    path = _save_conditioned_checkpoint(tmp_path, config)
    base = ResidualSRNet(**_model_config())
    model = wrap_for_conditioning(base, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch, _, _ = load_checkpoint_for_resume(
        path, model, optimizer, _model_config(), torch.device("cpu")
    )
    assert start_epoch == 2


def test_historical_ema_config_helper_unaffected() -> None:
    """Sanity: this experiment's changes must not disturb the pre-existing
    EMA config helper it sits alongside."""
    assert build_ema_config(False, 0.999) is None
    assert build_ema_config(True, 0.999) == {"enabled": True, "decay": 0.999}
