"""Fast, dataset-free tests for reconstruction loss selection (src/losses.py)."""

import math

import pytest
import torch
from torch import nn

from src.losses import (
    CharbonnierLoss,
    L1SSIMLoss,
    SSIMLoss,
    build_loss,
    build_loss_config,
    differentiable_ssim,
    loss_label,
)


def test_charbonnier_returns_a_scalar() -> None:
    loss = CharbonnierLoss(eps=1e-3)
    value = loss(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    assert value.ndim == 0


def test_charbonnier_identical_tensors_is_approximately_epsilon() -> None:
    eps = 1e-3
    loss = CharbonnierLoss(eps=eps)
    image = torch.rand(2, 1, 8, 8)
    value = loss(image, image.clone())
    assert math.isfinite(value.item())
    assert not math.isnan(value.item())
    assert value.item() == pytest.approx(eps, abs=1e-9)


def test_charbonnier_non_identical_tensors_produce_larger_loss() -> None:
    loss = CharbonnierLoss(eps=1e-3)
    image = torch.rand(2, 1, 8, 8)
    identical_loss = loss(image, image.clone())
    different_loss = loss(image, torch.rand(2, 1, 8, 8))
    assert different_loss.item() > identical_loss.item()


def test_charbonnier_is_finite_on_random_batches() -> None:
    loss = CharbonnierLoss(eps=1e-3)
    value = loss(torch.randn(4, 1, 16, 16) * 5, torch.randn(4, 1, 16, 16) * 5)
    assert math.isfinite(value.item())


def test_charbonnier_backward_pass_succeeds_with_finite_gradients() -> None:
    loss_fn = CharbonnierLoss(eps=1e-3)
    prediction = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 8, 8)
    loss = loss_fn(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_charbonnier_runs_on_cpu_tensors() -> None:
    loss_fn = CharbonnierLoss(eps=1e-3)
    prediction = torch.rand(1, 1, 4, 4, device="cpu")
    target = torch.rand(1, 1, 4, 4, device="cpu")
    value = loss_fn(prediction, target)
    assert value.device.type == "cpu"
    assert math.isfinite(value.item())


def test_charbonnier_matches_the_mathematical_formula() -> None:
    eps = 0.01
    loss_fn = CharbonnierLoss(eps=eps)
    prediction = torch.tensor([[0.1, 0.5], [0.9, -0.2]])
    target = torch.tensor([[0.0, 0.3], [1.2, 0.4]])
    expected = torch.sqrt((prediction - target) ** 2 + eps**2).mean()
    actual = loss_fn(prediction, target)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-8)


def test_charbonnier_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="eps"):
        CharbonnierLoss(eps=0.0)


def test_build_loss_config_defaults_to_l1() -> None:
    assert build_loss_config("l1") == {"name": "l1"}


def test_build_loss_config_charbonnier_respects_epsilon() -> None:
    config = build_loss_config("charbonnier", charbonnier_eps=5e-4)
    assert config == {"name": "charbonnier", "epsilon": 5e-4}


def test_build_loss_l1_constructs_l1_loss() -> None:
    loss_fn = build_loss({"name": "l1"})
    assert isinstance(loss_fn, nn.L1Loss)


def test_build_loss_charbonnier_constructs_charbonnier_loss_with_configured_epsilon() -> None:
    loss_fn = build_loss({"name": "charbonnier", "epsilon": 2e-3})
    assert isinstance(loss_fn, CharbonnierLoss)
    assert loss_fn.eps == 2e-3


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss({"name": "ssim"})


def test_loss_label_matches_expected_convention() -> None:
    assert loss_label("l1") == "L1"
    assert loss_label("charbonnier") == "Charbonnier"
    assert loss_label("l1_ssim") == "L1+SSIM"


# --- Differentiable SSIM (src/losses.py::differentiable_ssim, SSIMLoss) ---


def test_differentiable_ssim_identical_tensors_is_approximately_one() -> None:
    image = torch.rand(2, 1, 32, 32)
    value = differentiable_ssim(image, image.clone())
    assert value.item() == pytest.approx(1.0, abs=1e-5)


def test_ssim_loss_identical_tensors_is_approximately_zero() -> None:
    image = torch.rand(2, 1, 32, 32)
    loss = SSIMLoss()
    value = loss(image, image.clone())
    assert value.item() == pytest.approx(0.0, abs=1e-5)


def test_differentiable_ssim_different_tensors_is_lower() -> None:
    torch.manual_seed(0)
    image = torch.rand(2, 1, 32, 32)
    other = torch.rand(2, 1, 32, 32)
    identical_ssim = differentiable_ssim(image, image.clone())
    different_ssim = differentiable_ssim(image, other)
    assert different_ssim.item() < identical_ssim.item()


def test_differentiable_ssim_is_finite_on_random_batches() -> None:
    value = differentiable_ssim(torch.randn(4, 1, 16, 16) * 3, torch.randn(4, 1, 16, 16) * 3)
    assert math.isfinite(value.item())


def test_differentiable_ssim_backward_pass_succeeds_with_finite_gradients() -> None:
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    loss = 1.0 - differentiable_ssim(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_differentiable_ssim_works_with_single_channel_batched_tensor() -> None:
    prediction = torch.rand(3, 1, 20, 24)
    target = torch.rand(3, 1, 20, 24)
    value = differentiable_ssim(prediction, target)
    assert value.ndim == 0
    assert math.isfinite(value.item())


def test_differentiable_ssim_runs_on_cpu_tensors() -> None:
    prediction = torch.rand(1, 1, 16, 16, device="cpu")
    target = torch.rand(1, 1, 16, 16, device="cpu")
    value = differentiable_ssim(prediction, target)
    assert value.device.type == "cpu"
    assert math.isfinite(value.item())


def test_differentiable_ssim_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        differentiable_ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16))


# --- Composite L1+SSIM loss (src/losses.py::L1SSIMLoss) ---


def test_composite_loss_returns_a_scalar() -> None:
    loss = L1SSIMLoss(ssim_weight=0.1)
    value = loss(torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16))
    assert value.ndim == 0


def test_composite_loss_is_finite() -> None:
    loss = L1SSIMLoss(ssim_weight=0.1)
    value = loss(torch.randn(2, 1, 16, 16) * 4, torch.randn(2, 1, 16, 16) * 4)
    assert math.isfinite(value.item())


def test_composite_loss_matches_l1_plus_weighted_ssim_loss_on_synthetic_example() -> None:
    weight = 0.1
    torch.manual_seed(42)
    prediction = torch.rand(2, 1, 24, 24)
    target = torch.rand(2, 1, 24, 24)

    composite = L1SSIMLoss(ssim_weight=weight)
    actual = composite(prediction, target)

    expected_l1 = nn.L1Loss()(prediction, target)
    expected_ssim_loss = 1.0 - differentiable_ssim(prediction, target)
    expected = expected_l1 + weight * expected_ssim_loss

    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)


def test_composite_loss_backward_succeeds_through_both_components() -> None:
    loss_fn = L1SSIMLoss(ssim_weight=0.1)
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    loss = loss_fn(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad != 0).any()  # gradient actually flows, not a no-op


def test_composite_loss_with_zero_ssim_weight_equals_plain_l1() -> None:
    prediction = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    composite = L1SSIMLoss(ssim_weight=0.0)
    plain_l1 = nn.L1Loss()
    assert composite(prediction, target).item() == pytest.approx(
        plain_l1(prediction, target).item(), abs=1e-9
    )


def test_composite_loss_respects_configured_ssim_weight() -> None:
    prediction = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    low_weight = L1SSIMLoss(ssim_weight=0.01)(prediction, target)
    high_weight = L1SSIMLoss(ssim_weight=1.0)(prediction, target)
    # Different (non-zero) weight must change the composite value when SSIM loss > 0.
    assert low_weight.item() != pytest.approx(high_weight.item())


def test_composite_loss_rejects_negative_ssim_weight() -> None:
    with pytest.raises(ValueError, match="ssim_weight"):
        L1SSIMLoss(ssim_weight=-0.1)


# --- l1_ssim loss construction via build_loss_config/build_loss ---


def test_build_loss_config_l1_ssim_respects_ssim_weight() -> None:
    config = build_loss_config("l1_ssim", ssim_weight=0.25)
    assert config == {"name": "l1_ssim", "ssim_weight": 0.25}


def test_build_loss_config_l1_ssim_default_weight_is_point_one() -> None:
    config = build_loss_config("l1_ssim")
    assert config["ssim_weight"] == 0.1


def test_build_loss_l1_ssim_constructs_composite_loss_with_configured_weight() -> None:
    loss_fn = build_loss({"name": "l1_ssim", "ssim_weight": 0.3})
    assert isinstance(loss_fn, L1SSIMLoss)
    assert loss_fn.ssim_weight == 0.3


# --- MSE loss (Experiment 8: L1 -> MSE) ---


def test_mse_computes_mathematically_correct_value() -> None:
    loss_fn = build_loss({"name": "mse"})
    prediction = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 5.0]])
    # errors: 1, 0, 2, 2 -> squared: 1, 0, 4, 4 -> mean = 2.25
    expected = ((prediction - target) ** 2).mean()
    assert loss_fn(prediction, target).item() == pytest.approx(expected.item())
    assert loss_fn(prediction, target).item() == pytest.approx(2.25)


def test_mse_identical_prediction_and_target_is_zero() -> None:
    loss_fn = build_loss({"name": "mse"})
    image = torch.rand(2, 1, 16, 16)
    assert loss_fn(image, image.clone()).item() == pytest.approx(0.0, abs=1e-9)


def test_mse_is_differentiable_with_finite_gradients() -> None:
    loss_fn = build_loss({"name": "mse"})
    prediction = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 8, 8)
    loss = loss_fn(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_build_loss_config_mse() -> None:
    assert build_loss_config("mse") == {"name": "mse"}


def test_build_loss_mse_constructs_mse_loss() -> None:
    loss_fn = build_loss({"name": "mse"})
    assert isinstance(loss_fn, nn.MSELoss)


def test_loss_label_mse() -> None:
    assert loss_label("mse") == "MSE"
