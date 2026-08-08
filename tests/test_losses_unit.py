"""Fast, dataset-free tests for reconstruction loss selection (src/losses.py)."""

import math

import pytest
import torch
from torch import nn

from src.losses import CharbonnierLoss, build_loss, build_loss_config, loss_label


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
