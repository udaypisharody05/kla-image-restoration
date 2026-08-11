"""Fast, dataset-free tests for exponential moving average of model weights
(src/ema.py). No real dataset and no GPU are required.
"""

import pytest
import torch
from torch import nn

from src.ema import ExponentialMovingAverage
from src.models import ResidualSRNet


class _Scalar(nn.Module):
    """Minimal model: a single learnable scalar parameter -- makes the EMA
    arithmetic trivial to verify by hand."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


# --- Initialization ---


def test_ema_initializes_to_the_models_current_weights_not_zero() -> None:
    model = _Scalar(5.0)
    ema = ExponentialMovingAverage(model, decay=0.999)
    assert ema.shadow_model.weight.item() == 5.0


def test_ema_initialization_is_an_independent_copy() -> None:
    """Mutating the live model after construction must not move the shadow --
    they must be genuinely separate tensors, not aliased."""
    model = _Scalar(1.0)
    ema = ExponentialMovingAverage(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(99.0)
    assert ema.shadow_model.weight.item() == 1.0


def test_ema_rejects_decay_outside_open_unit_interval() -> None:
    model = _Scalar(1.0)
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(model, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(model, decay=0.0)


# --- Update formula (the numerical unit test from the task spec) ---


def test_ema_update_matches_hand_computed_value() -> None:
    """initial=1, live=3, decay=0.9 -> expected 0.9*1 + 0.1*3 = 1.2."""
    model = _Scalar(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    assert ema.shadow_model.weight.item() == pytest.approx(1.2)


def test_ema_update_formula_over_multiple_steps() -> None:
    """Two successive updates must compound correctly, not just reset to the
    latest live value."""
    model = _Scalar(0.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)  # 0.5*0 + 0.5*2 = 1.0
    assert ema.shadow_model.weight.item() == pytest.approx(1.0)
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update(model)  # 0.5*1.0 + 0.5*4 = 2.5
    assert ema.shadow_model.weight.item() == pytest.approx(2.5)


def test_ema_update_works_on_a_real_multi_parameter_model() -> None:
    torch.manual_seed(0)
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    initial_shadow = {name: p.clone() for name, p in ema.shadow_model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    for name, live_param in model.named_parameters():
        shadow_param = dict(ema.shadow_model.named_parameters())[name]
        expected = 0.9 * initial_shadow[name] + 0.1 * live_param
        assert torch.allclose(shadow_param, expected, atol=1e-6)


# --- No gradients / not optimizer-tracked / device handling ---


def test_ema_shadow_has_no_gradients() -> None:
    model = _Scalar(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    for parameter in ema.shadow_model.parameters():
        assert not parameter.requires_grad


def test_ema_update_produces_no_gradient_tracking() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.update(model)
    for parameter in ema.shadow_model.parameters():
        assert parameter.grad is None
        assert not parameter.requires_grad


def test_ema_parameters_are_not_in_the_optimizer() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for parameter in ema.shadow_model.parameters():
        assert id(parameter) not in optimizer_param_ids


def test_ema_update_does_not_alter_live_model_weights() -> None:
    model = _Scalar(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    assert model.weight.item() == 3.0  # unchanged by the EMA update itself


def test_ema_to_moves_shadow_to_target_device() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.to(torch.device("cpu"))
    for parameter in ema.shadow_model.parameters():
        assert parameter.device.type == "cpu"


# --- state_dict / load_state_dict ---


def test_ema_state_dict_round_trips() -> None:
    model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    state = ema.state_dict()

    fresh_model = ResidualSRNet(num_features=4, num_blocks=1, scale=2)
    restored_ema = ExponentialMovingAverage(fresh_model, decay=0.9)
    restored_ema.load_state_dict(state)
    for name, original in ema.shadow_model.named_parameters():
        restored = dict(restored_ema.shadow_model.named_parameters())[name]
        assert torch.equal(original, restored)
