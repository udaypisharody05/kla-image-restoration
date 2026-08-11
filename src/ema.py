"""Exponential moving average (EMA) of a model's trainable parameters.

Generic training infrastructure -- works with any ``nn.Module``, not tied to
``ResidualSRNet`` or any other specific architecture. Maintains a shadow copy
of the model's weights, updated once per optimizer step:

    ema_param <- decay * ema_param + (1 - decay) * live_param

The shadow model requires no gradients and is never touched by any optimizer
-- ``update()`` is the only thing that ever changes it, and it is called
manually by the training loop right after ``optimizer.step()``.
"""

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Holds a shadow (EMA) copy of *model*'s parameters and buffers.

    Initialized as a deep copy of *model*'s **current** state -- never zeros
    -- so the shadow starts out identical to whatever weights *model* has at
    construction time (freshly initialized weights for a from-scratch run, or
    the resumed weights when reconstructing an ``ExponentialMovingAverage``
    for ``--resume``; the resumed case then has its shadow immediately
    overwritten by ``load_state_dict`` with the checkpoint's actual saved EMA
    state, so this initial copy is only ever the operative one for a
    from-scratch run).
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow_model = copy.deepcopy(model).eval()
        for parameter in self.shadow_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move the shadow model's weights toward *model*'s current weights.

        Call once per optimizer step (i.e. once per training batch), after
        ``optimizer.step()`` has already been applied to *model* -- the first
        call therefore happens after the very first batch of training (or, on
        a resumed run, after the first batch following the resumed epoch;
        the shadow's prior trajectory was already restored via
        ``load_state_dict`` before that point).

        Floating-point parameters are updated with the standard EMA formula.
        Buffers (e.g. BatchNorm running statistics -- none of this project's
        architectures have any, but this is handled generically for any
        future one that might) are copied directly rather than averaged: a
        buffer is not a gradient-updated parameter, so "smoothing" it has no
        well-defined meaning independent of whatever bookkeeping the live
        model already does internally: the shadow simply mirrors it exactly.
        """
        shadow_params = dict(self.shadow_model.named_parameters())
        for name, live_param in model.named_parameters():
            shadow_param = shadow_params[name]
            shadow_param.mul_(self.decay).add_(live_param, alpha=1.0 - self.decay)

        shadow_buffers = dict(self.shadow_model.named_buffers())
        for name, live_buffer in model.named_buffers():
            shadow_buffers[name].copy_(live_buffer)

    def state_dict(self) -> dict:
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow_model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "ExponentialMovingAverage":
        self.shadow_model = self.shadow_model.to(device)
        return self
