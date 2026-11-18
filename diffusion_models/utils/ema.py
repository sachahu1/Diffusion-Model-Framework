import copy
from typing import Optional

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
  """Return the module underneath a ``torch.compile`` wrapper (if any).

  ``torch.compile`` returns an ``OptimizedModule`` that shares its parameters
  with the original module but prefixes every state dict key with
  ``_orig_mod.``. The EMA, checkpoints and optimizer must always deal with the
  original module so that keys stay stable whether or not compilation is on.
  """
  return getattr(model, "_orig_mod", model)


class ExponentialMovingAverage:
  def __init__(
    self,
    model: nn.Module,
    decay: float = 0.999,
    warmup_steps: int = 0,
  ):
    """Keeps an exponential moving average (EMA) of a model's weights.

    The EMA model is a frozen, eval-mode deep copy of the tracked model.
    The EMA is compile-aware: a ``torch.compile``-d model can be passed in, as
    both the copy and every update operate on the underlying module, and the
    EMA model itself is never compiled, so its keys stay free of
    ``_orig_mod.`` prefixes.
    Floating point parameters are averaged, while buffers (e.g. BatchNorm
    running statistics and integer counters) are copied verbatim from the
    tracked model, as averaging them is meaningless.

    Args:
      model: The model to track.
      decay: The decay of the moving average, in ``[0, 1)``.
      warmup_steps: When positive, the effective decay at update ``n`` is
        ``min(decay, (1 + n) / (warmup_steps + n))``. This makes the average
        follow the model closely at the start of training. ``0`` disables it.
    """
    if not 0.0 <= decay < 1.0:
      raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
    self.decay = decay
    """The target decay."""
    self.warmup_steps = warmup_steps
    """The warmup horizon, in updates."""
    self.num_updates = 0
    """The number of updates performed so far."""
    self.model = copy.deepcopy(unwrap_model(model))
    """The averaged model."""
    self.model.eval()
    self.model.requires_grad_(False)

  def get_decay(self) -> float:
    """The decay to use for the next update."""
    if self.warmup_steps > 0:
      n = self.num_updates
      return min(self.decay, (1 + n) / (self.warmup_steps + n))
    return self.decay

  @torch.no_grad()
  def update(self, model: nn.Module) -> None:
    """Update the averaged weights from the tracked model.

    Args:
      model: The tracked model (after an optimizer step). May be compiled.
    """
    model = unwrap_model(model)
    decay = self.get_decay()
    ema_parameters = dict(self.model.named_parameters())
    averaged_ema, averaged_model = [], []
    for name, parameter in model.named_parameters():
      if parameter.dtype.is_floating_point:
        averaged_ema.append(ema_parameters[name])
        averaged_model.append(parameter.detach())
    if averaged_ema:
      # ema <- ema + (1 - decay) * (model - ema), fused over all parameters.
      torch._foreach_lerp_(averaged_ema, averaged_model, 1.0 - decay)
    ema_buffers = dict(self.model.named_buffers())
    for name, buffer in model.named_buffers():
      ema_buffers[name].copy_(buffer)
    self.num_updates += 1

  def state_dict(self) -> dict:
    """The state dict of the averaged model."""
    return self.model.state_dict()

  def load_state_dict(
    self, state_dict: dict, num_updates: Optional[int] = None
  ) -> None:
    """Load averaged weights into the EMA model."""
    self.model.load_state_dict(state_dict)
    if num_updates is not None:
      self.num_updates = num_updates
