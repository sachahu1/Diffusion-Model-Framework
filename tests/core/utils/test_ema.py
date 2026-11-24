import pytest
import torch
from torch import nn

from diffusion_models.core.utils.ema import ExponentialMovingAverage


def _model():
  return nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))


def test_update_averages_parameters_and_copies_buffers():
  model = _model()
  ema = ExponentialMovingAverage(model, decay=0.5)
  assert not ema.model.training
  initial = ema.model[0].weight.clone()

  with torch.no_grad():
    model[0].weight.add_(1.0)
  model.train()
  model(torch.randn(8, 2))  # updates the BatchNorm running statistics
  ema.update(model)

  assert torch.allclose(ema.model[0].weight, initial + 0.5)
  assert torch.equal(ema.model[1].running_mean, model[1].running_mean)
  assert ema.model[1].num_batches_tracked == 1
  assert all(not p.requires_grad for p in ema.model.parameters())


def test_warmup_decay_ramps_up():
  ema = ExponentialMovingAverage(_model(), decay=0.99, warmup_steps=10)
  assert ema.get_decay() == pytest.approx(0.1)
  ema.num_updates = 10_000
  assert ema.get_decay() == 0.99


def test_invalid_decay():
  with pytest.raises(ValueError):
    ExponentialMovingAverage(_model(), decay=1.0)


def test_ema_is_compile_aware():
  """A compile wrapper exposes ``_orig_mod``; the EMA must see through it."""

  class FakeCompiled(nn.Module):
    def __init__(self, module):
      super().__init__()
      self._orig_mod = module

    def forward(self, *args):
      return self._orig_mod(*args)

  model = _model()
  wrapped = FakeCompiled(model)
  ema = ExponentialMovingAverage(wrapped, decay=0.5)
  assert not any(k.startswith("_orig_mod") for k in ema.state_dict())
  assert set(ema.state_dict()) == set(model.state_dict())

  initial = ema.model[0].weight.clone()
  with torch.no_grad():
    model[0].weight.add_(2.0)
  ema.update(wrapped)
  assert torch.allclose(ema.model[0].weight, initial + 1.0)


def test_real_torch_compile_keeps_keys_clean():
  model = _model()
  # The eager backend avoids needing a C++ toolchain; keys are the same.
  compiled = torch.compile(model, backend="eager")
  ema = ExponentialMovingAverage(compiled)
  assert set(ema.state_dict()) == set(model.state_dict())
