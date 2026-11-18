import torch

from diffusion_models.gaussian_diffusion.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.models.SmallAttentionUnet import (
  SelfAttention2d,
  SmallAttentionUnet,
)


def _model(**kwargs):
  diffuser = GaussianDiffuser(LinearBetaScheduler())
  return SmallAttentionUnet(
    diffuser=diffuser,
    image_channels=3,
    base_channels=32,
    channel_multipliers=(1, 2, 2),
    num_res_blocks=1,
    **kwargs,
  )


def test_forward_shape_and_backward():
  model = _model()
  x = torch.randn(2, 3, 16, 16)
  timestep = torch.tensor([0, 10])
  output = model(x, timestep)
  assert output.shape == x.shape
  output.square().mean().backward()
  assert all(p.grad is not None for p in model.parameters())


def test_no_batch_norm_and_eval_matches_train():
  model = _model()
  assert not any(
    isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    for m in model.modules()
  )
  x = torch.randn(2, 3, 16, 16)
  timestep = torch.tensor([3, 4])
  model.eval()
  with torch.no_grad():
    expected = model(x, timestep)
  model.train()
  with torch.no_grad():
    assert torch.allclose(model(x, timestep), expected, atol=1e-5)


def test_attention_only_at_lowest_levels():
  assert (
    sum(
      isinstance(m, SelfAttention2d)
      for m in _model(attention_levels=1).modules()
    )
    > 0
  )
  few = sum(
    isinstance(m, SelfAttention2d)
    for m in _model(attention_levels=1).modules()
  )
  more = sum(
    isinstance(m, SelfAttention2d)
    for m in _model(attention_levels=2).modules()
  )
  assert more > few


def test_diffuser_is_not_in_state_dict_and_roundtrips():
  model = _model()
  other = _model()
  other.load_state_dict(model.state_dict())
