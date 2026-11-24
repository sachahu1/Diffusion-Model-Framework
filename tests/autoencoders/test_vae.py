import pytest
import torch

from diffusion_models.autoencoders import ConvVae
from diffusion_models.core.losses import VaeLoss


def _vae(**kwargs):
  defaults = dict(
    base_channels=32,
    channel_multipliers=(1, 2, 2),
    latent_channels=4,
    num_res_blocks=1,
  )
  return ConvVae(**{**defaults, **kwargs})


def test_shapes_and_downsample_factor():
  vae = _vae()
  assert vae.downsample_factor == 4
  images = torch.randn(2, 3, 32, 32)
  output = vae(images)
  assert output.reconstruction.shape == images.shape
  assert output.mean.shape == (2, 4, 8, 8) == output.logvar.shape
  assert vae.latent_size(32) == 8
  assert vae.decode(output.mean).shape == images.shape


def test_default_is_f8():
  assert ConvVae().downsample_factor == 8


def test_encode_is_deterministic_unless_sampling():
  vae = _vae().eval()
  images = torch.randn(2, 3, 16, 16)
  assert torch.equal(vae.encode(images), vae.encode(images))
  assert not torch.equal(
    vae.encode(images, sample=True), vae.encode(images, sample=True)
  )


def test_forward_without_sampling_is_deterministic():
  vae = _vae().eval()
  images = torch.randn(2, 3, 16, 16)
  a = vae(images, sample_posterior=False).reconstruction
  b = vae(images, sample_posterior=False).reconstruction
  assert torch.equal(a, b)


def test_rejects_channels_not_divisible_by_groups():
  with pytest.raises(ValueError, match="divisible"):
    ConvVae(base_channels=48)


def _loss(loss_function, images, output):
  return loss_function(
    images, output.reconstruction, output.mean, output.logvar
  )


def test_vae_learns_to_reconstruct():
  torch.manual_seed(0)
  vae = _vae()
  loss_function = VaeLoss()
  optimizer = torch.optim.Adam(vae.parameters(), lr=2e-3)
  images = torch.rand(4, 3, 16, 16) * 2 - 1
  first = _loss(loss_function, images, vae(images)).loss.item()
  for _ in range(40):
    optimizer.zero_grad()
    loss = _loss(loss_function, images, vae(images)).loss
    loss.backward()
    optimizer.step()
  assert loss.item() < first * 0.7
