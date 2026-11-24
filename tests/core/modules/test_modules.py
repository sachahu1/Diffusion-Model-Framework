import pytest
import torch

from diffusion_models.core.modules import (
  Downsample,
  ResidualBlock,
  SelfAttention2d,
  SinusoidalPositionEmbeddings,
  Upsample,
)


def test_residual_block_without_time_conditioning():
  block = ResidualBlock(32, 64, num_groups=8)
  assert block.time_proj is None
  assert block(torch.randn(2, 32, 8, 8)).shape == (2, 64, 8, 8)


def test_residual_block_with_time_conditioning():
  block = ResidualBlock(32, 32, num_groups=8, time_emb_dim=16)
  x, t = torch.randn(2, 32, 8, 8), torch.randn(2, 16)
  assert block(x, t).shape == x.shape
  assert not torch.equal(block(x, t), block(x, torch.zeros_like(t)))
  with pytest.raises(ValueError, match="time conditioned"):
    block(x)


def test_state_dict_keys_are_stable():
  # Existing checkpoints depend on these names.
  keys = ResidualBlock(32, 64, 8, time_emb_dim=16).state_dict().keys()
  assert {
    "norm1.weight",
    "conv1.weight",
    "time_proj.weight",
    "skip.weight",
  } <= set(keys)
  keys = ResidualBlock(32, 64, 8).state_dict().keys()
  assert not any(k.startswith("time_proj") for k in keys)


def test_resampling_and_attention_shapes():
  x = torch.randn(2, 32, 8, 8)
  assert Downsample(32)(x).shape == (2, 32, 4, 4)
  assert Upsample(32)(x).shape == (2, 32, 16, 16)
  assert SelfAttention2d(32, 8)(x).shape == x.shape
  assert SinusoidalPositionEmbeddings(16)(torch.arange(3.0)).shape == (3, 16)


def test_patch_discriminator_gives_a_map_of_patch_logits():
  from diffusion_models.core.modules import PatchDiscriminator

  discriminator = PatchDiscriminator()
  assert discriminator(torch.randn(2, 3, 128, 128)).shape == (2, 1, 14, 14)
  small = PatchDiscriminator(base_channels=32, num_layers=2)
  assert small(torch.randn(2, 3, 32, 32)).shape == (2, 1, 6, 6)


def test_patch_discriminator_is_independent_across_the_batch():
  from diffusion_models.core.modules import PatchDiscriminator

  discriminator = PatchDiscriminator(base_channels=32, num_layers=2).eval()
  images = torch.randn(4, 3, 32, 32)
  together = discriminator(images)
  alone = discriminator(images[:1])
  assert torch.allclose(together[:1], alone, atol=1e-5)
