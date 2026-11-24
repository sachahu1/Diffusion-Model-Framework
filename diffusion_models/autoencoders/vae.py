from typing import Iterator, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.core.modules import (
  Downsample,
  ResidualBlock,
  SelfAttention2d,
  Upsample,
)


def _middle(channels: int, num_groups: int, dropout: float) -> nn.Sequential:
  return nn.Sequential(
    ResidualBlock(channels, channels, num_groups, dropout),
    SelfAttention2d(channels, num_groups),
    ResidualBlock(channels, channels, num_groups, dropout),
  )


class Encoder(nn.Module):
  def __init__(
    self,
    image_channels: int,
    channels: Sequence[int],
    out_channels: int,
    num_res_blocks: int,
    num_groups: int,
    dropout: float,
  ):
    super().__init__()
    self.conv_in = nn.Conv2d(image_channels, channels[0], 3, padding=1)
    layers = []
    current = channels[0]
    for level, level_channels in enumerate(channels):
      for _ in range(num_res_blocks):
        layers.append(
          ResidualBlock(current, level_channels, num_groups, dropout)
        )
        current = level_channels
      if level != len(channels) - 1:
        layers.append(Downsample(current))
    self.down = nn.Sequential(*layers)
    self.middle = _middle(current, num_groups, dropout)
    self.norm_out = nn.GroupNorm(num_groups, current)
    self.conv_out = nn.Conv2d(current, out_channels, 3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.middle(self.down(self.conv_in(x)))
    return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
  def __init__(
    self,
    image_channels: int,
    channels: Sequence[int],
    latent_channels: int,
    num_res_blocks: int,
    num_groups: int,
    dropout: float,
  ):
    super().__init__()
    current = channels[-1]
    self.conv_in = nn.Conv2d(latent_channels, current, 3, padding=1)
    self.middle = _middle(current, num_groups, dropout)
    layers = []
    for level, level_channels in reversed(list(enumerate(channels))):
      # One more block than the encoder, as in the usual VAE designs.
      for _ in range(num_res_blocks + 1):
        layers.append(
          ResidualBlock(current, level_channels, num_groups, dropout)
        )
        current = level_channels
      if level != 0:
        layers.append(Upsample(current))
    self.up = nn.Sequential(*layers)
    self.norm_out = nn.GroupNorm(num_groups, current)
    self.conv_out = nn.Conv2d(current, image_channels, 3, padding=1)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    h = self.up(self.middle(self.conv_in(z)))
    return self.conv_out(F.silu(self.norm_out(h)))


class ConvVae(BaseAutoencoder):
  def __init__(
    self,
    image_channels: int = 3,
    base_channels: int = 64,
    channel_multipliers: Sequence[int] = (1, 2, 4, 4),
    latent_channels: int = 4,
    num_res_blocks: int = 2,
    num_groups: int = 32,
    dropout: float = 0.0,
  ):
    """A convolutional VAE with GroupNorm residual blocks and bottleneck attention.

    Every level but the first halves the resolution, so the spatial
    downsampling is ``2 ** (len(channel_multipliers) - 1)``: the default
    ``(1, 2, 4, 4)`` maps a 128x128 image to a ``latent_channels x 16 x 16``
    latent (f8). The posterior is a diagonal Gaussian, regularised by a (small)
    KL term (see :class:`~.losses.VaeLoss`).

    Args:
      image_channels: The number of image channels.
      base_channels: The channel count at the highest resolution.
      channel_multipliers: Per level channel multiplier.
      latent_channels: The number of latent channels.
      num_res_blocks: The residual blocks per level in the encoder (the
        decoder has one more).
      num_groups: The number of GroupNorm groups.
      dropout: The dropout probability inside residual blocks.
    """
    super().__init__()
    channels = [base_channels * m for m in channel_multipliers]
    if any(c % num_groups for c in channels):
      raise ValueError(
        f"All channel counts {channels} must be divisible by {num_groups}."
      )
    self.latent_channels = latent_channels
    self.downsample_factor = 2 ** (len(channels) - 1)
    self.encoder = Encoder(
      image_channels,
      channels,
      2 * latent_channels,
      num_res_blocks,
      num_groups,
      dropout,
    )
    self.decoder = Decoder(
      image_channels,
      channels,
      latent_channels,
      num_res_blocks,
      num_groups,
      dropout,
    )

  def encode_distribution(
    self, images: torch.Tensor
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    mean, logvar = self.encoder(images).chunk(2, dim=1)
    # Keeps exp() finite in low precision and early in training.
    return mean, torch.clamp(logvar, -30.0, 20.0)

  def decode(self, latents: torch.Tensor) -> torch.Tensor:
    return self.decoder(latents)

  def decoder_parameters(self) -> Iterator[nn.Parameter]:
    return self.decoder.parameters()

  def last_layer(self) -> nn.Parameter:
    return self.decoder.conv_out.weight  # ty: ignore[invalid-explicit-return]
