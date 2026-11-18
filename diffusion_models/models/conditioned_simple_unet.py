# Diffusion Model taken from DeepFindr (https://www.youtube.com/@DeepFindr)
import math
from typing import Optional

import torch
from torch import nn

from diffusion_models.gaussian_diffusion.base_diffuser import BaseDiffuser
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.models.SimpleUnet import (
  SinusoidalPositionEmbeddings,
  Block,
)
from diffusion_models.models.base_diffusion_model import BaseDiffusionModel


class ConditionedSimpleUnet(BaseDiffusionModel):
  def __init__(
    self,
    diffuser: BaseDiffuser,
    image_channels: int,
    text_embedding_size: int,
  ):
    """A Simplified variant of the Unet architecture used in DDPM.

    Args:
      diffuser: A gaussian diffuser.
      image_channels: The number of image channels.
    """
    super().__init__(diffuser=diffuser)
    image_channels = image_channels
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    out_dim = image_channels
    time_emb_dim = 32

    # Text embeddings
    self.text_embedding_mlp = nn.Sequential(
      nn.Linear(text_embedding_size, time_emb_dim),
      nn.ReLU(),
    )

    # Time embedding
    self.time_mlp = nn.Sequential(
      SinusoidalPositionEmbeddings(time_emb_dim),
      nn.Linear(time_emb_dim, time_emb_dim),
      nn.ReLU(),
    )

    # Initial projection
    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

    # Downsample
    self.downs = nn.ModuleList(
      [
        Block(down_channels[i], down_channels[i + 1], time_emb_dim)
        for i in range(len(down_channels) - 1)
      ]
    )
    # Upsample
    self.ups = nn.ModuleList(
      [
        Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
        for i in range(len(up_channels) - 1)
      ]
    )

    # Edit: Corrected a bug found by Jakub C (see YouTube comment)
    self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

  def forward(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
  ):
    # if self.training:
    #   x = self.augmentations(x)

    # Embed time
    t = self.time_mlp(timestep)

    # Embed time
    if labels is not None:
      text_embedding = self.text_embedding_mlp(labels)
    else:
      text_embedding = torch.zeros_like(t)

    h = t + text_embedding

    # Initial conv
    x = self.conv0(x)
    # Unet
    residual_inputs = []
    for down in self.downs:
      x = down(x, h)
      residual_inputs.append(x)
    for up in self.ups:
      residual_x = residual_inputs.pop()
      # Add residual x as additional channels
      x = torch.cat((x, residual_x), dim=1)
      x = up(x, h)
    return self.output(x)
