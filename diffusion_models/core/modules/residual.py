from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    num_groups: int = 32,
    dropout: float = 0.0,
    time_emb_dim: Optional[int] = None,
  ):
    """A GroupNorm residual block, optionally conditioned on a timestep embedding.

    Args:
      in_channels: The input channels.
      out_channels: The output channels.
      num_groups: The number of GroupNorm groups.
      dropout: The dropout probability between the two convolutions.
      time_emb_dim: The size of the timestep embedding. When set, ``forward``
        takes the embedding and adds a projection of it to the features (as in
        a diffusion Unet). Leave it ``None`` for unconditioned uses (a VAE).
    """
    super().__init__()
    self.norm1 = nn.GroupNorm(num_groups, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.time_proj = (
      nn.Linear(time_emb_dim, out_channels)
      if time_emb_dim is not None
      else None
    )
    self.norm2 = nn.GroupNorm(num_groups, out_channels)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.skip = (
      nn.Conv2d(in_channels, out_channels, 1)
      if in_channels != out_channels
      else nn.Identity()
    )

  def forward(
    self, x: torch.Tensor, t: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    if self.time_proj is not None:
      if t is None:
        raise ValueError("This block is time conditioned: pass an embedding.")
      h = h + self.time_proj(F.silu(t))[:, :, None, None]
    h = self.conv2(self.dropout(F.silu(self.norm2(h))))
    return h + self.skip(x)
