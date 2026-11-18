from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_models.gaussian_diffusion.base_diffuser import BaseDiffuser
from diffusion_models.models.base_diffusion_model import BaseDiffusionModel
from diffusion_models.models.SimpleUnet import SinusoidalPositionEmbeddings


class ResidualBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    time_emb_dim: int,
    num_groups: int,
    dropout: float,
  ):
    """A GroupNorm residual block conditioned on the timestep embedding."""
    super().__init__()
    self.norm1 = nn.GroupNorm(num_groups, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.time_proj = nn.Linear(time_emb_dim, out_channels)
    self.norm2 = nn.GroupNorm(num_groups, out_channels)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.skip = (
      nn.Conv2d(in_channels, out_channels, 1)
      if in_channels != out_channels
      else nn.Identity()
    )

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    h = h + self.time_proj(F.silu(t))[:, :, None, None]
    h = self.conv2(self.dropout(F.silu(self.norm2(h))))
    return h + self.skip(x)


class SelfAttention2d(nn.Module):
  def __init__(self, channels: int, num_groups: int, num_heads: int = 4):
    """Multi-head self-attention over the spatial positions of a feature map."""
    super().__init__()
    self.num_heads = num_heads
    self.norm = nn.GroupNorm(num_groups, channels)
    self.qkv = nn.Conv2d(channels, 3 * channels, 1)
    self.proj = nn.Conv2d(channels, channels, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    qkv = self.qkv(self.norm(x))
    qkv = qkv.reshape(batch, 3, self.num_heads, -1, height * width)
    query, key, value = (
      qkv[:, i].transpose(-1, -2) for i in range(3)
    )  # (B, heads, HW, C / heads)
    h = F.scaled_dot_product_attention(query, key, value)
    h = h.transpose(-1, -2).reshape(batch, channels, height, width)
    return x + self.proj(h)


class Downsample(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _Stage(nn.Module):
  """A residual block optionally followed by self-attention."""

  def __init__(self, block: ResidualBlock, attention: Optional[nn.Module]):
    super().__init__()
    self.block = block
    self.attention = attention if attention is not None else nn.Identity()

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return self.attention(self.block(x, t))


class SmallAttentionUnet(BaseDiffusionModel):
  def __init__(
    self,
    diffuser: BaseDiffuser,
    image_channels: int,
    base_channels: int = 64,
    channel_multipliers: Sequence[int] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    attention_levels: int = 2,
    num_groups: int = 32,
    dropout: float = 0.0,
  ):
    """A small DDPM-style Unet with GroupNorm and low-resolution attention.

    Compared to :class:`~.SimpleUnet.SimpleUnet`, this model uses GroupNorm
    (so training and evaluation behave identically, which matters for
    sampling and EMA), residual blocks, and self-attention only at the lowest
    ``attention_levels`` resolutions. With the defaults it has roughly 25M
    parameters and fits comfortably on a small GPU at 64x64.

    Args:
      diffuser: A gaussian diffuser.
      image_channels: The number of image channels.
      base_channels: The channel count at the highest resolution.
      channel_multipliers: Per level channel multiplier. Every level but the
        first halves the resolution.
      num_res_blocks: The number of residual blocks per level (down path).
      attention_levels: The number of lowest-resolution levels (and the
        bottleneck) which get self-attention.
      num_groups: The number of GroupNorm groups.
      dropout: The dropout probability inside residual blocks.
    """
    super().__init__(diffuser=diffuser)
    channels = [base_channels * m for m in channel_multipliers]
    if any(c % num_groups for c in channels):
      raise ValueError(
        f"All channel counts {channels} must be divisible by {num_groups}."
      )
    time_emb_dim = base_channels * 4
    number_of_levels = len(channels)
    attention_start = number_of_levels - attention_levels

    def stage(in_ch: int, out_ch: int, level: int) -> _Stage:
      return _Stage(
        ResidualBlock(in_ch, out_ch, time_emb_dim, num_groups, dropout),
        SelfAttention2d(out_ch, num_groups)
        if level >= attention_start
        else None,
      )

    self.time_mlp = nn.Sequential(
      SinusoidalPositionEmbeddings(base_channels),
      nn.Linear(base_channels, time_emb_dim),
      nn.SiLU(),
      nn.Linear(time_emb_dim, time_emb_dim),
    )
    self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)

    self.down_stages = nn.ModuleList()
    self.downsamples = nn.ModuleList()
    skip_channels = [base_channels]
    current = base_channels
    for level, out_ch in enumerate(channels):
      for _ in range(num_res_blocks):
        self.down_stages.append(stage(current, out_ch, level))
        current = out_ch
        skip_channels.append(current)
      if level != number_of_levels - 1:
        self.downsamples.append(Downsample(current))
        skip_channels.append(current)

    self.mid_block1 = stage(current, current, number_of_levels - 1)
    self.mid_block2 = ResidualBlock(
      current, current, time_emb_dim, num_groups, dropout
    )

    self.up_stages = nn.ModuleList()
    self.upsamples = nn.ModuleList()
    for level, out_ch in reversed(list(enumerate(channels))):
      for _ in range(num_res_blocks + 1):
        self.up_stages.append(
          stage(current + skip_channels.pop(), out_ch, level)
        )
        current = out_ch
      if level != 0:
        self.upsamples.append(Upsample(current))

    self.num_res_blocks = num_res_blocks
    self.norm_out = nn.GroupNorm(num_groups, current)
    self.conv_out = nn.Conv2d(current, image_channels, 3, padding=1)

  def forward(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    t = self.time_mlp(timestep)
    x = self.conv_in(x)
    skips = [x]

    stages = iter(self.down_stages)
    for level in range(len(self.downsamples) + 1):
      for _ in range(self.num_res_blocks):
        x = next(stages)(x, t)
        skips.append(x)
      if level < len(self.downsamples):
        x = self.downsamples[level](x)
        skips.append(x)

    x = self.mid_block2(self.mid_block1(x, t), t)

    stages = iter(self.up_stages)
    for level in range(len(self.upsamples) + 1):
      for _ in range(self.num_res_blocks + 1):
        x = next(stages)(torch.cat((x, skips.pop()), dim=1), t)
      if level < len(self.upsamples):
        x = self.upsamples[level](x)

    return self.conv_out(F.silu(self.norm_out(x)))
