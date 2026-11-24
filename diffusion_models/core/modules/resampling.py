import torch
from torch import nn
from torch.nn import functional as F


class Downsample(nn.Module):
  def __init__(self, channels: int):
    """Halve the resolution with a strided convolution."""
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, channels: int):
    """Double the resolution (nearest neighbour) followed by a convolution."""
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
