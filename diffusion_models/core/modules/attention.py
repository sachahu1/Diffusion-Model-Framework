import torch
from torch import nn
from torch.nn import functional as F


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
