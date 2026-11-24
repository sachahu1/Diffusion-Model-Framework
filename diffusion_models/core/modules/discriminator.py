import torch
from torch import nn


class PatchDiscriminator(nn.Module):
  def __init__(
    self,
    image_channels: int = 3,
    base_channels: int = 64,
    num_layers: int = 3,
    num_groups: int = 32,
  ):
    """A PatchGAN discriminator: a map of real/fake logits, one per image patch.

    ``num_layers`` stride-2 convolutions (the first without normalisation) are
    followed by a stride-1 one and a 1-channel output convolution. With the
    default of 3 layers the receptive field is 70x70 pixels and a 128x128 image
    gives a 14x14 map. GroupNorm is used (not BatchNorm) so the logits of an
    image do not depend on the rest of the batch.

    Args:
      image_channels: The number of image channels.
      base_channels: The channels of the first convolution; doubled at every
        layer up to 8x.
      num_layers: The number of stride-2 convolutions.
      num_groups: The number of GroupNorm groups.
    """
    super().__init__()
    layers = [
      nn.Conv2d(image_channels, base_channels, 4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
    ]
    channels = base_channels
    for layer in range(1, num_layers + 1):
      out_channels = min(base_channels * 2**layer, base_channels * 8)
      # The last block keeps the resolution (stride 1).
      stride = 2 if layer < num_layers else 1
      layers += [  # ty: ignore[unsupported-operator]
        nn.Conv2d(channels, out_channels, 4, stride=stride, padding=1),
        nn.GroupNorm(num_groups, out_channels),
        nn.LeakyReLU(0.2),
      ]
      channels = out_channels
    layers.append(nn.Conv2d(channels, 1, 4, stride=1, padding=1))
    self.layers = nn.Sequential(*layers)

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """Return the patch logits, ``(B, 1, h, w)``: positive means real."""
    return self.layers(images)
