"""Building blocks shared by the models of the framework.

Task specific models (a diffusion Unet, a VAE, ...) are assembled from these
layers; only that assembly is specific to a task.
"""

from diffusion_models.core.modules.attention import SelfAttention2d
from diffusion_models.core.modules.discriminator import PatchDiscriminator
from diffusion_models.core.modules.embeddings import (
  SinusoidalPositionEmbeddings,
)
from diffusion_models.core.modules.residual import ResidualBlock
from diffusion_models.core.modules.resampling import Downsample, Upsample

__all__ = [
  "Downsample",
  "PatchDiscriminator",
  "ResidualBlock",
  "SelfAttention2d",
  "SinusoidalPositionEmbeddings",
  "Upsample",
]
