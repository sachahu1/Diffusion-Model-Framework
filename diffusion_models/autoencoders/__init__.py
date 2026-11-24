"""Autoencoders for latent diffusion, trained from scratch.

A latent diffusion model runs the diffusion process on the compact latents of
a frozen autoencoder instead of on pixels. This package holds the autoencoder
framework: the models (:class:`~.vae.ConvVae`) and a trainer; the losses live
in :mod:`diffusion_models.core.losses`.
Train one with ``diffusion-models train autoencoder``.
"""

from diffusion_models.autoencoders.base import (
  AutoencoderOutput,
  BaseAutoencoder,
)
from diffusion_models.autoencoders.vae import ConvVae

__all__ = [
  "AutoencoderOutput",
  "BaseAutoencoder",
  "ConvVae",
]
