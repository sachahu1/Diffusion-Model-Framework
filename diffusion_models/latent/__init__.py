"""Latent diffusion: diffusion models trained on the latents of an autoencoder.

The diffusion model itself is unchanged; it simply sees ``(C, h, w)`` latents
instead of images. This package adds the pieces around it:

- :mod:`~diffusion_models.latent.cache`: encode a dataset once into a file,
- :mod:`~diffusion_models.latent.dataset`: serve the cached (scaled) latents,
- :mod:`~diffusion_models.latent.trainer`: a diffusion trainer that decodes
  latents for logging and FID.
"""
