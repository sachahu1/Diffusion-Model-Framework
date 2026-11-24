"""Hydra-zen config stores for the training example.

Importing this package registers every config group (``beta_scheduler``,
``diffuser``, ``model``, ``autoencoder``, ``optimizer``, ``dataset``) with hydra-zen's store.
The top-level configs live next to the scripts that use them, e.g.
``examples/config/baseline.yaml``.
"""

from diffusion_models._store import (  # noqa: F401
  autoencoder,
  beta_scheduler,
  dataset,
  diffuser,
  discriminator,
  latent,
  loss,
  model,
  optimizer,
)
