"""Loss functions shared by every task.

Losses are plain callables; picking a sensible one for a task is up to the
user (see the ``loss`` config group). Regression losses take
``loss(prediction, target)`` (:data:`RECONSTRUCTION_LOSSES`). Composite
objectives such as :class:`VaeLoss` return a :class:`LossOutput`, the loss
together with its components for logging.
"""

from diffusion_models.core.losses.adversarial import (
  adaptive_weight,
  hinge_discriminator_loss,
  hinge_generator_loss,
)
from diffusion_models.core.losses.base import LossOutput
from diffusion_models.core.losses.kl import gaussian_kl_divergence
from diffusion_models.core.losses.reconstruction import RECONSTRUCTION_LOSSES
from diffusion_models.core.losses.vae import VaeLoss

__all__ = [
  "LossOutput",
  "RECONSTRUCTION_LOSSES",
  "VaeLoss",
  "adaptive_weight",
  "gaussian_kl_divergence",
  "hinge_discriminator_loss",
  "hinge_generator_loss",
]
