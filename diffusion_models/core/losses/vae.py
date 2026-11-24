from typing import Optional

import torch
from torch import nn

from diffusion_models.core.losses.base import LossOutput
from diffusion_models.core.losses.kl import gaussian_kl_divergence
from diffusion_models.core.losses.reconstruction import RECONSTRUCTION_LOSSES


class VaeLoss(nn.Module):
  def __init__(self, kl_weight: float = 1e-6, reconstruction_loss: str = "l1"):
    """The VAE objective: reconstruction plus a small KL regulariser.

    ``loss = reconstruction + kl_weight * KL / pixels``, with the KL summed over
    the latent of an image and normalised by the number of image values, like
    the reconstruction term. This makes ``kl_weight`` match the usual latent
    diffusion convention (``1e-6``): a tiny weight keeps the latent close to
    an autoencoder's (sharp reconstructions) while still bounding its scale.

    This is the place to add further terms (perceptual, adversarial) later.

    Args:
      kl_weight: The weight of the KL term.
      reconstruction_loss: One of :data:`~.reconstruction.RECONSTRUCTION_LOSSES`
        (``"l1"``, ``"mse"`` or ``"huber"``).
    """
    super().__init__()
    if reconstruction_loss not in RECONSTRUCTION_LOSSES:
      raise ValueError(
        f"reconstruction_loss must be one of {sorted(RECONSTRUCTION_LOSSES)}, "
        f"got {reconstruction_loss!r}."
      )
    self.kl_weight = kl_weight
    self.reconstruction_loss = reconstruction_loss

  def forward(
    self,
    images: torch.Tensor,
    reconstruction: torch.Tensor,
    mean: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
  ) -> LossOutput:
    """Compute the loss.

    Args:
      images: The target images.
      reconstruction: The reconstructed images.
      mean: The mean of the latent posterior.
      logvar: The log variance of the latent posterior (``None`` for a
        deterministic autoencoder, which has no KL term).
    """
    # Losses are computed in fp32 even under autocast.
    images = images.float()
    reconstruction = reconstruction.float()

    reconstruction_term = RECONSTRUCTION_LOSSES[self.reconstruction_loss](
      reconstruction, images
    )
    if logvar is None:
      kl_term = torch.zeros_like(reconstruction_term)
    else:
      kl_term = gaussian_kl_divergence(mean.float(), logvar.float())
      kl_term = kl_term / images[0].numel()

    loss = reconstruction_term + self.kl_weight * kl_term
    return LossOutput(
      loss=loss,
      metrics={
        "Loss": loss.detach(),
        "Reconstruction": reconstruction_term.detach(),
        "KL": kl_term.detach(),
      },
    )
