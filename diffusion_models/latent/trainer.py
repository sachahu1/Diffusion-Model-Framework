from typing import Any, Dict

import torch

from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.diffusion.models.base_diffusion_model import (
  BaseDiffusionModel,
)
from diffusion_models.diffusion.trainer import DiffusionTrainer
from diffusion_models.latent.dataset import LatentDataset


class LatentDiffusionTrainer(DiffusionTrainer):
  def __init__(
    self,
    autoencoder: BaseAutoencoder,
    dataset: LatentDataset,
    *args,
    **kwargs,
  ):
    """Trains a diffusion model on cached latents.

    Training is the plain diffusion objective on the latents of ``dataset``;
    the autoencoder is frozen and only used to turn samples into images for the
    tensorboard grids and FID/KID, so it never costs a training step.

    Args:
      autoencoder: The autoencoder the latents were encoded with (a decoder
        fine-tuned with a frozen encoder works too, the latents stay valid).
      dataset: The latents, see :class:`~.LatentDataset`.
      *args: Passed to :class:`~.DiffusionTrainer`.
      **kwargs: Passed to :class:`~.DiffusionTrainer`. ``evaluation_dataset``
        holds the *images* FID compares against.
    """
    super().__init__(*args, dataset=dataset, **kwargs)
    latent_channels = self._image_shape[0]
    if latent_channels != autoencoder.latent_channels:
      raise ValueError(
        f"The dataset has {latent_channels} latent channels but the "
        f"autoencoder produces {autoencoder.latent_channels}."
      )
    self.autoencoder = autoencoder.to(self.device).eval().requires_grad_(False)
    """The frozen autoencoder decoding samples."""
    self.normalization = dataset.normalization
    """Maps the dataset's latents to the diffusion model's data and back."""

  @torch.no_grad()
  def to_image(self, samples: torch.Tensor) -> torch.Tensor:
    latents = self.normalization.denormalize(samples.float())
    return self.autoencoder.decode(latents).clamp(-1, 1)

  def evaluate_samples(self, model: BaseDiffusionModel) -> Dict[str, float]:
    assert self.evaluator is not None
    return self.evaluator.evaluate(
      model,
      noise_shape=self._image_shape,
      decode=self.to_image,
    )

  def checkpoint_fields(self) -> Dict[str, Any]:
    return {
      **super().checkpoint_fields(),
      "latent_scale_factor": self.normalization.scale_values(),
      "latent_shift": self.normalization.shift_values(),
      "latent_whitening": self.normalization.whitening_values(),
    }
