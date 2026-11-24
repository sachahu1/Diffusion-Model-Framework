import pathlib
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from diffusion_models.latent.cache import (
  LatentCache,
  estimate_channel_statistics,
)
from diffusion_models.latent.normalization import LatentNormalization


class LatentDataset(Dataset):
  def __init__(
    self,
    path: Union[str, pathlib.Path],
    normalization: str = "channel",
    scale_factor: Optional[float] = None,
    sample: bool = True,
  ):
    """Serves the scaled latents of a :class:`~.LatentCache` file.

    It yields ``(latent, label)`` pairs like an image dataset does, so the
    diffusion trainer needs no change (and conditioning on ``label`` works as
    it would on images).

    Args:
      path: A cache written by ``diffusion-models encode``.
      normalization: ``"whiten"`` also decorrelates the channels (a rotation
        and a rescaling), so every direction of the latent has unit variance
        and the diffusion noise treats them alike. ``"channel"`` subtracts each channel's mean and divides by
        its std, so every channel is zero-mean and unit-variance. ``"global"``
        only multiplies by one factor (``scale_factor``), keeping the
        channels' relative means and scales.
      scale_factor: The factor of ``"global"``; defaults to the cache's, which
        makes the latents unit variance overall.
      sample: Draw the latent from its posterior instead of using the mean.
        A flipped view is picked at random when the cache has one.
    """
    cache = LatentCache.from_file(path)
    self.mean = cache.mean
    self.std = cache.std
    self.labels = cache.labels
    self.metadata = cache.metadata
    if normalization == "channel":
      channel_mean, channel_std = cache.channel_mean, cache.channel_std
      if (
        channel_mean is None or channel_std is None
      ):  # a cache of an older run
        channel_mean, channel_std = estimate_channel_statistics(
          cache.mean, cache.std
        )
      self.normalization = LatentNormalization(
        scale=1 / channel_std, shift=channel_mean
      )
    elif normalization == "whiten":
      self.normalization = self._whitening(cache)
    elif normalization == "global":
      factor = scale_factor if scale_factor is not None else cache.scale_factor
      self.normalization = LatentNormalization(scale=factor)
    else:
      raise ValueError(
        f"normalization must be 'channel', 'whiten' or 'global', not {normalization!r}."
      )
    """Maps cached latents to the diffusion model's data and back."""
    self.sample = sample

  @staticmethod
  def _whitening(
    cache: LatentCache, images: int = 20000
  ) -> LatentNormalization:
    """Centres the channels and decorrelates them, with unit variance each.

    The covariance across channels is estimated on the posterior means of the
    first ``images`` images (every spatial position is one observation).
    """
    values = cache.mean[:images, 0].float()
    channels = values.shape[1]
    flat = values.permute(0, 2, 3, 1).reshape(-1, channels).double()
    mean = flat.mean(0)
    covariance = torch.cov((flat - mean).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = (
      eigenvalues.clamp_min(1e-8).rsqrt().unsqueeze(1) * eigenvectors.T
    )
    return LatentNormalization(
      scale=1.0, shift=mean.float(), whitening=whitening.float()
    )

  def __len__(self) -> int:
    return self.mean.shape[0]

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    view = (
      int(torch.randint(self.mean.shape[1], ()))
      if self.mean.shape[1] > 1
      else 0
    )
    latent = self.mean[index, view].float()
    if self.sample:
      latent = latent + self.std[index, view].float() * torch.randn_like(
        latent
      )
    label = self.labels[index] if self.labels is not None else torch.tensor(0)
    return self.normalization.normalize(latent), label
