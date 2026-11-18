"""Frechet / Kernel Inception Distance for diffusion models."""

import logging
from typing import Dict
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from diffusion_models.models.base_diffusion_model import BaseDiffusionModel


class ImageQualityEvaluator:
  def __init__(
    self,
    real_dataset: Dataset,
    number_of_samples: int = 5000,
    batch_size: int = 100,
    device: str = "cuda",
    compute_kid: bool = True,
    kid_subset_size: int = 1000,
    seed: int = 0,
    num_workers: int = 4,
  ):
    """Scores generated images against real ones with FID (and KID).

    The Inception statistics of the real images are computed once, on the
    first call to :meth:`evaluate`, and reused afterwards. FID is biased by the
    number of samples, so only compare scores that use the same
    ``number_of_samples``. KID is unbiased, which makes it more reliable for
    small sample sizes.

    The real dataset must yield ``(image, label)`` pairs whose images are
    normalized to ``[-1, 1]`` (the same convention as the training data).

    Args:
      real_dataset: The reference images, ideally a held-out split.
      number_of_samples: How many real and generated images to compare.
      batch_size: The batch size used for sampling and feature extraction.
      device: The device used for sampling and the Inception network.
      compute_kid: Whether to also compute the Kernel Inception Distance.
      kid_subset_size: The KID subset size (at most ``number_of_samples``).
      seed: Seed of the sampling noise, so scores are comparable across runs.
      num_workers: Dataloader workers used to read the real images.
    """
    self.real_dataset = real_dataset
    """The reference images."""
    self.number_of_samples = min(number_of_samples, len(real_dataset))
    """The number of real and generated images compared."""
    self.batch_size = batch_size
    """The batch size used for sampling and feature extraction."""
    self.device = device
    """The device used for sampling and feature extraction."""
    self.seed = seed
    """The seed of the sampling noise."""
    self.num_workers = num_workers
    """Dataloader workers used to read the real images."""

    # ``reset_real_features=False`` keeps the real statistics across resets.
    self.fid = FrechetInceptionDistance(
      feature=2048, normalize=True, reset_real_features=False
    ).to(device)
    """The FID metric."""
    self.kid: Optional[KernelInceptionDistance] = None
    """The KID metric, when enabled."""
    if compute_kid:
      self.kid = KernelInceptionDistance(
        feature=2048,
        subset_size=min(kid_subset_size, self.number_of_samples),
        normalize=True,
        reset_real_features=False,
      ).to(device)
    self._real_ready = False

  @staticmethod
  def to_unit_range(images: torch.Tensor) -> torch.Tensor:
    """Map images from ``[-1, 1]`` to ``[0, 1]``."""
    return ((images.float() + 1) / 2).clamp(0, 1)

  def _update(self, images: torch.Tensor, real: bool) -> None:
    images = images.to(self.device)
    self.fid.update(images, real=real)
    if self.kid is not None:
      self.kid.update(images, real=real)

  def _prepare_real(self) -> None:
    logging.info(
      f"Computing Inception statistics of {self.number_of_samples} real images."
    )
    loader = DataLoader(
      Subset(self.real_dataset, range(self.number_of_samples)),
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
    )
    for images, _ in loader:
      self._update(self.to_unit_range(images), real=True)
    self._real_ready = True

  @torch.no_grad()
  def evaluate(self, model: BaseDiffusionModel) -> Dict[str, float]:
    """Generate images with the model and score them.

    The model is put in eval mode for sampling and restored afterwards.

    Args:
      model: The (EMA) model to evaluate, already on ``self.device``.

    Returns:
      A dictionary with ``"FID"`` and, when enabled, ``"KID"`` and
      ``"KID_std"``.
    """
    if not self._real_ready:
      self._prepare_real()

    image_shape = self.real_dataset[0][0].shape
    generator = torch.Generator(device=self.device).manual_seed(self.seed)
    was_training = model.training
    model.eval()
    try:
      remaining = self.number_of_samples
      while remaining > 0:
        count = min(self.batch_size, remaining)
        noise = torch.randn(
          (count, *image_shape), device=self.device, generator=generator
        )
        # ``denoise`` returns every step of the reverse process; the last
        # entry is the fully denoised image.
        images = model.denoise(noise)[-1]
        self._update(self.to_unit_range(images), real=False)
        remaining -= count
    finally:
      model.train(was_training)

    metrics = {"FID": float(self.fid.compute())}
    self.fid.reset()
    if self.kid is not None:
      kid_mean, kid_std = self.kid.compute()
      metrics["KID"] = float(kid_mean)
      metrics["KID_std"] = float(kid_std)
      self.kid.reset()
    return metrics
