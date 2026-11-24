"""Frechet / Kernel Inception Distance of generated images.

The evaluator is task agnostic: it scores any callable that produces images
(diffusion samples, autoencoder reconstructions, ...)."""

import logging
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm


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
    self.number_of_samples = min(
      number_of_samples, len(real_dataset)  # ty: ignore[invalid-argument-type]
    )
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
    with tqdm(total=self.number_of_samples, desc="Real images") as progress:
      for images, _ in loader:
        self._update(self.to_unit_range(images), real=True)
        progress.update(len(images))
    self._real_ready = True

  @torch.no_grad()
  def evaluate_generator(
    self,
    generate: Callable[[int, int, torch.Generator], torch.Tensor],
  ) -> Dict[str, float]:
    """Score the images produced by ``generate``.

    Args:
      generate: ``generate(start, count, generator) -> images``. It returns
        ``count`` images in ``[-1, 1]`` for the samples ``start`` to
        ``start + count`` of the evaluation. ``generator`` is seeded with
        ``self.seed``, for sources of randomness such as noise. Sources of
        images that need the real data (e.g. reconstructions) can use
        ``start`` and ``count`` to index it.

    Returns:
      A dictionary with ``"FID"`` and, when enabled, ``"KID"`` and
      ``"KID_std"``.
    """
    if not self._real_ready:
      self._prepare_real()

    generator = torch.Generator(device=self.device).manual_seed(self.seed)
    start = 0
    with tqdm(
      total=self.number_of_samples, desc="Generated images"
    ) as progress:
      while start < self.number_of_samples:
        count = min(self.batch_size, self.number_of_samples - start)
        images = generate(start, count, generator)
        self._update(self.to_unit_range(images), real=False)
        start += count
        progress.update(count)

    metrics = {"FID": float(self.fid.compute())}
    self.fid.reset()
    if self.kid is not None:
      kid_mean, kid_std = self.kid.compute()
      metrics["KID"] = float(kid_mean)
      metrics["KID_std"] = float(kid_std)
      self.kid.reset()
    return metrics

  @torch.no_grad()
  def evaluate(
    self,
    model: nn.Module,
    noise_shape: Optional[Sequence[int]] = None,
    decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  ) -> Dict[str, float]:
    """Sample images from a diffusion model and score them.

    The model is put in eval mode for sampling and restored afterwards.

    Args:
      model: The (EMA) diffusion model to evaluate, already on
        ``self.device``. It needs a ``denoise(noise)`` method returning every
        step of the reverse process.
      noise_shape: The shape of one noise sample; defaults to the shape of the
        real images. A latent diffusion model samples smaller latents.
      decode: Maps the final samples to images in ``[-1, 1]``, e.g. the
        decoder of a latent diffusion model.

    Returns:
      A dictionary with ``"FID"`` and, when enabled, ``"KID"`` and
      ``"KID_std"``.
    """
    shape = tuple(
      noise_shape if noise_shape is not None else self.real_dataset[0][0].shape
    )

    def sample(start: int, count: int, generator: torch.Generator):
      noise = torch.randn(
        (count, *shape), device=self.device, generator=generator
      )
      # The last step is the fully denoised image.
      samples = model.denoise(noise)[-1]
      return decode(samples) if decode is not None else samples

    was_training = model.training
    model.eval()
    # One bar over the samples (see ``evaluate_generator``), not one per batch.
    diffuser = getattr(model, "diffuser", None)
    show_progress = getattr(diffuser, "show_progress", None)
    if show_progress is not None:
      diffuser.show_progress = False  # ty: ignore[invalid-assignment]
    try:
      return self.evaluate_generator(sample)
    finally:
      model.train(was_training)
      if show_progress is not None:
        diffuser.show_progress = show_progress  # ty: ignore[invalid-assignment]
