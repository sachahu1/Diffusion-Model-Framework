"""Encode a dataset into latents once, so diffusion training never runs the encoder."""

import dataclasses
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_models.autoencoders.base import BaseAutoencoder

CACHE_VERSION = 1


@dataclass
class LatentCache:
  """The latent posteriors of a dataset.

  Every image is stored with the parameters of its posterior (``mean`` and
  ``std``), so training can draw a fresh latent each time it sees the image,
  and optionally also as its horizontal flip.
  """

  mean: torch.Tensor
  """``(N, V, C, h, w)``: the posterior means of ``V`` views (original, flipped)."""
  std: torch.Tensor
  """The posterior standard deviations, same shape as ``mean``."""
  scale_factor: float
  """Multiplies latents so that they have unit variance overall (the global option)."""
  channel_mean: Optional[torch.Tensor] = None
  """``(C,)``: the mean of each latent channel (for per-channel normalisation)."""
  channel_std: Optional[torch.Tensor] = None
  """``(C,)``: the standard deviation of each latent channel."""
  labels: Optional[torch.Tensor] = None
  """``(N, ...)``: what the dataset returned next to the image, for conditioning."""
  metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
  """Where the cache comes from (autoencoder checkpoint, flips, ...)."""
  version: int = CACHE_VERSION

  def __len__(self) -> int:
    return self.mean.shape[0]

  def to_file(self, path: Union[str, pathlib.Path]) -> None:
    """Write the cache atomically."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
      torch.save(dataclasses.asdict(self), temporary)
      os.replace(temporary, path)
    finally:
      temporary.unlink(missing_ok=True)

  @classmethod
  def from_file(cls, path: Union[str, pathlib.Path]) -> "LatentCache":
    # Memory mapped, so a cache bigger than the free RAM still loads.
    values = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if values.get("version") != CACHE_VERSION:
      raise ValueError(
        f"{path} is a latent cache of version {values.get('version')}, "
        f"expected {CACHE_VERSION}; rebuild it with `diffusion-models encode`."
      )
    return cls(**values)


def estimate_scale_factor(
  mean: torch.Tensor, std: torch.Tensor, chunk: int = 4096
) -> float:
  """``1 / std`` of latents drawn from the posteriors, over every element.

  A diffusion model assumes unit variance data (the noise it is mixed with
  has). Without rescaling, the noise schedule would be mismatched.
  """
  total = count = 0.0
  total_squares = 0.0
  generator = torch.Generator().manual_seed(0)
  for start in range(0, len(mean), chunk):
    m = mean[start : start + chunk].float()
    s = std[start : start + chunk].float()
    sample = m + s * torch.randn(m.shape, generator=generator)
    total += sample.sum().item()
    total_squares += sample.square().sum().item()
    count += sample.numel()
  average = total / count
  variance = total_squares / count - average**2
  return float(variance**-0.5)


def estimate_channel_statistics(
  mean: torch.Tensor, std: torch.Tensor, chunk: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor]:
  """The mean and std of each latent channel, on latents drawn from the posteriors."""
  generator = torch.Generator().manual_seed(0)
  channels = mean.shape[2]
  total = torch.zeros(channels, dtype=torch.float64)
  total_squares = torch.zeros(channels, dtype=torch.float64)
  count = 0
  for start in range(0, len(mean), chunk):
    m = mean[start : start + chunk].float()
    s = std[start : start + chunk].float()
    sample = (m + s * torch.randn(m.shape, generator=generator)).double()
    total += sample.sum(dim=(0, 1, 3, 4))
    total_squares += sample.square().sum(dim=(0, 1, 3, 4))
    count += sample.numel() // channels
  average = total / count
  variance = total_squares / count - average.square()
  return average.float(), variance.sqrt().float()


def _allocate(
  shape: Tuple[int, ...], scratch_directory: Optional[Union[str, pathlib.Path]]
) -> torch.Tensor:
  """An fp16 tensor, in memory or backed by a (unlinked) file in a directory."""
  if scratch_directory is None:
    return torch.empty(shape, dtype=torch.half)
  directory = pathlib.Path(scratch_directory)
  directory.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=directory, suffix=".scratch") as file:
    path = file.name
  size = 1
  for dimension in shape:
    size *= dimension
  tensor = torch.from_file(path, shared=True, size=size, dtype=torch.half)
  # The mapping outlives the name: the file disappears once the tensor does.
  os.unlink(path)
  return tensor.view(shape)


@torch.no_grad()
def build_latent_cache(
  autoencoder: BaseAutoencoder,
  dataset: Dataset,
  batch_size: int = 128,
  flip: bool = True,
  device: str = "cuda",
  num_workers: int = 4,
  metadata: Optional[Dict[str, Any]] = None,
  scratch_directory: Optional[Union[str, pathlib.Path]] = None,
) -> LatentCache:
  """Encode every image of ``dataset`` (``(image, label)`` pairs in ``[-1, 1]``).

  Args:
    autoencoder: The (trained) autoencoder.
    dataset: The images to encode, in the order they are cached.
    batch_size: Images per forward pass.
    flip: Also cache the horizontally flipped images. Training then picks a
      view at random, a free data augmentation. Turn it off for data with a
      left/right meaning (text, ...).
    device: Where to run the encoder.
    num_workers: Dataloader workers reading the images.
    metadata: Extra information stored in the cache.
    scratch_directory: Where to build the cache. By default it is built in
      memory; a directory on disk builds it in files the OS can evict, for
      caches larger than the free RAM (16 channels at 128px is ~7 GB).
  """
  autoencoder = autoencoder.to(device).eval().requires_grad_(False)
  loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
  )
  # Preallocated: concatenating batches would hold two copies.
  mean = std = None
  labels = []
  position = 0
  for done, (images, label) in enumerate(loader):
    images = images.to(device)
    views = [images, images.flip(-1)] if flip else [images]
    view_means, view_stds = [], []
    for view in views:
      view_mean, logvar = autoencoder.encode_distribution(view)
      view_std = (
        torch.exp(0.5 * logvar)
        if logvar is not None
        else torch.zeros_like(view_mean)
      )
      view_means.append(view_mean.float())
      view_stds.append(view_std.float())
    batch_mean = torch.stack(view_means, dim=1).half().cpu()
    batch_std = torch.stack(view_stds, dim=1).half().cpu()
    if mean is None:
      shape = (len(dataset), *batch_mean.shape[1:])  # ty: ignore[invalid-argument-type]
      mean = _allocate(shape, scratch_directory)
      std = _allocate(shape, scratch_directory)
    count = batch_mean.shape[0]
    mean[position : position + count] = batch_mean
    std[position : position + count] = batch_std  # ty: ignore[invalid-assignment]
    position += count
    # Copy: a shared memory batch holds a file descriptor until freed.
    labels.append(torch.as_tensor(label).clone())
    if done % 100 == 0:
      logging.info(f"Encoded {(done + 1) * batch_size}/{len(dataset)} images.")  # ty: ignore[invalid-argument-type]

  assert mean is not None and std is not None, "The dataset is empty."
  channel_mean, channel_std = estimate_channel_statistics(mean, std)
  return LatentCache(
    mean=mean,
    std=std,
    scale_factor=estimate_scale_factor(mean, std),
    channel_mean=channel_mean,
    channel_std=channel_std,
    labels=torch.cat(labels),
    metadata={
      "flip": flip,
      "latent_channels": autoencoder.latent_channels,
      "downsample_factor": autoencoder.downsample_factor,
      **(metadata or {}),
    },
  )
