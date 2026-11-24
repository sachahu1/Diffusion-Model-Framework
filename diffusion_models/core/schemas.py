from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfiguration:
  """A training configuration for simple experiment management."""

  training_name: str
  """The name of the training."""
  batch_size: int
  """The batch size used for training."""
  learning_rate: float
  """The learning rate of the training."""
  number_of_epochs: int
  """The number of epoch used for training."""
  checkpoint_rate: int = 100
  """The rate at which checkpoints are saved.."""
  mixed_precision_training: bool = False
  """Whether or not to use automatic mixed precision training."""
  mixed_precision_dtype: str = "bfloat16"
  """The autocast dtype when mixed precision is on: ``bfloat16`` (no gradient
  scaling needed) or ``float16`` (uses a ``GradScaler``)."""
  channels_last: bool = False
  """Whether to use the ``channels_last`` memory format for the model and its
  inputs. This speeds up mixed precision training on tensor cores but is very
  slow in fp32, so it should be combined with ``mixed_precision_training``."""
  gradient_clip: Optional[float] = None  # TODO: This is not complete yet
  """Whether or not to clip gradients."""
  lr_schedule: Optional[str] = None
  """How the learning rate changes over training: ``"cosine"`` or ``"linear"``
  decay from ``learning_rate`` to ``lr_min_ratio * learning_rate`` over the
  epochs from ``lr_schedule_start_epoch`` to ``number_of_epochs``. A constant
  rate when None. It is a function of the step, so resuming needs no state."""
  lr_warmup_steps: int = 0
  """Steps of linear warmup from zero at the start of the schedule."""
  lr_min_ratio: float = 0.0
  """The final learning rate as a fraction of ``learning_rate``."""
  lr_schedule_start_epoch: int = 0
  """The epoch the schedule starts at; before it the rate is constant. Set it to
  the epoch of a checkpoint to decay a finished run over the extra epochs."""
  ema_decay: Optional[float] = None
  """The decay of the exponential moving average of the weights. Disabled when None."""
  ema_warmup_steps: int = 0
  """When positive, ramps the EMA decay up as ``(1 + n) / (warmup + n)`` (capped at ``ema_decay``)."""
  latest_checkpoint_rate: Optional[int] = None
  """The rate (in epochs) at which the rolling ``latest.pt`` is overwritten. Disabled when None."""
  num_workers: int = 4
  """The number of dataloader worker processes. ``0`` loads data in the main process."""
  pin_memory: bool = True
  """Whether the dataloader copies batches into pinned memory (only used with CUDA)."""
  persistent_workers: bool = True
  """Whether dataloader workers are kept alive between epochs (needs ``num_workers > 0``)."""
  prefetch_factor: Optional[int] = None
  """Batches prefetched per worker. Uses the torch default when None (needs ``num_workers > 0``)."""
  compile: bool = False
  """Whether to ``torch.compile`` the model's forward pass during training.
  Checkpoints and the EMA always hold the uncompiled model's weights."""


@dataclass
class EvaluationConfiguration:
  """Settings of the FID / KID evaluation that runs during training."""

  fid_rate: Optional[int] = None
  """The rate (in epochs) at which FID/KID is computed and logged. Also computed after the last epoch. Disabled when None."""
  number_of_samples: int = 5000
  """The number of generated (and real) images compared. Only compare scores that use the same number."""
  batch_size: int = 100
  """The batch size used for sampling and feature extraction."""
  compute_kid: bool = True
  """Whether to also compute the Kernel Inception Distance."""
  kid_subset_size: int = 1000
  """The KID subset size."""
  seed: int = 0
  """The seed of the sampling noise, so scores are comparable across runs."""
  num_workers: int = 4
  """Dataloader workers used to read the real images."""


@dataclass
class LogConfiguration:
  """An object to manage logging configuration."""

  log_rate: int = 10
  """The rate at which training metrics are logged."""
  image_rate: int = 50
  """The rate at which images are generated for visualization. This can be used to validate model performance."""
  number_of_images: int = 5
  """The number of images to generate."""
  weight_rate: Optional[int] = None
  """The rate (in steps) at which weight histograms are logged. Disabled when None."""
