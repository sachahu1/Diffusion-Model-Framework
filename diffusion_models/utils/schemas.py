import dataclasses
import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch
from torch.cuda.amp import GradScaler

from diffusion_models.gaussian_diffusion.beta_schedulers import (
  BaseBetaScheduler,
)


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
  mixed_precision_training: bool = False  # TODO: This is not complete yet
  """Whether or not to use automatic mixed precision training."""
  gradient_clip: Optional[float] = None  # TODO: This is not complete yet
  """Whether or not to clip gradients."""


@dataclass
class LogConfiguration:
  """An object to manage logging configuration."""

  log_rate: int = 10
  """The rate at which training metrics are logged."""
  image_rate: int = 50
  """The rate at which images are generated for visualization. This can be used to validate model performance."""
  number_of_images: int = 5
  """The number of images to generate."""
  # metrics: Dict[str, float] # TODO: consider Dict[str, Callable]


@dataclass
class BetaSchedulerConfiguration:
  """A simplified beta scheduler configuration."""

  steps: int
  """The number of steps in the beta scheduler."""
  betas: torch.Tensor
  """The beta values."""
  alpha_bars: torch.Tensor
  """The alpha bar values."""


@dataclass
class Checkpoint:
  """A simplified checkpoint framework for easy saving and loading."""

  epoch: int
  """The current epoch."""
  model_state_dict: Dict[str, Any]
  """The model state dict."""
  optimizer_state_dict: Dict[str, Any]
  """The optimizer state dict."""
  scaler: Optional[GradScaler]
  """The GradScaler instance."""
  beta_scheduler_config: BetaSchedulerConfiguration
  """The beta scheduler configuration."""
  tensorboard_run_name: Optional[str] = None
  """The name of the tensorboard run."""
  image_channels: int = 3
  """The number of image channels used in the training."""
  loss: Optional[float] = (
    None  # TODO: remove legacy parameter and resave models
  )
  """The final loss value recorded.
  
  Note: 
    This is a legacy parameter and will be removed in a future release.
  
  """

  @classmethod
  def from_file(
    cls, file_path: str, map_location: Optional[str] = None
  ) -> "Checkpoint":
    """Load and instantiate a checkpoint from a file.

    Args:
      file_path: The path to the checkpoint file.
      map_location: A function, torch. device, string or a dict specifying how to remap storage location.

    Returns:
      A checkpoint instance.
    """
    checkpoint = torch.load(
      f=file_path, weights_only=True, map_location=map_location
    )
    checkpoint = cls(**checkpoint)
    beta_scheduler_config = BetaSchedulerConfiguration(
      **checkpoint.beta_scheduler_config
    )
    checkpoint.beta_scheduler_config = beta_scheduler_config
    return checkpoint

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
    """Saves a checkpoint to a file."""
    torch.save(dataclasses.asdict(self), file_path)


@dataclass
class OldCheckpoint:
  epoch: int
  model_state_dict: Dict[str, Any]
  optimizer_state_dict: Dict[str, Any]
  scaler: Optional[GradScaler]
  # beta_scheduler_config: BetaSchedulerConfiguration
  tensorboard_run_name: Optional[str] = None
  loss: Optional[float] = (
    None  # TODO: remove legacy parameter and resave models
  )

  @classmethod
  def from_file(cls, file_path: str) -> "OldCheckpoint":
    checkpoint = torch.load(f=file_path)
    return cls(**checkpoint)

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
    torch.save(dataclasses.asdict(self), file_path)

  def to_new_checkpoint(self, beta_scheduler: BaseBetaScheduler) -> Checkpoint:
    beta_scheduler_config = BetaSchedulerConfiguration(
      steps=beta_scheduler.steps,
      betas=beta_scheduler.betas,
      alpha_bars=beta_scheduler.alpha_bars,
    )
    return Checkpoint(
      **dataclasses.asdict(self), beta_scheduler_config=beta_scheduler_config
    )

@dataclass
class Timestep:
  current: torch.Tensor
  previous: Optional[torch.Tensor] = None
