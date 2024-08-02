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
  training_name: str
  batch_size: int
  learning_rate: float
  number_of_epochs: int

  log_rate: int = 10
  image_rate: int = 100
  checkpoint_rate: int = 100

  mixed_precision_training: bool = False  # TODO: This is not complete yet
  gradient_clip: Optional[float] = None  # TODO: This is not complete yet


@dataclass
class LogConfiguration:
  log_rate: int = 10
  image_rate: int = 50
  number_of_images: int = 5
  # metrics: Dict[str, float] # TODO: consider Dict[str, Callable]


@dataclass
class BetaSchedulerConfiguration:
  steps: int
  betas: torch.Tensor
  alpha_bars: torch.Tensor


@dataclass
class Checkpoint:
  epoch: int
  model_state_dict: Dict[str, Any]
  optimizer_state_dict: Dict[str, Any]
  scaler: Optional[GradScaler]
  beta_scheduler_config: BetaSchedulerConfiguration
  tensorboard_run_name: Optional[str] = None
  image_channels: int = 3
  loss: Optional[float] = (
    None  # TODO: remove legacy parameter and resave models
  )

  @classmethod
  def from_file(cls, file_path: str) -> "Checkpoint":
    checkpoint = torch.load(f=file_path)
    checkpoint = cls(**checkpoint)
    beta_scheduler_config = BetaSchedulerConfiguration(
      **checkpoint.beta_scheduler_config
    )
    checkpoint.beta_scheduler_config = beta_scheduler_config
    return checkpoint

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
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
