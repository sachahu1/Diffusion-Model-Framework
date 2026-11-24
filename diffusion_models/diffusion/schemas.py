import dataclasses
import logging
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Union

import torch

from diffusion_models.diffusion.diffusers.beta_schedulers import (
  BaseBetaScheduler,
)


class PredictionType(str, Enum):
  """What the model is trained to predict."""

  @classmethod
  def _missing_(cls, value):
    # Configs saved in checkpoints spell enums by member name ("V").
    if isinstance(value, str):
      for member in cls:
        if value.lower() in (member.name.lower(), member.value):
          return member
    return None

  Epsilon = "epsilon"
  """The noise added to the image."""
  V = "v"
  """The velocity, :math:`v = \\sqrt{\\bar{\\alpha}}\\epsilon - \\sqrt{1-\\bar{\\alpha}}x_0`."""


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
  beta_scheduler_config: BetaSchedulerConfiguration
  """The beta scheduler configuration."""
  scaler: Optional[Dict[str, Any]] = None
  """The GradScaler instance."""
  ema_model_state_dict: Optional[Dict[str, Any]] = None
  """The state dict of the EMA (exponential moving average) weights, if tracked."""
  ema_num_updates: Optional[int] = None
  """The number of EMA updates performed so far (needed to resume the EMA warmup)."""
  completed_epochs: Optional[int] = None
  """The number of fully completed epochs. Training resumes from here."""
  tensorboard_run_name: Optional[str] = None
  """The name of the tensorboard run."""
  image_channels: int = 3
  """The number of image channels used in the training."""
  prediction_type: str = "epsilon"
  """What the model predicts (``"epsilon"`` or ``"v"``). Legacy checkpoints are epsilon."""
  config_yaml: Optional[str] = None
  """The resolved configuration (YAML) of the run that produced this checkpoint."""
  latent_scale_factor: Optional[Union[float, List[float]]] = None
  """For a latent diffusion model: the model sees ``(latent - latent_shift) *
  latent_scale_factor``, per channel when this is a list."""
  latent_shift: Optional[List[float]] = None
  """The per-channel mean subtracted from the latents (none for a global scale)."""
  latent_whitening: Optional[List[List[float]]] = None
  """A ``(C, C)`` matrix applied across channels after the scale and shift
  (``normalization="whiten"``), ``None`` when the channels are only rescaled."""
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
      **checkpoint.beta_scheduler_config  # ty: ignore[invalid-argument-type]
    )
    checkpoint.beta_scheduler_config = beta_scheduler_config
    return checkpoint

  def get_model_state_dict(self, use_ema: bool = True) -> Dict[str, Any]:
    """Get the weights to load into a model.

    Args:
      use_ema: Prefer the EMA weights when the checkpoint has them. Falls back
        to the raw weights (with a warning) for checkpoints without EMA.

    Returns:
      A model state dict.
    """
    if use_ema:
      if self.ema_model_state_dict is not None:
        return self.ema_model_state_dict
      logging.warning("Checkpoint has no EMA weights; using the raw weights.")
    return self.model_state_dict

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
    """Saves a checkpoint to a file.

    The write is atomic: the checkpoint goes to a temporary file which then
    replaces the destination, so readers never see a partial file.
    """
    file_path = pathlib.Path(file_path)
    temporary_path = file_path.with_name(file_path.name + ".tmp")
    try:
      torch.save(dataclasses.asdict(self), temporary_path)
      os.replace(temporary_path, file_path)
    finally:
      temporary_path.unlink(missing_ok=True)


@dataclass
class OldCheckpoint:
  epoch: int
  model_state_dict: Dict[str, Any]
  optimizer_state_dict: Dict[str, Any]
  scaler: Optional[Dict[str, Any]]
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
