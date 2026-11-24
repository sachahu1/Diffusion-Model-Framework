import dataclasses
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class AutoencoderCheckpoint:
  """A checkpoint of an autoencoder training run."""

  epoch: int
  """The epoch the checkpoint was written at."""
  model_state_dict: Dict[str, Any]
  """The autoencoder weights."""
  optimizer_state_dict: Dict[str, Any]
  """The optimizer state."""
  completed_epochs: int = 0
  """The number of fully completed epochs. Training resumes from here."""
  ema_model_state_dict: Optional[Dict[str, Any]] = None
  """The EMA weights, if tracked."""
  ema_num_updates: Optional[int] = None
  """The number of EMA updates so far (to resume the EMA warmup)."""
  scaler: Optional[Dict[str, Any]] = None
  """The GradScaler state (float16 training only)."""
  image_channels: int = 3
  """The number of image channels."""
  config_yaml: Optional[str] = None
  """The resolved configuration (YAML) of the run, enough to rebuild the model."""
  tensorboard_run_name: Optional[str] = None
  """The tensorboard log directory of the run."""
  discriminator_state_dict: Optional[Dict[str, Any]] = None
  """The discriminator weights (adversarial training only)."""
  discriminator_optimizer_state_dict: Optional[Dict[str, Any]] = None
  """The discriminator optimizer state (adversarial training only)."""

  @classmethod
  def from_file(
    cls,
    file_path: Union[str, pathlib.Path],
    map_location: Optional[str] = None,
  ) -> "AutoencoderCheckpoint":
    """Load a checkpoint from a file."""
    return cls(
      **torch.load(f=file_path, weights_only=True, map_location=map_location)
    )

  def get_model_state_dict(self, use_ema: bool = True) -> Dict[str, Any]:
    """The weights to load into an autoencoder (the EMA ones when available)."""
    if use_ema:
      if self.ema_model_state_dict is not None:
        return self.ema_model_state_dict
      logging.warning("Checkpoint has no EMA weights; using the raw weights.")
    return self.model_state_dict

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
    """Save the checkpoint atomically (temporary file, then rename)."""
    file_path = pathlib.Path(file_path)
    temporary_path = file_path.with_name(file_path.name + ".tmp")
    try:
      torch.save(dataclasses.asdict(self), temporary_path)
      os.replace(temporary_path, file_path)
    finally:
      temporary_path.unlink(missing_ok=True)
