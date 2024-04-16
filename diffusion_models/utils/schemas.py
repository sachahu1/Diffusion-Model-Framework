import dataclasses
import pathlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import torch
from torch.cuda.amp import GradScaler


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


@dataclass
class LogConfiguration:
  log_rate: int = 10
  image_rate: int = 50
  number_of_images: int = 5
  # metrics: Dict[str, float] # TODO: consider Dict[str, Callable]


@dataclass
class Checkpoint:
  epoch: int
  model_state_dict: Dict[str, Any]
  optimizer_state_dict: Dict[str, Any]
  scaler: Optional[GradScaler]
  tensorboard_run_name: Optional[str] = None
  loss: Optional[float] = None # TODO: remove legacy parameter and resave models

  @classmethod
  def from_file(cls, file_path: str) -> "Checkpoint":
    checkpoint = torch.load(f=file_path)
    return cls(**checkpoint)

  def to_file(self, file_path: Union[str, pathlib.Path]) -> None:
    torch.save(
      dataclasses.asdict(self), file_path
    )
