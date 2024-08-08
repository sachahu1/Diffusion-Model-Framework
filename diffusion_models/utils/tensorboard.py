from typing import Dict
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardManager:
  def __init__(self, log_name: Optional[str] = None):
    """A tensorboard manager for simplified tensorboard logging.

    Args:
      log_name: The name of the tensorboard log run.
    """
    if log_name is None:
      log_name = "fill this"  ##########
    self.log_directory = f"../runs/{log_name}"
    """The directory where tensorboard logs are saved."""
    self.summary_writer = SummaryWriter(log_dir=self.log_directory)
    """The tensorboard summary writer."""

  def log_metrics(self, metrics: Dict[str, float], global_step: int):
    """Log metrics to tensorboard.

    Args:
      metrics: A dictionary mapping metric names to values.
      global_step: The step at which the metrics are recorded.
    """
    for metric_name, value in metrics.items():
      self.summary_writer.add_scalar(
        metric_name, value, global_step=global_step
      )

  def log_images(self, tag: str, images: torch.Tensor, timestep: int):
    """Log images to tensorboard.

    Args:
      tag: The name to give the images in tensorboard.
      images: A tensor representing the images to log.
      timestep: The timestep at which the images are produced.

    """
    self.summary_writer.add_images(
      tag=tag, img_tensor=images, global_step=timestep
    )
