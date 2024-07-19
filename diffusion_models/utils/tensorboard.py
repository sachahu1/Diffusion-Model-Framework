from typing import Dict
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardManager:
  def __init__(self, log_name: Optional[str] = None):
    if log_name is None:
      log_name = "fill this"  ##########
    self.log_directory = f"../runs/{log_name}"
    self.summary_writer = SummaryWriter(log_dir=self.log_directory)

  def log_metrics(self, metrics: Dict[str, float], global_step: int):
    for metric_name, value in metrics.items():
      self.summary_writer.add_scalar(
        metric_name, value, global_step=global_step
      )

  def log_images(self, tag: str, images: torch.Tensor, timestep: int):
    self.summary_writer.add_images(
      tag=tag, img_tensor=images, global_step=timestep
    )
