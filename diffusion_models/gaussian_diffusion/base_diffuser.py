import abc
from typing import List, Tuple

import torch

from diffusion_models.gaussian_diffusion.beta_schedulers import \
  BaseBetaScheduler


class BaseDiffuser(abc.ABC):
  def __init__(self, beta_scheduler: BaseBetaScheduler):
    self.beta_scheduler = beta_scheduler

  @abc.abstractmethod
  def diffuse_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass

  @abc.abstractmethod
  def denoise_batch(self, images: torch.Tensor, model) -> List[torch.Tensor]:
    pass

  @abc.abstractmethod
  def to(self, device: str = "cpu"):
    pass
