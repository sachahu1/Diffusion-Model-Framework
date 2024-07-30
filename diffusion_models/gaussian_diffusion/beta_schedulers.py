import abc

import torch


class BaseBetaScheduler:
  def __init__(self, steps: int):
    super().__init__()
    self.steps = steps
    self.betas = self.sample_betas()
    self.alpha_bars = self.compute_alpha_bar()

  @abc.abstractmethod
  def sample_betas(self):
    pass

  @abc.abstractmethod
  def compute_alpha_bar(self):
    pass

  def to(self, device: str):
    self.betas = self.betas.to(device)
    self.alpha_bars = self.alpha_bars.to(device)
    return self

  @classmethod
  def from_tensors(cls, steps: int, betas: torch.Tensor, alpha_bars: torch.Tensor):
    generic_beta_scheduler = cls(0)
    generic_beta_scheduler.steps = steps
    generic_beta_scheduler.betas = betas
    generic_beta_scheduler.alpha_bars = alpha_bars
    return generic_beta_scheduler

class LinearBetaScheduler(BaseBetaScheduler):
  def __init__(
    self,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    steps: int = 1000,
  ):
    self.beta_start = beta_start
    self.beta_end = beta_end
    super().__init__(steps=steps)

  def sample_betas(self):
    return torch.linspace(self.beta_start, self.beta_end, self.steps)

  def compute_alpha_bar(self):
    alphas = 1 - self.betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar
