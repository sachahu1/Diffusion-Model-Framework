import abc
import logging
from typing import Optional

import torch


class BaseBetaScheduler:
  def __init__(self, steps: int, enforce_zero_terminal_snr: bool = False):
    super().__init__()
    self.steps = steps
    self.betas = self.sample_betas()
    self.alpha_bars = self.compute_alpha_bar()

    if enforce_zero_terminal_snr:
      self.enforce_zero_terminal_snr()

  def enforce_zero_terminal_snr(self):
    alpha_bar_length = len(self.alpha_bars)

    # Convert betas to alphas_bar_sqrt
    alphas = 1 - self.betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
      alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    if len(alphas) == alpha_bar_length:
      self.betas = betas
      self.alpha_bars = alphas_bar
    else:
      logging.warning(
        "Got different alpha_bar length after enforcing zero SNR. Please check your beta scheduler"
      )

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
  def from_tensors(
    cls, steps: int, betas: torch.Tensor, alpha_bars: torch.Tensor
  ):
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
    enforce_zero_terminal_snr: bool = True,
  ):
    self.beta_start = beta_start
    self.beta_end = beta_end
    super().__init__(
      steps=steps,
      enforce_zero_terminal_snr=enforce_zero_terminal_snr,
    )

  def sample_betas(self):
    return torch.linspace(self.beta_start, self.beta_end, self.steps)

  def compute_alpha_bar(self):
    alphas = 1 - self.betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


class CosineBetaScheduler(BaseBetaScheduler):
  def __init__(
    self,
    offset: float = 0.008,
    steps: int = 1000,
    max_beta: Optional[float] = 0.999,
  ):
    self.offset = offset
    self.max_beta = max_beta
    self.steps = steps
    self._alpha_bars = self._compute_alpha_bar()
    self._betas = self._compute_betas()

    super().__init__(
      steps=steps,
    )

  def f(self, t: torch.Tensor):
    return (
      torch.cos(
        (((t / self.steps) + self.offset) / (1 + self.offset)) * (torch.pi / 2)
      )
      ** 2
    )

  def _compute_betas(self):
    betas = 1 - self._alpha_bars[1:] / self._alpha_bars[:-1]
    if self.max_beta:
      betas = torch.clip(betas, max=self.max_beta)
    return betas

  def _compute_alpha_bar(self):
    t = torch.linspace(0, self.steps, self.steps, dtype=torch.float32)
    return self.f(t) / self.f(torch.tensor([0], dtype=torch.float32))

  def sample_betas(self):
    return self._betas

  def compute_alpha_bar(self):
    return self._alpha_bars
