import abc
import logging
from typing import Optional

import torch


class BaseBetaScheduler:
  def __init__(self,
    steps: int,
    enforce_zero_terminal_snr: bool = False,
    initialize: bool = True,
  ):
    """Initializes a beta scheduler.

    BaseBetaScheduler is an abstract base class for different beta scheduler
    implementations. It defines the interface that all beta schedulers should
    adhere to.

    Args:
      steps: The number of steps for the beta.
      enforce_zero_terminal_snr: Whether to enforce zero terminal SNR inline
        with `"Common Diffusion Noise Schedules and Sample Steps are Flawed"
        <https://arxiv.org/abs/2305.08891>`_.\n
        Defaults to ``False``.
      initialize: Whether to initialize the beta scheduler. If this is
        set to ``False``, you will need to manually set ``self.betas`` and
        ``self.alpha_bars``. Otherwise, they are initialized using your
        ``sample_betas`` and ``compute_alpha_bar`` methods.

    Warnings:
      Do not instantiate this class directly. Instead, build your own Beta
      scheduler by inheriting from BaseBetaScheduler.
      (see :class:`~.LinearBetaScheduler`)
    """
    self.steps: int = steps
    """The number of steps for the beta scheduler."""
    self.betas = None
    """The :math:`\\beta` computed according to :meth:`~.BaseBetaScheduler.sample_betas`."""
    self.alpha_bars = None
    """The :math:`\\bar{\\alpha}` computed according to :meth:`~.BaseBetaScheduler.compute_alpha_bar`."""

    if initialize:
      self._initialize()
      if enforce_zero_terminal_snr:
        self.enforce_zero_terminal_snr()

  def _initialize(self):
      self.betas = self.sample_betas()
      self.alpha_bars = self.compute_alpha_bar()

  def enforce_zero_terminal_snr(self):
    """Enforce terminal SNR by adjusting :math:`\\beta` and :math:`\\bar{\\alpha}`.

    This method enforces zero terminal SNR according to
    `"Common Diffusion Noise Schedules and Sample Steps are Flawed"
    <https://arxiv.org/abs/2305.08891>`_.
    """
    alpha_bar_length = len(self.alpha_bars)

    # Convert betas to alphas_bar_sqrt
    alphas = 1 - self.betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_t
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
      alphas_bar_sqrt_0 - alphas_bar_sqrt_t
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
  def sample_betas(self) -> torch.Tensor:
    """Compute :math:`\\beta` for noise scheduling.

    Returns:
      A torch tensor of the :math:`\\beta` values.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_alpha_bar(self) -> torch.Tensor:
    """Compute :math:`\\bar{\\alpha}` for noise scheduling.

    Returns:
      A torch tensor of the :math:`\\bar{\\alpha}` values.
    """
    raise NotImplementedError()

  def to(self, device: str):
    """Moves the beta scheduler to the given device.

    Args:
      device: The device to which the method should move the object.
        Default is "cpu".

    """
    self.betas = self.betas.to(device)
    self.alpha_bars = self.alpha_bars.to(device)
    return self

  @classmethod
  def from_tensors(
    cls, steps: int, betas: torch.Tensor, alpha_bars: torch.Tensor
  ):
    """Instantiate a beta scheduler from tensors.

    Instantiate a beta scheduler from tensors. This is particularly useful for
    loading checkpoints.

    Args:
      steps: The number of steps for the beta scheduler.
      betas: The pre-computed beta values for the noise scheduler.
      alpha_bars: The pre-computed alpha bar values for the noise scheduler.

    Returns:

    """
    generic_beta_scheduler = cls(steps, initialize=False)
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
    """A Linear Beta scheduler.

    A simple linear beta scheduler with betas linearly spaced between
    ``beta_start`` and ``beta_end``.

    Args:
      beta_start: The starting value of the betas.
      beta_end: The end value of the betas.
      steps: The number of steps for the beta scheduler. This is also the number
        of betas.
      enforce_zero_terminal_snr: Whether to enforce zero terminal SNR.
    """
    self.beta_start: int = beta_start
    """The starting value of the betas."""
    self.beta_end: int = beta_end
    """The end value of the betas."""
    super().__init__(
      steps=steps,
      enforce_zero_terminal_snr=enforce_zero_terminal_snr,
    )

  def sample_betas(self) -> torch.Tensor:
    """Return linearly spaced betas between ``self.beta_start`` and ``self.beta_end``."""
    return torch.linspace(self.beta_start, self.beta_end, self.steps)

  def compute_alpha_bar(self):
    """Return :math:`\\bar{\\alpha}` computed from the beta values."""
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
    """A Cosine Beta scheduler.

    A Cosine Beta Scheduler based on the following formulas:

    .. math::
      :nowrap:

      \\begin{equation}
      \\left\\{ \\begin{aligned}
        \\bar{\\alpha}_t &= \\frac{f(t)}{f(0)} \\\\
        \\beta_t &= 1 - \\frac{\\bar{\\alpha}_t}{\\bar{\\alpha}_t -1}
      \\end{aligned} \\right.
      \\end{equation}

    where

      .. math::

        f(t) = \\cos(\\frac{t/T + s}{1 + s} * \\frac{\\pi}{2})^2

    where

      .. math::
        :nowrap:

        \\begin{equation}
        \\left\\{ \\begin{aligned}

          s & \\text{ is the offset} \\\\
          T & \\text{ is the number of steps}

        \\end{aligned} \\right.
        \\end{equation}

    Args:
      offset: The offset :math:`s` defined above.
      steps: The number of steps for the beta scheduler.
      max_beta: The maximum beta values. Higher values will be clipped.
    """
    self.offset: float = offset
    """The offset :math:`s` defined above."""
    self.max_beta: Optional[float] = max_beta
    """The maximum beta values. Higher values will be clipped."""
    self.steps: int = steps
    """The number of steps for the beta scheduler."""
    self._alpha_bars = self._compute_alpha_bar()
    self._betas = self._compute_betas()

    super().__init__(
      steps=steps,
    )

  def f(self, t: torch.Tensor) -> torch.Tensor:
    """A helper function to compute the :math:`\\bar{\\alpha}_t`.

    Args:
      t: The timestep to compute.

    Returns:

      .. math::

        f(t) = \\cos(\\frac{t/T + s}{1 + s} * \\frac{\\pi}{2})^2

    """
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
