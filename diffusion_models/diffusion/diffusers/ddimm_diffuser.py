from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from diffusion_models.diffusion.diffusers.base_diffuser import BaseDiffuser
from diffusion_models.diffusion.diffusers.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.diffusion.models.base_diffusion_model import (
  BaseDiffusionModel,
)
from diffusion_models.diffusion.schemas import (
  Checkpoint,
  PredictionType,
  Timestep,
)


class DenoisingMode(str, Enum):
  @classmethod
  def _missing_(cls, value):
    # Configs saved in checkpoints spell enums by member name ("Trailing").
    if isinstance(value, str):
      for member in cls:
        if value.lower() == member.name.lower():
          return member
    return None

  Linear = "linear"
  """Evenly spaced steps.

  Starts at the last timestep for v-prediction. For epsilon prediction it stops
  short of it (see :meth:`DdimDiffuser.steps`)."""
  Quadratic = "quadratic"
  """Steps densely spaced near the clean end. Same start rule as ``Linear``."""
  Trailing = "trailing"
  """Evenly spaced steps that end on the last (pure noise) timestep.

  Needed to sample models trained with zero terminal SNR properly. Requires
  v-prediction, as epsilon prediction is undefined at zero terminal SNR."""


class DdimDiffuser(BaseDiffuser):
  def __init__(
    self,
    beta_scheduler: BaseBetaScheduler,
    mode: DenoisingMode = DenoisingMode.Quadratic,
    number_of_steps: int = 20,
    prediction_type: PredictionType = PredictionType.Epsilon,
    clip_value: Optional[float] = 1.0,
    eta: float = 0.0,
    timestep_power: float = 1.0,
  ):
    """Initializes the class instance.

    Args:
      beta_scheduler (BaseBetaScheduler): The beta scheduler instance to be used.
      mode: How the sampling timesteps are spaced.
      number_of_steps: Number of denoising steps.
      prediction_type: Whether the model predicts the noise or the velocity.
      clip_value: See :class:`~.BaseDiffuser`; ``None`` disables the clamping.
      eta: The stochasticity of sampling: ``0`` is deterministic DDIM, ``1``
        is DDPM-like ancestral sampling. On the f8c16 latent diffusion model
        ``1.0`` lowered the FID from 38.1 to 27.4 at no training cost
        (``investigations/latent-diffusion-artifacts/experiments/21-*``).
      timestep_power: How training timesteps are drawn: ``t = steps * u**p``
        with ``u`` uniform, so ``1`` is uniform and ``2`` favours low noise
        (density proportional to ``1/sqrt(t)``). The weak latent directions only
        become visible at low noise, where uniform sampling spends few steps.

    """
    super().__init__(beta_scheduler, prediction_type, clip_value)

    if eta < 0:
      raise ValueError(f"eta must be non-negative, got {eta}.")
    if timestep_power <= 0:
      raise ValueError(
        f"timestep_power must be positive, got {timestep_power}."
      )
    self.timestep_power = timestep_power
    """Exponent of the training timestep distribution (1 is uniform)."""

    self.eta = eta
    """Sampling noise: 0 is deterministic DDIM, 1 is DDPM-like."""

    self.number_of_steps = number_of_steps
    """Number of steps to use in the denoising process."""

    self.mode = DenoisingMode(mode)
    """How the sampling timesteps are spaced."""

    self.device: str = "cpu"
    """The device to use. Defaults to cpu."""

  @property
  def steps(self) -> List[int]:
    if self.mode == DenoisingMode.Trailing:
      stride = self.beta_scheduler.steps / self.number_of_steps
      time_steps = (
        np.round(np.arange(1, self.number_of_steps + 1) * stride).astype(int)
        - 1
      )
      # No +1 shift: the last step is the final timestep.
      self._time_steps = time_steps
      self._time_steps_prev = np.concatenate([[0], time_steps[:-1]])
      self._check_first_step()
      return list(range(self.number_of_steps))[::-1]
    # Only v-prediction is defined at alpha_bar == 0.
    full_range = self._defined_at_zero_snr
    steps = self.beta_scheduler.steps
    n = self.number_of_steps
    if self.mode == DenoisingMode.Linear:
      if full_range:
        time_steps = np.linspace(0, steps - 2, n).round().astype(int)
      else:
        a = steps // n
        time_steps = np.asarray(list(range(0, steps, a)))
    elif full_range:
      time_steps = (np.linspace(0, np.sqrt(steps - 2), n) ** 2).round()
      time_steps = time_steps.astype(int)
    else:
      time_steps = (np.linspace(0, np.sqrt(steps * 0.8), n) ** 2).astype(int)
    # Rounding can repeat a value; force strictly increasing, keeping n steps.
    idx = np.arange(len(time_steps))
    time_steps = np.maximum.accumulate(time_steps - idx) + idx
    self._time_steps = time_steps + 1
    self._time_steps_prev = np.concatenate([[0], self._time_steps[:-1]])
    self._check_first_step()
    return list(range(self.number_of_steps))[::-1]

  @property
  def _defined_at_zero_snr(self) -> bool:
    """Whether the prediction type yields an x0 (v), valid at alpha_bar == 0."""
    return self.prediction_type != PredictionType.Epsilon

  def _check_first_step(self) -> None:
    first = int(self._time_steps[-1])
    alpha_bar = float(self.beta_scheduler.alpha_bars[first])
    if alpha_bar == 0.0 and not self._defined_at_zero_snr:
      raise ValueError(
        f"The first sampling timestep ({first}) has alpha_bar == 0 (zero "
        "terminal SNR), where epsilon prediction is undefined and would "
        "produce NaN. Use prediction_type V, or a mode that starts earlier "
        "(Linear or Quadratic)."
      )

  def get_timestep(self, number_of_images: int, idx: int) -> Timestep:
    timestep = torch.full(
      (number_of_images,), self._time_steps[idx], device=self.device
    )
    timestep_prev = torch.full(
      (number_of_images,), self._time_steps_prev[idx], device=self.device
    )
    return Timestep(
      current=timestep,
      previous=timestep_prev,
    )

  @classmethod
  def from_checkpoint(cls, checkpoint: Checkpoint) -> "DdimDiffuser":
    """Instantiate a DDIM Diffuser from a training checkpoint.

    Args:
      checkpoint: The training checkpoint object containing
        the trained model's parameters and configuration.

    Returns:
      An instance of the DdimDiffuser class initialized with the parameters
      loaded from the given checkpoint.
    """
    return cls(
      beta_scheduler=BaseBetaScheduler.from_tensors(
        steps=checkpoint.beta_scheduler_config.steps,
        betas=checkpoint.beta_scheduler_config.betas,
        alpha_bars=checkpoint.beta_scheduler_config.alpha_bars,
      ),
      prediction_type=PredictionType(checkpoint.prediction_type),
    )

  def to(self, device: str = "cpu"):
    """Moves the data to the specified device.

    This performs a similar behaviour to the `to` method of PyTorch. moving the
    GaussianDiffuser and the BetaScheduler to the specified device.

    Args:
      device: The device to which the method should move the object.
        Default is "cpu".

    Example:
      >>> ddim_diffuser = DdimDiffuser()
      >>> ddim_diffuser = ddim_diffuser.to(device="cuda")
    """
    self.device = device
    self.beta_scheduler = self.beta_scheduler.to(self.device)
    return self

  def _diffuse_batch(
    self, images: torch.Tensor, timesteps: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(images, device=self.device)

    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(dim=0, index=timesteps)

    alpha_bar_t = alpha_bar_t.reshape((-1, *((1,) * (len(images.shape) - 1))))

    mu = torch.sqrt(alpha_bar_t)
    sigma = torch.sqrt(1 - alpha_bar_t)
    images = mu * images + sigma * noise
    return images, noise, timesteps

  def diffuse_batch(
    self, images: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diffuse a batch of images.

    Diffuse the given batch of images by adding noise based on the beta scheduler.

    Args:
      images: Batch of images to diffuse.\n
        Shape should be ``(B, C, H, W)``.

    Returns:
      A tuple containing three tensors

        - images: Diffused batch of images.
        - noise: Noise added to the images.
        - timesteps: Timesteps used for diffusion.
    """
    steps = self.beta_scheduler.steps
    if self.timestep_power == 1.0:
      timesteps = torch.randint(
        0, steps, (images.shape[0],), device=self.device
      )
    else:
      u = torch.rand(images.shape[0], device=self.device)
      timesteps = (u**self.timestep_power * steps).long().clamp(max=steps - 1)
    return self._diffuse_batch(images, timesteps)

  @torch.no_grad()
  def _denoise_step(
    self,
    images: torch.Tensor,
    model: torch.nn.Module,
    timestep: Timestep,
    eta: float = 0.0,
  ) -> torch.Tensor:
    current_timestep = timestep.current
    previous_timestep = timestep.previous

    output = model(images, current_timestep)
    epsilon_theta, x0_theta = self.to_epsilon_and_x0(
      output, images, current_timestep
    )

    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(
      dim=0, index=current_timestep
    ).reshape(-1, 1, 1, 1)

    alpha_bar_t_prev = self.beta_scheduler.alpha_bars.gather(  # ty: ignore[no-matching-overload]
      dim=0, index=previous_timestep
    ).reshape(-1, 1, 1, 1)

    sigma = eta * torch.sqrt(
      (1 - alpha_bar_t_prev)
      / (1 - alpha_bar_t)
      * (1 - alpha_bar_t / alpha_bar_t_prev)
    )

    epsilon_t = torch.randn_like(images)

    if x0_theta is not None:
      # Division free form, valid where alpha_bar_t == 0.
      mu = (
        torch.sqrt(alpha_bar_t_prev) * x0_theta
        + torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * epsilon_theta
      )
      return mu + sigma * epsilon_t

    mu = (
      torch.sqrt(alpha_bar_t_prev / alpha_bar_t) * images
      + (
        torch.sqrt(1 - alpha_bar_t_prev - sigma**2)
        - torch.sqrt((alpha_bar_t_prev * (1 - alpha_bar_t)) / alpha_bar_t)
      )
      * epsilon_theta
    )
    return mu + sigma * epsilon_t

  def denoise_batch(
    self,
    images: torch.Tensor,
    model: "BaseDiffusionModel",
  ) -> List[torch.Tensor]:
    """Denoise a batch of images.

    This denoises a batch images. This is the image generation process.

    Args:
      images: A batch of noisy images.
      model: The model to be used for denoising.

    Returns:
      A list of tensors containing a batch of denoised images.
    """
    denoised_images = []
    for i in tqdm(
      self.steps, desc="Denoising", disable=not self.show_progress
    ):
      timestep = self.get_timestep(images.shape[0], idx=i)

      images = self._denoise_step(
        images,
        model=model,
        timestep=timestep,
        eta=self.eta,
      )
      images = self.clip(images)
      denoised_images.append(images)
    return denoised_images
