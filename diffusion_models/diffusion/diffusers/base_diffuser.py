import abc
from typing import List, Optional, TYPE_CHECKING
from typing import Tuple

import torch

from diffusion_models.diffusion.diffusers.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.diffusion.schemas import PredictionType, Timestep

if TYPE_CHECKING:
  from diffusion_models.diffusion.models.base_diffusion_model import (
    BaseDiffusionModel,
  )


class BaseDiffuser(abc.ABC):
  def __init__(
    self,
    beta_scheduler: BaseBetaScheduler,
    prediction_type: PredictionType = PredictionType.Epsilon,
    clip_value: Optional[float] = 1.0,
  ):
    """Initializes the object with the specified beta scheduler.

    BaseDiffuser is an abstract base class for different diffuser
    implementations. It defines the interface that all diffusers should adhere
    to.

    Args:
      beta_scheduler: The beta scheduler used by the diffuser.
      prediction_type: What the model is trained to predict, either the added
        noise (``"epsilon"``) or the velocity (``"v"``, see `"Progressive
        Distillation for Fast Sampling of Diffusion Models"
        <https://arxiv.org/abs/2202.00512>`_).
      clip_value: Samples are clamped to ``[-clip_value, clip_value]`` after
        every denoising step, which suits images normalised to ``[-1, 1]``.
        ``None`` disables it; it must be disabled for data that is not
        bounded, such as the unit-variance latents of a latent diffusion
        model (clamping them crushes the samples).

    Warnings:
      Do not instantiate this class directly. Instead, build your own Diffuser
      by inheriting from BaseDiffuser.
      (see :class:`~.gaussian_diffuser.GaussianDiffuser`)

    """
    self.show_progress: bool = True
    """Whether denoising shows a progress bar over its steps. FID evaluation
    turns it off and shows one bar over the samples instead."""
    self.clip_value: Optional[float] = clip_value
    """The bound samples are clamped to after each step, if any."""
    self.beta_scheduler: BaseBetaScheduler = beta_scheduler
    """The beta scheduler used by the diffuser."""
    self.prediction_type: PredictionType = PredictionType(prediction_type)
    """What the model predicts: the noise (epsilon) or the velocity (v)."""

  def _alpha_bar(self, timesteps: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    alpha_bar = self.beta_scheduler.alpha_bars.gather(dim=0, index=timesteps)
    return alpha_bar.reshape((-1, *((1,) * (ndim - 1))))

  def get_target(
    self, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
  ) -> torch.Tensor:
    """The regression target of the model for the configured prediction type.

    Args:
      images: The clean images.
      noise: The noise that was added to them (from :meth:`diffuse_batch`).
      timesteps: The timesteps used for diffusion.

    Returns:
      The noise for ``"epsilon"``, or the velocity
      :math:`v = \\sqrt{\\bar{\\alpha}_t}\\epsilon - \\sqrt{1 - \\bar{\\alpha}_t}x_0`
      for ``"v"``.
    """
    if self.prediction_type == PredictionType.Epsilon:
      return noise
    alpha_bar = self._alpha_bar(timesteps, images.ndim)
    return torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * images

  def to_epsilon_and_x0(
    self,
    output: torch.Tensor,
    images: torch.Tensor,
    timesteps: torch.Tensor,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert a model output into a noise estimate and a clean-image estimate.

    Args:
      output: The model prediction.
      images: The noisy images the model was given.
      timesteps: The timesteps the model was given.

    Returns:
      ``(epsilon, x0)``. ``x0`` is ``None`` for ``"epsilon"`` models, as it
      cannot be computed where :math:`\\bar{\\alpha}_t = 0`.
    """
    if self.prediction_type == PredictionType.Epsilon:
      return output, None
    alpha_bar = self._alpha_bar(timesteps, images.ndim)
    epsilon = (
      torch.sqrt(1 - alpha_bar) * images + torch.sqrt(alpha_bar) * output
    )
    x0 = torch.sqrt(alpha_bar) * images - torch.sqrt(1 - alpha_bar) * output
    return epsilon, x0

  @property
  @abc.abstractmethod
  def steps(self) -> List[int]:
    """Returns the list of steps used in the denoising process."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_timestep(self, number_of_images: int, idx: int) -> Timestep:
    """Get timestep information used for denoising."""
    raise NotImplementedError

  def _denoise_step(
    self, images: torch.Tensor, model: torch.nn.Module, timestep: Timestep
  ) -> torch.Tensor:
    """Denoise a batch of images by a single step.

    Used to step through generation one timestep at a time (see
    :meth:`~.DiffusionInference.get_generator`). Diffusers that support it
    override this method.

    Args:
      images: A tensor containing a batch of images to denoise.
      model: The model to be used for denoising.
      timestep: The current and previous timesteps.

    Returns:
      The images, denoised by one step.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def diffuse_batch(
    self, images: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diffuse a batch of images.

    Args:
      images: A tensor containing a batch of images.

    Returns:
      A tuple containing three tensors

        - images: Diffused batch of images.
        - noise: Noise added to the images.
        - timesteps: Timesteps used for diffusion.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def denoise_batch(
    self, images: torch.Tensor, model: "BaseDiffusionModel"
  ) -> List[torch.Tensor]:
    """Denoise a batch of images.

    Args:
      images: A tensor containing a batch of images to denoise.
      model: The model to be used for denoising.

    Returns:
      A list of tensors containing a batch of denoised images.
    """
    raise NotImplementedError()

  def clip(self, images: torch.Tensor) -> torch.Tensor:
    """Clamp ``images`` to the configured bound (a no-op without one)."""
    if self.clip_value is None:
      return images
    return torch.clamp(images, -self.clip_value, self.clip_value)

  @abc.abstractmethod
  def to(self, device: str = "cpu"):
    """Moves the data to the specified device.

    This performs a similar behaviour to the `to` method of PyTorch.

    Args:
      device: The device to which the method should move the data.
        It should be a string indicating the desired device.
        Default is "cpu".
    """
    raise NotImplementedError()
