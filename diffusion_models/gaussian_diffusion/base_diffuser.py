import abc
from typing import List, TYPE_CHECKING
from typing import Tuple

import torch

from diffusion_models.gaussian_diffusion.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.utils.schemas import Timestep

if TYPE_CHECKING:
  from diffusion_models.models.base_diffusion_model import BaseDiffusionModel


class BaseDiffuser(abc.ABC):
  def __init__(self, beta_scheduler: BaseBetaScheduler):
    """Initializes the object with the specified beta scheduler.

    BaseDiffuser is an abstract base class for different diffuser
    implementations. It defines the interface that all diffusers should adhere
    to.

    Args:
      beta_scheduler: The beta scheduler used by the diffuser.

    Warnings:
      Do not instantiate this class directly. Instead, build your own Diffuser
      by inheriting from BaseDiffuser.
      (see :class:`~.gaussian_diffuser.GaussianDiffuser`)

    """
    self.beta_scheduler: BaseBetaScheduler = beta_scheduler
    """The beta scheduler used by the diffuser."""

  @property
  @abc.abstractmethod
  def steps(self) -> List[int]:
    """Returns the list of steps used in the denoising process."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_timestep(self, number_of_images: int, idx: int) -> Timestep:
    """Get timestep information used for denoising."""
    raise NotImplementedError

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
