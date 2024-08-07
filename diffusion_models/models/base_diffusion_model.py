import abc
from typing import List, Tuple

import torch
from torch import nn

from diffusion_models.gaussian_diffusion.base_diffuser import BaseDiffuser


class BaseDiffusionModel(nn.Module, abc.ABC):
  def __init__(self, diffuser: BaseDiffuser):
    """Initializes the object with the specified diffuser.

    BaseDiffusionModel is an abstract base class for different diffusion models
    implementations. It defines the interface that all diffusion models should
    adhere to.

    Args:
      diffuser: The diffuser to use for the diffusion model.

    Warnings:
      Do not instantiate this class directly. Instead, build your own diffusion
      model by inheriting from BaseDiffusionModel.
      (see :class:`~.SimpleUnet.SimpleUnet`)

    """
    super().__init__()
    self.diffuser: BaseDiffuser = diffuser
    """A diffuser to be used by the diffusion model."""

  def diffuse(
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
    return self.diffuser.diffuse_batch(images=images)

  def denoise(self, images: torch.Tensor) -> List[torch.Tensor]:
    """Denoise a batch of images.

        Args:
          images: A tensor containing a batch of images to denoise.

        Returns:
          A list of tensors containing a batch of denoised images.
        """
    return self.diffuser.denoise_batch(images=images, model=self)

  @abc.abstractmethod
  def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Forward pass of the diffusion model.

    The forward pass of the diffusion model, predicting the noise at a single
    step.

    Args:
      x: A batch of noisy images.
      timestep: The timesteps of each image in the batch.

    Returns:
      A tensor representing the noise predicted for each image.
    """
    raise NotImplementedError

  def to(self, device: str = "cpu") -> "BaseDiffusionModel":
    """Moves the model to the specified device.

    This performs a similar behaviour to the `to` method of PyTorch. moving the
    DiffusionModel and all related artifacts to the specified device.

    Args:
      device: The device to which the method should move the object.
        Default is "cpu".
    """
    new_self = super(BaseDiffusionModel, self).to(device)
    new_self.diffuser = new_self.diffuser.to(device)
    return new_self

  def compile(self, *args, **kwargs):
    """Compiles the diffusion model.

    This performs a similar behaviour to the `compile` method of PyTorch.

    Returns:
      A compiled diffusion model.
    """
    model = torch.compile(self, *args, **kwargs)
    model.to = self.to
    return model
