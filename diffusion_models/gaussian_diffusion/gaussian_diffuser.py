from typing import TYPE_CHECKING, List, Tuple

import torch
from tqdm import tqdm

from diffusion_models.gaussian_diffusion.base_diffuser import BaseDiffuser
from diffusion_models.gaussian_diffusion.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.utils.schemas import Checkpoint, Timestep

if TYPE_CHECKING:
  from diffusion_models.models.base_diffusion_model import BaseDiffusionModel


class GaussianDiffuser(BaseDiffuser):
  def __init__(self, beta_scheduler: BaseBetaScheduler):
    """Initializes the class instance.

    Args:
      beta_scheduler (BaseBetaScheduler): The beta scheduler instance to be used.

    """
    super().__init__(beta_scheduler)
    self.device: str = "cpu"
    """The device to use. Defaults to cpu."""

  @property
  def steps(self) -> List[int]:
    return list(range(self.beta_scheduler.steps))[::-1]

  def get_timestep(self, number_of_images: int, idx: int):
    timestep = torch.full((number_of_images,), idx, device=self.device)
    return Timestep(
      current=timestep,
    )

  @classmethod
  def from_checkpoint(cls, checkpoint: Checkpoint) -> "GaussianDiffuser":
    """Instantiate a Gaussian Diffuser from a training checkpoint.

    Args:
      checkpoint: The training checkpoint object containing
        the trained model's parameters and configuration.

    Returns:
      An instance of the GaussianDiffuser class initialized with the parameters
      loaded from the given checkpoint.
    """
    return cls(
      beta_scheduler=BaseBetaScheduler.from_tensors(
        steps=checkpoint.beta_scheduler_config.steps,
        betas=checkpoint.beta_scheduler_config.betas,
        alpha_bars=checkpoint.beta_scheduler_config.alpha_bars,
      )
    )

  def to(self, device: str = "cpu"):
    """Moves the data to the specified device.

    This performs a similar behaviour to the `to` method of PyTorch. moving the
    GaussianDiffuser and the BetaScheduler to the specified device.

    Args:
      device: The device to which the method should move the object.
        Default is "cpu".

    Example:
      >>> gaussian_diffuser = GaussianDiffuser()
      >>> gaussian_diffuser = gaussian_diffuser.to(device="cuda")
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
    timesteps = torch.randint(
      0, self.beta_scheduler.steps, (images.shape[0],), device=self.device
    )
    return self._diffuse_batch(images, timesteps)

  @torch.no_grad()
  def _denoise_step(
    self, images: torch.Tensor, model: torch.nn.Module, timestep: Timestep
  ) -> torch.Tensor:
    current_timestep = timestep.current
    beta_t = self.beta_scheduler.betas[current_timestep].reshape(-1, 1, 1, 1)
    alpha_t = 1 - beta_t
    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(
      dim=0, index=current_timestep
    ).reshape(-1, 1, 1, 1)
    mu = (1 / torch.sqrt(alpha_t)) * (
      images
      - model(images, current_timestep)
      * (beta_t / torch.sqrt(1 - alpha_bar_t))
    )

    if current_timestep[0] == 0:
      return mu
    else:
      sigma = torch.sqrt(beta_t) * torch.randn_like(images)
      return mu + sigma

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
    for i in tqdm(self.steps, desc="Denoising"):
      timestep = self.get_timestep(images.shape[0], idx=i)
      images = self._denoise_step(images, model=model, timestep=timestep)
      images = torch.clamp(images, -1.0, 1.0)
      denoised_images.append(images)
    return denoised_images
