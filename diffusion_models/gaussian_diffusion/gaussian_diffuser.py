from typing import List

import torch
from tqdm import tqdm

from diffusion_models.gaussian_diffusion.base_diffuser import BaseDiffuser
from diffusion_models.gaussian_diffusion.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.utils.schemas import Checkpoint


class GaussianDiffuser(BaseDiffuser):
  def __init__(self, beta_scheduler: BaseBetaScheduler):
    super().__init__(beta_scheduler)
    self.device = "cuda"

  @classmethod
  def from_checkpoint(cls, checkpoint: Checkpoint) -> "GaussianDiffuser":
    return cls(
      beta_scheduler=BaseBetaScheduler.from_tensors(
        steps=checkpoint.beta_scheduler_config.steps,
        betas=checkpoint.beta_scheduler_config.betas,
        alpha_bars=checkpoint.beta_scheduler_config.alpha_bars,
      )
    )

  def to(self, device: str = "cpu"):
    self.device = device
    self.beta_scheduler = self.beta_scheduler.to(self.device)
    return self

  def diffuse_batch(self, images):
    timesteps = torch.randint(
      0, self.beta_scheduler.steps, (images.shape[0],), device=self.device
    )
    noise = torch.randn_like(images, device=self.device)

    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(dim=0, index=timesteps)

    alpha_bar_t = alpha_bar_t.reshape((-1, *((1,) * (len(images.shape) - 1))))

    mu = torch.sqrt(alpha_bar_t)
    sigma = torch.sqrt(1 - alpha_bar_t)
    images = mu * images + sigma * noise
    return images, noise, timesteps

  @torch.no_grad()
  def _denoise_step(
    self, images: torch.Tensor, model: torch.nn.Module, timestep: torch.Tensor
  ) -> torch.Tensor:
    beta_t = self.beta_scheduler.betas[timestep].reshape(-1, 1, 1, 1)
    alpha_t = 1 - beta_t
    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(
      dim=0, index=timestep
    ).reshape(-1, 1, 1, 1)
    mu = (1 / torch.sqrt(alpha_t)) * (
      images - model(images, timestep) * (beta_t / torch.sqrt(1 - alpha_bar_t))
    )

    if timestep[0] == 0:
      return mu
    else:
      sigma = torch.sqrt(beta_t) * torch.randn_like(images)
      return mu + sigma

  def denoise_batch(
    self,
    images: torch.Tensor,
    model: torch.nn.Module,
  ) -> List[torch.Tensor]:
    denoised_images = []
    for i in tqdm(range(self.beta_scheduler.steps)[::-1], desc="Denoising"):
      timestep = torch.full((images.shape[0],), i, device=self.device)
      images = self._denoise_step(images, model=model, timestep=timestep)
      images = torch.clamp(images, -1.0, 1.0)
      denoised_images.append(images)
    return denoised_images
