import logging
from typing import List, Callable, Tuple

import torch
import torchvision.utils
from torchvision import transforms
from tqdm import tqdm

from diffusion_models.gaussian_diffusion.beta_schedulers import \
  BaseBetaScheduler


class GaussianDiffuser:
  def __init__(self, beta_scheduler: BaseBetaScheduler):
    self.beta_scheduler = beta_scheduler
    self.device = "cuda"

  def to(self, device: str):
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
  def _denoise_step(self, images: torch.Tensor, model: torch.nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    beta_t = self.beta_scheduler.betas[timestep].reshape(-1, 1, 1, 1)
    alpha_t = 1 - beta_t
    alpha_bar_t = self.beta_scheduler.alpha_bars.gather(
      dim=0, index=timestep
    ).reshape(-1, 1, 1, 1)
    mu = (1 / torch.sqrt(alpha_t)) * (
      images - model(images, timestep) * (beta_t / torch.sqrt(1 - alpha_bar_t)))

    if timestep[0] == 0:
      return mu
    else:
      sigma = torch.sqrt(beta_t) * torch.randn_like(images)
      return mu + sigma

  def denoise_batch(self,
    images: torch.Tensor,
    model: torch.nn.Module,
  ) -> List[torch.Tensor]:
    denoised_images = []
    for i in tqdm(range(0, self.beta_scheduler.steps)[::-1], desc="Denoising"):
      timestep = torch.full((images.shape[0],), i, device=self.device)
      images = self._denoise_step(images, model=model, timestep=timestep)
      images = torch.clamp(images, -1.0, 1.0)
      denoised_images.append(images)
    return denoised_images

class DiffusionInference:
  def __init__(self,
    gaussian_diffuser: GaussianDiffuser,
    model: torch.nn.Module,
    reverse_transforms: Callable = lambda x: x,
    image_shape: Tuple[int, int] = (3, 64),
    device: str = "cuda",
  ):
    self.image_channels = image_shape[0]
    self.image_size = image_shape[1]

    self.gaussian_diffuser = gaussian_diffuser.to(device)
    self.model = model.to(device)

    self.reverse_transforms = reverse_transforms

    self.device = device

  def _visualise_images(self, denoised_images: torch.Tensor):
    reverse_transformed_images = self.reverse_transforms(denoised_images)
    image_grid = torchvision.utils.make_grid(reverse_transformed_images, nrow=5)
    pil_images = transforms.ToPILImage()(image_grid)
    return pil_images

  def generate(self, number_of_images: int, save_gif: bool = False):
    images = torch.randn(
      (number_of_images, self.image_channels, self.image_size, self.image_size),
      device=self.device
    )

    denoised_images = self.gaussian_diffuser.denoise_batch(images, self.model)

    if save_gif:
      pil_images = []
      for timestep, images in enumerate(denoised_images):
        pil_image_grid = self._visualise_images(images)
        pil_images.append(pil_image_grid)

      images = pil_images[1::2]
      frame_one = pil_images[0]

      logging.info("Saving GIF")
      frame_one.save(
        "generated.gif", format="GIF", append_images=images,
        save_all=True, duration=[10 for i in range(len(images))] + [2000],
        loop=0
      )

    else:
      pil_image_grid = self._visualise_images(denoised_images[-1])
      return pil_image_grid
