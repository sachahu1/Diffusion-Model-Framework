import logging
from typing import Callable, Tuple

import torch
from PIL.Image import Image
from torchvision import transforms
from torchvision.utils import make_grid

from diffusion_models.models.base_diffusion_model import BaseDiffusionModel


class DiffusionInference:
  def __init__(
    self,
    model: BaseDiffusionModel,
    reverse_transforms: Callable = lambda x: x,
    image_shape: Tuple[int, int] = (3, 64),
    device: str = "cuda",
  ):
    """A diffusion inference framework.

    This is a simplified inference framework to easily start using your
    diffusion model.

    Args:
      model: The trained diffusion model.
      reverse_transforms: A set of reverse transforms.
      image_shape: The shape of the image to produce. This is a tuple where the
        first value is the number of channels and the second is the size of the
        image. Images are expected to be square.
      device: The device to run the inference on.
    """
    self.image_channels = image_shape[0]
    """The number of channels of the image."""
    self.image_size = image_shape[1]
    """The size of the image."""

    self.model = model.to(device)
    """The trained diffusion model."""

    self.reverse_transforms = reverse_transforms
    """The set of reverse transforms."""
    self.device = device
    """The device to run the inference on."""

  def _visualise_images(self, denoised_images: torch.Tensor):
    reverse_transformed_images = self.reverse_transforms(denoised_images)
    image_grid = make_grid(reverse_transformed_images, nrow=5)
    pil_images = transforms.ToPILImage()(image_grid)
    return pil_images

  def generate(
    self, number_of_images: int, save_gif: bool = False
  ) -> Image:
    """Generate a batch of images.

    Args:
      number_of_images: The number of images to generate.
      save_gif: Whether to save the generation process as a GIF.

    Returns:
      A PIL image of the generated images stacked together.
    """
    images = torch.randn(
      (
        number_of_images,
        self.image_channels,
        self.image_size,
        self.image_size,
      ),
      device=self.device,
    )

    denoised_images = self.model.denoise(images)

    if save_gif:
      pil_images = []
      for timestep, images in enumerate(denoised_images):
        pil_image_grid = self._visualise_images(images)
        pil_images.append(pil_image_grid)

      images = pil_images[1::2]
      frame_one = pil_images[0]

      logging.info("Saving GIF")
      frame_one.save(
        "generated.gif",
        format="GIF",
        append_images=images,
        save_all=True,
        duration=[10 for i in range(len(images))] + [2000],
        loop=0,
      )

    else:
      pil_image_grid = self._visualise_images(denoised_images[-1])
      return pil_image_grid

  def get_generator(self, number_of_images: int = 1):
    """An image generator.

    This method is a generator that will generate a batch of images. At
    every call, the generator denoises the images by one more step until the
    image is fully generated. This can be particularly useful for running the
    image generation step by step or for a streaming API.

    Args:
      number_of_images: The number of images the generator should generate.
    """
    images = torch.randn(
      (
        number_of_images,
        self.image_channels,
        self.image_size,
        self.image_size,
      ),
      device=self.device,
    )

    for i in range(self.model.diffuser.beta_scheduler.steps)[::-1]:
      timestep = torch.full((images.shape[0],), i, device=self.device)
      images = self.model.diffuser._denoise_step(
        images, model=self.model, timestep=timestep
      )
      images = torch.clamp(images, -1.0, 1.0)

      yield self._visualise_images(images)
