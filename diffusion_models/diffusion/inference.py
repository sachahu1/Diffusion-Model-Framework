"""Config-driven inference, run through the ``diffusion-models infer`` CLI.

With only the simple options (``checkpoint``, ``number_of_images``, ``steps``,
``eta``, ``use_ema``, ``save_gif``, ``device``, ``seed``) everything is rebuilt
from the config stored in the checkpoint. Any other override composes the model
from hydra configs instead, so use the overrides of the training run::

  diffusion-models infer checkpoint=<path> number_of_images=16 steps=50
  diffusion-models infer checkpoint=<path> +config=small_attention_ema save_gif=true

A latent diffusion checkpoint also needs ``autoencoder.checkpoint=<path>``.
Images (and the optional GIF) are written to ``outputs/infer/<timestamp>``.
"""

__all__ = []

import logging
import pathlib
from typing import Optional, Union

import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate, store, zen
from omegaconf import OmegaConf
from torchvision.transforms import v2

from diffusion_models import _store  # noqa: F401  (registers the config groups)
from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.diffusion.diffusion_inference import DiffusionInference
from diffusion_models.latent.normalization import LatentNormalization
from diffusion_models.diffusion.models.base_diffusion_model import (
  BaseDiffusionModel,
)
from diffusion_models.diffusion.schemas import Checkpoint


@torch.no_grad()
def infer(
  model: BaseDiffusionModel,
  checkpoint: str,
  device: str,
  image_size: int,
  image_channels: int,
  reverse_image_size: int,
  number_of_images: int,
  use_ema: bool,
  save_gif: bool,
  steps: Optional[int],
  autoencoder: Optional[BaseAutoencoder] = None,
  eval_image_channels: int = 3,
  output_directory: Optional[pathlib.Path] = None,
) -> None:
  """Instantiated by hydra-zen from the config, then run."""
  if output_directory is None:
    output_directory = pathlib.Path(HydraConfig.get().runtime.output_dir)

  # Skip the optimizer state and keep the file on the CPU.
  weights = Checkpoint.from_file(
    checkpoint, map_location="cpu", load_optimizer=False
  )
  model.load_state_dict(weights.get_model_state_dict(use_ema=use_ema))
  model = model.to(device).eval()
  if steps is not None:
    if not hasattr(model.diffuser, "number_of_steps"):
      raise ValueError("This diffuser has no configurable number of steps.")
    model.diffuser.number_of_steps = steps  # ty: ignore[invalid-assignment]

  # A latent diffusion model samples latents, which the autoencoder decodes.
  decode = None
  output_channels = image_channels
  if weights.latent_scale_factor is not None:
    if autoencoder is None:
      raise ValueError(
        "This is a latent diffusion checkpoint: pass +config=latent_celeba "
        "and autoencoder.checkpoint=<autoencoder checkpoint>."
      )
    autoencoder = autoencoder.to(device).eval()
    normalization = LatentNormalization(
      scale=weights.latent_scale_factor,
      shift=weights.latent_shift,
      whitening=weights.latent_whitening,
    )
    output_channels = eval_image_channels

    def decode(samples: torch.Tensor) -> torch.Tensor:
      return autoencoder.decode(
        normalization.denormalize(samples.float())
      ).clamp(-1, 1)

  inference = DiffusionInference(
    model=model,
    image_shape=(image_channels, image_size),
    device=device,
    decode=decode,
    reverse_transforms=v2.Compose(
      [
        v2.Normalize(
          mean=[-1.0] * output_channels, std=[2.0] * output_channels
        ),
        v2.Resize((reverse_image_size, reverse_image_size)),
      ]
    ),
  )
  if save_gif:
    gif_path = output_directory / "generated.gif"
    inference.generate(number_of_images, save_gif=True, gif_path=str(gif_path))
    print(f"Saved {gif_path}")
  else:
    image_path = output_directory / "generated.png"
    image = inference.generate(number_of_images)
    assert image is not None
    image.save(image_path)
    print(f"Saved {image_path}")


def generate_from_checkpoint(
  checkpoint: Union[str, pathlib.Path],
  output_directory: Union[str, pathlib.Path],
  number_of_images: int = 25,
  steps: Optional[int] = None,
  eta: Optional[float] = None,
  use_ema: bool = True,
  save_gif: bool = False,
  device: str = "cuda",
  seed: Optional[int] = None,
) -> None:
  """Generate images from a checkpoint, rebuilding everything from its config.

  Like ``diffusion-models fid``, this needs no hydra overrides: the model,
  diffuser, latent size and autoencoder are those of the run that produced the
  checkpoint, read from the config stored in it.

  Args:
    checkpoint: A checkpoint saved by ``diffusion-models train diffuser``.
    output_directory: Where ``generated.png`` (or ``generated.gif``) is saved.
    number_of_images: How many images to generate.
    steps: Overrides the number of DDIM sampling steps.
    eta: Overrides the sampling stochasticity (``0`` deterministic).
    use_ema: Use the EMA weights when the checkpoint has them.
    save_gif: Save the denoising process as a GIF instead of a picture.
    device: The device to run on.
    seed: Seeds the sampling noise when given.

  Raises:
    ValueError: If the checkpoint has no embedded config, or the diffuser has
      no ``eta`` to override.
  """
  weights = Checkpoint.from_file(
    str(checkpoint), map_location="cpu", load_optimizer=False
  )
  if weights.config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt: "
      "use `diffusion-models infer +config=<preset> checkpoint=<path>` instead."
    )
  configuration = OmegaConf.create(weights.config_yaml)

  model = instantiate(configuration.model)
  if eta is not None:
    if not hasattr(model.diffuser, "eta"):
      raise ValueError("This diffuser has no configurable eta.")
    model.diffuser.eta = eta
  autoencoder = None
  if weights.latent_scale_factor is not None:
    if (
      configuration.get("autoencoder") is None  # ty: ignore[invalid-argument-type]
    ):
      raise ValueError("A latent checkpoint's config needs an autoencoder.")
    autoencoder = instantiate(configuration.autoencoder)
  if seed is not None:
    torch.manual_seed(seed)

  output_directory = pathlib.Path(output_directory)
  output_directory.mkdir(parents=True, exist_ok=True)
  # What the hydra route records: the options used and an ``inference.log``.
  (output_directory / "options.yaml").write_text(
    OmegaConf.to_yaml(
      OmegaConf.create(
        {
          "checkpoint": str(checkpoint),
          "number_of_images": number_of_images,
          "steps": steps,
          "eta": eta,
          "use_ema": use_ema,
          "save_gif": save_gif,
          "device": device,
          "seed": seed,
        }
      )
    )
  )
  handler = logging.FileHandler(output_directory / "inference.log")
  handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
  )
  root = logging.getLogger()
  previous_level = root.level
  root.addHandler(handler)
  root.setLevel(min(previous_level or logging.INFO, logging.INFO))
  try:
    logging.info(f"Generating {number_of_images} images from {checkpoint}.")
    infer(
      model=model,
      checkpoint=str(checkpoint),
      device=device,
      image_size=configuration.image_size,
      image_channels=configuration.image_channels,
      reverse_image_size=configuration.get(
        "reverse_image_size"  # ty: ignore[invalid-argument-type]
      )
      or configuration.image_size,
      number_of_images=number_of_images,
      use_ema=use_ema,
      save_gif=save_gif,
      steps=steps,
      autoencoder=autoencoder,
      eval_image_channels=configuration.get(
        "eval_image_channels", 3  # ty: ignore[invalid-argument-type]
      ),
      output_directory=output_directory,
    )
    logging.info(f"Saved the images to {output_directory}.")
  finally:
    root.removeHandler(handler)
    root.setLevel(previous_level)
    handler.close()


def main() -> None:
  """Run the inference with hydra, reading overrides from ``sys.argv``."""
  store.add_to_hydra_store()
  zen(infer).hydra_main(
    config_name="infer",
    config_path="pkg://diffusion_models.conf",
    version_base="1.3",
  )
