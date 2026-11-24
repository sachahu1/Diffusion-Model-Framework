"""Score a saved training checkpoint with FID / KID."""

import pathlib
from typing import Dict
from typing import Optional
from typing import Union

import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf

from diffusion_models.core.evaluation.fid import ImageQualityEvaluator
from diffusion_models.diffusion.schemas import Checkpoint
from diffusion_models.latent.normalization import LatentNormalization


def evaluate_checkpoint(
  checkpoint_path: Union[str, pathlib.Path],
  number_of_samples: int = 10000,
  batch_size: int = 100,
  device: str = "cuda",
  use_ema: bool = True,
  compute_kid: bool = True,
  kid_subset_size: int = 1000,
  steps: Optional[int] = None,
  seed: int = 0,
  num_workers: int = 4,
  clip: bool = True,
  eta: Optional[float] = None,
) -> Dict[str, float]:
  """Compute FID/KID of a checkpoint produced by ``diffusion-models train diffuser``.

  The model and the reference dataset are rebuilt from the resolved config
  stored in the checkpoint. The reference images are the config's
  ``eval_dataset`` when it has one, otherwise its ``dataset`` with
  ``split="valid"`` when that dataset has a ``split`` option.

  Args:
    checkpoint_path: The checkpoint to evaluate.
    number_of_samples: The number of real and generated images compared.
    batch_size: The batch size used for sampling and feature extraction.
    device: The device to use.
    use_ema: Prefer the EMA weights when the checkpoint has them.
    compute_kid: Whether to also compute KID.
    kid_subset_size: The KID subset size.
    steps: Overrides the number of DDIM sampling steps, when the diffuser has
      that option.
    eta: Overrides the sampling stochasticity of the diffuser (``0`` is
      deterministic DDIM, ``1`` DDPM-like), when the diffuser has that option.
    seed: The seed of the sampling noise.
    num_workers: Dataloader workers used to read the real images.
    clip: Clamp samples to the diffuser's ``clip_value`` after every step.
      Turn it off to score a latent diffusion checkpoint trained before the
      preset disabled it: latents are unbounded and clamping crushes them.

  Returns:
    The ``FID`` (and ``KID``) scores.
  """
  checkpoint = Checkpoint.from_file(
    str(checkpoint_path), map_location="cpu", load_optimizer=False
  )
  if checkpoint.config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt."
    )
  configuration = OmegaConf.create(checkpoint.config_yaml)

  model = instantiate(configuration.model)
  model.load_state_dict(checkpoint.get_model_state_dict(use_ema=use_ema))
  model = model.to(device).eval()
  if not clip:
    model.diffuser.clip_value = None
  if steps is not None:
    if not hasattr(model.diffuser, "number_of_steps"):
      raise ValueError("This diffuser has no configurable number of steps.")
    model.diffuser.number_of_steps = steps
  if eta is not None:
    if not hasattr(model.diffuser, "eta"):
      raise ValueError("This diffuser has no configurable eta.")
    model.diffuser.eta = eta

  if (
    configuration.get("eval_dataset")  # ty: ignore[invalid-argument-type]
    is not None
  ):
    real_configuration = configuration.eval_dataset
  else:
    real_configuration = OmegaConf.create(
      OmegaConf.to_container(configuration.dataset)
    )
    if "split" in real_configuration:
      real_configuration.split = "valid"
  real_dataset = instantiate(real_configuration)

  evaluator = ImageQualityEvaluator(
    real_dataset=real_dataset,
    number_of_samples=number_of_samples,
    batch_size=batch_size,
    device=device,
    compute_kid=compute_kid,
    kid_subset_size=kid_subset_size,
    seed=seed,
    num_workers=num_workers,
  )
  noise_shape = None
  decode = None
  if checkpoint.latent_scale_factor is not None:
    # A latent diffusion model: sample latents and decode them.
    if (
      configuration.get("autoencoder") is None  # ty: ignore[invalid-argument-type]
    ):
      raise ValueError("A latent checkpoint's config needs an autoencoder.")
    autoencoder = instantiate(configuration.autoencoder).to(device).eval()
    normalization = LatentNormalization(
      scale=checkpoint.latent_scale_factor,
      shift=checkpoint.latent_shift,
      whitening=checkpoint.latent_whitening,
    )
    noise_shape = (
      checkpoint.image_channels,
      *[configuration.image_size] * 2,
    )

    def decode(samples: torch.Tensor) -> torch.Tensor:
      return autoencoder.decode(
        normalization.denormalize(samples.float())
      ).clamp(-1, 1)

  with torch.no_grad():
    return evaluator.evaluate(model, noise_shape=noise_shape, decode=decode)
