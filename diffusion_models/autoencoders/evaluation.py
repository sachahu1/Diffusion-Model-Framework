"""Score the reconstructions of a saved autoencoder checkpoint with FID / KID.

The *reconstruction FID* is the best FID a latent diffusion model built on the
autoencoder could reach: the generative model cannot beat the quality of the
images its decoder can produce.
"""

import pathlib
from typing import Dict, Union

import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf

from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint
from diffusion_models.core.evaluation.fid import ImageQualityEvaluator


def evaluate_autoencoder_checkpoint(
  checkpoint_path: Union[str, pathlib.Path],
  number_of_samples: int = 10000,
  batch_size: int = 50,
  device: str = "cuda",
  use_ema: bool = True,
  compute_kid: bool = True,
  kid_subset_size: int = 1000,
  num_workers: int = 4,
) -> Dict[str, float]:
  """Compute the reconstruction FID/KID of a ``train autoencoder`` checkpoint.

  The autoencoder and the reference images (the config's ``eval_dataset``) are
  rebuilt from the config stored in the checkpoint. Each reference image is
  encoded (posterior mean) and decoded, and the reconstructions are compared
  with the reference images.

  Args:
    checkpoint_path: The checkpoint to evaluate.
    number_of_samples: The number of images compared.
    batch_size: The batch size.
    device: The device to use.
    use_ema: Prefer the EMA weights when the checkpoint has them.
    compute_kid: Whether to also compute KID.
    kid_subset_size: The KID subset size.
    num_workers: Dataloader workers used to read the images.

  Returns:
    The ``FID`` (and ``KID``) scores.
  """
  checkpoint = AutoencoderCheckpoint.from_file(
    checkpoint_path, map_location=device
  )
  if checkpoint.config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt."
    )
  configuration = OmegaConf.create(checkpoint.config_yaml)
  if (
    configuration.get("eval_dataset") is None  # ty: ignore[invalid-argument-type]
  ):
    raise ValueError("The checkpoint's config has no eval_dataset.")

  model = instantiate(configuration.autoencoder)
  model.load_state_dict(checkpoint.get_model_state_dict(use_ema=use_ema))
  model = model.to(device).eval()
  dataset = instantiate(configuration.eval_dataset)

  evaluator = ImageQualityEvaluator(
    real_dataset=dataset,
    number_of_samples=number_of_samples,
    batch_size=batch_size,
    device=device,
    compute_kid=compute_kid,
    kid_subset_size=kid_subset_size,
    num_workers=num_workers,
  )

  def reconstruct(start: int, count: int, generator: torch.Generator):
    images = torch.stack([dataset[i][0] for i in range(start, start + count)])
    return model(images.to(device), sample_posterior=False).reconstruction

  with torch.no_grad():
    return evaluator.evaluate_generator(reconstruct)
