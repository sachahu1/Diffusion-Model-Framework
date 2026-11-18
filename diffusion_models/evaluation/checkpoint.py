"""Score a saved training checkpoint with FID / KID."""

import pathlib
from typing import Dict
from typing import Optional
from typing import Union

import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf

from diffusion_models.evaluation.fid import ImageQualityEvaluator
from diffusion_models.utils.schemas import Checkpoint


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
) -> Dict[str, float]:
  """Compute FID/KID of a checkpoint produced by ``diffusion-models train``.

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
    seed: The seed of the sampling noise.
    num_workers: Dataloader workers used to read the real images.

  Returns:
    The ``FID`` (and ``KID``) scores.
  """
  checkpoint = Checkpoint.from_file(str(checkpoint_path), map_location=device)
  if checkpoint.config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt."
    )
  configuration = OmegaConf.create(checkpoint.config_yaml)

  model = instantiate(configuration.model)
  model.load_state_dict(checkpoint.get_model_state_dict(use_ema=use_ema))
  model = model.to(device).eval()
  if steps is not None:
    if not hasattr(model.diffuser, "number_of_steps"):
      raise ValueError("This diffuser has no configurable number of steps.")
    model.diffuser.number_of_steps = steps

  if configuration.get("eval_dataset") is not None:  # ty: ignore[invalid-argument-type]
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
  with torch.no_grad():
    return evaluator.evaluate(model)
