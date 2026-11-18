"""Config-driven inference, run through the ``diffusion-models infer`` CLI.

Rebuilds a model from a hydra config, loads the weights of a training
checkpoint and generates images. The config is composed like for training
(``diffusion_models/conf/baseline.yaml``, extended by ``conf/infer.yaml``), so
use the same overrides as the run that produced the checkpoint. Install the
extra dependencies with ``uv sync --group training``, then run::

  diffusion-models infer checkpoint=runs/<name>/<timestamp>/checkpoints/latest.pt \\
    +config=small_attention_ema
  diffusion-models infer checkpoint=<path> +config=small_attention_ema \\
    number_of_images=16 save_gif=true steps=50 device=cpu

The images (and the optional GIF) are written to ``outputs/infer/<timestamp>``.
To extend this further, you might want to look into compiling your model to
ONNX, TensorRT, OpenVino or other formats.
"""

__all__ = []

import pathlib
from typing import Optional

import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, zen
from torchvision.transforms import v2

from diffusion_models import _store  # noqa: F401  (registers the config groups)
from diffusion_models.diffusion_inference import DiffusionInference
from diffusion_models.models.base_diffusion_model import BaseDiffusionModel
from diffusion_models.utils.schemas import Checkpoint


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
) -> None:
  """Instantiated by hydra-zen from the config, then run."""
  output_directory = pathlib.Path(HydraConfig.get().runtime.output_dir)

  weights = Checkpoint.from_file(checkpoint, map_location=device)
  model.load_state_dict(weights.get_model_state_dict(use_ema=use_ema))
  model = model.to(device).eval()
  if steps is not None:
    if not hasattr(model.diffuser, "number_of_steps"):
      raise ValueError("This diffuser has no configurable number of steps.")
    model.diffuser.number_of_steps = steps

  inference = DiffusionInference(
    model=model,
    image_shape=(image_channels, image_size),
    device=device,
    reverse_transforms=v2.Compose(
      [
        v2.Normalize(mean=[-1.0] * image_channels, std=[2.0] * image_channels),
        v2.Resize((reverse_image_size, reverse_image_size)),
      ]
    ),
  )
  with torch.no_grad():
    if save_gif:
      gif_path = output_directory / "generated.gif"
      inference.generate(number_of_images, save_gif=True, gif_path=str(gif_path))
      print(f"Saved {gif_path}")
    else:
      image_path = output_directory / "generated.png"
      inference.generate(number_of_images).save(image_path)
      print(f"Saved {image_path}")


def main() -> None:
  """Run the inference with hydra, reading overrides from ``sys.argv``."""
  store.add_to_hydra_store()
  zen(infer).hydra_main(
    config_name="infer",
    config_path="pkg://diffusion_models.conf",
    version_base="1.3",
  )
