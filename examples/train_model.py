"""Running a config-driven training script.

This example configures a training run with `hydra <https://hydra.cc>`_ and
`hydra-zen <https://mit-ll-responsible-ai.github.io/hydra-zen>`_. Every run
gets its own directory (``runs/<name>/<timestamp>``) holding:

- ``.hydra/``: the resolved config and the command line overrides,
- ``tensorboard/``: metrics, images, weight histograms and the config as text,
- ``checkpoints/``: checkpoints which embed the resolved config as YAML,
- ``run_info.json``: the git commit and seed.

The standard config is ``examples/config/baseline.yaml`` and the overridable
config groups are registered in ``diffusion_models/_store``. Install the extra dependencies with ``uv sync --group training``, then run::

  python -m examples.train_model
  python -m examples.train_model training.batch_size=64 optimizer=adopt
  python -m examples.train_model diffuser=ddpm beta_scheduler=cosine
  python -m examples.train_model dataset=fake device=cpu  # smoke test
  python -m examples.train_model --multirun optimizer.lr=1e-4,2e-4

.. literalinclude:: /../../examples/train_model.py
   :language: python
   :linenos:
   :lines: 26-
"""

__all__ = []

import json
import subprocess

import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, zen
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torchvision import datasets
from torchvision.transforms import v2

from diffusion_models.diffusion_trainer import DiffusionTrainer
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)
from diffusion_models import _store  # noqa: F401  (registers the config groups)


# --- Task -------------------------------------------------------------------


def _git_info() -> dict:
  def run(*args: str) -> str:
    return subprocess.run(
      ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()

  return {
    "commit": run("rev-parse", "HEAD"),
    "dirty": bool(run("status", "-s")),
  }


def train(
  model: SimpleUnet,
  dataset: datasets.VisionDataset,
  optimizer,
  training: TrainingConfiguration,
  logging: LogConfiguration,
  seed: int,
  device: str,
  reverse_image_size: int,
) -> None:
  """Instantiated by hydra-zen from the config, then run."""
  # The raw (uninstantiated) config is the source of truth for the run.
  run_config = OmegaConf.to_yaml(_RAW_CONFIG, resolve=True)
  output_directory = HydraConfig.get().runtime.output_dir

  torch.manual_seed(seed)
  with open(f"{output_directory}/run_info.json", "w") as file:
    json.dump({"seed": seed, "git": _git_info()}, file, indent=2)

  print("Num params: ", sum(p.numel() for p in model.parameters()))

  trainer = DiffusionTrainer(
    model=model,
    dataset=dataset,
    optimizer=optimizer(model.parameters()),
    reverse_transforms=v2.Compose(
      [
        v2.Normalize(mean=[-1.0] * 3, std=[2.0] * 3),
        v2.Resize((reverse_image_size, reverse_image_size)),
      ]
    ),
    training_configuration=training,
    loss_function=F.mse_loss,
    log_configuration=logging,
    device=device,
    output_directory=output_directory,
    run_config=run_config,
  )
  trainer.train()


_RAW_CONFIG: DictConfig


def _capture_config(cfg: DictConfig) -> None:
  """Keeps the raw config around; hydra-zen only hands over instantiated fields."""
  global _RAW_CONFIG
  _RAW_CONFIG = cfg


if __name__ == "__main__":
  store.add_to_hydra_store()
  zen(train, pre_call=_capture_config).hydra_main(
    config_name="baseline", config_path="config", version_base="1.3"
  )
