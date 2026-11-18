"""Config-driven training, run through the ``diffusion-models train`` CLI.

A run is configured with `hydra <https://hydra.cc>`_ and
`hydra-zen <https://mit-ll-responsible-ai.github.io/hydra-zen>`_. Every run
gets its own directory (``runs/<name>/<timestamp>``) holding:

- ``.hydra/``: the resolved config and the command line overrides,
- ``tensorboard/``: metrics, images, weight histograms and the config as text,
- ``checkpoints/``: checkpoints which embed the resolved config as YAML,
- ``run_info.json``: the git commit and seed.

The standard config is ``diffusion_models/conf/baseline.yaml``, presets live in
``diffusion_models/conf/config/`` and the overridable config groups are
registered in ``diffusion_models/_store``. Install the extra dependencies with
``pip install diffusion-model-framework[training]``, then run::

  diffusion-models train
  diffusion-models train training.batch_size=64 optimizer=adopt
  diffusion-models train +config=small_attention_ema
  diffusion-models train diffuser=ddpm beta_scheduler=cosine
  diffusion-models train dataset=fake device=cpu  # smoke test
  diffusion-models train resume_from=runs/<name>/<timestamp>/checkpoints/latest.pt
  diffusion-models train --multirun optimizer.lr=1e-4,2e-4
"""

__all__ = []

import json
import subprocess
from typing import Optional

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
  EvaluationConfiguration,
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
  resume_from: Optional[str] = None,
  evaluation: Optional[EvaluationConfiguration] = None,
  eval_dataset: Optional[datasets.VisionDataset] = None,
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
    resume_from=resume_from,
    evaluation_configuration=evaluation,
    evaluation_dataset=eval_dataset,
  )
  trainer.train()


_RAW_CONFIG: DictConfig


def _capture_config(cfg: DictConfig) -> None:
  """Keeps the raw config around; hydra-zen only hands over instantiated fields."""
  global _RAW_CONFIG
  _RAW_CONFIG = cfg


def main() -> None:
  """Run the training with hydra, reading overrides from ``sys.argv``."""
  store.add_to_hydra_store()
  zen(train, pre_call=_capture_config).hydra_main(
    config_name="baseline",
    config_path="pkg://diffusion_models.conf",
    version_base="1.3",
  )
