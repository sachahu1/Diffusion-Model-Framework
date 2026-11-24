"""Train an autoencoder with hydra-zen, e.g. the VAE of a latent diffusion model.

Run it through the command line interface::

  diffusion-models train autoencoder
  diffusion-models train autoencoder +config=vae_f8 training.batch_size=16
  diffusion-models train autoencoder resume_from=runs/<name>/<timestamp>/checkpoints/latest.pt

A second stage fine-tunes the decoder of a trained autoencoder against a
PatchGAN discriminator to sharpen its reconstructions::

  diffusion-models train autoencoder +config=vae_f8_gan \
    autoencoder.checkpoint=runs/vae_f8/<timestamp>/checkpoints/latest.pt
"""

__all__ = []

import json
from typing import Optional

import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, zen
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets
from torchvision.transforms import v2

from diffusion_models import _store  # noqa: F401  (registers the config groups)
from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.autoencoders.adversarial_trainer import (
  AdversarialAutoencoderTrainer,
)
from diffusion_models.autoencoders.schemas import (
  AdversarialConfiguration,
  ValidationConfiguration,
)
from diffusion_models.autoencoders.trainer import AutoencoderTrainer
from diffusion_models.diffusion.training import _git_info
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)


def train_autoencoder(
  autoencoder: BaseAutoencoder,
  dataset: datasets.VisionDataset,
  optimizer,
  training: TrainingConfiguration,
  loss,
  logging: LogConfiguration,
  validation: ValidationConfiguration,
  seed: int,
  device: str,
  resume_from: Optional[str] = None,
  eval_dataset: Optional[datasets.VisionDataset] = None,
  discriminator=None,
  discriminator_optimizer=None,
  adversarial: Optional[AdversarialConfiguration] = None,
) -> None:
  """Instantiated by hydra-zen from the config, then run.

  With a ``discriminator`` the autoencoder is fine-tuned adversarially.
  """
  run_config = OmegaConf.to_yaml(_RAW_CONFIG, resolve=True)
  output_directory = HydraConfig.get().runtime.output_dir

  torch.manual_seed(seed)
  with open(f"{output_directory}/run_info.json", "w") as file:
    json.dump({"seed": seed, "git": _git_info()}, file, indent=2)

  print("Num params: ", sum(p.numel() for p in autoencoder.parameters()))

  common = dict(
    dataset=dataset,
    training_configuration=training,
    loss=loss,
    log_configuration=logging,
    validation_configuration=validation,
    validation_dataset=eval_dataset,
    # Normalize(0.5, 0.5) maps [0, 1] -> [-1, 1]; this undoes it.
    reverse_transforms=v2.Normalize(mean=[-1.0] * 3, std=[2.0] * 3),
    device=device,
    output_directory=output_directory,
    run_config=run_config,
    resume_from=resume_from,
  )
  if discriminator is None:
    trainer = AutoencoderTrainer(
      model=autoencoder,
      optimizer=optimizer(autoencoder.parameters()),
      **common,  # ty: ignore[invalid-argument-type]
    )
  else:
    if discriminator_optimizer is None:
      raise ValueError("An adversarial run needs a discriminator_optimizer.")
    adversarial = adversarial or AdversarialConfiguration()
    trained = (
      autoencoder.decoder_parameters()
      if adversarial.trainable == "decoder"
      else autoencoder.parameters()
    )
    trainer = AdversarialAutoencoderTrainer(
      model=autoencoder,
      optimizer=optimizer(trained),
      discriminator=discriminator,
      discriminator_optimizer=discriminator_optimizer(
        discriminator.parameters()
      ),
      adversarial_configuration=adversarial,
      **common,  # ty: ignore[invalid-argument-type]
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
  zen(train_autoencoder, pre_call=_capture_config).hydra_main(
    config_name="autoencoder_baseline",
    config_path="pkg://diffusion_models.conf",
    version_base="1.3",
  )
