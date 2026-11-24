import logging
import pathlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.autoencoders.schemas import (
  AdversarialConfiguration,
  ValidationConfiguration,
)
from diffusion_models.autoencoders.trainer import AutoencoderTrainer
from diffusion_models.core.losses import (
  LossOutput,
  adaptive_weight,
  hinge_discriminator_loss,
  hinge_generator_loss,
)
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)


class AdversarialAutoencoderTrainer(AutoencoderTrainer):
  def __init__(
    self,
    model: BaseAutoencoder,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
    discriminator: nn.Module,
    discriminator_optimizer: torch.optim.Optimizer,
    adversarial_configuration: AdversarialConfiguration = AdversarialConfiguration(),
    loss: Optional[Callable[..., LossOutput]] = None,
    log_configuration: LogConfiguration = LogConfiguration(),
    validation_configuration: ValidationConfiguration = ValidationConfiguration(),
    validation_dataset: Optional[Dataset] = None,
    reverse_transforms: Callable = lambda x: x,
    device: str = "cuda",
    output_directory: Optional[Union[str, pathlib.Path]] = None,
    run_config: Optional[str] = None,
    resume_from: Optional[Union[str, pathlib.Path]] = None,
  ):
    """Fine-tunes an autoencoder against a PatchGAN discriminator.

    Each step reconstructs a batch once, then updates

    1. the autoencoder with ``reconstruction + KL + w * adversarial``, where the
       discriminator is frozen and ``w`` ramps up and (optionally) adapts to the
       gradients at the decoder's last layer; then
    2. the discriminator with the hinge loss on real images and (detached)
       reconstructions.

    By default only the decoder is trained and the encoder is frozen, so the
    latents of the stage the autoencoder was loaded from (see
    :meth:`~.BaseAutoencoder.load`) remain valid. The EMA tracks the
    autoencoder only. Mixed precision supports bfloat16 (or none).

    Args:
      model: The autoencoder, typically loaded from an earlier stage.
      dataset: The training images.
      optimizer: The autoencoder's optimizer. For a decoder-only fine-tune it
        should hold ``model.decoder_parameters()``.
      training_configuration: The training configuration.
      discriminator: A network mapping images to patch logits.
      discriminator_optimizer: The discriminator's optimizer.
      adversarial_configuration: The adversarial settings.
      loss: The autoencoder's reconstruction objective (see
        :class:`~.AutoencoderTrainer`).
      log_configuration: See :class:`~.AutoencoderTrainer`.
      validation_configuration: See :class:`~.AutoencoderTrainer`.
      validation_dataset: See :class:`~.AutoencoderTrainer`.
      reverse_transforms: See :class:`~.AutoencoderTrainer`.
      device: The device.
      output_directory: See :class:`~.AutoencoderTrainer`.
      run_config: See :class:`~.AutoencoderTrainer`.
      resume_from: A checkpoint of this stage to resume from: it restores the
        discriminator and its optimizer too.
    """
    if adversarial_configuration.trainable not in ("decoder", "all"):
      raise ValueError(
        "trainable must be 'decoder' or 'all', got "
        f"{adversarial_configuration.trainable!r}."
      )
    if (
      training_configuration.mixed_precision_training
      and training_configuration.mixed_precision_dtype == "float16"
    ):
      raise ValueError(
        "Adversarial training supports bfloat16 or no mixed precision, not float16."
      )
    # The discriminator is needed to resume, so resume after it exists.
    super().__init__(
      model=model,
      dataset=dataset,
      optimizer=optimizer,
      training_configuration=training_configuration,
      loss=loss,
      log_configuration=log_configuration,
      validation_configuration=validation_configuration,
      validation_dataset=validation_dataset,
      reverse_transforms=reverse_transforms,
      device=device,
      output_directory=output_directory,
      run_config=run_config,
      resume_from=None,
    )
    if (
      adversarial_configuration.latent_noise > 0
      and adversarial_configuration.trainable != "decoder"
    ):
      raise ValueError("latent_noise needs trainable='decoder'.")
    self.adversarial = adversarial_configuration
    """The adversarial settings."""
    self.discriminator = discriminator.to(device)
    """The discriminator."""
    self.discriminator_optimizer = discriminator_optimizer
    """The discriminator's optimizer."""
    if training_configuration.channels_last:
      self.discriminator.to(memory_format=torch.channels_last)  # ty: ignore[no-matching-overload]

    if adversarial_configuration.trainable == "decoder":
      self.model.requires_grad_(False)
      for parameter in self.model.decoder_parameters():
        parameter.requires_grad_(True)

    self._encode_distribution: Callable = self.model.encode_distribution
    self._decode: Callable = self.model.decode
    self._discriminate: Callable = self.discriminator
    if training_configuration.compile:
      self._encode_distribution = torch.compile(self.model.encode_distribution)
      self._decode = torch.compile(self.model.decode)
      self._discriminate = torch.compile(self.discriminator)

    if resume_from is not None:
      self.load_checkpoint(resume_from)

  def checkpoint_fields(self) -> Dict[str, Any]:
    return {
      **super().checkpoint_fields(),
      "discriminator_state_dict": self.discriminator.state_dict(),
      "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
    }

  def load_extra_state(self, checkpoint: Any) -> None:
    if checkpoint.discriminator_state_dict is None:
      logging.warning(
        "Checkpoint has no discriminator; it starts from scratch."
      )
      return
    self.discriminator.load_state_dict(checkpoint.discriminator_state_dict)
    if checkpoint.discriminator_optimizer_state_dict is not None:
      self.discriminator_optimizer.load_state_dict(
        checkpoint.discriminator_optimizer_state_dict
      )

  def _generate(
    self, images: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Reconstruct ``images``; the encoder is frozen for a decoder-only fine-tune."""
    if self.adversarial.trainable == "all":
      output = self._train_model(images)
      return output.reconstruction, output.mean, output.logvar
    with torch.no_grad():
      mean, logvar = self._encode_distribution(images)
    latents = mean
    if logvar is not None:
      latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)  # ty: ignore[invalid-argument-type]
    if self.adversarial.latent_noise > 0:
      channel_std = latents.float().std(dim=(0, 2, 3), keepdim=True)
      level = torch.rand((latents.shape[0], 1, 1, 1), device=latents.device)
      latents = latents + (
        self.adversarial.latent_noise
        * level
        * channel_std
        * torch.randn_like(latents)
      ).to(latents.dtype)
    return self._decode(latents), mean, logvar

  def optimization_step(
    self, batch: Any, global_step: int
  ) -> Dict[str, torch.Tensor]:
    configuration = self.training_configuration
    adversarial = self.adversarial
    # The schedule drives the autoencoder's optimizer, not the discriminator's.
    self.apply_learning_rate(global_step)
    images, _ = batch
    images = self.to_device(images)

    # 1. The autoencoder, against a frozen discriminator.
    self.optimizer.zero_grad(set_to_none=True)
    self.discriminator.requires_grad_(False)
    try:
      with self.autocast():
        reconstruction, mean, logvar = self._generate(images)
        reconstruction_loss = self.loss(images, reconstruction, mean, logvar)
        fake_logits = self._discriminate(reconstruction)
      adversarial_loss = hinge_generator_loss(fake_logits.float())
      since_start = global_step - adversarial.start_step + 1
      if since_start <= 0:
        ramp = 0.0
      elif adversarial.warmup_steps > 0:
        ramp = min(1.0, since_start / adversarial.warmup_steps)
      else:
        ramp = 1.0
      balance = (
        adaptive_weight(
          reconstruction_loss.loss,
          adversarial_loss,
          self.model.last_layer(),
          max_weight=adversarial.max_adaptive_weight,
        )
        if adversarial.adaptive
        else torch.ones((), device=images.device)
      )
      weight = ramp * adversarial.weight * balance
      generator_loss = reconstruction_loss.loss + weight * adversarial_loss
      self.scaler.scale(generator_loss).backward()
    finally:
      self.discriminator.requires_grad_(True)

    metrics = dict(reconstruction_loss.metrics)
    if configuration.gradient_clip is not None:
      self.scaler.unscale_(self.optimizer)
      metrics["GradNorm"] = torch.nn.utils.clip_grad_norm_(
        [p for p in self.model.parameters() if p.requires_grad],
        max_norm=configuration.gradient_clip,
      )
    self.scaler.step(self.optimizer)
    self.scaler.update()
    if self.ema is not None:
      self.ema.update(self.model)

    # 2. The discriminator, on real images and detached reconstructions.
    self.discriminator_optimizer.zero_grad(set_to_none=True)
    with self.autocast():
      real_logits = self._discriminate(images)
      fake_logits = self._discriminate(reconstruction.detach())
    discriminator_loss = hinge_discriminator_loss(
      real_logits.float(), fake_logits.float()
    )
    discriminator_loss.backward()
    self.discriminator_optimizer.step()

    mean = mean.detach().float()
    metrics.update(
      {
        "LatentMean": mean.mean(),
        "LatentStd": mean.std(),
        "Adversarial/Generator": adversarial_loss.detach(),
        "Adversarial/Discriminator": discriminator_loss.detach(),
        "Adversarial/RealLogit": real_logits.detach().float().mean(),
        "Adversarial/FakeLogit": fake_logits.detach().float().mean(),
        "Adversarial/Weight": weight.detach(),
      }
    )
    return metrics
