import logging
import pathlib
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid

from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint
from diffusion_models.autoencoders.schemas import ValidationConfiguration
from diffusion_models.core.losses import LossOutput, VaeLoss
from diffusion_models.core.trainer import BaseTrainer
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)


class AutoencoderTrainer(BaseTrainer):
  checkpoint_class = AutoencoderCheckpoint
  metric_prefix = "Train/"

  def __init__(
    self,
    model: BaseAutoencoder,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
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
    """A trainer for autoencoders (e.g. the VAE of a latent diffusion model).

    The training loop, mixed precision, EMA, checkpoints and logging are
    inherited from :class:`~.BaseTrainer`; this class adds the reconstruction
    objective, a validation pass and reconstruction grids.

    Args:
      model: The autoencoder to train.
      dataset: The training images; items are ``(image, target)``.
      optimizer: The optimizer.
      training_configuration: The training configuration.
      loss: ``loss(images, reconstruction, mean, logvar) -> LossOutput``.
        Defaults to :class:`~diffusion_models.core.losses.VaeLoss`.
      log_configuration: ``log_rate`` and ``image_rate`` are in steps;
        ``number_of_images`` is the number of reconstructions shown.
      validation_configuration: When and on how many images to validate.
      validation_dataset: Held-out images for validation and for the
        reconstruction grid. Without it, there is no validation and the grid
        uses training images.
      reverse_transforms: Maps model-range images to ``[0, 1]`` for display.
      device: The device.
      output_directory: Where this run writes its checkpoints (under
        ``checkpoints``) and tensorboard logs (under ``tensorboard``).
      run_config: The resolved configuration (YAML), logged to tensorboard and
        stored in every checkpoint.
      resume_from: A checkpoint to resume from.
    """
    super().__init__(
      model=model,
      dataset=dataset,
      optimizer=optimizer,
      training_configuration=training_configuration,
      log_configuration=log_configuration,
      device=device,
      output_directory=output_directory,
      run_config=run_config,
      resume_from=resume_from,
    )
    self.model: BaseAutoencoder
    self.loss = loss if loss is not None else VaeLoss()
    if isinstance(self.loss, torch.nn.Module):
      self.loss.to(device)
    self.validation_configuration = validation_configuration
    self.reverse_transforms = reverse_transforms
    self._image_shape = dataset[0][0].shape

    self._validation_dataset = validation_dataset
    self.evaluator = None
    """The reconstruction FID evaluator, when ``fid_rate`` is configured."""
    if validation_configuration.fid_rate is not None:
      if validation_dataset is None:
        raise ValueError("The reconstruction FID needs a validation_dataset.")
      # Imported lazily: torchmetrics is an optional dependency.
      from diffusion_models.core.evaluation.fid import ImageQualityEvaluator

      self.evaluator = ImageQualityEvaluator(
        real_dataset=validation_dataset,
        number_of_samples=validation_configuration.fid_samples,
        batch_size=validation_configuration.batch_size,
        device=device,
        compute_kid=False,
        num_workers=validation_configuration.num_workers,
      )

    self.validation_dataloader: Optional[DataLoader] = None
    display_dataset = dataset
    if validation_dataset is not None:
      size = len(validation_dataset)  # ty: ignore[invalid-argument-type]
      count = min(validation_configuration.number_of_samples, size)
      stride = max(1, size // count)
      subset = Subset(validation_dataset, list(range(0, size, stride))[:count])
      self.validation_dataloader = self._build_dataloader(
        subset,
        batch_size=validation_configuration.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=validation_configuration.num_workers,
      )
      display_dataset = subset
    # A fixed set of images, so reconstructions are comparable across epochs.
    number_of_images = min(
      log_configuration.number_of_images,
      len(display_dataset),  # ty: ignore[invalid-argument-type]
    )
    self._display_images = (
      torch.stack(
        [
          display_dataset[i][0] for i in range(number_of_images)
        ]
      ).to(device)
      if number_of_images > 0
      else None
    )

  def training_step(self, batch: Any) -> LossOutput:
    images, _ = batch
    images = self.to_device(images)
    output = self._train_model(images)
    loss = self.loss(images, output.reconstruction, output.mean, output.logvar)
    mean = output.mean.detach().float()
    return LossOutput(
      loss=loss.loss,
      metrics={
        **loss.metrics,
        "LatentMean": mean.mean(),
        "LatentStd": mean.std(),
      },
    )

  def checkpoint_fields(self) -> Dict[str, Any]:
    return dict(image_channels=self._image_shape[0])

  def log_images(self, global_step: int) -> None:
    """Log originals (top row) against reconstructions (bottom row)."""
    if self._display_images is None:
      return
    model = self.ema.model if self.ema is not None else self.model
    reconstruction = model(self._display_images, sample_posterior=False)[0]
    images = torch.cat([self._display_images, reconstruction.float()])
    grid = make_grid(
      self.reverse_transforms(images).clamp(0.0, 1.0),
      nrow=len(self._display_images),
    )
    self.tensorboard_manager.summary_writer.add_image(
      "Reconstructions (top: original)", grid, global_step=global_step
    )

  @torch.no_grad()
  def on_epoch_end(self) -> None:
    self._maybe_validate()
    self._maybe_evaluate_fid()

  def _maybe_validate(self) -> None:
    rate = self.validation_configuration.rate
    if (
      self.validation_dataloader is None
      or rate is None
      or self._completed_epochs % rate != 0
    ):
      return
    metrics = self.validate()
    self.tensorboard_manager.log_metrics(
      {f"Val/{name}": value for name, value in metrics.items()},
      global_step=self._completed_epochs * len(self.dataloader),
    )
    logging.info(f"Epoch {self._completed_epochs}: {metrics}")

  def _maybe_evaluate_fid(self) -> None:
    """Log the FID of the reconstructions of validation images, when due.

    PSNR rewards blurry averages, so it is a poor guide once an adversarial
    term is used; FID compares the distribution of reconstructions with the
    real images instead.
    """
    rate = self.validation_configuration.fid_rate
    if self.evaluator is None or rate is None:
      return
    is_last = (
      self._completed_epochs == self.training_configuration.number_of_epochs
    )
    if self._completed_epochs % rate != 0 and not is_last:
      return
    dataset = self._validation_dataset
    assert dataset is not None
    model = self.ema.model if self.ema is not None else self.model

    def reconstruct(start: int, count: int, generator: torch.Generator):
      images = torch.stack(
        [
          dataset[i][0] for i in range(start, start + count)  # ty: ignore[not-subscriptable]
        ]
      )
      return model(
        images.to(self.device), sample_posterior=False
      ).reconstruction

    was_training = model.training
    model.eval()
    try:
      metrics = self.evaluator.evaluate_generator(reconstruct)
    finally:
      model.train(was_training)
    self.tensorboard_manager.log_metrics(
      {f"Val/{name}": value for name, value in metrics.items()},
      global_step=self._completed_epochs * len(self.dataloader),
    )
    logging.info(f"Epoch {self._completed_epochs}: {metrics}")

  @torch.no_grad()
  def validate(self) -> Dict[str, float]:
    """Reconstruct the validation images (posterior mean, EMA weights).

    Returns:
      ``L1``, ``PSNR`` (images span ``[-1, 1]``), the normalised ``KL`` and the
      standard deviation of the latent means (``LatentStd``; its inverse is the
      scale a latent diffusion model should apply to the latents).
    """
    assert self.validation_dataloader is not None
    model = self.ema.model if self.ema is not None else self.model
    was_training = model.training
    model.eval()
    totals = {"l1": 0.0, "mse": 0.0, "kl": 0.0, "n": 0}
    latent_sum = latent_square_sum = 0.0
    latent_count = 0
    try:
      for images, _ in self.validation_dataloader:
        images = images.to(self.device)
        with torch.autocast(
          device_type=torch.device(self.device).type,
          dtype=self._autocast_dtype,
          enabled=self.training_configuration.mixed_precision_training,
        ):
          output = model(images, sample_posterior=False)
        reconstruction = output.reconstruction.float()
        count = images.shape[0]
        totals["l1"] += float((reconstruction - images).abs().mean()) * count
        totals["mse"] += float((reconstruction - images).pow(2).mean()) * count
        totals["kl"] += (
          float(
            self.loss(
              images, output.reconstruction, output.mean, output.logvar
            ).metrics["KL"]
          )
          * count
        )
        totals["n"] += count
        mean = output.mean.float()
        latent_sum += float(mean.sum())
        latent_square_sum += float(mean.pow(2).sum())
        latent_count += mean.numel()
    finally:
      model.train(was_training)
    n = totals["n"]
    mse = totals["mse"] / n
    latent_mean = latent_sum / latent_count
    return {
      "L1": totals["l1"] / n,
      # The images span [-1, 1], so the peak-to-peak range is 2.
      "PSNR": float(10.0 * torch.log10(torch.tensor(4.0 / mse))),
      "KL": totals["kl"] / n,
      "LatentStd": (latent_square_sum / latent_count - latent_mean**2) ** 0.5,
    }
