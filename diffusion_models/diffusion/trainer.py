import logging
import pathlib
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from diffusion_models.core.losses import LossOutput
from diffusion_models.core.trainer import BaseTrainer
from diffusion_models.diffusion.models.base_diffusion_model import (
  BaseDiffusionModel,
)
from diffusion_models.core.schemas import (
  EvaluationConfiguration,
  LogConfiguration,
  TrainingConfiguration,
)
from diffusion_models.diffusion.schemas import (
  BetaSchedulerConfiguration,
  Checkpoint,
  PredictionType,
)


class DiffusionTrainer(BaseTrainer):
  checkpoint_class = Checkpoint

  def __init__(
    self,
    model: BaseDiffusionModel,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
    loss_function: Callable = F.l1_loss,
    # scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,
    log_configuration: LogConfiguration = LogConfiguration(),
    reverse_transforms: Callable = lambda x: x,
    device: str = "cuda",
    output_directory: Optional[Union[str, pathlib.Path]] = None,
    run_config: Optional[str] = None,
    resume_from: Optional[Union[str, pathlib.Path]] = None,
    evaluation_configuration: Optional[EvaluationConfiguration] = None,
    evaluation_dataset: Optional[Dataset] = None,
  ):
    """A diffusion trainer framework.

    This is a simplified framework for training a diffusion models. The
    training loop, mixed precision, EMA, checkpoints and logging are
    inherited from :class:`~.BaseTrainer`; this class adds the diffusion
    objective, sampling for visualisation and the FID/KID evaluation.

    Args:
      model: A diffusion model.
      dataset: A dataset to train on.
      optimizer: The optimizer to use.
      training_configuration: The training configuration to use.
      loss_function: ``loss_function(target, prediction)``.
      log_configuration: The logging configuration to use.
      reverse_transforms: The reverse transforms to use.
      device: The device to use.
      output_directory: Where to write this run's artifacts. When provided,
        checkpoints go to ``<output_directory>/checkpoints`` and tensorboard
        logs to ``<output_directory>/tensorboard``. Otherwise the legacy
        ``../checkpoints/<name>`` and ``../runs/<name>`` locations are used.
      run_config: The resolved configuration (YAML) of the run. It is logged
        to tensorboard and stored in every checkpoint for reproducibility.
      resume_from: A checkpoint to resume training from. The model, optimizer,
        grad scaler and EMA are restored and training continues from the
        epoch after the last completed one. Resuming happens at an epoch
        boundary; progress within an interrupted epoch is lost.
      evaluation_configuration: When ``fid_rate`` is set, FID (and KID) of
        samples from the EMA model (or the raw model without EMA) is computed
        every that many epochs, and after the last epoch, then logged to
        tensorboard. Needs the ``evaluation`` extra.
      evaluation_dataset: The real reference images for the evaluation,
        ideally a held-out split.
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
    self.model: BaseDiffusionModel
    self.loss_function = loss_function
    """The loss function to use."""
    self.reverse_transforms = reverse_transforms
    """A set of reverse transforms."""
    self._image_shape = dataset[0][0].shape

    self.evaluation_configuration = evaluation_configuration
    """The FID/KID evaluation settings."""
    self.evaluator = None
    """The FID/KID evaluator, when ``fid_rate`` is configured."""
    if (
      evaluation_configuration is not None
      and evaluation_configuration.fid_rate is not None
    ):
      if evaluation_dataset is None:
        raise ValueError("FID evaluation needs an evaluation_dataset.")
      # Imported lazily: torchmetrics is an optional dependency.
      from diffusion_models.core.evaluation.fid import ImageQualityEvaluator

      self.evaluator = ImageQualityEvaluator(
        real_dataset=evaluation_dataset,
        number_of_samples=evaluation_configuration.number_of_samples,
        batch_size=evaluation_configuration.batch_size,
        device=device,
        compute_kid=evaluation_configuration.compute_kid,
        kid_subset_size=evaluation_configuration.kid_subset_size,
        seed=evaluation_configuration.seed,
        num_workers=evaluation_configuration.num_workers,
      )

  def training_step(self, batch: Any) -> LossOutput:
    images, _ = batch
    images = images.to(self.device)
    noisy_images, noise, timesteps = self.model.diffuse(images=images)
    prediction = self._train_model(self.to_device(noisy_images), timesteps)
    target = self.model.diffuser.get_target(images, noise, timesteps)
    loss = self.loss_function(target, prediction)
    return LossOutput(loss=loss, metrics={"Loss": loss.detach()})

  def checkpoint_fields(self) -> Dict[str, Any]:
    diffuser = self.model.diffuser
    return dict(
      image_channels=self._image_shape[0],
      prediction_type=PredictionType(diffuser.prediction_type).value,
      beta_scheduler_config=BetaSchedulerConfiguration(
        steps=diffuser.beta_scheduler.steps,
        betas=diffuser.beta_scheduler.betas,
        alpha_bars=diffuser.beta_scheduler.alpha_bars,
      ),
    )

  def check_checkpoint(self, checkpoint: Checkpoint) -> None:
    expected = PredictionType(self.model.diffuser.prediction_type).value
    if checkpoint.prediction_type != expected:
      raise ValueError(
        f"Checkpoint was trained to predict {checkpoint.prediction_type!r} "
        f"but the diffuser predicts {expected!r}; refusing to resume."
      )

  def on_epoch_end(self) -> None:
    """Compute and log FID/KID when due (after checkpoints are saved)."""
    if self.evaluator is None or self.evaluation_configuration is None:
      return
    rate = self.evaluation_configuration.fid_rate
    is_last = (
      self._completed_epochs == self.training_configuration.number_of_epochs
    )
    if rate is None or self._completed_epochs % rate != 0 and not is_last:
      return
    # Evaluate the EMA weights when available: they are what you would ship.
    model = self.ema.model if self.ema is not None else self.model
    metrics = self.evaluate_samples(model)  # ty: ignore[invalid-argument-type]
    # ``evaluate`` restores the model's training mode.
    global_step = self._completed_epochs * len(self.dataloader)
    self.tensorboard_manager.log_metrics(
      metrics={f"Eval/{name}": value for name, value in metrics.items()},
      global_step=global_step,
    )
    logging.info(f"Epoch {self._completed_epochs}: {metrics}")

  def evaluate_samples(self, model: BaseDiffusionModel) -> Dict[str, float]:
    """Score the samples of ``model`` against the reference images."""
    assert self.evaluator is not None
    return self.evaluator.evaluate(model)

  def to_image(self, samples: torch.Tensor) -> torch.Tensor:
    """Map samples of the diffusion model to images in ``[-1, 1]``.

    The identity here; a latent diffusion trainer decodes them.
    """
    return samples

  # Snapshots of the denoising process shown in the ``Denoising`` grid.
  _TRAJECTORY_ROWS = 6

  def log_images(self, global_step: int) -> None:
    """Log samples, as two image series with a slider over the epochs.

    - ``Samples``: the final images.
    - ``Denoising``: the same images at evenly spaced steps of the reverse
      process, one row per step, from the first (top, still noisy) to the
      sample (bottom).

    The slider is the number of completed epochs (this is called right after
    the first step of an epoch), so ``Samples`` at 40 is what the model
    generated after 40 epochs, like ``epoch_40.pt``. Logging more than once
    per epoch puts several grids on the same epoch.
    """
    image_channels, image_height, image_width = self._image_shape
    noise = torch.randn(
      (
        self.log_configuration.number_of_images,
        image_channels,
        image_height,
        image_width,
      ),
      device=self.device,
    )
    # EMA weights when available; the last step is the sample.
    trajectory = (
      self.ema.model if self.ema is not None else self.model
    ).denoise(noise)
    epoch = global_step // len(self.dataloader)

    def to_pictures(images: torch.Tensor) -> torch.Tensor:
      return self.reverse_transforms(self.to_image(images))

    self.tensorboard_manager.log_image_grid(
      "Samples", to_pictures(trajectory[-1]), global_step=epoch
    )
    rows = min(self._TRAJECTORY_ROWS, len(trajectory))
    snapshots = [
      trajectory[round(i * (len(trajectory) - 1) / max(rows - 1, 1))]
      for i in range(rows)
    ]
    self.tensorboard_manager.log_image_grid(
      "Denoising",
      torch.cat([to_pictures(snapshot) for snapshot in snapshots]),
      global_step=epoch,
      columns=len(noise),
    )
