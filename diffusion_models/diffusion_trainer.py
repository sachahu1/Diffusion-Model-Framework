import contextlib
import logging
import pathlib
import signal
from typing import Callable, Dict, Optional, Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_models.models.base_diffusion_model import BaseDiffusionModel
from diffusion_models.utils.ema import ExponentialMovingAverage
from diffusion_models.utils.ema import unwrap_model
from diffusion_models.utils.schemas import (
  BetaSchedulerConfiguration,
  Checkpoint,
  EvaluationConfiguration,
  LogConfiguration,
  PredictionType,
  TrainingConfiguration,
)
from diffusion_models.utils.tensorboard import TensorboardManager


class DiffusionTrainer:
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

    This is a simplified framework for training a diffusion models.

    Args:
      model: A diffusion model.
      dataset: A dataset to train on.
      optimizer: The optimizer to use.
      training_configuration: The training configuration to use.
      loss_function: The loss function to use.
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
        tensorboard. Needs the ``evaluation`` dependency group.
      evaluation_dataset: The real reference images for the evaluation,
        ideally a held-out split.
    """
    self.model = model.to(device)
    """The diffusion model to use."""
    self.optimizer = optimizer
    """The optimizer to use."""
    self.loss_function = loss_function
    """The loss function to use."""
    self.training_configuration = training_configuration
    """The training configuration to use."""
    # self.scheduler = scheduler
    self.device = device
    """The device to use."""

    num_workers = training_configuration.num_workers
    self.dataloader = DataLoader(
      dataset=dataset,
      batch_size=training_configuration.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=num_workers,
      # Pinned memory only helps host-to-CUDA copies.
      pin_memory=training_configuration.pin_memory
      and torch.device(device).type == "cuda",
      persistent_workers=training_configuration.persistent_workers
      and num_workers > 0,
      prefetch_factor=training_configuration.prefetch_factor
      if num_workers > 0
      else None,
    )
    """A torch dataloader."""

    self._image_shape = dataset[0][0].shape

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if training_configuration.mixed_precision_dtype not in dtypes:
      raise ValueError(
        "mixed_precision_dtype must be one of "
        f"{sorted(dtypes)}, got {training_configuration.mixed_precision_dtype!r}."
      )
    self._autocast_dtype = dtypes[training_configuration.mixed_precision_dtype]
    """The dtype used by autocast when mixed precision is on."""

    self.scaler = torch.amp.GradScaler(
      device=torch.device(device).type,
      # bfloat16 has the range of fp32, so only fp16 needs loss scaling.
      enabled=training_configuration.mixed_precision_training
      and self._autocast_dtype == torch.float16,
    )
    """A torch GradScaler object (a no-op unless training in float16)."""

    if training_configuration.channels_last:
      if not training_configuration.mixed_precision_training:
        logging.warning(
          "channels_last is very slow without mixed precision training."
        )
      # ``BaseDiffusionModel.to`` does not accept ``memory_format``.
      torch.nn.Module.to(self.model, memory_format=torch.channels_last)

    self.log_configuration = log_configuration
    """A LogConfiguration object."""

    if output_directory is not None:
      output_directory = pathlib.Path(output_directory)
      self.checkpoint_path = output_directory / "checkpoints"
      log_directory = output_directory / "tensorboard"
    else:
      self.checkpoint_path = (
        pathlib.Path("../checkpoints")
        / self.training_configuration.training_name
      )
      log_directory = None
    """The path to save checkpoints."""

    self.checkpoint_path.mkdir(parents=True, exist_ok=True)
    self.tensorboard_manager = TensorboardManager(
      log_name=self.training_configuration.training_name,
      log_directory=log_directory,
    )
    """A tensorboard manager instance."""

    self.run_config = run_config
    """The resolved run configuration (YAML), if any."""
    if run_config is not None:
      self.tensorboard_manager.log_text(
        "config", f"```yaml\n{run_config}\n```"
      )

    self.reverse_transforms = reverse_transforms
    """A set of reverse transforms."""

    self.ema: Optional[ExponentialMovingAverage] = None
    """The EMA of the model weights, when ``ema_decay`` is configured."""
    if training_configuration.ema_decay is not None:
      self.ema = ExponentialMovingAverage(
        self.model,
        decay=training_configuration.ema_decay,
        warmup_steps=training_configuration.ema_warmup_steps,
      )

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
      from diffusion_models.evaluation.fid import ImageQualityEvaluator

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

    self._completed_epochs = 0
    """The number of fully completed epochs."""

    self._train_model: Callable = self.model
    """The callable used for the training forward pass (possibly compiled)."""
    if training_configuration.compile:
      # ``self.model`` stays the plain module: the optimizer, EMA and
      # checkpoints keep unprefixed state dict keys. The compiled wrapper
      # shares its parameters, so updating one updates the other.
      self._train_model = torch.compile(self.model)

    if resume_from is not None:
      self.load_checkpoint(resume_from)

    torch.backends.cudnn.benchmark = True

  def load_checkpoint(self, file_path: Union[str, pathlib.Path]) -> None:
    """Restore the training state from a checkpoint.

    Args:
      file_path: The checkpoint to restore (e.g. ``latest.pt`` or
        ``interrupted.pt``).
    """
    checkpoint = Checkpoint.from_file(file_path, map_location=self.device)
    expected = PredictionType(self.model.diffuser.prediction_type).value
    if checkpoint.prediction_type != expected:
      raise ValueError(
        f"Checkpoint was trained to predict {checkpoint.prediction_type!r} "
        f"but the diffuser predicts {expected!r}; refusing to resume."
      )
    unwrap_model(self.model).load_state_dict(checkpoint.model_state_dict)
    self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    if checkpoint.scaler is not None and self.scaler.is_enabled():
      self.scaler.load_state_dict(checkpoint.scaler)
    if self.ema is not None:
      if checkpoint.ema_model_state_dict is not None:
        self.ema.load_state_dict(
          checkpoint.ema_model_state_dict,
          num_updates=checkpoint.ema_num_updates,
        )
      else:
        logging.warning(
          "Checkpoint has no EMA weights; the EMA restarts from the raw weights."
        )
        self.ema.load_state_dict(checkpoint.model_state_dict)
    self._completed_epochs = (
      checkpoint.completed_epochs
      if checkpoint.completed_epochs is not None
      else checkpoint.epoch
    )
    logging.info(
      f"Resumed from {file_path} after {self._completed_epochs} epochs."
    )

  def save_checkpoint(self, epoch: int, checkpoint_name: str):
    """Save a checkpoint (atomically).

    Args:
      epoch: The current epoch. ``final.pt``, ``latest.pt`` and
        ``interrupted.pt`` store the number of completed epochs.
      checkpoint_name: The name of the checkpoint.
    """
    checkpoint = Checkpoint(
      epoch=epoch,
      completed_epochs=self._completed_epochs,
      model_state_dict=self.model.state_dict(),
      ema_model_state_dict=self.ema.state_dict()
      if self.ema is not None
      else None,
      ema_num_updates=self.ema.num_updates if self.ema is not None else None,
      optimizer_state_dict=self.optimizer.state_dict(),
      scaler=self.scaler.state_dict() if self.scaler.is_enabled() else None,
      image_channels=self._image_shape[0],
      prediction_type=PredictionType(
        self.model.diffuser.prediction_type
      ).value,
      beta_scheduler_config=BetaSchedulerConfiguration(
        steps=self.model.diffuser.beta_scheduler.steps,
        betas=self.model.diffuser.beta_scheduler.betas,
        alpha_bars=self.model.diffuser.beta_scheduler.alpha_bars,
      ),
      tensorboard_run_name=self.tensorboard_manager.summary_writer.log_dir,
      config_yaml=self.run_config,
    )
    checkpoint.to_file(self.checkpoint_path / checkpoint_name)

  @contextlib.contextmanager
  def _sigterm_as_interrupt(self):
    """Treat SIGTERM like Ctrl-C so an interrupted checkpoint gets saved."""

    def handler(signum, frame):
      raise KeyboardInterrupt

    try:
      previous = signal.signal(signal.SIGTERM, handler)
    except ValueError:  # Not in the main thread.
      yield
      return
    try:
      yield
    finally:
      signal.signal(signal.SIGTERM, previous)

  def train(self):
    """Start the diffusion training.

    If the run is interrupted (``KeyboardInterrupt`` or SIGTERM), an
    ``interrupted.pt`` checkpoint is saved before the interruption is
    re-raised. When ``latest_checkpoint_rate`` is set, ``latest.pt`` is
    overwritten every that many epochs.
    """
    try:
      with self._sigterm_as_interrupt():
        self._train()
    except KeyboardInterrupt:
      self.save_checkpoint(
        epoch=self._completed_epochs, checkpoint_name="interrupted.pt"
      )
      logging.warning(
        f"Interrupted after {self._completed_epochs} completed epochs; "
        f"saved {self.checkpoint_path / 'interrupted.pt'}"
      )
      raise

  def _train(self):
    self.model.train()
    configuration = self.training_configuration
    for epoch in range(self._completed_epochs, configuration.number_of_epochs):
      for step, batch in enumerate(
        tqdm(self.dataloader, desc=f"Epoch={epoch}")
      ):
        global_step = epoch * len(self.dataloader) + step

        images, _ = batch
        images = images.to(self.device)

        noisy_images, noise, timesteps = self.model.diffuse(images=images)
        if configuration.channels_last:
          noisy_images = noisy_images.contiguous(
            memory_format=torch.channels_last
          )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
          device_type=torch.device(self.device).type,
          dtype=self._autocast_dtype,
          enabled=configuration.mixed_precision_training,
        ):
          prediction = self._train_model(noisy_images, timesteps)
          target = self.model.diffuser.get_target(images, noise, timesteps)
          loss = self.loss_function(target, prediction)

        self.scaler.scale(loss).backward()

        if configuration.gradient_clip is not None:
          # Unscales the gradients of optimizer's assigned params in-place
          self.scaler.unscale_(self.optimizer)

          # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
          torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=configuration.gradient_clip,
          )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema is not None:
          self.ema.update(self.model)

        self.log_to_tensorboard(
          metrics={
            "Loss": loss,
          },
          global_step=global_step,
        )
      self._completed_epochs = epoch + 1
      if epoch % configuration.checkpoint_rate == 0:
        self.save_checkpoint(epoch=epoch, checkpoint_name=f"epoch_{epoch}.pt")
      if (
        configuration.latest_checkpoint_rate is not None
        and self._completed_epochs % configuration.latest_checkpoint_rate == 0
      ):
        self.save_checkpoint(
          epoch=self._completed_epochs, checkpoint_name="latest.pt"
        )
      self._maybe_evaluate()
    self.save_checkpoint(
      epoch=configuration.number_of_epochs,
      checkpoint_name="final.pt",
    )

  def _maybe_evaluate(self) -> None:
    """Compute and log FID/KID when due (after checkpoints are saved)."""
    if self.evaluator is None:
      return
    rate = self.evaluation_configuration.fid_rate
    is_last = (
      self._completed_epochs == self.training_configuration.number_of_epochs
    )
    if self._completed_epochs % rate != 0 and not is_last:
      return
    # Evaluate the EMA weights when available: they are what you would ship.
    model = self.ema.model if self.ema is not None else self.model
    metrics = self.evaluator.evaluate(model)
    # ``evaluate`` restores the model's training mode.
    global_step = self._completed_epochs * len(self.dataloader)
    self.tensorboard_manager.log_metrics(
      metrics={f"Eval/{name}": value for name, value in metrics.items()},
      global_step=global_step,
    )
    logging.info(f"Epoch {self._completed_epochs}: {metrics}")

  @torch.no_grad()
  def log_to_tensorboard(self, metrics: Dict[str, float], global_step: int):
    """Log to tensorboard.

    This method logs some useful metrics and visualizations to tensorboard.

    Args:
      metrics: A dictionary mapping metric names to values.
      global_step: The current global step.
    """
    was_training = self.model.training
    self.model.eval()
    try:
      self._log_to_tensorboard(metrics=metrics, global_step=global_step)
    finally:
      # Logging switches to eval mode (BatchNorm, etc.); always restore.
      self.model.train(was_training)

  @torch.no_grad()
  def _log_to_tensorboard(self, metrics: Dict[str, float], global_step: int):
    if global_step % self.log_configuration.log_rate == 0:
      self.tensorboard_manager.log_metrics(
        metrics=metrics, global_step=global_step
      )

    if (
      self.log_configuration.weight_rate is not None
      and global_step % self.log_configuration.weight_rate == 0
    ):
      self.tensorboard_manager.log_weights(
        model=self.model, global_step=global_step
      )

    if (global_step % self.log_configuration.image_rate == 0) and (
      self.log_configuration.number_of_images > 0
    ):
      image_channels, image_height, image_width = self._image_shape
      images = torch.randn(
        (
          self.log_configuration.number_of_images,
          image_channels,
          image_height,
          image_width,
        ),
        device=self.device,
      )
      # Sample from the EMA weights when available; they are far less noisy.
      images = (
        self.ema.model if self.ema is not None else self.model
      ).denoise(images)
      for step, images in enumerate(images[::-1]):
        self.tensorboard_manager.log_images(
          tag=f"Images at timestep {global_step}",
          images=self.reverse_transforms(images),
          timestep=step,
        )
