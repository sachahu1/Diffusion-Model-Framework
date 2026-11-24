import abc
import logging
import math
import pathlib
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_models.core.losses.base import LossOutput
from diffusion_models.core.utils.ema import (
  ExponentialMovingAverage,
  unwrap_model,
)
from diffusion_models.core.utils.interrupts import sigterm_as_interrupt
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)
from diffusion_models.core.utils.tensorboard import TensorboardManager


class BaseTrainer(abc.ABC):
  checkpoint_class: Any
  """The checkpoint dataclass (needs ``from_file`` and ``to_file``)."""
  metric_prefix: str = ""
  """Prepended to the name of every training metric logged to tensorboard."""

  def __init__(
    self,
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
    log_configuration: LogConfiguration = LogConfiguration(),
    device: str = "cuda",
    output_directory: Optional[Union[str, pathlib.Path]] = None,
    run_config: Optional[str] = None,
    resume_from: Optional[Union[str, pathlib.Path]] = None,
  ):
    """A generic trainer: everything but the objective is shared.

    A concrete trainer implements :meth:`training_step` and :meth:`log_images`, and
    optionally :meth:`on_epoch_end`, :meth:`checkpoint_fields` and
    :meth:`check_checkpoint`. It sets its own attributes *after* calling
    ``super().__init__``, so :meth:`check_checkpoint` may only rely on the
    attributes defined here.

    Args:
      model: The model to train.
      dataset: The training dataset. Its items are passed to
        :meth:`training_step` as batches.
      optimizer: The optimizer.
      training_configuration: The training configuration.
      log_configuration: When to log metrics, images and weights.
      device: The device.
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
    """
    self.model = model.to(device)
    """The model being trained."""
    self.optimizer = optimizer
    """The optimizer to use."""
    self._base_learning_rates = [
      group["lr"] for group in optimizer.param_groups
    ]
    """The rate of every parameter group before any schedule (and before a
    resumed optimizer state overwrites it)."""
    self._check_learning_rate_schedule(training_configuration)
    self.training_configuration = training_configuration
    """The training configuration to use."""
    self.log_configuration = log_configuration
    """A LogConfiguration object."""
    self.device = device
    """The device to use."""
    self.run_config = run_config
    """The resolved run configuration (YAML), if any."""

    self.dataloader = self._build_dataloader(
      dataset,
      batch_size=training_configuration.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=training_configuration.num_workers,
    )
    """The training dataloader."""

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
      torch.nn.Module.to(  # ty: ignore[no-matching-overload]
        self.model, memory_format=torch.channels_last
      )

    if output_directory is not None:
      output_directory = pathlib.Path(output_directory)
      self.checkpoint_path = output_directory / "checkpoints"
      log_directory = output_directory / "tensorboard"
    else:
      self.checkpoint_path = (
        pathlib.Path("../checkpoints") / training_configuration.training_name
      )
      log_directory = None
    """The path to save checkpoints."""
    self.checkpoint_path.mkdir(parents=True, exist_ok=True)
    self.tensorboard_manager = TensorboardManager(
      log_name=training_configuration.training_name,
      log_directory=log_directory,
    )
    """A tensorboard manager instance."""
    if run_config is not None:
      self.tensorboard_manager.log_text(
        "config", f"```yaml\n{run_config}\n```"
      )

    self.ema: Optional[ExponentialMovingAverage] = None
    """The EMA of the model weights, when ``ema_decay`` is configured."""
    if training_configuration.ema_decay is not None:
      self.ema = ExponentialMovingAverage(
        self.model,
        decay=training_configuration.ema_decay,
        warmup_steps=training_configuration.ema_warmup_steps,
      )

    self._completed_epochs = 0
    """The number of fully completed epochs."""

    self._train_model: Callable = self.model
    """The callable used for the training forward pass (possibly compiled)."""
    if training_configuration.compile:
      # ``self.model`` stays plain to keep unprefixed state dict keys.
      self._train_model = torch.compile(self.model)

    if resume_from is not None:
      self.load_checkpoint(resume_from)

    torch.backends.cudnn.benchmark = True

  @abc.abstractmethod
  def training_step(self, batch: Any) -> LossOutput:
    """Compute the loss of a batch.

    This is where the inputs, the forward pass (use ``self._train_model``) and
    the objective live. It runs under autocast when mixed precision is on.
    Use :meth:`to_device` for the tensors fed to the model.

    Args:
      batch: An item of the dataloader.

    Returns:
      The loss to optimise and detached metrics to log.
    """
    raise NotImplementedError

  def log_images(self, global_step: int) -> None:
    """Log visualisations to tensorboard (every ``image_rate`` steps).

    The model is in eval mode and gradients are disabled. Prefer the EMA
    model (``self.ema.model``) when available.
    """

  def on_epoch_end(self) -> None:
    """Run after the epoch's checkpoints are saved (validation, FID, ...)."""

  def checkpoint_fields(self) -> Dict[str, Any]:
    """Model specific fields for the checkpoint, e.g. its configuration."""
    return {}

  def check_checkpoint(self, checkpoint: Any) -> None:
    """Raise if ``checkpoint`` cannot be resumed by this trainer.

    Called before any state is restored.
    """

  def load_extra_state(self, checkpoint: Any) -> None:
    """Restore the state saved by :meth:`checkpoint_fields` (e.g. a second model).

    Called after the model, optimizer, scaler and EMA are restored.
    """

  @staticmethod
  def _check_learning_rate_schedule(
    configuration: TrainingConfiguration,
  ) -> None:
    if configuration.lr_schedule not in (None, "cosine", "linear"):
      raise ValueError(
        "lr_schedule must be None, 'cosine' or 'linear', "
        f"got {configuration.lr_schedule!r}."
      )
    if not 0.0 <= configuration.lr_min_ratio <= 1.0:
      raise ValueError("lr_min_ratio must be between 0 and 1.")
    if configuration.lr_warmup_steps < 0:
      raise ValueError("lr_warmup_steps cannot be negative.")
    if (
      not 0
      <= configuration.lr_schedule_start_epoch
      < max(configuration.number_of_epochs, 1)
    ):
      raise ValueError(
        "lr_schedule_start_epoch must be in [0, number_of_epochs)."
      )

  def learning_rate_factor(self, global_step: int) -> Optional[float]:
    """The learning rate at ``global_step`` as a fraction of the base rate.

    ``None`` when the rate is constant. A pure function of the step, so a
    resumed run continues the schedule with no saved state.
    """
    configuration = self.training_configuration
    if configuration.lr_schedule is None:
      return None
    steps_per_epoch = len(self.dataloader)
    start = configuration.lr_schedule_start_epoch * steps_per_epoch
    total = configuration.number_of_epochs * steps_per_epoch
    step = global_step - start
    if step < 0:
      return 1.0
    warmup = configuration.lr_warmup_steps
    if step < warmup:
      return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - start - warmup))
    if configuration.lr_schedule == "cosine":
      decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
      decay = 1.0 - progress
    ratio = configuration.lr_min_ratio
    return ratio + (1.0 - ratio) * decay

  def apply_learning_rate(self, global_step: int) -> Optional[float]:
    """Set the optimizer's learning rate for ``global_step``.

    Returns the rate of the first parameter group, or None without a schedule.
    """
    factor = self.learning_rate_factor(global_step)
    if factor is None:
      return None
    for group, base in zip(
      self.optimizer.param_groups, self._base_learning_rates
    ):
      group["lr"] = base * factor
    return self.optimizer.param_groups[0]["lr"]

  def optimization_step(
    self, batch: Any, global_step: int
  ) -> Dict[str, torch.Tensor]:
    """Update the model on a batch: the part of a step that differs between tasks.

    The default is the standard step: compute the loss with
    :meth:`training_step`, back-propagate, clip, step the optimizer and update
    the EMA. Override it for training that updates several models (e.g. a
    generator and a discriminator).

    Args:
      batch: An item of the dataloader.
      global_step: The index of the step since the start of the training.

    Returns:
      The detached metrics to log.
    """
    configuration = self.training_configuration
    learning_rate = self.apply_learning_rate(global_step)
    self.optimizer.zero_grad(set_to_none=True)
    with self.autocast():
      output = self.training_step(batch)

    self.scaler.scale(output.loss).backward()
    metrics = dict(output.metrics)
    if learning_rate is not None:
      metrics["LearningRate"] = torch.tensor(learning_rate)
    if configuration.gradient_clip is not None:
      # Unscales the gradients of optimizer's assigned params in-place
      self.scaler.unscale_(self.optimizer)
      # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
      metrics["GradNorm"] = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), max_norm=configuration.gradient_clip
      )
    self.scaler.step(self.optimizer)
    self.scaler.update()

    if self.ema is not None:
      self.ema.update(self.model)
    return metrics

  def autocast(self):
    """The autocast context of the configured mixed precision (a no-op when off)."""
    return torch.autocast(
      device_type=torch.device(self.device).type,
      dtype=self._autocast_dtype,
      enabled=self.training_configuration.mixed_precision_training,
    )

  def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the device, in ``channels_last`` format if configured."""
    tensor = tensor.to(self.device)
    if self.training_configuration.channels_last and tensor.ndim == 4:
      tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor

  def _build_dataloader(
    self,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
  ) -> DataLoader:
    configuration = self.training_configuration
    return DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      num_workers=num_workers,
      # Pinned memory only helps host-to-CUDA copies.
      pin_memory=configuration.pin_memory
      and torch.device(self.device).type == "cuda",
      persistent_workers=configuration.persistent_workers and num_workers > 0,
      prefetch_factor=configuration.prefetch_factor
      if num_workers > 0
      else None,
    )

  def load_checkpoint(self, file_path: Union[str, pathlib.Path]) -> None:
    """Restore the training state from a checkpoint.

    Args:
      file_path: The checkpoint to restore (e.g. ``latest.pt`` or
        ``interrupted.pt``).
    """
    checkpoint = self.checkpoint_class.from_file(
      file_path, map_location=self.device
    )
    self.check_checkpoint(checkpoint)
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
    self.load_extra_state(checkpoint)
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
    checkpoint = self.checkpoint_class(
      epoch=epoch,
      completed_epochs=self._completed_epochs,
      model_state_dict=self.model.state_dict(),
      optimizer_state_dict=self.optimizer.state_dict(),
      ema_model_state_dict=self.ema.state_dict()
      if self.ema is not None
      else None,
      ema_num_updates=self.ema.num_updates if self.ema is not None else None,
      scaler=self.scaler.state_dict() if self.scaler.is_enabled() else None,
      tensorboard_run_name=self.tensorboard_manager.summary_writer.log_dir,
      config_yaml=self.run_config,
      **self.checkpoint_fields(),
    )
    checkpoint.to_file(self.checkpoint_path / checkpoint_name)

  def train(self):
    """Start the training.

    If the run is interrupted (``KeyboardInterrupt`` or SIGTERM), an
    ``interrupted.pt`` checkpoint is saved before the interruption is
    re-raised. When ``latest_checkpoint_rate`` is set, ``latest.pt`` is
    overwritten every that many epochs.
    """
    try:
      with sigterm_as_interrupt():
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

        metrics = self.optimization_step(batch, global_step)

        self.log_to_tensorboard(metrics=metrics, global_step=global_step)

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
      self.on_epoch_end()
    self.save_checkpoint(
      epoch=configuration.number_of_epochs,
      checkpoint_name="final.pt",
    )

  @torch.no_grad()
  def log_to_tensorboard(
    self, metrics: Dict[str, torch.Tensor], global_step: int
  ):
    """Log metrics, weights and images to tensorboard, when they are due.

    Args:
      metrics: The training metrics of the step.
      global_step: The current global step.
    """
    configuration = self.log_configuration
    if global_step % configuration.log_rate == 0:
      self.tensorboard_manager.log_metrics(
        metrics={
          f"{self.metric_prefix}{name}": float(value)
          for name, value in metrics.items()
        },
        global_step=global_step,
      )

    log_weights = (
      configuration.weight_rate is not None
      and global_step % configuration.weight_rate == 0
    )
    log_images = (
      global_step % configuration.image_rate == 0
      and configuration.number_of_images > 0
    )
    if not (log_weights or log_images):
      return

    was_training = self.model.training
    self.model.eval()
    try:
      if log_weights:
        self.tensorboard_manager.log_weights(
          model=self.model, global_step=global_step
        )
      if log_images:
        self.log_images(global_step)
    finally:
      # Logging switches to eval mode (BatchNorm, etc.); always restore.
      self.model.train(was_training)
