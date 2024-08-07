import pathlib
from typing import Callable, Dict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_models.models.base_diffusion_model import BaseDiffusionModel
from diffusion_models.utils.schemas import BetaSchedulerConfiguration, \
  Checkpoint, LogConfiguration, TrainingConfiguration
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

    self.dataloader = DataLoader(
      dataset=dataset,
      batch_size=training_configuration.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=16,
      pin_memory=True,
      persistent_workers=True,
    )
    """A torch dataloader."""

    self._image_shape = dataset[0][0].shape

    self.scaler = torch.amp.GradScaler(
      device=device
      # init_scale=8192,
    )
    """A torch GradScaler object."""

    self.log_configuration = log_configuration
    """A LogConfiguration object."""

    self.checkpoint_path = (
      pathlib.Path("../checkpoints")
      / self.training_configuration.training_name
    )
    """The path to save checkpoints."""

    self.checkpoint_path.mkdir(exist_ok=True)
    self.tensorboard_manager = TensorboardManager(
      log_name=self.training_configuration.training_name,
    )
    """A tensorboard manager instance."""

    self.reverse_transforms = reverse_transforms
    """A set of reverse transforms."""

    torch.backends.cudnn.benchmark = True

  def save_checkpoint(self, epoch: int, checkpoint_name: str):
    """Save a checkpoint.

    Args:
      epoch: The current epoch.
      checkpoint_name: The name of the checkpoint.
    """
    checkpoint = Checkpoint(
      epoch=epoch,
      model_state_dict=self.model.state_dict(),
      optimizer_state_dict=self.optimizer.state_dict(),
      scaler=self.scaler.state_dict()
      if self.training_configuration.mixed_precision_training
      else None,
      image_channels=self._image_shape[0],
      beta_scheduler_config=BetaSchedulerConfiguration(
        steps=self.model.diffuser.beta_scheduler.steps,
        betas=self.model.diffuser.beta_scheduler.betas,
        alpha_bars=self.model.diffuser.beta_scheduler.alpha_bars,
      ),
      tensorboard_run_name=self.tensorboard_manager.summary_writer.log_dir,
    )
    checkpoint.to_file(self.checkpoint_path / checkpoint_name)

  def train(self):
    """Start the diffusion training."""
    self.model.train()
    for epoch in range(self.training_configuration.number_of_epochs):
      for step, batch in enumerate(
        tqdm(self.dataloader, desc=f"Epoch={epoch}")
      ):
        global_step = epoch * len(self.dataloader) + step

        images, _ = batch
        images = images.to(self.device)

        noisy_images, noise, timesteps = self.model.diffuse(images=images)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
          device_type=self.device,
          enabled=self.training_configuration.mixed_precision_training,
        ):
          prediction = self.model(noisy_images, timesteps)
          loss = self.loss_function(noise, prediction)

        self.scaler.scale(loss).backward()

        if self.training_configuration.gradient_clip is not None:
          # Unscales the gradients of optimizer's assigned params in-place
          self.scaler.unscale_(self.optimizer)

          # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
          torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.training_configuration.gradient_clip,
          )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log_to_tensorboard(
          metrics={
            "Loss": loss,
          },
          global_step=global_step,
        )
      if epoch % self.training_configuration.checkpoint_rate == 0:
        self.save_checkpoint(epoch=epoch, checkpoint_name=f"epoch_{epoch}.pt")
    self.save_checkpoint(
      epoch=self.training_configuration.number_of_epochs,
      checkpoint_name="final.pt",
    )

  @torch.no_grad()
  def log_to_tensorboard(self, metrics: Dict[str, float], global_step: int):
    """Log to tensorboard.

    This method logs some useful metrics and visualizations to tensorboard.

    Args:
      metrics: A dictionary mapping metric names to values.
      global_step: The current global step.
    """
    self.model.eval()
    if global_step % self.log_configuration.log_rate == 0:
      self.tensorboard_manager.log_metrics(
        metrics=metrics, global_step=global_step
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
      images = self.model.denoise(images)
      for step, images in enumerate(images[::-1]):
        self.tensorboard_manager.log_images(
          tag=f"Images at timestep {global_step}",
          images=self.reverse_transforms(images),
          timestep=step,
        )
