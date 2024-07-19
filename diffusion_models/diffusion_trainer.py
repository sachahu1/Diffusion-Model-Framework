import pathlib
from typing import Callable
from typing import Dict
from typing import Optional

import torch
from torch import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from diffusion_models.gaussian_diffusion.beta_schedulers import (
  BaseBetaScheduler,
)
from diffusion_models.gaussian_diffusion.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.utils.schemas import Checkpoint
from diffusion_models.utils.schemas import LogConfiguration
from diffusion_models.utils.schemas import TrainingConfiguration
from diffusion_models.utils.tensorboard import TensorboardManager


class DiffusionTrainer:
  def __init__(
    self,
    model: torch.nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
    loss_function: Callable = F.l1_loss,
    beta_scheduler: BaseBetaScheduler = LinearBetaScheduler(),
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    log_configuration: LogConfiguration = LogConfiguration(),
    reverse_transforms: Callable = lambda x: x,
    device: str = "cuda",
  ):
    self.model = model.to(device)
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.training_configuration = training_configuration
    self.scheduler = scheduler
    self.device = device

    self.beta_scheduler = beta_scheduler
    self.gaussian_diffuser = GaussianDiffuser(
      beta_scheduler=beta_scheduler,
    ).to(device)

    self.dataloader = DataLoader(
      dataset=dataset,
      batch_size=training_configuration.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=16,
      pin_memory=True,
      persistent_workers=True,
    )

    self._image_shape = dataset[0][0].shape

    self.scaler = torch.cuda.amp.GradScaler()

    self.log_configuration = log_configuration

    self.checkpoint_path = (
      pathlib.Path("../checkpoints")
      / self.training_configuration.training_name
    )

    self.checkpoint_path.mkdir(exist_ok=True)
    self.tensorboard_manager = TensorboardManager(
      log_name=self.training_configuration.training_name,
    )

    self.reverse_transforms = reverse_transforms

  def save_checkpoint(self, epoch: int, checkpoint_name: str):
    checkpoint = Checkpoint(
      epoch=epoch,
      model_state_dict=self.model.state_dict(),
      optimizer_state_dict=self.optimizer.state_dict(),
      scaler=self.scaler.state_dict()
      if self.training_configuration.mixed_precision_training
      else None,
      tensorboard_run_name=self.tensorboard_manager.summary_writer.log_dir,
    )
    checkpoint.to_file(self.checkpoint_path / checkpoint_name)

  def train(self):
    self.model.train()
    for epoch in range(self.training_configuration.number_of_epochs):
      for step, batch in enumerate(
        tqdm(self.dataloader, desc=f"Epoch={epoch}")
      ):
        global_step = epoch * len(self.dataloader) + step

        images, _ = batch
        images = images.to(self.device)

        noisy_images, noise, timesteps = self.gaussian_diffuser.diffuse_batch(
          images=images
        )

        with amp.autocast(
          device_type=self.device,
          enabled=self.training_configuration.mixed_precision_training,
        ):
          prediction = self.model(noisy_images, timesteps)
          loss = self.loss_function(noise, prediction)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

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
      images = self.gaussian_diffuser.denoise_batch(images, self.model)
      for step, images in enumerate(images[::-1]):
        self.tensorboard_manager.log_images(
          tag=f"Images at timestep {global_step}",
          images=self.reverse_transforms(images),
          timestep=step,
        )
