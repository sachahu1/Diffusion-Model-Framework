import torch
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms

from diffusion_models.diffusion_trainer import DiffusionTrainer
from diffusion_models.gaussian_diffusion.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import LogConfiguration
from diffusion_models.utils.schemas import TrainingConfiguration


if __name__ == "__main__":
  image_size = 64
  image_channels = 3

  training_configuration = TrainingConfiguration(
    batch_size=128,
    learning_rate=2 * 10e-4,
    number_of_epochs=500,
    training_name="SimpleTraining",
    checkpoint_rate=100,
  )
  log_configuration = LogConfiguration(
    log_rate=10,
    image_rate=5000,
    number_of_images=5,
  )
  beta_scheduler = LinearBetaScheduler(
    beta_start=0.0001,
    beta_end=0.02,
    steps=1000,
  )
  model = SimpleUnet(image_channels=image_channels)
  print("Num params: ", sum(p.numel() for p in model.parameters()))

  model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

  # Define Image Transforms and Reverse Transforms
  image_transforms = transforms.Compose(
    [
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: (x * 2) - 1),
    ]
  )
  reverse_transforms = transforms.Compose(
    [
      transforms.Lambda(lambda x: (x + 1) / 2),
      transforms.Resize((128, 128)),
    ]
  )

  # Define Dataset
  dataset = datasets.CelebA(
    root="../data", download=False, transform=image_transforms, split="train"
  )

  # Instantiate DiffusionTrainer
  trainer = DiffusionTrainer(
    model=model,
    dataset=dataset,
    optimizer=Adam(
      model.parameters(), lr=training_configuration.learning_rate
    ),
    training_configuration=training_configuration,
    loss_function=F.l1_loss,
    beta_scheduler=beta_scheduler,
    scheduler=None,
    log_configuration=log_configuration,
    reverse_transforms=reverse_transforms,
    device="cuda",
  )

  # Launch training
  trainer.train()
