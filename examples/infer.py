import torch
from torchvision import transforms

from diffusion_models.gaussian_diffusion.beta_schedulers import \
  LinearBetaScheduler
from diffusion_models.gaussian_diffusion.gaussian_diffuser import \
  DiffusionInference, GaussianDiffuser
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import Checkpoint

if __name__ == '__main__':

  checkpoint_file_path = "../checkpoints/SimpleTraining/model_50.pt"
  image_channels = 3

  gaussian_diffuser = GaussianDiffuser(
    beta_scheduler=LinearBetaScheduler(
      beta_start=0.0001,
      beta_end=0.02,
      steps=1000,
    )
  )

  model = SimpleUnet(
    image_channels=image_channels,
  )
  model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

  checkpoint = Checkpoint.from_file(checkpoint_file_path)

  model.load_state_dict(checkpoint.model_state_dict)

  reverse_transforms = transforms.Compose(
    [
      transforms.Lambda(lambda x: (x + 1) / 2),
      transforms.Resize((128, 128)),
    ]
  )

  inference = DiffusionInference(
    gaussian_diffuser=gaussian_diffuser,
    model=model,
    reverse_transforms=reverse_transforms,
    device="cuda"
  )
  inference.generate(number_of_images=25, save_gif=True)
