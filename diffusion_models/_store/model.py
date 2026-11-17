from hydra_zen import builds, store

from diffusion_models.models.SimpleUnet import SimpleUnet

model_store = store(group="model")

model_store(
  builds(
    SimpleUnet,
    diffuser="${diffuser}",
    image_channels="${image_channels}",
  ),
  name="simple_unet",
)
