from hydra_zen import builds, store

from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.models.SmallAttentionUnet import SmallAttentionUnet

model_store = store(group="model")

model_store(
  builds(
    SimpleUnet,
    diffuser="${diffuser}",
    image_channels="${image_channels}",
  ),
  name="simple_unet",
)

model_store(
  builds(
    SmallAttentionUnet,
    diffuser="${diffuser}",
    image_channels="${image_channels}",
    populate_full_signature=True,
    hydra_convert="all",
  ),
  name="small_attention_unet",
)
