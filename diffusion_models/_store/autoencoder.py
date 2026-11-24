from hydra_zen import builds, store
from omegaconf import MISSING

from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.autoencoders.vae import ConvVae

autoencoder_store = store(group="autoencoder")

autoencoder_store(
  builds(
    ConvVae,
    image_channels="${image_channels}",
    populate_full_signature=True,
  ),
  name="conv_vae",
)

# An autoencoder rebuilt from ``autoencoder.checkpoint``, to fine-tune.
autoencoder_store(
  builds(
    BaseAutoencoder.load,
    checkpoint=MISSING,
    use_ema=True,
    populate_full_signature=True,
  ),
  name="pretrained",
)
