from hydra_zen import builds, store

from diffusion_models.core.modules import PatchDiscriminator

discriminator_store = store(group="discriminator")

discriminator_store(
  builds(
    PatchDiscriminator,
    image_channels="${image_channels}",
    populate_full_signature=True,
  ),
  name="patch",
)
