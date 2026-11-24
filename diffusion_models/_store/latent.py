from hydra_zen import builds, store
from omegaconf import MISSING

from diffusion_models.latent.dataset import LatentDataset

dataset_store = store(group="dataset")

# Cached latents, written by `diffusion-models encode`; give `dataset.path=<file>`.
dataset_store(
  builds(LatentDataset, path=MISSING, populate_full_signature=True),
  name="latents",
)
