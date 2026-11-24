from hydra_zen import builds, store

from diffusion_models.diffusion.diffusers.ddimm_diffuser import DdimDiffuser
from diffusion_models.diffusion.diffusers.gaussian_diffuser import (
  GaussianDiffuser,
)

diffuser_store = store(group="diffuser")

diffuser_store(
  builds(
    DdimDiffuser,
    beta_scheduler="${beta_scheduler}",
    populate_full_signature=True,
  ),
  name="ddim",
)
diffuser_store(
  builds(
    GaussianDiffuser,
    beta_scheduler="${beta_scheduler}",
    populate_full_signature=True,
  ),
  name="ddpm",
)
