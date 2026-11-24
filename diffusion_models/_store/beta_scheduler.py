from hydra_zen import builds, store

from diffusion_models.diffusion.diffusers.beta_schedulers import (
  CosineBetaScheduler,
  LinearBetaScheduler,
)

beta_scheduler_store = store(group="beta_scheduler")

beta_scheduler_store(
  builds(LinearBetaScheduler, populate_full_signature=True), name="linear"
)
beta_scheduler_store(
  builds(CosineBetaScheduler, populate_full_signature=True), name="cosine"
)
