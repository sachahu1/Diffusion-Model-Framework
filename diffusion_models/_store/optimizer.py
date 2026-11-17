from hydra_zen import builds, store
from torch.optim import AdamW

from diffusion_models.utils.adopt import ADOPT

optimizer_store = store(group="optimizer")

# Partials: the model parameters are only known at runtime.
# ``hydra_convert="all"`` yields plain tuples/lists instead of OmegaConf
# containers, keeping optimizer state loadable with ``weights_only=True``.
optimizer_store(
  builds(
    AdamW,
    lr=2e-4,
    populate_full_signature=True,
    zen_partial=True,
    hydra_convert="all",
  ),
  name="adamw",
)
optimizer_store(
  builds(
    ADOPT,
    lr=2e-4,
    populate_full_signature=True,
    zen_partial=True,
    hydra_convert="all",
  ),
  name="adopt",
)
