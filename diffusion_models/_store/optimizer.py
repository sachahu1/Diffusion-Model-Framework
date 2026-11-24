from hydra_zen import builds, store
from torch.optim import AdamW

from diffusion_models.core.utils.adopt import ADOPT

optimizer_store = store(group="optimizer")

# Plain containers keep optimizer state loadable with ``weights_only=True``.
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
