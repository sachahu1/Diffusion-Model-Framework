from hydra_zen import builds, store
from torch.nn import functional as F

from diffusion_models.core.losses import VaeLoss

loss_store = store(group="loss")

# Regression losses ``loss(prediction, target)``.
loss_store(builds(F.mse_loss, zen_partial=True), name="mse")
loss_store(builds(F.l1_loss, zen_partial=True), name="l1")
loss_store(builds(F.huber_loss, delta=1.0, zen_partial=True), name="huber")

# The autoencoder objective (reconstruction + KL).
loss_store(builds(VaeLoss, populate_full_signature=True), name="vae")
