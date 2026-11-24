from typing import Callable, Dict

import torch
from torch.nn import functional as F

RECONSTRUCTION_LOSSES: Dict[str, Callable[..., torch.Tensor]] = {
  "l1": F.l1_loss,
  "mse": F.mse_loss,
  "huber": F.huber_loss,
}
"""Element-wise regression losses, ``loss(prediction, target)``, by name."""
