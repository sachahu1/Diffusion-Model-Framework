from typing import Dict, NamedTuple

import torch


class LossOutput(NamedTuple):
  """A loss and its components, for logging."""

  loss: torch.Tensor
  """The scalar to backpropagate."""
  metrics: Dict[str, torch.Tensor]
  """Detached scalars, logged to tensorboard."""
