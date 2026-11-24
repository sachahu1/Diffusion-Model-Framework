import torch


def gaussian_kl_divergence(
  mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
  """KL divergence of a diagonal Gaussian from a standard normal.

  Args:
    mean: The mean, ``(B, ...)``.
    logvar: The log variance, same shape.

  Returns:
    The KL of every sample summed over its dimensions, averaged over the
    batch: a scalar.
  """
  kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
  return kl.flatten(1).sum(dim=1).mean()
