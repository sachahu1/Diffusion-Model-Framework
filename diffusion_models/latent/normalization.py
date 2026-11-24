from typing import Optional, Sequence, Union

import torch


class LatentNormalization:
  def __init__(
    self,
    scale: Union[float, Sequence[float], torch.Tensor],
    shift: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    whitening: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
  ):
    """``normalized = (latent - shift) * scale``, per channel or global.

    A diffusion model assumes zero-mean, unit-variance data, which the raw
    latents of an autoencoder are not. Channels can differ a lot in both.

    Args:
      scale: A float (all channels) or one value per channel.
      shift: The same for the mean to subtract; ``None`` subtracts nothing.
      whitening: An optional ``(C, C)`` matrix ``W`` applied across channels
        after the scale: ``normalized = W @ ((latent - shift) * scale)``. With
        the inverse square root of the channel covariance it decorrelates the
        channels and gives every direction unit variance.
    """
    self.scale = self._as_channels(scale)
    """``(C, 1, 1)`` or ``(1, 1, 1)``."""
    self.shift = (
      self._as_channels(shift) if shift is not None else torch.zeros(1, 1, 1)
    )

    self.whitening = (
      torch.as_tensor(whitening, dtype=torch.float32)
      if whitening is not None
      else None
    )
    self._unwhitening = (
      torch.linalg.inv(self.whitening) if self.whitening is not None else None
    )

  @staticmethod
  def _as_channels(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1, 1, 1)

  def normalize(self, latents: torch.Tensor) -> torch.Tensor:
    """Latents, ``(..., C, h, w)``, to what the diffusion model sees."""
    shift = self.shift.to(latents.device)
    scale = self.scale.to(latents.device)
    normalized = (latents - shift) * scale
    if self.whitening is not None:
      normalized = self._mix(self.whitening, normalized)
    return normalized

  @staticmethod
  def _mix(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,...chw->...ohw", matrix.to(values.device), values)

  def denormalize(self, samples: torch.Tensor) -> torch.Tensor:
    """What the diffusion model produces back to latents for the decoder."""
    shift = self.shift.to(samples.device)
    scale = self.scale.to(samples.device)
    if self._unwhitening is not None:
      samples = self._mix(self._unwhitening, samples)
    return samples / scale + shift

  def scale_values(self) -> Union[float, list]:
    """For checkpoints: a float for a global scale, else one value per channel."""
    values = self.scale.flatten().tolist()
    return values[0] if len(values) == 1 else values

  def shift_values(self) -> Optional[list]:
    """For checkpoints: the per-channel shifts, ``None`` when there is none."""
    if not bool(self.shift.any()):
      return None
    return self.shift.flatten().tolist()

  def whitening_values(self) -> Optional[list]:
    """For checkpoints: the whitening matrix as nested lists, or ``None``."""
    return self.whitening.tolist() if self.whitening is not None else None
