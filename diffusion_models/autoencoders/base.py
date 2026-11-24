import abc
import pathlib
from typing import Iterator, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn


class AutoencoderOutput(NamedTuple):
  """The result of an autoencoder forward pass."""

  reconstruction: torch.Tensor
  """The reconstructed images, same shape as the input."""
  mean: torch.Tensor
  """The mean of the latent posterior, ``(B, latent_channels, h, w)``."""
  logvar: Optional[torch.Tensor]
  """The log variance of the latent posterior (``None`` for a deterministic autoencoder)."""


class BaseAutoencoder(nn.Module, abc.ABC):
  """The interface every autoencoder of the framework implements.

  A latent diffusion model only needs :meth:`encode` and :meth:`decode`
  (plus :attr:`latent_channels` and :attr:`downsample_factor` to size its
  UNet), so any architecture that fulfils this interface can be swapped in.
  Images are in ``[-1, 1]``.
  """

  latent_channels: int
  """The number of channels of the latent."""
  downsample_factor: int
  """The spatial downsampling between images and latents."""

  @abc.abstractmethod
  def encode_distribution(
    self, images: torch.Tensor
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode images into the parameters of the latent posterior.

    Args:
      images: A batch of images, ``(B, C, H, W)``.

    Returns:
      ``(mean, logvar)``. ``logvar`` is ``None`` for deterministic encoders.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def decode(self, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents back to images.

    Args:
      latents: A batch of latents, ``(B, latent_channels, h, w)``.

    Returns:
      A batch of images, unclamped.
    """
    raise NotImplementedError

  def encode(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """Encode images into latents.

    Args:
      images: A batch of images.
      sample: Draw from the posterior instead of taking its mean.
    """
    mean, logvar = self.encode_distribution(images)
    if sample and logvar is not None:
      return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    return mean

  def forward(
    self, images: torch.Tensor, sample_posterior: bool = True
  ) -> AutoencoderOutput:
    """Encode then decode ``images``.

    Args:
      images: A batch of images.
      sample_posterior: Whether to sample the latent (training) or use the
        posterior mean (reconstruction).
    """
    mean, logvar = self.encode_distribution(images)
    latents = mean
    if sample_posterior and logvar is not None:
      latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    return AutoencoderOutput(self.decode(latents), mean, logvar)

  def decoder_parameters(self) -> Iterator[nn.Parameter]:
    """The parameters of the decoder (what a decoder-only fine-tune trains)."""
    raise NotImplementedError

  def last_layer(self) -> nn.Parameter:
    """The weight of the decoder's output layer.

    The adversarial loss balances the reconstruction and the adversarial
    gradients at this layer.
    """
    raise NotImplementedError

  @classmethod
  def load(
    cls,
    checkpoint: Union[str, pathlib.Path],
    use_ema: bool = True,
    map_location: str = "cpu",
  ) -> "BaseAutoencoder":
    """Rebuild an autoencoder from a training checkpoint.

    The architecture comes from the configuration stored in the checkpoint, so
    only the file is needed. This is how a later stage starts from an earlier
    one: ``autoencoder.checkpoint=<path>`` with the ``pretrained`` option of the
    ``autoencoder`` config group.

    Args:
      checkpoint: A checkpoint written by ``diffusion-models train autoencoder``.
      use_ema: Use the EMA weights when the checkpoint has them.
      map_location: Where to load the weights.

    Returns:
      The autoencoder with the checkpoint's weights, in training mode.
    """
    from hydra_zen import instantiate
    from omegaconf import OmegaConf

    from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint

    state = AutoencoderCheckpoint.from_file(
      checkpoint, map_location=map_location
    )
    if state.config_yaml is None:
      raise ValueError(
        f"{checkpoint} has no embedded config, so the model cannot be rebuilt."
      )
    model = instantiate(OmegaConf.create(state.config_yaml).autoencoder)
    if not isinstance(model, cls):
      raise TypeError(
        f"{checkpoint} holds a {type(model).__name__}, not a {cls.__name__}."
      )
    model.load_state_dict(state.get_model_state_dict(use_ema=use_ema))
    return model

  def latent_size(self, image_size: int) -> int:
    """The spatial size of the latent of a square image of ``image_size``."""
    return image_size // self.downsample_factor
