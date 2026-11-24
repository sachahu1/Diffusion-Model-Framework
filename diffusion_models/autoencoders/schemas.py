from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationConfiguration:
  """Settings of the validation pass of an autoencoder training run."""

  rate: Optional[int] = 1
  """The rate (in epochs) at which the validation runs. Disabled when None."""
  number_of_samples: int = 2000
  """The number of (evenly spaced) validation images used."""
  batch_size: int = 64
  """The validation batch size."""
  num_workers: int = 2
  """Dataloader workers used to read the validation images."""
  fid_rate: Optional[int] = None
  """The rate (in epochs) at which the reconstruction FID is computed and logged
  (``Val/FID``). Needs the ``evaluation`` extra. Disabled when None."""
  fid_samples: int = 2000
  """The number of validation images used for the reconstruction FID."""


@dataclass
class AdversarialConfiguration:
  """Settings of the adversarial (PatchGAN) fine-tuning of an autoencoder."""

  weight: float = 0.5
  """The weight of the adversarial term (multiplies the adaptive weight)."""
  start_step: int = 0
  """The adversarial term only reaches the autoencoder from this step on. Until
  then the discriminator trains alone against the (unchanged) reconstructions,
  so it can learn before the autoencoder starts to exploit it."""
  warmup_steps: int = 1000
  """After ``start_step`` the adversarial weight ramps up linearly over this
  many steps."""
  adaptive: bool = True
  """Balance the reconstruction and adversarial gradients at the decoder's last
  layer, so the adversarial term cannot swamp the reconstruction."""
  max_adaptive_weight: float = 1e4
  """The upper bound of the adaptive weight."""
  latent_noise: float = 0.0
  """Robustness of the decoder to imperfect latents (decoder-only fine-tune).
  Every sampled latent gets Gaussian noise with a standard deviation of
  ``u * latent_noise`` times the batch's per-channel latent std, with ``u``
  drawn uniformly from [0, 1] per image. The target stays the clean image, so
  the decoder learns to map the off-manifold latents a diffusion model produces
  to plausible images. ``0`` disables it."""
  trainable: str = "decoder"
  """``"decoder"`` fine-tunes the decoder only (the encoder is frozen, so the
  latents of the earlier stage stay valid); ``"all"`` trains the whole VAE."""
