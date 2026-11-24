import pytest
import torch

from diffusion_models.core.losses import (
  RECONSTRUCTION_LOSSES,
  LossOutput,
  VaeLoss,
  gaussian_kl_divergence,
)


def test_kl_is_zero_for_a_standard_normal():
  zeros = torch.zeros(3, 4, 2, 2)
  assert gaussian_kl_divergence(zeros, zeros).item() == pytest.approx(0.0)


def test_kl_sums_over_dimensions_and_averages_over_the_batch():
  mean = torch.full((2, 1, 1, 1), 2.0)  # 0.5 * 2**2 = 2 per sample
  assert gaussian_kl_divergence(mean, torch.zeros_like(mean)).item() == (
    pytest.approx(2.0)
  )
  mean = torch.full((2, 3), 2.0)  # three dimensions per sample
  assert gaussian_kl_divergence(mean, torch.zeros_like(mean)).item() == (
    pytest.approx(6.0)
  )


def test_perfect_reconstruction_with_a_standard_normal_posterior_is_free():
  images = torch.rand(2, 3, 8, 8)
  zeros = torch.zeros(2, 4, 2, 2)
  loss = VaeLoss(kl_weight=1.0)(images, images, zeros, zeros.clone())
  assert isinstance(loss, LossOutput)
  assert loss.loss.item() == pytest.approx(0.0)
  assert set(loss.metrics) == {"Loss", "Reconstruction", "KL"}


def test_vae_kl_is_normalised_by_the_number_of_image_values():
  images = torch.zeros(1, 3, 4, 4)  # 48 values
  mean = torch.ones(1, 1, 1, 1) * 2.0  # KL = 2
  loss = VaeLoss(kl_weight=1.0)(images, images, mean, torch.zeros_like(mean))
  assert loss.metrics["KL"].item() == pytest.approx(2.0 / 48)
  assert loss.loss.item() == pytest.approx(2.0 / 48)


def test_vae_without_a_variance_has_no_kl_term():
  images = torch.zeros(1, 3, 4, 4)
  loss = VaeLoss(kl_weight=1.0)(images, images, torch.ones(1, 1, 1, 1))
  assert loss.metrics["KL"].item() == 0.0


@pytest.mark.parametrize(
  "name,expected", [("l1", 0.5), ("mse", 0.25), ("huber", 0.125)]
)
def test_reconstruction_terms(name, expected):
  images = torch.zeros(1, 3, 4, 4)
  reconstruction = torch.full_like(images, 0.5)
  z = torch.zeros(1, 1, 1, 1)
  loss = VaeLoss(kl_weight=0.0, reconstruction_loss=name)(
    images, reconstruction, z, z
  )
  assert loss.loss.item() == pytest.approx(expected)


def test_unknown_reconstruction_loss_is_rejected():
  with pytest.raises(ValueError, match="reconstruction_loss"):
    VaeLoss(reconstruction_loss="lpips")


def test_regression_losses_are_symmetric_callables():
  a, b = torch.randn(4, 3), torch.randn(4, 3)
  for loss in RECONSTRUCTION_LOSSES.values():
    assert loss(a, b).item() == pytest.approx(loss(b, a).item())
