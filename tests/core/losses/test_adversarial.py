import pytest
import torch

from diffusion_models.core.losses import (
  adaptive_weight,
  hinge_discriminator_loss,
  hinge_generator_loss,
)


def test_discriminator_hinge_is_zero_when_it_is_confident_and_right():
  real, fake = torch.full((4,), 2.0), torch.full((4,), -2.0)
  assert hinge_discriminator_loss(real, fake).item() == 0.0


def test_discriminator_hinge_penalises_wrong_and_unsure_logits():
  assert hinge_discriminator_loss(
    torch.zeros(4), torch.zeros(4)
  ).item() == pytest.approx(1.0)  # 0.5 * (1 + 1)
  assert hinge_discriminator_loss(
    torch.full((4,), -1.0), torch.full((4,), 1.0)
  ).item() == pytest.approx(2.0)  # 0.5 * (2 + 2)


def test_generator_hinge_rewards_high_logits():
  assert hinge_generator_loss(torch.full((3,), 2.0)).item() == -2.0
  assert hinge_generator_loss(torch.full((3,), -1.0)).item() == 1.0


def test_adaptive_weight_equalises_the_gradient_norms():
  last_layer = torch.nn.Parameter(torch.randn(5))
  reconstruction = (last_layer * 3.0).sum()  # gradient norm 3 * sqrt(5)
  adversarial = (last_layer * 0.5).sum()  # gradient norm 0.5 * sqrt(5)
  weight = adaptive_weight(reconstruction, adversarial, last_layer)
  assert weight.item() == pytest.approx(6.0, rel=1e-3)
  assert not weight.requires_grad
  # After weighting, both gradients have the same norm.
  scaled = torch.autograd.grad(weight * adversarial, last_layer)[0].norm()
  assert scaled.item() == pytest.approx(
    torch.autograd.grad(reconstruction, last_layer)[0].norm().item(), rel=1e-3
  )


def test_adaptive_weight_is_bounded():
  last_layer = torch.nn.Parameter(torch.ones(3))
  weight = adaptive_weight(
    (last_layer * 1e6).sum(),
    (last_layer * 1e-6).sum(),
    last_layer,
    max_weight=50.0,
  )
  assert weight.item() == 50.0
  # The graph is still usable afterwards (retain_graph).
  loss = (last_layer * 2.0).sum()
  adaptive_weight(loss, loss, last_layer)
  loss.backward()
