import torch


def hinge_discriminator_loss(
  real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
  """The hinge loss of the discriminator: real logits above +1, fake below -1."""
  return 0.5 * (
    torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
  )


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
  """The hinge loss of the generator: raise the logits of its fake images."""
  return -fake_logits.mean()


def adaptive_weight(
  reconstruction_loss: torch.Tensor,
  adversarial_loss: torch.Tensor,
  last_layer: torch.Tensor,
  max_weight: float = 1e4,
  epsilon: float = 1e-4,
) -> torch.Tensor:
  """Balance the two losses by the norm of their gradients at ``last_layer``.

  As in VQGAN: the returned weight makes the adversarial gradient at the
  decoder's output layer as large as the reconstruction gradient, so the
  adversarial term can sharpen the images without overwhelming the
  reconstruction.

  Args:
    reconstruction_loss: The reconstruction (nll) loss.
    adversarial_loss: The generator's adversarial loss.
    last_layer: The weight of the decoder's output layer.
    max_weight: The upper bound of the weight.
    epsilon: Stabilises the division.

  Returns:
    A detached scalar in ``[0, max_weight]``.
  """
  reconstruction_gradient = torch.autograd.grad(
    reconstruction_loss, last_layer, retain_graph=True
  )[0]
  adversarial_gradient = torch.autograd.grad(
    adversarial_loss, last_layer, retain_graph=True
  )[0]
  weight = reconstruction_gradient.norm() / (
    adversarial_gradient.norm() + epsilon
  )
  return weight.clamp(0.0, max_weight).detach()
