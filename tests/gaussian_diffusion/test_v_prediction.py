import pytest
import torch

from diffusion_models.gaussian_diffusion.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.gaussian_diffusion.ddimm_diffuser import (
  DdimDiffuser,
  DenoisingMode,
)
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)


def _diffused(diffuser, batch=4):
  images = torch.rand(batch, 3, 8, 8) * 2 - 1
  noisy, noise, timesteps = diffuser.diffuse_batch(images)
  return images, noisy, noise, timesteps


def test_epsilon_target_is_the_noise():
  diffuser = DdimDiffuser(LinearBetaScheduler())
  images, _, noise, timesteps = _diffused(diffuser)
  assert torch.equal(diffuser.get_target(images, noise, timesteps), noise)


def test_v_target_roundtrips_to_epsilon_and_x0():
  diffuser = DdimDiffuser(LinearBetaScheduler(), prediction_type="v")
  images, noisy, noise, timesteps = _diffused(diffuser)
  v = diffuser.get_target(images, noise, timesteps)
  epsilon, x0 = diffuser.to_epsilon_and_x0(v, noisy, timesteps)
  assert torch.allclose(epsilon, noise, atol=1e-4)
  assert torch.allclose(x0, images, atol=1e-4)


def test_v_target_is_finite_at_zero_terminal_snr():
  diffuser = DdimDiffuser(LinearBetaScheduler(), prediction_type="v")
  images = torch.rand(2, 3, 8, 8)
  timesteps = torch.full((2,), diffuser.beta_scheduler.steps - 1)
  noisy, noise, _ = diffuser._diffuse_batch(images, timesteps)
  # At the last step the image is pure noise, so v == -x0.
  assert torch.allclose(
    diffuser.get_target(images, noise, timesteps), -images, atol=1e-4
  )


def test_trailing_steps_reach_the_last_timestep():
  diffuser = DdimDiffuser(
    LinearBetaScheduler(), mode=DenoisingMode.Trailing, number_of_steps=20
  )
  assert diffuser.steps[0] == 19  # first denoising step is the last index
  assert diffuser._time_steps[-1] == 999
  assert len(set(diffuser._time_steps)) == 20
  assert diffuser._time_steps.max() < diffuser.beta_scheduler.steps


@pytest.mark.parametrize("mode", list(DenoisingMode))
def test_oracle_v_model_recovers_the_image(mode):
  """A perfect v model, sampled deterministically, returns the clean image."""
  diffuser = DdimDiffuser(
    LinearBetaScheduler(),
    mode=mode,
    number_of_steps=10,
    prediction_type="v",
  )
  image = torch.rand(2, 3, 8, 8) * 2 - 1
  noise = torch.randn_like(image)
  alpha_bars = diffuser.beta_scheduler.alpha_bars

  def oracle(x, t):
    # The model knows the clean image: v = sqrt(ab) * eps - sqrt(1-ab) * x0,
    # with eps implied by the current x.
    ab = alpha_bars[t].reshape(-1, 1, 1, 1)
    eps = (x - torch.sqrt(ab) * image) / torch.sqrt(1 - ab)
    return torch.sqrt(ab) * eps - torch.sqrt(1 - ab) * image

  steps = diffuser.steps
  # Start from the diffused image at the first sampling timestep.
  t0 = torch.full((2,), int(diffuser._time_steps[steps[0]]))
  x, _, _ = diffuser._diffuse_batch(image, t0)
  for i in steps:
    x = diffuser._denoise_step(x, oracle, diffuser.get_timestep(2, i))
  # The sampler ends at timestep index 0 (alpha_bar = 0.9999), leaving ~1% noise.
  assert torch.allclose(x, image, atol=0.1)


def test_ddpm_v_step_matches_epsilon_step():
  scheduler = LinearBetaScheduler()
  eps_diffuser = GaussianDiffuser(scheduler)
  v_diffuser = GaussianDiffuser(scheduler, prediction_type="v")
  images, noisy, noise, _ = _diffused(eps_diffuser)
  timesteps = torch.full((4,), 500)
  noisy, noise, _ = eps_diffuser._diffuse_batch(images, timesteps)
  v = v_diffuser.get_target(images, noise, timesteps)
  step = lambda diffuser, out: diffuser._denoise_step(  # noqa: E731
    noisy, lambda *_: out, v_diffuser.get_timestep(4, 500)
  )
  torch.manual_seed(0)
  a = step(eps_diffuser, noise)
  torch.manual_seed(0)
  b = step(v_diffuser, v)
  assert torch.allclose(a, b, atol=1e-4)


def test_enums_accept_names_from_a_saved_config():
  # Checkpoints embed the config with enums spelled by member name.
  diffuser = DdimDiffuser(
    LinearBetaScheduler(), mode="Trailing", prediction_type="V"
  )
  assert diffuser.mode is DenoisingMode.Trailing
  assert diffuser.prediction_type.value == "v"
  assert DdimDiffuser(LinearBetaScheduler(), mode="Quadratic").mode is (
    DenoisingMode.Quadratic
  )
