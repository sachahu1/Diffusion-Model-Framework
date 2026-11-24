import pytest
import torch

from diffusion_models.diffusion.diffusers.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.diffusion.diffusers.ddimm_diffuser import (
  DdimDiffuser,
  DenoisingMode,
)
from diffusion_models.diffusion.diffusers.gaussian_diffuser import (
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
    LinearBetaScheduler(),
    mode=DenoisingMode.Trailing,
    number_of_steps=20,
    prediction_type="v",
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
    # An oracle model: v = sqrt(ab) * eps - sqrt(1-ab) * x0.
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


def _scheduler():
  return LinearBetaScheduler(1e-4, 0.02, 1000, enforce_zero_terminal_snr=True)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v"])
@pytest.mark.parametrize("number_of_steps", [10, 20, 50])
def test_quadratic_steps_are_strictly_increasing(
  prediction_type, number_of_steps
):
  diffuser = DdimDiffuser(
    _scheduler(),
    mode=DenoisingMode.Quadratic,
    number_of_steps=number_of_steps,
    prediction_type=prediction_type,
  )
  diffuser.steps
  assert len(diffuser._time_steps) == number_of_steps
  assert (diffuser._time_steps > diffuser._time_steps_prev).all()
  assert (diffuser._time_steps[1:] > diffuser._time_steps[:-1]).all()


def test_epsilon_quadratic_start_is_unchanged():
  diffuser = DdimDiffuser(
    _scheduler(), mode=DenoisingMode.Quadratic, number_of_steps=50
  )
  diffuser.steps
  assert diffuser._time_steps[-1] == 801
  assert list(diffuser._time_steps[:5]) == [1, 2, 3, 4, 6]
  assert list(diffuser._time_steps[4:]) == [
    6, 9, 12, 17, 22, 27, 34, 41, 48, 57, 66, 75, 86, 97, 108, 121, 134, 147,
    162, 177, 192, 209, 226, 243, 262, 281, 300, 321, 342, 363, 386, 409, 432,
    457, 482, 507, 534, 561, 588, 617, 646, 675, 706, 737, 768, 801,
  ]  # fmt: skip


def test_epsilon_linear_start_is_unchanged():
  diffuser = DdimDiffuser(
    _scheduler(), mode=DenoisingMode.Linear, number_of_steps=50
  )
  diffuser.steps
  assert diffuser._time_steps[-1] == 981


@pytest.mark.parametrize(
  "mode", [DenoisingMode.Linear, DenoisingMode.Quadratic]
)
def test_v_prediction_starts_at_the_last_timestep(mode):
  diffuser = DdimDiffuser(
    _scheduler(), mode=mode, number_of_steps=50, prediction_type="v"
  )
  diffuser.steps
  assert diffuser._time_steps[-1] == 999
  assert diffuser._time_steps[0] == 1


@pytest.mark.parametrize("mode", list(DenoisingMode))
@pytest.mark.parametrize("prediction_type", ["epsilon", "v"])
def test_last_sampled_step_ends_at_previous_zero(mode, prediction_type):
  if mode is DenoisingMode.Trailing and prediction_type == "epsilon":
    pytest.skip("raises, see test below")
  diffuser = DdimDiffuser(
    _scheduler(),
    mode=mode,
    number_of_steps=20,
    prediction_type=prediction_type,
  )
  steps = diffuser.steps
  assert diffuser.get_timestep(1, steps[-1]).previous.item() == 0


def test_trailing_with_epsilon_raises_instead_of_returning_nan():
  diffuser = DdimDiffuser(
    _scheduler(), mode=DenoisingMode.Trailing, number_of_steps=10
  )
  with pytest.raises(ValueError, match="prediction_type V"):
    diffuser.denoise_batch(torch.randn(1, 3, 8, 8), lambda x, t: x)


def test_eta_defaults_to_deterministic_and_is_validated():
  assert DdimDiffuser(_scheduler()).eta == 0.0
  with pytest.raises(ValueError, match="eta"):
    DdimDiffuser(_scheduler(), eta=-0.1)


def test_eta_makes_sampling_stochastic_and_zero_keeps_it_deterministic():
  class Model:
    def __call__(self, x, t):
      return torch.zeros_like(x)

  start = torch.randn(2, 1, 4, 4)

  def sample(eta, seed):
    diffuser = DdimDiffuser(
      _scheduler(), number_of_steps=5, clip_value=None, eta=eta
    )
    torch.manual_seed(seed)  # only the step noise depends on the seed
    out = start
    for i in diffuser.steps:
      out = diffuser._denoise_step(
        out, Model(), diffuser.get_timestep(2, i), eta=diffuser.eta
      )
    return out

  assert torch.equal(sample(0.0, 0), sample(0.0, 1))
  assert not torch.equal(sample(1.0, 0), sample(1.0, 1))


def test_timestep_power_skews_training_timesteps_towards_low_noise():
  scheduler = LinearBetaScheduler(steps=1000)
  uniform = DdimDiffuser(scheduler)
  skewed = DdimDiffuser(scheduler, timestep_power=2.0)
  images = torch.zeros(20000, 1, 2, 2)
  _, _, t_uniform = uniform.diffuse_batch(images)
  _, _, t_skewed = skewed.diffuse_batch(images)
  assert 0.45 < (t_uniform < 500).float().mean() < 0.55
  assert 0.65 < (t_skewed < 500).float().mean() < 0.75
  assert t_skewed.max() <= 999
  with pytest.raises(ValueError):
    DdimDiffuser(scheduler, timestep_power=0.0)
