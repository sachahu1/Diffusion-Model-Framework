import pathlib

import pytest
import torch
from click.testing import CliRunner
from omegaconf import OmegaConf

from diffusion_models import cli
from diffusion_models.diffusion.diffusers.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.diffusion.inference import generate_from_checkpoint
from diffusion_models.diffusion.models.SmallAttentionUnet import (
  SmallAttentionUnet,
)
from diffusion_models.diffusion.diffusers.ddimm_diffuser import DdimDiffuser
from diffusion_models.diffusion.schemas import (
  BetaSchedulerConfiguration,
  Checkpoint,
)


def _model(channels: int = 3):
  return SmallAttentionUnet(
    diffuser=DdimDiffuser(
      LinearBetaScheduler(), number_of_steps=2, clip_value=1.0
    ),
    image_channels=channels,
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    num_groups=8,
  )


def _checkpoint(path: pathlib.Path, with_config: bool = True) -> pathlib.Path:
  model = _model()
  scheduler = LinearBetaScheduler()
  configuration = {
    "image_size": 8,
    "image_channels": 3,
    "reverse_image_size": 8,
    "model": {
      "_target_": "diffusion_models.diffusion.models.SmallAttentionUnet.SmallAttentionUnet",
      "_convert_": "all",
      "diffuser": {
        "_target_": "diffusion_models.diffusion.diffusers.ddimm_diffuser.DdimDiffuser",
        "beta_scheduler": {
          "_target_": "diffusion_models.diffusion.diffusers.beta_schedulers.LinearBetaScheduler"
        },
        "number_of_steps": 20,
        "clip_value": 1.0,
      },
      "image_channels": 3,
      "base_channels": 32,
      "channel_multipliers": [1, 2],
      "num_res_blocks": 1,
      "num_groups": 8,
    },
  }
  Checkpoint(
    epoch=1,
    model_state_dict=model.state_dict(),
    optimizer_state_dict={},
    beta_scheduler_config=BetaSchedulerConfiguration(
      steps=scheduler.steps,
      betas=scheduler.betas,
      alpha_bars=scheduler.alpha_bars,
    ),
    config_yaml=OmegaConf.to_yaml(OmegaConf.create(configuration))
    if with_config
    else None,
  ).to_file(path)
  return path


def test_images_are_generated_from_the_config_in_the_checkpoint(tmp_path):
  checkpoint = _checkpoint(tmp_path / "model.pt")
  generate_from_checkpoint(
    checkpoint,
    output_directory=tmp_path / "out",
    number_of_images=4,
    steps=2,
    eta=1.0,
    device="cpu",
    seed=0,
  )
  assert (tmp_path / "out" / "generated.png").exists()
  log = (tmp_path / "out" / "inference.log").read_text()
  assert "Generating 4 images" in log and "Saved the images" in log
  assert "eta: 1.0" in (tmp_path / "out" / "options.yaml").read_text()


def test_a_checkpoint_without_config_is_refused(tmp_path):
  checkpoint = _checkpoint(tmp_path / "model.pt", with_config=False)
  with pytest.raises(ValueError, match="no embedded config"):
    generate_from_checkpoint(
      checkpoint, output_directory=tmp_path / "out", device="cpu"
    )


@pytest.mark.parametrize(
  "overrides,expected",
  [
    (("checkpoint=a.pt",), {"checkpoint": "a.pt"}),
    (
      ("checkpoint=a.pt", "steps=50", "eta=1.0", "use_ema=false"),
      {"checkpoint": "a.pt", "steps": 50, "eta": 1.0, "use_ema": False},
    ),
    (("steps=50",), None),  # no checkpoint
    (("checkpoint=a.pt", "+config=x"), None),
    (("checkpoint=a.pt", "autoencoder.checkpoint=b.pt"), None),
    (("checkpoint=a.pt", "steps=many"), None),
    (("--help",), None),
  ],
)
def test_only_simple_overrides_use_the_checkpoint_config(overrides, expected):
  assert cli._simple_infer_options(overrides) == expected


def test_infer_uses_the_checkpoint_config_for_simple_overrides(
  tmp_path, monkeypatch
):
  checkpoint = _checkpoint(tmp_path / "model.pt")
  monkeypatch.chdir(tmp_path)
  hydra_calls = []
  monkeypatch.setattr(
    cli, "_run_hydra", lambda *args: hydra_calls.append(args)
  )
  result = CliRunner().invoke(
    cli.main,
    ["infer", f"checkpoint={checkpoint}", "number_of_images=4", "device=cpu"],
  )
  assert result.exit_code == 0, result.output
  assert not hydra_calls
  assert list((tmp_path / "outputs" / "infer").glob("*/generated.png"))


def test_infer_composes_the_hydra_config_for_other_overrides(
  tmp_path, monkeypatch
):
  checkpoint = _checkpoint(tmp_path / "model.pt")
  hydra_calls = []
  monkeypatch.setattr(
    cli, "_run_hydra", lambda *args: hydra_calls.append(args)
  )
  CliRunner().invoke(
    cli.main,
    ["infer", f"checkpoint={checkpoint}", "+config=small_attention_ema"],
  )
  assert hydra_calls and hydra_calls[0][3] == (
    f"checkpoint={checkpoint}",
    "+config=small_attention_ema",
  )


def test_infer_falls_back_to_hydra_without_a_stored_config(
  tmp_path, monkeypatch
):
  checkpoint = _checkpoint(tmp_path / "model.pt", with_config=False)
  hydra_calls = []
  monkeypatch.setattr(
    cli, "_run_hydra", lambda *args: hydra_calls.append(args)
  )
  CliRunner().invoke(cli.main, ["infer", f"checkpoint={checkpoint}"])
  assert hydra_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_checkpoint_is_read_on_the_cpu_not_the_gpu(tmp_path, monkeypatch):
  calls = []
  original = Checkpoint.from_file.__func__

  def spy(cls, file_path, map_location=None, **options):
    calls.append(map_location)
    return original(cls, file_path, map_location, **options)

  monkeypatch.setattr(Checkpoint, "from_file", classmethod(spy))
  generate_from_checkpoint(
    _checkpoint(tmp_path / "model.pt"),
    tmp_path / "out",
    number_of_images=4,
    device="cuda",
  )
  assert calls and all(location == "cpu" for location in calls)


def test_inference_does_not_load_the_optimizer_state(tmp_path):
  path = _checkpoint(tmp_path / "model.pt")
  full = Checkpoint.from_file(str(path))
  light = Checkpoint.from_file(str(path), load_optimizer=False)
  assert light.optimizer_state_dict == {}
  assert light.scaler is None
  assert light.model_state_dict.keys() == full.model_state_dict.keys()
  assert light.completed_epochs == full.completed_epochs


def test_stepwise_generator_samples_with_the_diffusers_eta():
  from diffusion_models.diffusion.diffusion_inference import (
    DiffusionInference,
  )

  def last_frame(eta):
    model = _model().eval()
    torch.manual_seed(0)
    model.load_state_dict(_model().state_dict())
    model.diffuser.eta = eta
    inference = DiffusionInference(
      model=model,
      image_shape=(3, 8),
      device="cpu",
      reverse_transforms=lambda x: x,
    )
    torch.manual_seed(1)
    return [
      torch.tensor(list(frame.getdata()))
      for frame in inference.get_generator(2)
    ][-1]

  assert not torch.equal(last_frame(0.0), last_frame(1.0))
