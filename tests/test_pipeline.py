import json
import pathlib

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from diffusion_models import cli, pipeline

PRESET = "+pipeline=latent_celeba_f8c16"


@pytest.fixture
def runner():
  return CliRunner()


@pytest.fixture
def run_directory(tmp_path):
  return tmp_path / "run"


def _config(stages, **values):
  return OmegaConf.create({"name": "t", "stages": stages, **values})


def test_the_latent_pipeline_wires_its_stages_together(run_directory):
  config = pipeline._compose("pipeline", [PRESET], ())
  stages = pipeline.compose_stages(config, run_directory, ())

  assert [s.name for s in stages] == ["vae", "gan", "encode", "diffuser"]
  vae, gan, encode, diffuser = stages
  # The shared values reach the stages, which agree on the latent's shape.
  assert vae.config["autoencoder"]["latent_channels"] == 16
  assert vae.config["loss"]["kl_weight"] == 1e-3
  assert diffuser.config["image_channels"] == 16
  assert diffuser.config["model"]["image_channels"] == 16
  # Each stage reads the output of the previous one.
  assert gan.config["autoencoder"]["checkpoint"] == str(
    run_directory / "vae" / "checkpoints" / "final.pt"
  )
  assert encode.config["checkpoint"] == str(
    run_directory / "gan" / "checkpoints" / "final.pt"
  )
  assert diffuser.config["dataset"]["path"] == encode.config["output"]
  assert (
    diffuser.config["autoencoder"]["checkpoint"] == encode.config["checkpoint"]
  )
  assert "pipeline" not in diffuser.config  # the context is not left behind


def test_shared_values_can_be_overridden_from_the_command_line(run_directory):
  config = pipeline._compose(
    "pipeline", [PRESET, "latent_channels=32", "kl_weight=0.01"], ()
  )
  vae, _, _, diffuser = pipeline.compose_stages(config, run_directory, ())
  assert vae.config["autoencoder"]["latent_channels"] == 32
  assert vae.config["loss"]["kl_weight"] == 0.01
  assert diffuser.config["image_channels"] == 32


def test_a_typo_in_a_stage_is_reported_with_its_path(run_directory):
  config = pipeline._compose(
    "pipeline", [PRESET, "+stages.diffuser.config.training.btach_size=4"], ()
  )
  with pytest.raises(pipeline.PipelineError, match="training.btach_size"):
    pipeline.compose_stages(config, run_directory, ())


def test_a_forgotten_value_is_reported(run_directory):
  # `autoencoder=pretrained` needs a checkpoint, which this stage never sets.
  config = _config(
    {
      "gan": {
        "task": "autoencoder",
        "preset": "vae_f8_gan",
        "config": {"training": {"number_of_epochs": 1}},
      }
    }
  )
  with pytest.raises(pipeline.PipelineError, match="checkpoint is missing"):
    pipeline.compose_stages(config, run_directory, ())


@pytest.mark.parametrize(
  "stages,message",
  [
    ({}, "no stages"),
    ({"a": {"preset": "x"}}, "needs a `task`"),
    ({"a": {"task": "train"}}, "needs a `task`"),
    ({"a": {"task": "diffuser", "colour": 1}}, "unknown keys"),
    ({"a": {"task": "encode"}}, "needs `config.checkpoint`"),
    (
      {"a": {"task": "encode", "config": {"checkpoint": "c", "oops": 1}}},
      "unknown encode options",
    ),
    (
      {"a": {"task": "encode", "preset": "x", "config": {"checkpoint": "c"}}},
      "takes only `config`",
    ),
  ],
)
def test_a_malformed_pipeline_is_rejected(run_directory, stages, message):
  with pytest.raises(pipeline.PipelineError, match=message):
    pipeline.compose_stages(_config(stages), run_directory, ())


def test_check_config_finds_unknown_options_and_missing_values():
  config = {
    "training": {
      "_target_": "diffusion_models.core.schemas.TrainingConfiguration",
      "batch_size": 4,
      "btach_size": 4,
    },
    "path": "???",
  }
  problems = pipeline.check_config(config)
  assert any("btach_size" in problem for problem in problems)
  assert any("path is missing" in problem for problem in problems)
  assert pipeline.check_config({"training": {"batch_size": 4}}) == []


ENCODE_ONLY = {
  "first": {"task": "encode", "config": {"checkpoint": "a.pt"}},
  "second": {
    "task": "encode",
    "config": {"checkpoint": "${pipeline.stages.first.dir}/b.pt"},
  },
}


@pytest.fixture
def fake_runner(monkeypatch):
  """Replaces the stage processes by a list of the commands that were run.

  The pipeline is composed from its overrides: ``[root]`` or ``[root, run to
  resume]``.
  """
  commands = []
  monkeypatch.setattr(
    pipeline, "_run_command", lambda command: commands.append(command) or 0
  )
  monkeypatch.setattr(
    pipeline,
    "_compose",
    lambda task, overrides, directories: _config(
      ENCODE_ONLY,
      root=overrides[0],
      resume_from=overrides[1] if len(overrides) > 1 else None,
    ),
  )
  return commands


def test_stages_run_in_order_and_are_marked_complete(tmp_path, fake_runner):
  pipeline.run_pipeline([str(tmp_path)])

  assert [c[3] for c in fake_runner] == ["encode", "encode"]
  assert fake_runner[0][4] == "a.pt"
  (run,) = (tmp_path / "t").iterdir()
  # The second stage is told where the first one wrote.
  assert fake_runner[1][4] == str(run / "first" / "b.pt")
  assert (run / "first" / pipeline.COMPLETE_MARKER).exists()
  status = json.loads((run / pipeline.STATUS_FILE).read_text())
  assert {name: s["status"] for name, s in status.items()} == {
    "first": "complete",
    "second": "complete",
  }
  assert (run / "pipeline.yaml").exists()
  assert (run / "first" / "stage.yaml").exists()


def test_resuming_skips_finished_stages(tmp_path, fake_runner):
  pipeline.run_pipeline([str(tmp_path)])
  (run,) = (tmp_path / "t").iterdir()
  (run / "second" / pipeline.COMPLETE_MARKER).unlink()
  fake_runner.clear()

  pipeline.run_pipeline([str(tmp_path), str(run)])

  assert len(fake_runner) == 1  # only `second` runs again
  assert fake_runner[0][4] == str(run / "first" / "b.pt")


def test_resuming_an_interrupted_training_continues_from_its_checkpoint(
  tmp_path,
):
  stage = pipeline.Stage(
    name="a",
    task="diffuser",
    directory=tmp_path / "a",
    config={"resume_from": None},
    command=[],
  )
  (stage.directory / "checkpoints").mkdir(parents=True)
  (stage.directory / "checkpoints" / "latest.pt").write_bytes(b"")
  pipeline._prepare(stage)
  assert stage.config["resume_from"] == str(
    stage.directory / "checkpoints" / "latest.pt"
  )
  assert "latest.pt" in stage.config_file.read_text()


def test_a_failing_stage_stops_the_pipeline(
  tmp_path, monkeypatch, fake_runner
):
  monkeypatch.setattr(pipeline, "_run_command", lambda command: 3)
  with pytest.raises(pipeline.PipelineError, match="exit code 3"):
    pipeline.run_pipeline([str(tmp_path)])
  (run,) = (tmp_path / "t").iterdir()
  assert not (run / "first" / pipeline.COMPLETE_MARKER).exists()
  status = json.loads((run / pipeline.STATUS_FILE).read_text())
  assert status["first"]["status"].startswith("failed")
  assert status["second"]["status"] == "pending"


def test_the_resume_hint_does_not_repeat_resume_from(tmp_path):
  command = pipeline._resume_command(
    ["+pipeline=x", "resume_from=old"], tmp_path
  )
  assert command == (
    f"diffusion-models train pipeline +pipeline=x resume_from={tmp_path}"
  )


def test_a_dry_run_writes_nothing(tmp_path, fake_runner):
  pipeline.run_pipeline([str(tmp_path)], dry_run=True)
  assert fake_runner == []
  assert not (tmp_path / "t").exists()


def test_the_command_forwards_overrides_and_flags(runner, monkeypatch):
  calls = []
  monkeypatch.setattr(
    pipeline,
    "run_pipeline",
    lambda overrides, directories, dry_run: calls.append(
      (overrides, directories, dry_run)
    ),
  )
  result = runner.invoke(
    cli.main,
    ["train", "pipeline", PRESET, "kl_weight=0.1", "--dry-run"],
  )
  assert result.exit_code == 0, result.output
  assert calls == [((PRESET, "kl_weight=0.1"), (), True)]


def test_the_pipeline_presets_are_listed(runner):
  result = runner.invoke(cli.main, ["configs", "list", "pipeline"])
  assert result.exit_code == 0, result.output
  assert "+pipeline=" in result.output
  assert "latent_celeba_f8c16" in result.output
