import pytest
from click.shell_completion import BashComplete, FishComplete, ZshComplete
from click.testing import CliRunner

from diffusion_models import cli


@pytest.fixture
def runner():
  return CliRunner()


def test_command_tree(runner):
  result = runner.invoke(cli.main, ["--help"])
  for command in ["train", "configs", "completion", "fid", "rfid", "infer"]:
    assert command in result.output
  result = runner.invoke(cli.main, ["train", "--help"])
  assert "diffuser" in result.output and "autoencoder" in result.output


@pytest.mark.parametrize(
  "task,module",
  [
    ("diffuser", "diffusion_models.diffusion.training"),
    ("autoencoder", "diffusion_models.autoencoders.training"),
  ],
)
def test_train_forwards_everything_to_hydra(runner, monkeypatch, task, module):
  calls = []
  monkeypatch.setattr(cli, "_run_hydra", lambda *args: calls.append(args))
  result = runner.invoke(
    cli.main,
    ["train", task, "+config=x", "model=y", "--multirun", "training.a=1"],
  )
  assert result.exit_code == 0, result.output
  command, target, extra, overrides, directories = calls[0]
  assert command == f"train {task}"
  assert target == module
  assert overrides == ("+config=x", "model=y", "--multirun", "training.a=1")
  assert directories == ()


def test_train_needs_a_task(runner):
  result = runner.invoke(cli.main, ["train", "+config=x"])
  assert result.exit_code != 0


def test_configs_list(runner):
  result = runner.invoke(cli.main, ["configs", "list", "autoencoder"])
  assert result.exit_code == 0, result.output
  assert "loss=" in result.output and "[vae]" in result.output
  assert "+config=" in result.output and "vae_f8" in result.output
  assert "diffuser:" not in result.output


def test_configs_show_composes_a_preset(runner):
  result = runner.invoke(
    cli.main,
    [
      "configs",
      "show",
      "autoencoder",
      "+config=vae_f8",
      "loss.reconstruction_loss=mse",
    ],
  )
  assert result.exit_code == 0, result.output
  assert "base_channels: 32" in result.output
  assert "training_name: vae_f8" in result.output
  assert "reconstruction_loss: mse" in result.output
  assert "${image_size}" in result.output
  resolved = runner.invoke(
    cli.main, ["configs", "show", "autoencoder", "--resolve"]
  )
  assert "${" not in resolved.output.split("hydra:")[0]


def test_configs_show_with_an_extra_config_directory(runner, tmp_path):
  (tmp_path / "config").mkdir()
  (tmp_path / "config" / "mine.yaml").write_text(
    "# @package _global_\ntraining:\n  training_name: mine\n"
  )
  result = runner.invoke(
    cli.main,
    [
      "configs",
      "show",
      "diffuser",
      "--config-dir",
      str(tmp_path),
      "+config=mine",
    ],
  )
  assert result.exit_code == 0, result.output
  assert "training_name: mine" in result.output


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_script(runner, shell):
  result = runner.invoke(cli.main, ["completion", shell])
  assert result.exit_code == 0
  assert "_DIFFUSION_MODELS_COMPLETE" in result.output


@pytest.mark.parametrize(
  "tokens,expected",
  [
    (["model", "=", "sm"], ["model=sm"]),
    (["model", "="], ["model="]),
    (["+config", "=", ""], ["+config="]),
    (["a=1", "b", "=", "2"], ["a=1", "b=2"]),
    (["model=sm"], ["model=sm"]),
    (["x"], ["x"]),
  ],
)
def test_join_split_equals(tokens, expected):
  assert cli._join_split_equals(tokens) == expected


def _complete(shell, monkeypatch, words, cword):
  monkeypatch.setenv("_DIFFUSION_MODELS_COMPLETE", f"{shell}_complete")
  monkeypatch.setenv("COMP_WORDS", words)
  monkeypatch.setenv("COMP_CWORD", cword)
  complete = {"bash": BashComplete, "zsh": ZshComplete, "fish": FishComplete}[
    shell
  ](cli.main, {}, "diffusion-models", "_DIFFUSION_MODELS_COMPLETE")
  args, incomplete = complete.get_completion_args()
  return [item.value for item in complete.get_completions(args, incomplete)]


def test_completes_subcommands(monkeypatch):
  assert _complete("bash", monkeypatch, "diffusion-models train ", "2") == [
    "autoencoder",
    "diffuser",
    "pipeline",
  ]


def test_bash_completes_the_value_after_a_split_equals(monkeypatch):
  # Bash hands over ``model``, ``=``, ``sm``: only the value is replaced.
  assert _complete(
    "bash", monkeypatch, "diffusion-models train diffuser model = sm", "5"
  ) == ["small_attention_unet"]
  assert _complete(
    "bash", monkeypatch, "diffusion-models train diffuser model =", "4"
  ) == ["simple_unet", "small_attention_unet"]


def test_bash_completes_presets_and_keys(monkeypatch):
  presets = _complete(
    "bash", monkeypatch, "diffusion-models train autoencoder +config =", "4"
  )
  assert "vae_f8" in presets and "small_attention_ema" in presets
  assert _complete(
    "bash", monkeypatch, "diffusion-models train autoencoder training.ba", "3"
  ) == ["training.batch_size="]


def test_zsh_and_fish_complete_whole_overrides(monkeypatch):
  assert _complete(
    "zsh", monkeypatch, "diffusion-models train diffuser model=sm", "3"
  ) == ["model=small_attention_unet"]
  assert _complete(
    "fish", monkeypatch, "diffusion-models train autoencoder loss=", "loss="
  ) == ["loss=huber", "loss=l1", "loss=mse", "loss=vae"]


def test_configs_show_completes_for_the_chosen_task(monkeypatch):
  assert _complete(
    "bash",
    monkeypatch,
    "diffusion-models configs show autoencoder loss = ",
    "5",
  ) == ["huber", "l1", "mse", "vae"]
  assert "ddpm" in _complete(
    "bash",
    monkeypatch,
    "diffusion-models configs show diffuser diffuser = ",
    "5",
  )


def test_configs_show_rejects_a_loss_that_does_not_fit_the_task(runner):
  # A diffusion regression loss has no ``kl_weight``, which the VAE config sets.
  result = runner.invoke(
    cli.main, ["configs", "show", "autoencoder", "loss=l1"]
  )
  assert result.exit_code != 0


def test_train_refreshes_the_completion_cache_before_hydra_starts(monkeypatch):
  from diffusion_models import _completion_cache

  events = []

  class Entry:
    @staticmethod
    def main():
      events.append("hydra")

  monkeypatch.setattr(cli.importlib, "import_module", lambda name: Entry)
  monkeypatch.setattr(
    _completion_cache, "refresh", lambda: events.append("refresh")
  )
  cli._run_hydra("train diffuser", "any.module", "training", ("a=1",), ())
  assert events == ["refresh", "hydra"]


def test_the_adversarial_preset_composes_with_a_checkpoint(runner):
  result = runner.invoke(
    cli.main,
    [
      "configs",
      "show",
      "autoencoder",
      "+config=vae_f8_gan",
      "autoencoder.checkpoint=runs/x/latest.pt",
    ],
  )
  assert result.exit_code == 0, result.output
  assert "BaseAutoencoder.load" in result.output
  assert "checkpoint: runs/x/latest.pt" in result.output
  assert "PatchDiscriminator" in result.output
  assert "trainable: decoder" in result.output
  assert "discriminator_optimizer:" in result.output
