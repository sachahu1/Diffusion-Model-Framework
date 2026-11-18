"""The ``diffusion-models`` command line interface.

Hydra does the heavy lifting for training and inference, so everything after the
sub-command is forwarded to it untouched::

  diffusion-models train +config=small_attention_ema training.batch_size=64
  diffusion-models train --help  # hydra's own help, listing the config groups
  diffusion-models infer checkpoint=<path> +config=small_attention_ema
"""

import importlib
import pathlib
import sys
from typing import Tuple

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
  """Tools for training and running diffusion models."""


def _run_hydra(command, module, extra, overrides, config_directories):
  """Forwards ``overrides`` to the hydra entry point ``module.main``."""
  try:
    entry_point = importlib.import_module(module).main
  except ModuleNotFoundError as error:  # pragma: no cover
    raise click.ClickException(
      f"Missing {extra} dependency ({error.name}). "
      f"Install them with `pip install diffusion-model-framework[{extra}]`."
    ) from error

  arguments = list(overrides)
  if config_directories:
    paths = ",".join(f"file://{path.resolve()}" for path in config_directories)
    arguments.append(f"hydra.searchpath=[{paths}]")
  # Hydra reads its overrides straight from sys.argv.
  sys.argv = [f"{sys.argv[0]} {command}", *arguments]
  entry_point()


_HYDRA_COMMAND = dict(
  context_settings={
    "ignore_unknown_options": True,
    "allow_extra_args": True,
    # Let ``--help`` reach hydra, which knows about the config groups.
    "help_option_names": [],
  },
  add_help_option=False,
)
_config_dir_option = click.option(
  "--config-dir",
  "config_directories",
  multiple=True,
  type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
  help="An extra directory to search for hydra configs and `+config=` presets.",
)


@main.command(**_HYDRA_COMMAND)
@_config_dir_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train(config_directories: Tuple[pathlib.Path, ...], overrides) -> None:
  """Train a diffusion model.

  OVERRIDES are hydra overrides and flags, e.g. `+config=small_attention_ema`,
  `training.batch_size=64`, `resume_from=<checkpoint>` or `--multirun`.
  Pass `--help` to list the available config groups.
  """
  _run_hydra(
    "train", "diffusion_models.training", "training", overrides, config_directories
  )


@main.command(**_HYDRA_COMMAND)
@_config_dir_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def infer(config_directories: Tuple[pathlib.Path, ...], overrides) -> None:
  """Generate images with a trained checkpoint.

  OVERRIDES are hydra overrides: `checkpoint=<path>` is required and the model
  config should match the training run, e.g. `+config=small_attention_ema`.
  Other options: `number_of_images=25`, `save_gif=true`, `steps=50`,
  `use_ema=false`, `device=cpu`. Pass `--help` to list the config groups.
  """
  _run_hydra(
    "infer", "diffusion_models.inference", "inference", overrides, config_directories
  )


@main.command()
@click.argument(
  "checkpoint",
  type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option("--samples", default=10000, show_default=True, help="Images.")
@click.option("--batch-size", default=100, show_default=True)
@click.option("--steps", type=int, help="Override the DDIM sampling steps.")
@click.option("--device", default="cuda", show_default=True)
@click.option("--ema/--no-ema", default=True, show_default=True)
@click.option("--kid/--no-kid", default=True, show_default=True)
@click.option("--seed", default=0, show_default=True)
def fid(
  checkpoint, samples, batch_size, steps, device, ema, kid, seed
) -> None:
  """Compute FID (and KID) of a training CHECKPOINT.

  Needs the `evaluation` extra. Only compare scores
  that use the same number of samples.
  """
  try:
    from diffusion_models.evaluation.checkpoint import evaluate_checkpoint
  except ModuleNotFoundError as error:  # pragma: no cover
    raise click.ClickException(
      f"Missing evaluation dependency ({error.name}). "
      "Install them with `pip install diffusion-model-framework[evaluation]`."
    ) from error

  metrics = evaluate_checkpoint(
    checkpoint,
    number_of_samples=samples,
    batch_size=batch_size,
    device=device,
    use_ema=ema,
    compute_kid=kid,
    steps=steps,
    seed=seed,
  )
  for name, value in metrics.items():
    click.echo(f"{name}: {value:.4f}")


if __name__ == "__main__":
  main()
