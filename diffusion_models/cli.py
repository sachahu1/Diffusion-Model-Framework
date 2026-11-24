"""The ``diffusion-models`` command line interface.

Hydra does the heavy lifting for training and inference, so everything after the
task is forwarded to it untouched::

  diffusion-models train diffuser +config=small_attention_ema training.batch_size=64
  diffusion-models train autoencoder +config=vae_f8
  diffusion-models train pipeline +pipeline=latent_celeba_f8c16  # all the stages
  diffusion-models train diffuser --help  # hydra's own help, listing the config groups
  diffusion-models infer checkpoint=<path> +config=small_attention_ema
  diffusion-models fid <diffusion checkpoint>    # FID/KID of generated samples
  diffusion-models rfid <autoencoder checkpoint>  # reconstruction FID
  diffusion-models encode <autoencoder checkpoint> -o latents.pt  # latent cache

Discover and check configs, and get tab completion::

  diffusion-models configs list diffuser   # groups, options and presets
  diffusion-models configs show autoencoder +config=vae_f8 loss=l1
  eval "$(diffusion-models completion bash)"  # also zsh and fish
"""

import importlib
import os
import pathlib
import sys
from datetime import datetime
from typing import List, Optional, Tuple

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

  # The store is registered now: keep the completion snapshot current.
  from diffusion_models import _completion_cache

  _completion_cache.refresh()

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


def _bash_current_word() -> str:
  """The word under the cursor as bash sees it (click drops a lone ``=``)."""
  from click.shell_completion import split_arg_string

  words = split_arg_string(os.environ.get("COMP_WORDS", ""))
  index = int(os.environ.get("COMP_CWORD", len(words)))
  return words[index] if index < len(words) else ""


def _complete_overrides(task: Optional[str] = None):
  """Shell completion of hydra overrides (groups, options, presets, keys).

  Args:
    task: The task whose config to complete; by default the command's ``task``
      argument.
  """

  def complete(ctx, param, incomplete):
    from click.shell_completion import CompletionItem

    selected = task or ctx.params.get("task")
    if selected is None:
      return []
    in_bash = os.environ.get("_DIFFUSION_MODELS_COMPLETE", "").startswith(
      "bash"
    )
    if in_bash:
      incomplete = _bash_current_word()
    tokens = [*(ctx.params.get("overrides") or ()), incomplete]
    if in_bash:
      # Bash splits words at "=": ["model", "=", "sm"] is one override.
      tokens = _join_split_equals(tokens)
    *previous, current = tokens
    try:
      from diffusion_models import _configs

      matches = _configs.complete_overrides(
        selected,
        previous,
        current,
        config_directories=ctx.params.get("config_directories") or (),
      )
    except Exception:  # Completion must never break the shell.
      return []
    if in_bash and "=" in current:
      # Only the text after the "=" is the word bash replaces.
      prefix = current.rsplit("=", 1)[0] + "="
      matches = [m[len(prefix) :] for m in matches if m.startswith(prefix)]
    return [CompletionItem(m) for m in matches]

  return complete


def _join_split_equals(tokens: List[str]) -> List[str]:
  """Re-join the words bash split around ``=`` (``a``, ``=``, ``b`` -> ``a=b``)."""
  joined: List[str] = []
  index = 0
  while index < len(tokens):
    token = tokens[index]
    if token == "=" and joined:
      joined[-1] += "="
      if index + 1 < len(tokens):
        joined[-1] += tokens[index + 1]
        index += 1
    elif token.endswith("=") or not joined or not joined[-1].endswith("="):
      joined.append(token)
    else:
      joined[-1] += token
    index += 1
  return joined


def _overrides_argument(task: str):
  return click.argument(
    "overrides",
    nargs=-1,
    type=click.UNPROCESSED,
    shell_complete=_complete_overrides(task),
  )


@main.group()
def train() -> None:
  """Train a model (`diffuser`, `autoencoder`) or a `pipeline` of stages."""


@train.command(name="diffuser", **_HYDRA_COMMAND)
@_config_dir_option
@_overrides_argument("diffuser")
def train_diffuser(
  config_directories: Tuple[pathlib.Path, ...], overrides
) -> None:
  """Train a diffusion model.

  OVERRIDES are hydra overrides and flags, e.g. `+config=small_attention_ema`,
  `training.batch_size=64`, `resume_from=<checkpoint>` or `--multirun`.
  Pass `--help` to list the available config groups, or run
  `diffusion-models configs list diffuser`.
  """
  _run_hydra(
    "train diffuser",
    "diffusion_models.diffusion.training",
    "training",
    overrides,
    config_directories,
  )


@train.command(name="autoencoder", **_HYDRA_COMMAND)
@_config_dir_option
@_overrides_argument("autoencoder")
def train_autoencoder(
  config_directories: Tuple[pathlib.Path, ...], overrides
) -> None:
  """Train an autoencoder (the VAE of a latent diffusion model).

  OVERRIDES are hydra overrides and flags, e.g. `+config=vae_f8`,
  `training.batch_size=16`, `image_size=128` or `resume_from=<checkpoint>`.
  Pass `--help` to list the available config groups, or run
  `diffusion-models configs list autoencoder`.
  """
  _run_hydra(
    "train autoencoder",
    "diffusion_models.autoencoders.training",
    "training",
    overrides,
    config_directories,
  )


@train.command(
  name="pipeline",
  context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@_config_dir_option
@click.option(
  "--dry-run",
  is_flag=True,
  help="Print each stage's command and resolved config instead of running.",
)
@_overrides_argument("pipeline")
def train_pipeline(
  config_directories: Tuple[pathlib.Path, ...], dry_run: bool, overrides
) -> None:
  """Train several stages in sequence from one config.

  A pipeline lists stages (an autoencoder, its GAN fine-tune, the latent
  encoding, a diffusion model, ...) with their full config; they run one after
  the other, each in its own process. OVERRIDES are hydra overrides of the
  pipeline config: select one with `+pipeline=latent_celeba_f8c16`, change a
  stage with e.g. `stages.diffuser.config.training.number_of_epochs=50`, and
  continue an earlier run with `resume_from=<run directory>`. See
  `diffusion-models configs list pipeline` and `diffusion_models/pipeline.py`.
  """
  from diffusion_models.pipeline import run_pipeline

  run_pipeline(overrides, config_directories, dry_run=dry_run)


_SIMPLE_INFER_OPTIONS = {
  "checkpoint": str,
  "number_of_images": int,
  "steps": int,
  "eta": float,
  "use_ema": lambda value: value.lower() in ("true", "1", "yes"),
  "save_gif": lambda value: value.lower() in ("true", "1", "yes"),
  "device": str,
  "seed": int,
}


def _simple_infer_options(overrides) -> Optional[dict]:
  """The options of ``infer`` when ``overrides`` are only the simple ones.

  ``None`` when there is no ``checkpoint=`` or anything else is overridden (a
  config group, ``+config=``, ``autoencoder.checkpoint=``, ``--help``, ...): the
  model must then be composed from the hydra configs.
  """
  options = {}
  for override in overrides:
    name, separator, value = override.partition("=")
    if not separator or name not in _SIMPLE_INFER_OPTIONS:
      return None
    try:
      options[name] = _SIMPLE_INFER_OPTIONS[name](value)
    except ValueError:
      return None
  return options if "checkpoint" in options else None


@main.command(**_HYDRA_COMMAND)
@_config_dir_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def infer(config_directories: Tuple[pathlib.Path, ...], overrides) -> None:
  """Generate images with a trained checkpoint.

  OVERRIDES are hydra overrides: `checkpoint=<path>` is required and the model
  config should match the training run, e.g. `+config=small_attention_ema`.
  Other options: `number_of_images=25`, `save_gif=true`, `steps=50`,
  `use_ema=false`, `device=cpu`. Pass `--help` to list the config groups.

  With only the simple options (`checkpoint`, `number_of_images`, `steps`,
  `eta`, `use_ema`, `save_gif`, `device`, `seed`), the model, latent size and
  autoencoder are rebuilt from the config stored in the checkpoint, as
  `diffusion-models fid` does. Any other override composes the model from the
  hydra configs, as before.
  """
  options = _simple_infer_options(overrides)
  if options is not None and not config_directories:
    from diffusion_models.diffusion.inference import generate_from_checkpoint
    from diffusion_models.diffusion.schemas import Checkpoint

    checkpoint = options["checkpoint"]
    if Checkpoint.from_file(
      checkpoint, map_location="cpu", load_optimizer=False
    ).config_yaml:
      output = pathlib.Path("outputs/infer") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
      )
      generate_from_checkpoint(output_directory=output, **options)
      click.echo(f"Saved to {output}")
      return
  _run_hydra(
    "infer",
    "diffusion_models.diffusion.inference",
    "inference",
    overrides,
    config_directories,
  )


@main.command()
@click.argument(
  "checkpoint",
  type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
  "-o",
  "--output",
  type=click.Path(dir_okay=False, path_type=pathlib.Path),
  help="The cache file; defaults to <checkpoint>.latents.pt.",
)
@click.option("--batch-size", default=128, show_default=True)
@click.option(
  "--flip/--no-flip",
  default=True,
  show_default=True,
  help="Also cache the flipped images, a free augmentation.",
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--ema/--no-ema", default=True, show_default=True)
@click.option("--num-workers", default=4, show_default=True)
def encode(checkpoint, output, batch_size, flip, device, ema, num_workers):
  """Encode the training images of an autoencoder CHECKPOINT into latents.

  The images are the config's `dataset` (the autoencoder's training data). The
  cache is what `diffusion-models train diffuser +config=latent_celeba
  dataset.path=<cache>` trains on, so diffusion training never runs the
  encoder. Latents of a decoder-only fine-tune (the GAN stage) are those of
  its first stage: either checkpoint encodes to the same cache.
  """
  import torch
  from hydra_zen import instantiate
  from omegaconf import OmegaConf

  from diffusion_models.autoencoders.base import BaseAutoencoder
  from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint
  from diffusion_models.latent.cache import build_latent_cache

  state = AutoencoderCheckpoint.from_file(checkpoint, map_location="cpu")
  if state.config_yaml is None:
    raise click.ClickException("The checkpoint has no embedded config.")
  dataset = instantiate(OmegaConf.create(state.config_yaml).dataset)
  autoencoder = BaseAutoencoder.load(checkpoint, use_ema=ema)
  output = output or checkpoint.with_name(checkpoint.name + ".latents.pt")

  cache = build_latent_cache(
    autoencoder,
    dataset,
    batch_size=batch_size,
    flip=flip,
    device=device,
    num_workers=num_workers,
    metadata={"autoencoder_checkpoint": str(checkpoint)},
    # Built next to the output, on disk: the cache can exceed the free RAM.
    scratch_directory=output.parent,
  )
  cache.to_file(output)
  click.echo(
    f"Wrote {output}: {len(cache)} images, latents "
    f"{tuple(cache.mean.shape[2:])}, global scale factor "
    f"{cache.scale_factor:.4f}, channel means "
    f"{[round(v, 2) for v in cache.channel_mean.tolist()]}, channel stds "  # ty: ignore[missing-union-attribute]
    f"{[round(v, 2) for v in cache.channel_std.tolist()]}, mean posterior "  # ty: ignore[missing-union-attribute]
    f"std {cache.std.mean(dtype=torch.float32).item():.4f}"
  )


@main.command()
@click.argument(
  "checkpoint",
  type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option("--samples", default=10000, show_default=True, help="Images.")
@click.option("--batch-size", default=100, show_default=True)
@click.option("--steps", type=int, help="Override the DDIM sampling steps.")
@click.option(
  "--eta",
  type=float,
  help="Override the sampling stochasticity (0: deterministic DDIM, 1: DDPM-like).",
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--ema/--no-ema", default=True, show_default=True)
@click.option("--kid/--no-kid", default=True, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option(
  "--clip/--no-clip",
  default=True,
  show_default=True,
  help="Clamp samples to the diffuser's clip value after each step. "
  "Latent diffusion needs --no-clip (the preset trains with it off).",
)
def fid(
  checkpoint, samples, batch_size, steps, eta, device, ema, kid, seed, clip
) -> None:
  """Compute FID (and KID) of a training CHECKPOINT.

  Needs the `evaluation` extra. Only compare scores
  that use the same number of samples.
  """
  try:
    from diffusion_models.diffusion.evaluation import evaluate_checkpoint
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
    clip=clip,
    eta=eta,
  )
  for name, value in metrics.items():
    click.echo(f"{name}: {value:.4f}")


@main.command()
@click.argument(
  "checkpoint",
  type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option("--samples", default=10000, show_default=True, help="Images.")
@click.option("--batch-size", default=50, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--ema/--no-ema", default=True, show_default=True)
@click.option("--kid/--no-kid", default=True, show_default=True)
def rfid(checkpoint, samples, batch_size, device, ema, kid) -> None:
  """Compute the reconstruction FID (and KID) of an autoencoder CHECKPOINT.

  Each validation image is encoded and decoded and compared with the original:
  the best FID a latent diffusion model on this autoencoder could reach.
  Needs the `evaluation` extra.
  """
  try:
    from diffusion_models.autoencoders.evaluation import (
      evaluate_autoencoder_checkpoint,
    )
  except ModuleNotFoundError as error:  # pragma: no cover
    raise click.ClickException(
      f"Missing evaluation dependency ({error.name}). "
      "Install them with `pip install diffusion-model-framework[evaluation]`."
    ) from error

  metrics = evaluate_autoencoder_checkpoint(
    checkpoint,
    number_of_samples=samples,
    batch_size=batch_size,
    device=device,
    use_ema=ema,
    compute_kid=kid,
  )
  for name, value in metrics.items():
    click.echo(f"{name}: {value:.4f}")


_TASKS = ["diffuser", "autoencoder", "pipeline"]
_task_argument = click.argument("task", type=click.Choice(_TASKS))


@main.group()
def configs() -> None:
  """Discover and check the hydra configs."""


@configs.command(name="list")
@click.argument("task", type=click.Choice(_TASKS), required=False)
@_config_dir_option
def configs_list(task, config_directories) -> None:
  """List the config groups, their options and the `+config=` presets.

  Without TASK, both tasks are listed. The bracketed option is the default.
  """
  from diffusion_models import _configs

  tasks = [task] if task else list(_configs.TASKS)
  for name in tasks:
    options = _configs.list_options(name, config_directories)
    click.secho(f"{name}: {_configs.TASKS[name].description}", bold=True)
    for group, info in options[
      "groups"
    ].items():  # ty: ignore[missing-attribute]
      shown = ", ".join(
        f"[{option}]" if option == info["default"] else option
        for option in info["options"]
      )
      click.echo(f"  {group}=  {shown}")
    click.echo(
      f"  +{_configs.TASKS[name].preset_group}=  "
      f"{', '.join(options['presets'])}"  # ty: ignore[no-matching-overload]
    )
    click.echo()


@configs.command(name="show", **_HYDRA_COMMAND)
@click.argument("task", type=click.Choice(_TASKS))
@_config_dir_option
@click.option(
  "--resolve",
  is_flag=True,
  help="Resolve interpolations such as ${image_size}.",
)
@click.argument(
  "overrides",
  nargs=-1,
  type=click.UNPROCESSED,
  shell_complete=_complete_overrides(),
)
def configs_show(task, config_directories, resolve, overrides) -> None:
  """Print the config a training would use, e.g. `show diffuser +config=x`."""
  from diffusion_models import _configs

  click.echo(
    _configs.compose_config(
      task, overrides, config_directories, resolve=resolve
    ),
    nl=False,
  )


@main.command()
@click.argument(
  "shell", type=click.Choice(["bash", "zsh", "fish"]), default="bash"
)
def completion(shell: str) -> None:
  """Print the shell completion script.

  Completes commands, options and hydra overrides: config groups
  (`model=<TAB>`), their options, `+config=` presets and config keys.

  \b
    eval "$(diffusion-models completion bash)"   # in ~/.bashrc
    eval "$(diffusion-models completion zsh)"    # in ~/.zshrc
    diffusion-models completion fish > ~/.config/fish/completions/diffusion-models.fish
  """
  from click.shell_completion import get_completion_class

  completion_class = get_completion_class(shell)
  assert completion_class is not None
  click.echo(
    completion_class(
      main, {}, "diffusion-models", "_DIFFUSION_MODELS_COMPLETE"
    ).source()
  )


if __name__ == "__main__":
  main()
