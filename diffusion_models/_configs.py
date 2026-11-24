"""Hydra config discovery, composition and shell completion for the CLI.

Everything here imports torch (through the config store), so the CLI only
imports this module lazily.
"""

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from hydra import compose, initialize_config_dir
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.core.config_search_path import SearchPathQuery
from hydra.core.global_hydra import GlobalHydra
from hydra.plugins.completion_plugin import DefaultCompletionPlugin
from omegaconf import OmegaConf

from diffusion_models import _completion_cache


def _conf_directory() -> pathlib.Path:
  """The directory of the packaged configs.

  Used instead of a ``pkg://`` source so that discovery does not depend on how
  the package loader exposes its resources.
  """
  import diffusion_models.conf as conf

  return pathlib.Path(conf.__path__[0])


PRESET_GROUP = "config"
"""The config group holding the ``+config=<preset>`` presets."""


@dataclass(frozen=True)
class Task:
  """A trainable task and its primary hydra config."""

  config_name: str
  description: str
  preset_group: str = PRESET_GROUP
  """The config group of the task's ``+<group>=<preset>`` presets."""


TASKS: Dict[str, Task] = {
  "diffuser": Task("baseline", "a diffusion model"),
  "autoencoder": Task(
    "autoencoder_baseline",
    "an autoencoder (the VAE of a latent diffusion model)",
  ),
  "pipeline": Task(
    "pipeline_baseline",
    "a pipeline of stages trained in sequence (see `train pipeline`)",
    preset_group="pipeline",
  ),
}


def _register() -> None:
  """Register the config groups with hydra (imports torch)."""
  from hydra_zen import store

  from diffusion_models import _store  # noqa: F401  (registers the groups)

  store.add_to_hydra_store(overwrite_ok=True)


def _search_path_overrides(
  config_directories: Sequence[pathlib.Path],
) -> List[str]:
  if not config_directories:
    return []
  paths = ",".join(f"file://{path.resolve()}" for path in config_directories)
  return [f"hydra.searchpath=[{paths}]"]


def _loader(
  config_directories: Sequence[pathlib.Path],
  exported: Optional[pathlib.Path] = None,
) -> ConfigLoaderImpl:
  search_path = create_config_search_path(f"file://{_conf_directory()}")
  # Hydra requires its schema source to stay last: add ours after ``main``.
  after_main = SearchPathQuery(provider="main", path=None)
  extra = [("exported", exported)] if exported is not None else []
  extra += [("cli", directory.resolve()) for directory in config_directories]
  for provider, directory in reversed(extra):
    search_path.append(
      provider=provider, path=f"file://{directory}", anchor=after_main
    )
  return ConfigLoaderImpl(config_search_path=search_path)


def _task(name: str) -> Task:
  if name not in TASKS:
    raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}.")
  return TASKS[name]


def complete_overrides(
  task: str,
  previous: Sequence[str],
  incomplete: str,
  config_directories: Sequence[pathlib.Path] = (),
) -> List[str]:
  """Complete the hydra override being typed.

  Args:
    task: ``"diffuser"``, ``"autoencoder"`` or ``"pipeline"``.
    previous: The overrides already on the command line.
    incomplete: The (possibly empty) override being typed.
    config_directories: Extra directories of configs and presets.

  Returns:
    Matching overrides: config groups (``model=``), their options
    (``model=small_attention_unet``), presets (``+config=vae_f8``), config keys
    (``training.batch_size=``) and values.
  """
  # Completes over the exported snapshot of the registered groups: no torch.
  exported = _completion_cache.ensure()
  line = " ".join([*previous, incomplete])
  plugin = DefaultCompletionPlugin(_loader(config_directories, exported))
  return plugin._query(_task(task).config_name, line)


def list_options(
  task: str, config_directories: Sequence[pathlib.Path] = ()
) -> Dict[str, object]:
  """The config groups, their options and the presets of a task.

  Returns:
    ``{"groups": {group: {"default": option, "options": [...]}},
    "presets": [...]}``.
  """
  exported = _completion_cache.ensure()
  config_name = _task(task).config_name
  preset_group = _task(task).preset_group
  overrides = _search_path_overrides([exported, *config_directories])
  GlobalHydra.instance().clear()
  with initialize_config_dir(str(_conf_directory()), version_base="1.3"):
    cfg = compose(config_name, overrides=overrides, return_hydra_config=True)
    loader = GlobalHydra.instance().config_loader()

    def options(group: str) -> List[str]:
      # The search path override only applies when passed along.
      return sorted(
        loader.get_group_options(
          group, config_name=config_name, overrides=overrides
        )
      )

    groups = {}
    for group, selected in cfg.hydra.runtime.choices.items():
      if group.startswith("hydra") or group == preset_group:
        continue
      groups[group] = {"default": selected, "options": options(group)}
    presets = options(preset_group)
  GlobalHydra.instance().clear()
  return {"groups": groups, "presets": presets}


def compose_config(
  task: str,
  overrides: Sequence[str] = (),
  config_directories: Sequence[pathlib.Path] = (),
  resolve: bool = False,
) -> str:
  """Compose a task's config with ``overrides`` and render it as YAML."""
  _register()
  GlobalHydra.instance().clear()
  with initialize_config_dir(str(_conf_directory()), version_base="1.3"):
    cfg = compose(
      _task(task).config_name,
      overrides=[*overrides, *_search_path_overrides(config_directories)],
    )
  GlobalHydra.instance().clear()
  return OmegaConf.to_yaml(cfg, resolve=resolve)
