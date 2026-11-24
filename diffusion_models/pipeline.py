"""Train several stages in sequence from one config: ``diffusion-models train pipeline``.

A pipeline is a hydra config with shared values and an ordered ``stages`` mapping::

  diffusion-models train pipeline +pipeline=latent_celeba_f8c16 --dry-run

  stages:
    vae:
      task: autoencoder                # autoencoder, diffuser or encode
      preset: vae_f8                   # optional ``+config=<preset>`` to start from
      overrides: [optimizer=adopt]     # optional raw hydra overrides
      config:                          # merged on top of the stage's config
        training: {number_of_epochs: 30}

Inside a stage, ``${pipeline.<shared value>}``, ``${pipeline.dir}`` and
``${pipeline.stages.<name>.dir}`` read the pipeline, which is how a stage picks up
the output of an earlier one.

Each stage runs as a separate process in ``<root>/<name>/<timestamp>/<stage>``, and
all are checked before the first one starts. Resume a failed run with
``resume_from=<run directory>``: finished stages are skipped.
"""

import dataclasses
import datetime
import inspect
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import click
from omegaconf import OmegaConf

__all__ = ["run_pipeline"]

TRAINING_TASKS = ("autoencoder", "diffuser")
STAGE_TASKS = (*TRAINING_TASKS, "encode")
STAGE_KEYS = {"task", "preset", "overrides", "config"}
COMPLETE_MARKER = ".complete"
STATUS_FILE = "pipeline.json"

ENCODE_DEFAULTS: Dict[str, Any] = {
  "checkpoint": None,
  "output": None,
  "batch_size": 128,
  "flip": True,
  "device": "cuda",
  "ema": True,
  "num_workers": 4,
}
"""The options of an ``encode`` stage, those of ``diffusion-models encode``.
``output`` defaults to ``latents.pt`` in the stage's directory."""


class PipelineError(click.ClickException):
  """A problem with the pipeline's config or run."""


@dataclasses.dataclass
class Stage:
  """A stage, composed and ready to run."""

  name: str
  task: str
  directory: pathlib.Path
  config: Dict[str, Any]
  """The resolved config (the options of ``encode`` for an encode stage)."""
  command: List[str]
  complete: bool = False

  @property
  def config_file(self) -> pathlib.Path:
    return self.directory / "stage.yaml"


def _compose(
  task: str,
  overrides: Sequence[str],
  config_directories: Sequence[pathlib.Path],
) -> Any:
  from diffusion_models import _configs

  try:
    text = _configs.compose_config(task, overrides, config_directories)
  except Exception as error:  # hydra's errors are explicit enough
    raise PipelineError(
      f"Cannot compose the {task} config with {list(overrides)}: {error}"
    ) from error
  return OmegaConf.create(text)


def _context(
  pipeline_config, run_directory: pathlib.Path, names: Sequence[str]
) -> Dict[str, Any]:
  """What a stage reads as ``${pipeline.*}``."""
  shared = OmegaConf.masked_copy(
    pipeline_config, [key for key in pipeline_config if key != "stages"]
  )
  context = OmegaConf.to_container(shared, resolve=True)
  assert isinstance(context, dict)
  context["dir"] = str(run_directory)
  context["stages"] = {
    name: {"dir": str(run_directory / name)} for name in names
  }
  return context  # ty: ignore[invalid-explicit-return]


def _latest_checkpoint(directory: pathlib.Path) -> Optional[pathlib.Path]:
  candidates = [
    path
    for path in (
      directory / "checkpoints" / "latest.pt",
      directory / "checkpoints" / "interrupted.pt",
    )
    if path.exists()
  ]
  return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _walk_targets(node: Any, path: str = ""):
  """Every ``(path, mapping)`` with a ``_target_`` in a nested config."""
  if isinstance(node, dict):
    if isinstance(node.get("_target_"), str):
      yield path, node
    for key, value in node.items():
      yield from _walk_targets(value, f"{path}.{key}" if path else str(key))
  elif isinstance(node, list):
    for index, value in enumerate(node):
      yield from _walk_targets(value, f"{path}[{index}]")


def _missing_values(node: Any, path: str = "") -> List[str]:
  if isinstance(node, dict):
    return [
      problem
      for key, value in node.items()
      for problem in _missing_values(value, f"{path}.{key}" if path else key)
    ]
  if isinstance(node, list):
    return [
      problem
      for index, value in enumerate(node)
      for problem in _missing_values(value, f"{path}[{index}]")
    ]
  return [path] if node == "???" else []


def check_config(config: Dict[str, Any]) -> List[str]:
  """Cheap problems of a resolved training config, found without instantiating it.

  Catches what would otherwise fail when the stage starts: values still
  ``???``, and keys that the target of a ``_target_`` node does not accept (a
  typo such as ``training.btach_size``).
  """
  from hydra.utils import get_object

  problems = [f"{path} is missing" for path in _missing_values(config)]
  for path, node in _walk_targets(config):
    target = node["_target_"]
    try:
      signature = inspect.signature(get_object(target))
    except (ValueError, TypeError):  # no introspectable signature
      continue
    except Exception as error:
      problems.append(f"{path}: cannot import {target} ({error})")
      continue
    parameters = signature.parameters
    if "_args_" in node or any(
      p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
      continue
    for key in node:
      if not key.startswith("_") and key not in parameters:
        problems.append(f"{path}.{key} is not an option of {target}")
  return problems


def _training_stage_config(
  name: str,
  spec: Dict[str, Any],
  context: Dict[str, Any],
  config_directories: Sequence[pathlib.Path],
) -> Dict[str, Any]:
  overrides = list(spec.get("overrides") or [])
  if spec.get("preset"):
    overrides.insert(0, f"+config={spec['preset']}")
  composed = _compose(spec["task"], overrides, config_directories)
  merged = OmegaConf.merge(
    composed, {"pipeline": context}, spec.get("config") or {}
  )
  try:
    resolved = OmegaConf.to_container(merged, resolve=True)
  except Exception as error:
    raise PipelineError(
      f"Stage {name!r}: cannot resolve its config: {error}"
    ) from error
  assert isinstance(resolved, dict)
  resolved.pop("pipeline")
  return resolved  # ty: ignore[invalid-explicit-return]


def _encode_stage_config(
  name: str, spec: Dict[str, Any], context: Dict[str, Any], directory
) -> Dict[str, Any]:
  if spec.get("preset") or spec.get("overrides"):
    raise PipelineError(
      f"Stage {name!r}: an encode stage takes only `config`, not "
      "`preset` or `overrides`."
    )
  unknown = set(spec.get("config") or {}) - set(ENCODE_DEFAULTS)
  if unknown:
    raise PipelineError(
      f"Stage {name!r}: unknown encode options {sorted(unknown)}; "
      f"choose from {sorted(ENCODE_DEFAULTS)}."
    )
  merged = OmegaConf.merge(
    {**ENCODE_DEFAULTS, "output": str(directory / "latents.pt")},
    {"pipeline": context},
    spec.get("config") or {},
  )
  resolved = OmegaConf.to_container(merged, resolve=True)
  assert isinstance(resolved, dict)
  resolved.pop("pipeline")
  if not resolved["checkpoint"]:
    raise PipelineError(
      f"Stage {name!r}: an encode stage needs `config.checkpoint`, the "
      "autoencoder checkpoint to encode with."
    )
  return resolved  # ty: ignore[invalid-explicit-return]


def _command(task: str, directory: pathlib.Path, config: Dict[str, Any]):
  python = [sys.executable, "-m", "diffusion_models.cli"]
  if task == "encode":
    return [
      *python,
      "encode",
      str(config["checkpoint"]),
      "-o",
      str(config["output"]),
      "--batch-size",
      str(config["batch_size"]),
      "--flip" if config["flip"] else "--no-flip",
      "--device",
      str(config["device"]),
      "--ema" if config["ema"] else "--no-ema",
      "--num-workers",
      str(config["num_workers"]),
    ]
  return [
    *python,
    "train",
    task,
    f"--config-path={directory}",
    "--config-name=stage",
    f"hydra.run.dir={directory}",
  ]


def compose_stages(
  pipeline_config,
  run_directory: pathlib.Path,
  config_directories: Sequence[pathlib.Path] = (),
) -> List[Stage]:
  """Compose and check every stage of a pipeline config.

  Args:
    pipeline_config: The composed pipeline config.
    run_directory: The run's directory; each stage gets a subdirectory.
    config_directories: Extra directories of configs and presets.

  Raises:
    PipelineError: For a malformed pipeline or an invalid stage, naming all the
      problems of the stage.
  """
  specs = OmegaConf.to_container(pipeline_config.get("stages"), resolve=False)
  if not specs:
    raise PipelineError(
      "The pipeline has no stages: select one with `+pipeline=<name>` "
      "(see `diffusion-models configs list pipeline`)."
    )
  assert isinstance(specs, dict)
  context = _context(pipeline_config, run_directory, list(specs))  # ty: ignore[invalid-argument-type]

  stages = []
  for name, spec in specs.items():
    if not isinstance(spec, dict) or spec.get("task") not in STAGE_TASKS:
      raise PipelineError(
        f"Stage {name!r} needs a `task`, one of {list(STAGE_TASKS)}."
      )
    unknown = set(spec) - STAGE_KEYS
    if unknown:
      raise PipelineError(
        f"Stage {name!r} has unknown keys {sorted(unknown)}; "
        f"choose from {sorted(STAGE_KEYS)}."
      )
    directory = run_directory / name  # ty: ignore[unsupported-operator]
    task = spec["task"]
    if task == "encode":
      config = _encode_stage_config(name, spec, context, directory)  # ty: ignore[invalid-argument-type]
    else:
      config = _training_stage_config(name, spec, context, config_directories)  # ty: ignore[invalid-argument-type]
      problems = check_config(config)
      if problems:
        raise PipelineError(
          f"Stage {name!r} is invalid:\n  - " + "\n  - ".join(problems)
        )
    stages.append(
      Stage(
        name=name,  # ty: ignore[invalid-argument-type]
        task=task,
        directory=directory,
        config=config,
        command=_command(task, directory, config),
        complete=(directory / COMPLETE_MARKER).exists(),
      )
    )
  return stages


def _run_command(command: Sequence[str]) -> int:
  """Run a stage's process in the foreground and return its exit code.

  On Ctrl-C the process gets the signal too (same process group): wait for it,
  so a trainer can write its ``interrupted.pt``, instead of killing it.
  """
  process = subprocess.Popen(list(command))
  try:
    return process.wait()
  except KeyboardInterrupt:
    process.wait()
    raise


def _write_status(run_directory: pathlib.Path, status: Dict[str, Any]) -> None:
  (run_directory / STATUS_FILE).write_text(json.dumps(status, indent=2))


def _format_duration(seconds: float) -> str:
  hours, rest = divmod(int(seconds), 3600)
  minutes, seconds_ = divmod(rest, 60)
  return f"{hours}h{minutes:02d}m{seconds_:02d}s"


def _prepare(stage: Stage) -> None:
  """Write the stage's config; an interrupted stage resumes from its checkpoint."""
  stage.directory.mkdir(parents=True, exist_ok=True)
  if stage.task in TRAINING_TASKS and not stage.complete:
    checkpoint = _latest_checkpoint(stage.directory)
    if checkpoint is not None:
      stage.config["resume_from"] = str(checkpoint)
  text = OmegaConf.to_yaml(OmegaConf.create(stage.config))
  if stage.complete and stage.config_file.exists():
    if stage.config_file.read_text() != text:
      click.secho(
        f"Stage {stage.name!r} is complete but its config changed since: it "
        "is not re-run. Use a new run directory to train it again.",
        fg="yellow",
      )
    return
  stage.config_file.write_text(text)


def _resume_command(
  overrides: Sequence[str], run_directory: pathlib.Path
) -> str:
  """The command that resumes ``run_directory`` (replacing any ``resume_from``)."""
  kept = [item for item in overrides if not item.startswith("resume_from=")]
  return " ".join(
    ["diffusion-models train pipeline", *kept, f"resume_from={run_directory}"]
  )


def run_pipeline(
  overrides: Sequence[str] = (),
  config_directories: Sequence[pathlib.Path] = (),
  dry_run: bool = False,
) -> None:
  """Compose a pipeline from hydra ``overrides`` and run its stages in order.

  Args:
    overrides: Hydra overrides of the pipeline config, e.g.
      ``+pipeline=latent_celeba_f8c16`` or
      ``stages.diffuser.config.training.number_of_epochs=50``.
    config_directories: Extra directories of configs and presets.
    dry_run: Print every stage's command and resolved config instead of running.
  """
  pipeline_config = _compose("pipeline", overrides, config_directories)
  resume_from = pipeline_config.get("resume_from")
  if resume_from:
    run_directory = pathlib.Path(resume_from).resolve()
    if not run_directory.is_dir():
      raise PipelineError(f"resume_from={resume_from}: no such directory.")
  else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = (
      pathlib.Path(pipeline_config.get("root", "runs"))
      / str(pipeline_config.get("name", "pipeline"))
      / timestamp
    ).resolve()

  stages = compose_stages(pipeline_config, run_directory, config_directories)

  if dry_run:
    click.secho(f"Pipeline run directory: {run_directory}", bold=True)
    for stage in stages:
      state = " (complete, would be skipped)" if stage.complete else ""
      click.secho(f"\n== {stage.name}: {stage.task}{state}", bold=True)
      click.echo(" ".join(stage.command))
      click.echo(OmegaConf.to_yaml(OmegaConf.create(stage.config)))
    return

  run_directory.mkdir(parents=True, exist_ok=True)
  (run_directory / "pipeline.yaml").write_text(
    OmegaConf.to_yaml(pipeline_config)
  )
  status: Dict[str, Any] = {
    stage.name: {"task": stage.task, "status": "pending"} for stage in stages
  }
  for stage in stages:
    _prepare(stage)
    if stage.complete:
      status[stage.name]["status"] = "skipped (already complete)"
  _write_status(run_directory, status)

  click.secho(f"Pipeline run directory: {run_directory}", bold=True)
  for number, stage in enumerate(stages, start=1):
    header = f"[{number}/{len(stages)}] {stage.name} ({stage.task})"
    if stage.complete:
      click.secho(f"{header}: already complete, skipping.", fg="green")
      continue
    click.secho(f"{header}: {stage.directory}", bold=True)
    entry = status[stage.name]
    entry.update(status="running", started=datetime.datetime.now().isoformat())
    _write_status(run_directory, status)
    started = time.monotonic()
    try:
      code = _run_command(stage.command)
    except KeyboardInterrupt:
      entry["status"] = "interrupted"
      _write_status(run_directory, status)
      raise PipelineError(
        f"Interrupted during stage {stage.name!r}. Resume with: "
        f"{_resume_command(overrides, run_directory)}"
      ) from None
    seconds = time.monotonic() - started
    entry["seconds"] = round(seconds)
    if code != 0:
      entry["status"] = f"failed (exit code {code})"
      _write_status(run_directory, status)
      raise PipelineError(
        f"Stage {stage.name!r} failed with exit code {code}. Fix it and "
        f"resume with: {_resume_command(overrides, run_directory)}"
      )
    (stage.directory / COMPLETE_MARKER).write_text(
      datetime.datetime.now().isoformat()
    )
    entry["status"] = "complete"
    _write_status(run_directory, status)
    click.secho(f"{header}: done in {_format_duration(seconds)}.", fg="green")

  click.secho("Pipeline complete.", bold=True)
  for stage in stages:
    click.echo(f"  {stage.name}: {stage.directory}")
