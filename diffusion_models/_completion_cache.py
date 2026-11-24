"""A torch-free snapshot of the registered configs, for fast shell completion.

Registering the config groups imports torch (seconds per TAB press), so they are
exported once to YAML in a cache directory, keyed by a fingerprint of the sources
and library versions, and Hydra completes over those. Nothing heavy is imported.
"""

import hashlib
import importlib.metadata
import os
import pathlib
import shutil
from typing import Optional

CACHE_VERSION = 1
"""Bump to invalidate every existing snapshot (e.g. when the export changes)."""

_COMPLETE_MARKER = ".complete"
_LIBRARIES = ("hydra-core", "hydra-zen", "omegaconf", "torch", "torchvision")


def cache_root() -> pathlib.Path:
  """The directory holding the snapshots (``DIFFUSION_MODELS_CACHE_DIR`` overrides it)."""
  override = os.environ.get("DIFFUSION_MODELS_CACHE_DIR")
  if override:
    return pathlib.Path(override)
  base = os.environ.get("XDG_CACHE_HOME") or str(
    pathlib.Path.home() / ".cache"
  )
  return pathlib.Path(base) / "diffusion-models" / "completion"


def _package_directory() -> pathlib.Path:
  return pathlib.Path(__file__).resolve().parent


def fingerprint(package_directory: Optional[pathlib.Path] = None) -> str:
  """A short hash of everything the exported configs depend on.

  Args:
    package_directory: The sources to hash; the installed package by default.
  """
  package_directory = package_directory or _package_directory()
  digest = hashlib.sha256(f"v{CACHE_VERSION}".encode())
  for library in _LIBRARIES:
    try:
      version = importlib.metadata.version(library)
    except importlib.metadata.PackageNotFoundError:
      version = "-"
    digest.update(f"{library}=={version};".encode())
  for path in sorted(package_directory.rglob("*.py")):
    if "__pycache__" in path.parts:
      continue
    stat = path.stat()
    relative = path.relative_to(package_directory).as_posix()
    digest.update(f"{relative}:{stat.st_mtime_ns}:{stat.st_size};".encode())
  return digest.hexdigest()[:16]


def current_directory() -> Optional[pathlib.Path]:
  """The up to date snapshot, or ``None`` when it is missing or stale."""
  directory = cache_root() / fingerprint()
  return directory if (directory / _COMPLETE_MARKER).exists() else None


def export(destination: pathlib.Path) -> None:
  """Write every registered config group to ``destination`` as YAML.

  This registers the config store, so it imports torch.
  """
  from hydra.core.config_store import ConfigStore
  from hydra_zen import store
  from omegaconf import OmegaConf

  from diffusion_models import _store  # noqa: F401  (registers the groups)

  store.add_to_hydra_store(overwrite_ok=True)

  def write(group: dict, directory: pathlib.Path) -> None:
    for name, entry in group.items():
      if isinstance(entry, dict):
        write(entry, directory / name)
        continue
      if entry.group is None:  # a top level node, not a group option
        continue
      directory.mkdir(parents=True, exist_ok=True)
      header = f"# @package {entry.package}\n" if entry.package else ""
      (directory / name).write_text(header + OmegaConf.to_yaml(entry.node))

  repo = ConfigStore.instance().repo
  # Hydra registers its own nodes under ``hydra``; those are not ours.
  write({k: v for k, v in repo.items() if k != "hydra"}, destination)


def ensure() -> pathlib.Path:
  """The directory of the up to date snapshot, building it when needed.

  Building imports torch (once per change of the sources); reusing the
  snapshot does not.
  """
  directory = current_directory()
  if directory is not None:
    return directory

  root = cache_root()
  target = root / fingerprint()
  root.mkdir(parents=True, exist_ok=True)
  temporary = root / f"{target.name}.tmp-{os.getpid()}"
  shutil.rmtree(temporary, ignore_errors=True)
  try:
    temporary.mkdir()
    export(temporary)
    (temporary / _COMPLETE_MARKER).touch()
    try:
      os.replace(temporary, target)
    except OSError:
      # Another process published the same snapshot first.
      pass
  finally:
    shutil.rmtree(temporary, ignore_errors=True)
  for old in root.iterdir():  # Drop the snapshots of earlier sources.
    if old.name != target.name and old.is_dir():
      shutil.rmtree(old, ignore_errors=True)
  return target


def refresh() -> None:
  """Bring the snapshot up to date, ignoring any failure (it is a cache)."""
  try:
    ensure()
  except Exception:
    pass
