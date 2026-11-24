"""Backwards compatibility for modules that moved.

The package was reorganised into ``diffusion_models.core`` (task agnostic
building blocks) and ``diffusion_models.diffusion`` (everything diffusion
specific). The old import paths keep working: importing one returns the very
same module object as its new location (so class identity is preserved) and
emits a :class:`DeprecationWarning`. This also resolves the old ``_target_``
paths stored in the configuration of existing checkpoints.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings
from typing import Dict, Optional, Tuple

#: Old module path -> new module path. A prefix also covers its submodules.
RENAMES: Dict[str, str] = {
  "diffusion_models.evaluation.checkpoint": "diffusion_models.diffusion.evaluation",
  "diffusion_models.models._modules": "diffusion_models.core.modules",
  "diffusion_models.gaussian_diffusion": "diffusion_models.diffusion.diffusers",
  "diffusion_models.models": "diffusion_models.diffusion.models",
  "diffusion_models.base_trainer": "diffusion_models.core.trainer",
  "diffusion_models.diffusion_trainer": "diffusion_models.diffusion.trainer",
  "diffusion_models.diffusion_inference": "diffusion_models.diffusion.diffusion_inference",
  "diffusion_models.training": "diffusion_models.diffusion.training",
  "diffusion_models.inference": "diffusion_models.diffusion.inference",
  "diffusion_models.autoencoder_training": "diffusion_models.autoencoders.training",
  "diffusion_models.evaluation": "diffusion_models.core.evaluation",
  "diffusion_models.utils": "diffusion_models.core.utils",
}

#: Old modules that were split in two: the union of the new modules' names.
MERGED: Dict[str, Tuple[str, ...]] = {
  "diffusion_models.utils.schemas": (
    "diffusion_models.core.schemas",
    "diffusion_models.diffusion.schemas",
  ),
}


def _new_name(fullname: str) -> Optional[str]:
  for old, new in sorted(RENAMES.items(), key=lambda item: -len(item[0])):
    if fullname == old or fullname.startswith(old + "."):
      return new + fullname[len(old) :]
  return None


class _MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
  """Serves old module paths with their relocated module objects."""

  def __init__(self) -> None:
    self._original: Dict[str, Tuple[object, object]] = {}
    self._merged: Dict[str, types.ModuleType] = {}

  def _merged_module(self, fullname: str) -> types.ModuleType:
    if fullname not in self._merged:
      module = types.ModuleType(fullname)
      for source in MERGED[fullname]:
        origin = importlib.import_module(source)
        module.__dict__.update(
          {k: v for k, v in vars(origin).items() if not k.startswith("__")}
        )
      self._merged[fullname] = module
    return self._merged[fullname]

  def find_spec(self, fullname, path=None, target=None):
    if fullname in MERGED or _new_name(fullname) is not None:
      return importlib.util.spec_from_loader(fullname, self)
    return None

  def create_module(self, spec):
    fullname = spec.name
    if fullname in MERGED:
      module = self._merged_module(fullname)
      new_name = " and ".join(MERGED[fullname])
    else:
      new_name = _new_name(fullname)
      module = importlib.import_module(new_name)  # ty: ignore[invalid-argument-type]
      # Importing sets ``__spec__`` to this alias; remember the real one.
      self._original[fullname] = (module.__spec__, module.__loader__)
      # ``from old_package import merged_module`` looks up an attribute.
      for merged_name in MERGED:
        parent, _, leaf = merged_name.rpartition(".")
        if parent == fullname:
          setattr(module, leaf, self._merged_module(merged_name))
    warnings.warn(
      f"{fullname} has moved to {new_name}; the old import path is deprecated.",
      DeprecationWarning,
      stacklevel=2,
    )
    return module

  def exec_module(self, module):
    # Undo the spec the import machinery overwrote with the alias'.
    original = self._original.pop(module.__spec__.name, None)
    if original is not None:
      module.__spec__, module.__loader__ = original  # type: ignore[assignment]


_FINDER = _MovedModuleFinder()


def install() -> None:
  """Register the import hook (idempotent)."""
  if not any(isinstance(f, _MovedModuleFinder) for f in sys.meta_path):
    # Ahead of the path finder, which would load a second copy.
    sys.meta_path.insert(0, _FINDER)
