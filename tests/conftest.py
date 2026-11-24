import pytest


@pytest.fixture(scope="session", autouse=True)
def plain_loaders_for_resources():
  """Let ``importlib.resources`` traverse packages that pytest 6 rewrites.

  hydra-core's pytest plugin registers ``hydra`` for assertion rewriting. Old
  pytest versions then wrap the package loader in an object without a resource
  reader, which breaks hydra's ``pkg://`` config sources. Only test modules
  need rewriting, so give the other packages their plain loader back.
  """
  import importlib.machinery
  import sys

  for name, module in list(sys.modules.items()):
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    if (
      type(loader).__name__ == "AssertionRewritingHook"
      and spec.origin
      and spec.origin.endswith(".py")
      and not name.startswith("tests")
    ):
      spec.loader = importlib.machinery.SourceFileLoader(name, spec.origin)
      module.__loader__ = spec.loader


@pytest.fixture(scope="session", autouse=True)
def isolated_completion_cache(tmp_path_factory):
  """Keep the tests away from the user's completion cache."""
  import os

  previous = os.environ.get("DIFFUSION_MODELS_CACHE_DIR")
  os.environ["DIFFUSION_MODELS_CACHE_DIR"] = str(
    tmp_path_factory.mktemp("completion-cache")
  )
  yield
  if previous is None:
    os.environ.pop("DIFFUSION_MODELS_CACHE_DIR", None)
  else:
    os.environ["DIFFUSION_MODELS_CACHE_DIR"] = previous


def pytest_configure(config):
  # The live store and its exported snapshot would register a group twice.
  config.addinivalue_line(
    "filterwarnings",
    "ignore:(?s).*is validated against ConfigStore schema:UserWarning",
  )
