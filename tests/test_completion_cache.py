import os
import pathlib
import subprocess
import sys

import pytest
import yaml
from hydra.plugins.completion_plugin import DefaultCompletionPlugin

from diffusion_models import _completion_cache as cache
from diffusion_models import _configs

QUERIES = [
  ("diffuser", [], "mo"),
  ("diffuser", [], "model="),
  ("diffuser", [], "+config="),
  ("diffuser", [], "training.batch_size="),
  ("diffuser", ["model=small_attention_unet"], "model.num_res"),
  ("diffuser", [], "diffuser="),
  ("diffuser", [], "diffuser.mode="),
  ("autoencoder", ["+config=vae_f8"], "autoencoder.base_ch"),
  ("autoencoder", [], "loss."),
  ("autoencoder", [], "optimizer.lr="),
]


def test_export_writes_every_group_option_as_yaml(tmp_path):
  cache.export(tmp_path)
  groups = {p.name for p in tmp_path.iterdir() if p.is_dir()}
  assert {"model", "loss", "dataset", "diffuser", "autoencoder"} <= groups
  assert "hydra" not in groups
  loaded = yaml.safe_load(
    (tmp_path / "model" / "small_attention_unet.yaml").read_text()
  )
  assert loaded["_target_"].endswith("SmallAttentionUnet")
  assert loaded["base_channels"] == 64
  assert (tmp_path / "loss" / "vae.yaml").exists()


@pytest.mark.parametrize("task,previous,incomplete", QUERIES)
def test_cached_completion_matches_the_live_store(task, previous, incomplete):
  _configs._register()
  live = DefaultCompletionPlugin(_configs._loader(()))._query(
    _configs.TASKS[task].config_name, " ".join([*previous, incomplete])
  )
  assert live  # the queries are meaningful
  assert _configs.complete_overrides(task, previous, incomplete) == live


def test_snapshot_is_reused_until_the_sources_change(tmp_path, monkeypatch):
  monkeypatch.setenv("DIFFUSION_MODELS_CACHE_DIR", str(tmp_path))
  exports = []
  monkeypatch.setattr(cache, "export", lambda destination: exports.append(1))
  monkeypatch.setattr(cache, "fingerprint", lambda *a: "aaaa")

  first = cache.ensure()
  assert cache.ensure() == first == cache.current_directory()
  assert len(exports) == 1

  monkeypatch.setattr(cache, "fingerprint", lambda *a: "bbbb")
  assert cache.current_directory() is None  # stale
  second = cache.ensure()
  assert second.name == "bbbb" and len(exports) == 2
  # Earlier snapshots are pruned and no temporary directory is left.
  assert [p.name for p in tmp_path.iterdir()] == ["bbbb"]


def test_fingerprint_follows_the_sources(tmp_path):
  (tmp_path / "a.py").write_text("x = 1\n")
  before = cache.fingerprint(tmp_path)
  assert cache.fingerprint(tmp_path) == before
  (tmp_path / "a.py").write_text("x = 22\n")
  assert cache.fingerprint(tmp_path) != before
  (tmp_path / "b.py").write_text("")
  assert cache.fingerprint(tmp_path) != cache.fingerprint(tmp_path / "..")


def test_a_failed_export_leaves_no_snapshot(tmp_path, monkeypatch):
  monkeypatch.setenv("DIFFUSION_MODELS_CACHE_DIR", str(tmp_path))
  monkeypatch.setattr(cache, "export", lambda destination: 1 / 0)
  with pytest.raises(ZeroDivisionError):
    cache.ensure()
  assert cache.current_directory() is None
  assert list(tmp_path.iterdir()) == []
  cache.refresh()  # a cache never raises


def test_completion_needs_neither_torch_nor_the_store(tmp_path):
  environment = {**os.environ, "DIFFUSION_MODELS_CACHE_DIR": str(tmp_path)}
  cache_dir = pathlib.Path(tmp_path)
  # Build the snapshot in this process, then complete in a fresh one.
  os.environ["DIFFUSION_MODELS_CACHE_DIR"], previous = (
    str(tmp_path),
    os.environ.get("DIFFUSION_MODELS_CACHE_DIR"),
  )
  try:
    cache.ensure()
  finally:
    os.environ["DIFFUSION_MODELS_CACHE_DIR"] = previous
  assert any(cache_dir.iterdir())
  script = (
    "import sys\n"
    "from diffusion_models import _configs\n"
    "print(_configs.complete_overrides('diffuser', [], 'model='))\n"
    "print(sorted(m for m in ('torch', 'torchvision', 'hydra_zen') if m in sys.modules))\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", script],
    env=environment,
    capture_output=True,
    text=True,
    check=True,
  )
  matches, loaded = result.stdout.strip().splitlines()[-2:]
  assert "model=small_attention_unet" in matches
  assert loaded == "[]"


def test_extra_config_directories_are_completed_too(tmp_path):
  (tmp_path / "config").mkdir()
  (tmp_path / "config" / "mine.yaml").write_text(
    "# @package _global_\nseed: 3\n"
  )
  presets = _configs.complete_overrides(
    "diffuser", [], "+config=", config_directories=[tmp_path]
  )
  assert "+config=mine" in presets and "+config=vae_f8" in presets


def test_list_options_uses_the_snapshot_and_extra_presets(tmp_path):
  (tmp_path / "config").mkdir()
  (tmp_path / "config" / "mine.yaml").write_text(
    "# @package _global_\nseed: 3\n"
  )
  options = _configs.list_options("autoencoder", [tmp_path])
  assert options["groups"]["loss"]["default"] == "vae"
  assert "mine" in options["presets"]
