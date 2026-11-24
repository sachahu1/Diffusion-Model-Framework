import pytest
import torch
from omegaconf import OmegaConf

from diffusion_models.autoencoders import ConvVae
from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint

CONFIG = OmegaConf.to_yaml(
  OmegaConf.create(
    {
      "autoencoder": {
        "_target_": "diffusion_models.autoencoders.vae.ConvVae",
        "base_channels": 32,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "latent_channels": 2,
      }
    }
  )
)


def _model(seed):
  torch.manual_seed(seed)
  return ConvVae(
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    latent_channels=2,
  )


def _save(path, raw, ema=None, config=CONFIG):
  AutoencoderCheckpoint(
    epoch=3,
    completed_epochs=3,
    model_state_dict=raw.state_dict(),
    optimizer_state_dict={},
    ema_model_state_dict=ema.state_dict() if ema is not None else None,
    config_yaml=config,
  ).to_file(path)


def test_load_rebuilds_the_architecture_from_the_checkpoint(tmp_path):
  raw = _model(0)
  _save(tmp_path / "a.pt", raw)
  model = BaseAutoencoder.load(tmp_path / "a.pt", use_ema=False)
  assert isinstance(model, ConvVae)
  assert model.latent_channels == 2
  for key, value in raw.state_dict().items():
    assert torch.equal(model.state_dict()[key], value)


def test_load_prefers_the_ema_weights(tmp_path):
  raw, ema = _model(0), _model(1)
  _save(tmp_path / "a.pt", raw, ema)
  from_ema = ConvVae.load(tmp_path / "a.pt")
  from_raw = ConvVae.load(tmp_path / "a.pt", use_ema=False)
  key = "decoder.conv_out.weight"
  assert torch.equal(from_ema.state_dict()[key], ema.state_dict()[key])
  assert torch.equal(from_raw.state_dict()[key], raw.state_dict()[key])


def test_load_needs_an_embedded_config(tmp_path):
  _save(tmp_path / "a.pt", _model(0), config=None)
  with pytest.raises(ValueError, match="no embedded config"):
    BaseAutoencoder.load(tmp_path / "a.pt")


def test_load_checks_the_type_of_the_model(tmp_path):
  _save(tmp_path / "a.pt", _model(0))

  class Other(BaseAutoencoder):
    def encode_distribution(self, images):
      raise NotImplementedError

    def decode(self, latents):
      raise NotImplementedError

  with pytest.raises(TypeError, match="not a Other"):
    Other.load(tmp_path / "a.pt")


def test_loaded_model_exposes_the_decoder_hooks(tmp_path):
  model = _model(0)
  assert model.last_layer() is model.decoder.conv_out.weight
  decoder = {id(p) for p in model.decoder_parameters()}
  encoder = {id(p) for p in model.encoder.parameters()}
  assert decoder and not decoder & encoder
