import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (
  EventAccumulator,
)
from torch.utils.data import TensorDataset

from diffusion_models.autoencoders import ConvVae
from diffusion_models.autoencoders.adversarial_trainer import (
  AdversarialAutoencoderTrainer,
)
from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint
from diffusion_models.autoencoders.schemas import (
  AdversarialConfiguration,
  ValidationConfiguration,
)
from diffusion_models.core.modules import PatchDiscriminator
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)


def _trainer(
  tmp_path,
  epochs=2,
  resume_from=None,
  trainable="decoder",
  adversarial=None,
  **configuration,
):
  torch.manual_seed(0)
  model = ConvVae(
    base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1
  )
  discriminator = PatchDiscriminator(base_channels=32, num_layers=2)
  generator_parameters = (
    model.decoder_parameters()
    if trainable == "decoder"
    else model.parameters()
  )
  return AdversarialAutoencoderTrainer(
    model=model,
    dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
    optimizer=torch.optim.AdamW(generator_parameters, lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="test",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=epochs,
      **{"num_workers": 0, "gradient_clip": 1.0, **configuration},
    ),
    discriminator=discriminator,
    discriminator_optimizer=torch.optim.AdamW(
      discriminator.parameters(), lr=1e-3
    ),
    adversarial_configuration=adversarial
    or AdversarialConfiguration(warmup_steps=2, trainable=trainable),
    log_configuration=LogConfiguration(
      log_rate=1, image_rate=1000, number_of_images=2
    ),
    validation_configuration=ValidationConfiguration(
      rate=1, number_of_samples=4, batch_size=2, num_workers=0
    ),
    validation_dataset=TensorDataset(
      torch.randn(6, 3, 16, 16), torch.zeros(6)
    ),
    device="cpu",
    output_directory=tmp_path,
    resume_from=resume_from,
  )


def _snapshot(module):
  return {k: v.clone() for k, v in module.state_dict().items()}


def _changed(before, module):
  return any(
    not torch.equal(before[k], v) for k, v in module.state_dict().items()
  )


def test_decoder_only_freezes_the_encoder_and_trains_the_rest(tmp_path):
  trainer = _trainer(tmp_path, ema_decay=0.9)
  encoder, decoder, discriminator = (
    _snapshot(trainer.model.encoder),
    _snapshot(trainer.model.decoder),
    _snapshot(trainer.discriminator),
  )
  trainer.train()
  assert not _changed(encoder, trainer.model.encoder)  # frozen
  assert _changed(decoder, trainer.model.decoder)
  assert _changed(discriminator, trainer.discriminator)
  assert all(not p.requires_grad for p in trainer.model.encoder.parameters())
  assert all(p.requires_grad for p in trainer.discriminator.parameters())


def test_training_all_updates_the_encoder_too(tmp_path):
  trainer = _trainer(tmp_path, trainable="all")
  encoder = _snapshot(trainer.model.encoder)
  trainer.train()
  assert _changed(encoder, trainer.model.encoder)


def test_adversarial_metrics_are_logged_and_the_ramp_applies(tmp_path):
  trainer = _trainer(tmp_path, epochs=1)
  trainer.train()
  trainer.tensorboard_manager.summary_writer.flush()
  accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
  accumulator.Reload()
  scalars = set(accumulator.Tags()["scalars"])
  assert {
    "Train/Adversarial/Generator",
    "Train/Adversarial/Discriminator",
    "Train/Adversarial/RealLogit",
    "Train/Adversarial/FakeLogit",
    "Train/Adversarial/Weight",
    "Train/Reconstruction",
    "Train/GradNorm",
  } <= scalars
  weights = [e.value for e in accumulator.Scalars("Train/Adversarial/Weight")]
  assert all(w >= 0 and w == w for w in weights)  # finite, non negative


def test_fixed_weight_without_the_adaptive_balance(tmp_path):
  adversarial = AdversarialConfiguration(
    adaptive=False, weight=0.25, warmup_steps=0
  )
  trainer = _trainer(tmp_path, epochs=1, adversarial=adversarial)
  metrics = trainer.optimization_step(next(iter(trainer.dataloader)), 0)
  assert metrics["Adversarial/Weight"].item() == pytest.approx(0.25)


def test_checkpoint_holds_the_discriminator_and_resume_restores_it(tmp_path):
  trainer = _trainer(
    tmp_path / "a", epochs=1, latest_checkpoint_rate=1, ema_decay=0.9
  )
  trainer.train()
  saved = AutoencoderCheckpoint.from_file(
    tmp_path / "a" / "checkpoints" / "latest.pt"
  )
  assert saved.discriminator_state_dict is not None
  assert saved.discriminator_optimizer_state_dict is not None

  resumed = _trainer(
    tmp_path / "b",
    epochs=2,
    ema_decay=0.9,
    resume_from=tmp_path / "a" / "checkpoints" / "latest.pt",
  )
  for key, value in trainer.discriminator.state_dict().items():
    assert torch.equal(resumed.discriminator.state_dict()[key], value)
  assert resumed.discriminator_optimizer.state_dict()["state"]
  assert resumed._completed_epochs == 1
  resumed.train()
  assert resumed._completed_epochs == 2


def _first_stage(tmp_path):
  """Train a plain autoencoder one epoch; its checkpoint embeds a config."""
  from omegaconf import OmegaConf

  from diffusion_models.autoencoders.trainer import AutoencoderTrainer

  torch.manual_seed(0)
  model = ConvVae(
    base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1
  )
  run_config = OmegaConf.to_yaml(
    OmegaConf.create(
      {
        "autoencoder": {
          "_target_": "diffusion_models.autoencoders.vae.ConvVae",
          "base_channels": 32,
          "channel_multipliers": [1, 2],
          "num_res_blocks": 1,
        }
      }
    )
  )
  AutoencoderTrainer(
    model=model,
    dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="stage1",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=1,
      num_workers=0,
      latest_checkpoint_rate=1,
      ema_decay=0.9,
    ),
    device="cpu",
    output_directory=tmp_path,
    run_config=run_config,
  ).train()
  return tmp_path / "checkpoints" / "latest.pt"


def test_second_stage_starts_from_a_loaded_first_stage(tmp_path):
  from diffusion_models.autoencoders.base import BaseAutoencoder

  checkpoint = _first_stage(tmp_path / "stage1")
  stage1 = BaseAutoencoder.load(checkpoint)
  encoder = _snapshot(stage1.encoder)
  decoder = _snapshot(stage1.decoder)

  discriminator = PatchDiscriminator(base_channels=32, num_layers=2)
  trainer = AdversarialAutoencoderTrainer(
    model=stage1,
    dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
    optimizer=torch.optim.AdamW(stage1.decoder_parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="stage2",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=1,
      num_workers=0,
    ),
    discriminator=discriminator,
    discriminator_optimizer=torch.optim.AdamW(
      discriminator.parameters(), lr=1e-3
    ),
    adversarial_configuration=AdversarialConfiguration(warmup_steps=1),
    device="cpu",
    output_directory=tmp_path / "stage2",
  )
  # A fresh run: nothing is inherited but the weights.
  assert trainer._completed_epochs == 0
  assert not trainer.optimizer.state_dict()["state"]
  trainer.train()
  # The encoder, hence every latent of the first stage, is untouched.
  assert not _changed(encoder, stage1.encoder)
  assert _changed(decoder, stage1.decoder)


def test_resume_from_is_for_checkpoints_of_the_same_stage(tmp_path):
  # A first-stage optimizer covers the whole VAE, this stage's only the decoder.
  checkpoint = _first_stage(tmp_path / "stage1")
  with pytest.raises(ValueError, match="parameter group"):
    _trainer(tmp_path / "gan", resume_from=checkpoint)


def test_rejects_fp16_and_unknown_trainable_modes(tmp_path):
  with pytest.raises(ValueError, match="float16"):
    _trainer(
      tmp_path / "a",
      mixed_precision_training=True,
      mixed_precision_dtype="float16",
    )
  with pytest.raises(ValueError, match="trainable"):
    _trainer(
      tmp_path / "b", adversarial=AdversarialConfiguration(trainable="encoder")
    )


def test_compiled_adversarial_training_keeps_plain_checkpoint_keys(
  tmp_path, monkeypatch
):
  real_compile = torch.compile
  monkeypatch.setattr(
    torch, "compile", lambda m, *a, **k: real_compile(m, backend="eager")
  )
  trainer = _trainer(tmp_path, epochs=1, compile=True, ema_decay=0.9)
  trainer.train()
  saved = AutoencoderCheckpoint.from_file(
    tmp_path / "checkpoints" / "final.pt"
  )
  for state in (saved.model_state_dict, saved.discriminator_state_dict):
    assert not any("_orig_mod" in key for key in state)


def test_latent_noise_perturbs_the_decoder_input_but_not_the_target(tmp_path):
  noisy = _trainer(
    tmp_path / "noisy",
    adversarial=AdversarialConfiguration(latent_noise=0.5, warmup_steps=2),
  )
  clean = _trainer(tmp_path / "clean")
  images = torch.randn(4, 3, 16, 16)
  seen = []
  for trainer in (noisy, clean):
    trainer.model.eval()  # deterministic encoder: only the noise differs
    trainer._decode = lambda z, seen=seen: (
      seen.append(z) or torch.zeros(z.shape[0], 3, 16, 16)
    )
  torch.manual_seed(0)
  noisy._generate(images)
  torch.manual_seed(0)
  clean._generate(images)
  assert not torch.equal(seen[0], seen[1])
  assert (seen[0] - seen[1]).abs().max() > 0


def test_latent_noise_needs_a_decoder_only_fine_tune(tmp_path):
  with pytest.raises(ValueError, match="latent_noise"):
    _trainer(
      tmp_path,
      trainable="all",
      adversarial=AdversarialConfiguration(latent_noise=0.1, trainable="all"),
    )
