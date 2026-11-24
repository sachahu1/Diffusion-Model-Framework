import pytest
import torch
from torch.utils.data import TensorDataset

from diffusion_models.autoencoders import ConvVae
from diffusion_models.autoencoders.checkpoint import AutoencoderCheckpoint
from diffusion_models.core.losses import VaeLoss
from diffusion_models.autoencoders.schemas import ValidationConfiguration
from diffusion_models.autoencoders.trainer import AutoencoderTrainer
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)


def _trainer(tmp_path, loss=None, resume_from=None, epochs=3, **configuration):
  model = ConvVae(
    base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1
  )
  return AutoencoderTrainer(
    model=model,
    dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="test",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=epochs,
      **{"num_workers": 0, **configuration},
    ),
    loss=loss,
    log_configuration=LogConfiguration(
      log_rate=1, image_rate=1, number_of_images=2
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


def test_trains_and_writes_checkpoints(tmp_path):
  trainer = _trainer(
    tmp_path, ema_decay=0.9, latest_checkpoint_rate=1, checkpoint_rate=100
  )
  trainer.train()
  checkpoints = tmp_path / "checkpoints"
  assert {p.name for p in checkpoints.iterdir()} == {
    "epoch_0.pt",
    "latest.pt",
    "final.pt",
  }
  latest = AutoencoderCheckpoint.from_file(checkpoints / "latest.pt")
  assert latest.completed_epochs == 3
  assert latest.ema_num_updates == 6
  assert latest.get_model_state_dict(True) is latest.ema_model_state_dict
  assert latest.get_model_state_dict(False) is latest.model_state_dict


def test_validate_returns_metrics(tmp_path):
  trainer = _trainer(tmp_path)
  metrics = trainer.validate()
  assert set(metrics) == {"L1", "PSNR", "KL", "LatentStd"}
  assert metrics["L1"] > 0 and metrics["LatentStd"] > 0


def test_validate_restores_training_mode(tmp_path):
  trainer = _trainer(tmp_path)
  trainer.model.train()
  trainer.validate()
  assert trainer.model.training


def test_logs_validation_scalars_and_reconstructions(tmp_path):
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  trainer = _trainer(tmp_path, epochs=1)
  trainer.train()
  trainer.tensorboard_manager.summary_writer.flush()
  accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
  accumulator.Reload()
  scalars = set(accumulator.Tags()["scalars"])
  assert {"Train/Loss", "Train/KL", "Train/LatentStd", "Val/PSNR"} <= scalars
  assert accumulator.Tags()["images"]


def test_interrupt_saves_a_resumable_checkpoint(tmp_path):
  calls = 0
  base = VaeLoss()

  def interrupting_loss(images, reconstruction, mean, logvar):
    nonlocal calls
    if mean.requires_grad:  # training steps only, not validation
      calls += 1
      if calls == 5:  # the first step of the third epoch (2 steps each)
        raise KeyboardInterrupt
    return base(images, reconstruction, mean, logvar)

  trainer = _trainer(tmp_path / "a", loss=interrupting_loss)
  with pytest.raises(KeyboardInterrupt):
    trainer.train()
  interrupted = tmp_path / "a" / "checkpoints" / "interrupted.pt"
  assert AutoencoderCheckpoint.from_file(interrupted).completed_epochs == 2

  resumed = _trainer(tmp_path / "b", resume_from=interrupted)
  assert resumed._completed_epochs == 2
  resumed.train()
  final = AutoencoderCheckpoint.from_file(
    tmp_path / "b" / "checkpoints" / "final.pt"
  )
  assert final.completed_epochs == 3


def test_resume_restores_weights_and_ema(tmp_path):
  trainer = _trainer(tmp_path / "a", ema_decay=0.9, latest_checkpoint_rate=1)
  trainer.train()
  resumed = _trainer(
    tmp_path / "b",
    ema_decay=0.9,
    resume_from=tmp_path / "a" / "checkpoints" / "latest.pt",
  )
  for key, value in trainer.model.state_dict().items():
    assert torch.equal(resumed.model.state_dict()[key], value)
  assert resumed.ema.num_updates == trainer.ema.num_updates
  assert resumed._completed_epochs == 3


def test_float16_enables_the_grad_scaler_only(tmp_path):
  assert not _trainer(
    tmp_path / "a", mixed_precision_training=True
  ).scaler.is_enabled()
  assert not _trainer(tmp_path / "b").scaler.is_enabled()


def test_rejects_unknown_precision(tmp_path):
  with pytest.raises(ValueError, match="mixed_precision_dtype"):
    _trainer(tmp_path, mixed_precision_dtype="int8")


def test_compiled_training_keeps_plain_checkpoint_keys(tmp_path, monkeypatch):
  real_compile = torch.compile
  monkeypatch.setattr(
    torch, "compile", lambda m, *a, **k: real_compile(m, backend="eager")
  )
  trainer = _trainer(tmp_path, epochs=1, compile=True, ema_decay=0.9)
  trainer.train()
  checkpoint = AutoencoderCheckpoint.from_file(
    tmp_path / "checkpoints" / "final.pt"
  )
  assert not any("_orig_mod" in k for k in checkpoint.model_state_dict)
  assert not any("_orig_mod" in k for k in checkpoint.ema_model_state_dict)


def test_reconstruction_fid_is_logged_at_its_rate(tmp_path, monkeypatch):
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  from diffusion_models.core.evaluation import fid as fid_module
  from tests.core.evaluation.test_fid import FakeMetric

  FakeMetric.instances = []
  monkeypatch.setattr(fid_module, "FrechetInceptionDistance", FakeMetric)

  model = ConvVae(
    base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1
  )
  trainer = AutoencoderTrainer(
    model=model,
    dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="fid",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=3,
      num_workers=0,
    ),
    validation_configuration=ValidationConfiguration(
      rate=1,
      number_of_samples=4,
      batch_size=2,
      num_workers=0,
      fid_rate=2,
      fid_samples=4,
    ),
    validation_dataset=TensorDataset(
      torch.randn(6, 3, 16, 16), torch.zeros(6)
    ),
    device="cpu",
    output_directory=tmp_path / "fid",
  )
  trainer.train()
  trainer.tensorboard_manager.summary_writer.flush()
  accumulator = EventAccumulator(str(tmp_path / "fid" / "tensorboard"))
  accumulator.Reload()
  steps = [
    e.step // len(trainer.dataloader) for e in accumulator.Scalars("Val/FID")
  ]
  assert steps == [2, 3]  # every 2 epochs, and after the last
  assert FakeMetric.instances[0].real_updates == 4
  assert FakeMetric.instances[0].fake_updates == 8  # two evaluations of 4


def test_reconstruction_fid_needs_a_validation_dataset(tmp_path):
  model = ConvVae(
    base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1
  )
  with pytest.raises(ValueError, match="validation_dataset"):
    AutoencoderTrainer(
      model=model,
      dataset=TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8)),
      optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
      training_configuration=TrainingConfiguration(
        training_name="x",
        batch_size=4,
        learning_rate=1e-3,
        number_of_epochs=1,
        num_workers=0,
      ),
      validation_configuration=ValidationConfiguration(fid_rate=1),
      device="cpu",
      output_directory=tmp_path,
    )
