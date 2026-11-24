import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from diffusion_models.diffusion.trainer import DiffusionTrainer
from diffusion_models.diffusion.diffusers.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.diffusion.diffusers.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.diffusion.models.SmallAttentionUnet import (
  SmallAttentionUnet,
)
from diffusion_models.core.schemas import (
  LogConfiguration,
  TrainingConfiguration,
)
from diffusion_models.diffusion.schemas import Checkpoint


def _trainer(
  tmp_path,
  loss_function=F.mse_loss,
  resume_from=None,
  prediction_type="epsilon",
  log_configuration=None,
  scheduler_steps=1000,
  **configuration,
):
  model = SmallAttentionUnet(
    diffuser=GaussianDiffuser(
      LinearBetaScheduler(steps=scheduler_steps),
      prediction_type=prediction_type,
    ),
    image_channels=3,
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
  )
  dataset = TensorDataset(torch.randn(8, 3, 8, 8), torch.zeros(8))
  return DiffusionTrainer(
    model=model,
    dataset=dataset,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="test",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=3,
      **{"num_workers": 0, **configuration},
    ),
    loss_function=loss_function,
    log_configuration=log_configuration
    or LogConfiguration(number_of_images=0),
    device="cpu",
    output_directory=tmp_path,
    resume_from=resume_from,
  )


def test_ema_and_latest_checkpoints(tmp_path):
  trainer = _trainer(
    tmp_path, ema_decay=0.9, latest_checkpoint_rate=1, checkpoint_rate=100
  )
  trainer.train()

  checkpoints = tmp_path / "checkpoints"
  assert (checkpoints / "final.pt").is_file()
  assert not list(checkpoints.glob("*.tmp"))
  latest = Checkpoint.from_file(checkpoints / "latest.pt")
  assert latest.epoch == 3
  assert latest.ema_model_state_dict is not None
  assert trainer.ema.num_updates == 6
  raw = trainer.model.state_dict()
  assert any(
    not torch.equal(raw[k], v) for k, v in latest.ema_model_state_dict.items()
  )
  assert (
    latest.get_model_state_dict(use_ema=True) is latest.ema_model_state_dict
  )
  assert latest.get_model_state_dict(use_ema=False) is latest.model_state_dict


def test_no_ema_by_default(tmp_path):
  trainer = _trainer(tmp_path)
  trainer.train()
  checkpoint = Checkpoint.from_file(tmp_path / "checkpoints" / "final.pt")
  assert trainer.ema is None
  assert checkpoint.ema_model_state_dict is None
  assert not (tmp_path / "checkpoints" / "latest.pt").exists()


def test_interrupt_saves_checkpoint(tmp_path):
  calls = {"count": 0}

  def interrupting_loss(target, prediction):
    calls["count"] += 1
    if calls["count"] == 3:  # Epoch 1 (0-indexed), first step
      raise KeyboardInterrupt
    return F.mse_loss(target, prediction)

  trainer = _trainer(tmp_path, loss_function=interrupting_loss, ema_decay=0.9)
  with pytest.raises(KeyboardInterrupt):
    trainer.train()

  checkpoints = tmp_path / "checkpoints"
  interrupted = Checkpoint.from_file(checkpoints / "interrupted.pt")
  assert interrupted.epoch == 1
  assert interrupted.ema_model_state_dict is not None
  assert not (checkpoints / "final.pt").exists()


def test_old_checkpoint_without_ema_loads(tmp_path):
  trainer = _trainer(tmp_path)
  trainer.save_checkpoint(epoch=1, checkpoint_name="a.pt")
  path = tmp_path / "checkpoints" / "a.pt"
  data = torch.load(path, weights_only=True)
  data.pop("ema_model_state_dict")
  torch.save(data, path)
  checkpoint = Checkpoint.from_file(path)
  assert checkpoint.ema_model_state_dict is None
  assert checkpoint.get_model_state_dict() is checkpoint.model_state_dict


def test_resume_matches_uninterrupted_run(tmp_path):
  torch.manual_seed(0)
  first = _trainer(tmp_path / "a", ema_decay=0.9, latest_checkpoint_rate=1)
  first.training_configuration.number_of_epochs = 2
  first.train()
  latest = tmp_path / "a" / "checkpoints" / "latest.pt"
  assert Checkpoint.from_file(latest).completed_epochs == 2
  assert Checkpoint.from_file(latest).ema_num_updates == 4

  resumed = _trainer(tmp_path / "b", ema_decay=0.9, resume_from=latest)
  assert resumed._completed_epochs == 2
  assert resumed.ema.num_updates == 4
  for key, value in first.model.state_dict().items():
    assert torch.equal(resumed.model.state_dict()[key], value)
  for key, value in first.ema.state_dict().items():
    assert torch.equal(resumed.ema.state_dict()[key], value)
  assert resumed.optimizer.state_dict()["state"]

  resumed.train()  # number_of_epochs=3 -> exactly one more epoch
  final = Checkpoint.from_file(tmp_path / "b" / "checkpoints" / "final.pt")
  assert final.completed_epochs == 3
  assert resumed.ema.num_updates == 6


def test_resume_from_interrupted_checkpoint(tmp_path):
  calls = {"count": 0}

  def interrupting_loss(target, prediction):
    calls["count"] += 1
    if calls["count"] == 3:
      raise KeyboardInterrupt
    return F.mse_loss(target, prediction)

  trainer = _trainer(tmp_path / "a", loss_function=interrupting_loss)
  with pytest.raises(KeyboardInterrupt):
    trainer.train()
  resumed = _trainer(
    tmp_path / "b",
    resume_from=tmp_path / "a" / "checkpoints" / "interrupted.pt",
  )
  assert resumed._completed_epochs == 1
  resumed.train()
  assert (tmp_path / "b" / "checkpoints" / "final.pt").is_file()


def test_resume_legacy_checkpoint_without_completed_epochs(tmp_path):
  trainer = _trainer(tmp_path / "a")
  trainer.save_checkpoint(epoch=2, checkpoint_name="old.pt")
  path = tmp_path / "a" / "checkpoints" / "old.pt"
  data = torch.load(path, weights_only=True)
  data.pop("completed_epochs")
  data.pop("ema_num_updates")
  torch.save(data, path)
  assert _trainer(tmp_path / "b", resume_from=path)._completed_epochs == 2


def test_resume_into_ema_run_from_checkpoint_without_ema(tmp_path):
  _trainer(tmp_path / "a").save_checkpoint(epoch=1, checkpoint_name="a.pt")
  resumed = _trainer(
    tmp_path / "b",
    ema_decay=0.9,
    resume_from=tmp_path / "a" / "checkpoints" / "a.pt",
  )
  for key, value in resumed.model.state_dict().items():
    assert torch.equal(resumed.ema.state_dict()[key], value)


def test_dataloader_settings(tmp_path):
  loader = _trainer(tmp_path / "a", num_workers=0).dataloader
  assert loader.num_workers == 0
  assert not loader.persistent_workers
  assert not loader.pin_memory  # device="cpu"

  loader = _trainer(
    tmp_path / "b", num_workers=2, persistent_workers=False
  ).dataloader
  assert loader.num_workers == 2
  assert not loader.persistent_workers


def test_compiled_training_keeps_clean_keys(tmp_path, monkeypatch):
  # Use the eager backend so the test does not need a C++ toolchain.
  real_compile = torch.compile
  monkeypatch.setattr(
    torch, "compile", lambda m: real_compile(m, backend="eager")
  )
  trainer = _trainer(tmp_path, compile=True, ema_decay=0.9)
  trainer.training_configuration.number_of_epochs = 1
  trainer.train()
  checkpoint = Checkpoint.from_file(tmp_path / "checkpoints" / "final.pt")
  assert not any("_orig_mod" in k for k in checkpoint.model_state_dict)
  assert not any("_orig_mod" in k for k in checkpoint.ema_model_state_dict)


def test_bfloat16_has_no_grad_scaler_and_trains(tmp_path):
  trainer = _trainer(
    tmp_path, mixed_precision_training=True, mixed_precision_dtype="bfloat16"
  )
  assert not trainer.scaler.is_enabled()
  trainer.train()
  checkpoint = Checkpoint.from_file(tmp_path / "checkpoints" / "final.pt")
  assert checkpoint.scaler is None


def test_float16_enables_grad_scaler(tmp_path):
  trainer = _trainer(
    tmp_path, mixed_precision_training=True, mixed_precision_dtype="float16"
  )
  assert trainer._autocast_dtype == torch.float16
  # GradScaler is disabled on CPU (with a warning) but requested for CUDA.
  assert trainer.training_configuration.mixed_precision_dtype == "float16"


def test_invalid_mixed_precision_dtype(tmp_path):
  with pytest.raises(ValueError, match="mixed_precision_dtype"):
    _trainer(tmp_path, mixed_precision_dtype="float8")


def test_channels_last(tmp_path):
  trainer = _trainer(
    tmp_path,
    channels_last=True,
    mixed_precision_training=True,
    ema_decay=0.9,
  )
  weights = [p for p in trainer.model.parameters() if p.dim() == 4]
  assert weights
  assert all(
    p.is_contiguous(memory_format=torch.channels_last) for p in weights
  )
  trainer.train()
  assert (tmp_path / "checkpoints" / "final.pt").is_file()


def test_v_prediction_trains_and_is_stored_in_the_checkpoint(tmp_path):
  trainer = _trainer(tmp_path, prediction_type="v")
  trainer.train()
  checkpoint = Checkpoint.from_file(tmp_path / "checkpoints" / "final.pt")
  assert checkpoint.prediction_type == "v"
  assert (
    Checkpoint.from_file(tmp_path / "checkpoints" / "final.pt").prediction_type
    == "v"
  )


def test_resume_refuses_a_different_prediction_type(tmp_path):
  _trainer(tmp_path / "a", prediction_type="v").train()
  with pytest.raises(ValueError, match="predict"):
    _trainer(
      tmp_path / "b",
      prediction_type="epsilon",
      resume_from=tmp_path / "a" / "checkpoints" / "final.pt",
    )


def test_images_are_one_series_with_a_slider_over_the_epochs(tmp_path):
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  trainer = _trainer(
    tmp_path,
    # Two steps per epoch: an image grid after the first step of each epoch.
    log_configuration=LogConfiguration(image_rate=2, number_of_images=3),
    scheduler_steps=20,
  )
  trainer.train()
  trainer.tensorboard_manager.summary_writer.flush()

  accumulator = EventAccumulator(
    trainer.tensorboard_manager.log_directory,
    size_guidance={"images": 0},
  )
  accumulator.Reload()
  tags = accumulator.Tags()["images"]
  # One tab per series, not one per epoch.
  assert sorted(tags) == ["Denoising", "Samples"]
  # The slider is the number of completed epochs.
  for tag in tags:
    assert [image.step for image in accumulator.Images(tag)] == [0, 1, 2]


def test_the_logged_sample_is_the_last_step_of_the_reverse_process(tmp_path):
  import io

  from PIL import Image
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  trainer = _trainer(
    tmp_path,
    log_configuration=LogConfiguration(image_rate=2, number_of_images=2),
    scheduler_steps=20,
  )
  shape = (2, *trainer._image_shape)
  # Black (first step) to white (the sample): a reversed order would show black.
  steps = [torch.zeros(shape), torch.full(shape, 0.5), torch.ones(shape)]
  trainer.model.denoise = lambda noise: steps
  trainer.log_images(global_step=0)
  trainer.tensorboard_manager.summary_writer.flush()

  accumulator = EventAccumulator(
    trainer.tensorboard_manager.log_directory, size_guidance={"images": 0}
  )
  accumulator.Reload()

  def brightness(tag):
    (event,) = accumulator.Images(tag)
    return Image.open(io.BytesIO(event.encoded_image_string)).convert("L")

  assert brightness("Samples").getpixel((5, 5)) > 240  # white: the last step
  rows = brightness("Denoising")
  height = rows.height
  # The first row is the first step (black), the last row the sample (white).
  assert rows.getpixel((5, 5)) < 15
  assert rows.getpixel((5, height - 5)) > 240


def test_learning_rate_schedule_values(tmp_path):
  # 8 samples with batch 4 and drop_last: 2 steps per epoch, 3 epochs = 6 steps.
  constant = _trainer(tmp_path / "constant")
  assert constant.learning_rate_factor(3) is None

  cosine = _trainer(
    tmp_path / "cosine", lr_schedule="cosine", lr_warmup_steps=2
  )
  factors = [cosine.learning_rate_factor(step) for step in range(6)]
  assert factors[:2] == [0.5, 1.0]  # linear warmup
  assert factors[2] == pytest.approx(1.0)  # the decay starts at the peak
  assert factors == sorted(factors[:2]) + sorted(factors[2:], reverse=True)
  assert factors[-1] < 0.15

  linear = _trainer(
    tmp_path / "linear", lr_schedule="linear", lr_min_ratio=0.1
  )
  assert linear.learning_rate_factor(0) == pytest.approx(1.0)
  assert (
    linear.learning_rate_factor(5) > 0.1
  )  # approaches, never below, the floor


def test_learning_rate_schedule_can_start_at_a_later_epoch(tmp_path):
  trainer = _trainer(tmp_path, lr_schedule="linear", lr_schedule_start_epoch=1)
  # Constant through epoch 0 (steps 0-1), then decays over epochs 1-2.
  assert [trainer.learning_rate_factor(s) for s in (0, 1, 2)] == [
    1.0,
    1.0,
    1.0,
  ]
  assert trainer.learning_rate_factor(3) < 1.0
  assert trainer.learning_rate_factor(5) < trainer.learning_rate_factor(3)


def test_the_schedule_is_applied_logged_and_survives_a_resume(tmp_path):
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  first = _trainer(
    tmp_path / "first",
    lr_schedule="cosine",
    latest_checkpoint_rate=1,
    log_configuration=LogConfiguration(log_rate=1, number_of_images=0),
  )
  first.train()
  first.tensorboard_manager.summary_writer.flush()
  # The optimizer holds the rate of the last step, far below the base 1e-3.
  assert first.optimizer.param_groups[0]["lr"] < 1.5e-4
  accumulator = EventAccumulator(first.tensorboard_manager.log_directory)
  accumulator.Reload()
  rates = [e.value for e in accumulator.Scalars("LearningRate")]
  assert rates[0] == pytest.approx(1e-3) and rates == sorted(
    rates, reverse=True
  )

  # A resumed run reads the base rate from the config, not the decayed state.
  resumed = _trainer(
    tmp_path / "second",
    lr_schedule="cosine",
    resume_from=tmp_path / "first" / "checkpoints" / "latest.pt",
  )
  assert resumed._base_learning_rates == [1e-3]
  assert resumed.learning_rate_factor(0) == pytest.approx(1.0)


def test_invalid_learning_rate_schedules_are_refused(tmp_path):
  with pytest.raises(ValueError, match="lr_schedule"):
    _trainer(tmp_path, lr_schedule="step")
  with pytest.raises(ValueError, match="lr_min_ratio"):
    _trainer(tmp_path, lr_schedule="cosine", lr_min_ratio=2.0)
  with pytest.raises(ValueError, match="lr_schedule_start_epoch"):
    _trainer(tmp_path, lr_schedule="cosine", lr_schedule_start_epoch=3)
