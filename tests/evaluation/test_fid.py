import pytest
import torch
from torch.utils.data import TensorDataset

from diffusion_models.diffusion_trainer import DiffusionTrainer
from diffusion_models.evaluation import fid as fid_module
from diffusion_models.gaussian_diffusion.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.gaussian_diffusion.ddimm_diffuser import DdimDiffuser
from diffusion_models.models.SmallAttentionUnet import SmallAttentionUnet
from diffusion_models.utils.schemas import (
  EvaluationConfiguration,
  LogConfiguration,
  TrainingConfiguration,
)


class FakeMetric:
  """Stands in for the torchmetrics Inception metrics (no weights download)."""

  instances = []

  def __init__(self, *args, **kwargs):
    self.kwargs = kwargs
    self.real_updates = 0
    self.fake_updates = 0
    self.resets = 0
    self.ranges = []
    FakeMetric.instances.append(self)

  def to(self, device):
    return self

  def update(self, images, real):
    assert images.dtype == torch.float32
    self.ranges.append((images.min().item(), images.max().item()))
    if real:
      self.real_updates += images.shape[0]
    else:
      self.fake_updates += images.shape[0]

  def compute(self):
    if self.kwargs.get("subset_size") is not None:
      return torch.tensor(0.5), torch.tensor(0.1)
    return torch.tensor(12.0)

  def reset(self):
    self.resets += 1


@pytest.fixture(autouse=True)
def fake_metrics(monkeypatch):
  FakeMetric.instances = []
  monkeypatch.setattr(fid_module, "FrechetInceptionDistance", FakeMetric)
  monkeypatch.setattr(fid_module, "KernelInceptionDistance", FakeMetric)


def _model():
  return SmallAttentionUnet(
    diffuser=DdimDiffuser(LinearBetaScheduler(), number_of_steps=3),
    image_channels=3,
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
  )


def _dataset(size=8):
  return TensorDataset(torch.rand(size, 3, 8, 8) * 2 - 1, torch.zeros(size))


def test_evaluator_scores_and_caches_real_statistics():
  evaluator = fid_module.ImageQualityEvaluator(
    real_dataset=_dataset(8),
    number_of_samples=6,
    batch_size=4,
    device="cpu",
    kid_subset_size=4,
    num_workers=0,
  )
  fid, kid = FakeMetric.instances
  model = _model()
  model.train()

  metrics = evaluator.evaluate(model)
  assert metrics == pytest.approx({"FID": 12.0, "KID": 0.5, "KID_std": 0.1})
  assert fid.real_updates == 6 and fid.fake_updates == 6
  assert kid.real_updates == 6 and kid.fake_updates == 6
  assert model.training  # restored
  # Images handed to the metrics are in [0, 1].
  assert all(0.0 <= low and high <= 1.0 for low, high in fid.ranges)

  evaluator.evaluate(model)
  assert fid.real_updates == 6  # real statistics are computed only once
  assert fid.fake_updates == 12
  assert fid.resets == 2


def test_number_of_samples_is_capped_by_dataset_size():
  evaluator = fid_module.ImageQualityEvaluator(
    real_dataset=_dataset(4),
    number_of_samples=100,
    device="cpu",
    num_workers=0,
  )
  assert evaluator.number_of_samples == 4


def test_without_kid():
  evaluator = fid_module.ImageQualityEvaluator(
    real_dataset=_dataset(4),
    number_of_samples=4,
    device="cpu",
    compute_kid=False,
    num_workers=0,
  )
  assert list(evaluator.evaluate(_model())) == ["FID"]


def test_sampling_noise_is_seeded():
  seen = []

  class RecordingModel(SmallAttentionUnet):
    def denoise(self, images):
      seen.append(images.clone())
      return super().denoise(images)

  def run(seed):
    model = RecordingModel(
      diffuser=DdimDiffuser(LinearBetaScheduler(), number_of_steps=3),
      image_channels=3,
      base_channels=32,
      channel_multipliers=(1, 2),
      num_res_blocks=1,
    )
    fid_module.ImageQualityEvaluator(
      real_dataset=_dataset(4),
      number_of_samples=2,
      device="cpu",
      seed=seed,
      num_workers=0,
    ).evaluate(model)

  run(1)
  run(1)
  run(2)
  assert torch.equal(seen[0], seen[1])
  assert not torch.equal(seen[0], seen[2])


def _trainer(tmp_path, **evaluation):
  model = _model()
  return DiffusionTrainer(
    model=model,
    dataset=_dataset(8),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="test",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=5,
      num_workers=0,
      ema_decay=0.9,
    ),
    loss_function=torch.nn.functional.mse_loss,
    log_configuration=LogConfiguration(number_of_images=0),
    device="cpu",
    output_directory=tmp_path,
    evaluation_configuration=EvaluationConfiguration(
      num_workers=0, number_of_samples=4, batch_size=4, **evaluation
    ),
    evaluation_dataset=_dataset(4),
  )


def test_trainer_logs_fid_at_rate_and_after_last_epoch(tmp_path):
  trainer = _trainer(tmp_path, fid_rate=2)
  logged = []
  original = trainer.tensorboard_manager.log_metrics

  def record(metrics, global_step):
    if any(name.startswith("Eval/") for name in metrics):
      logged.append((global_step, sorted(metrics)))
    original(metrics, global_step)

  trainer.tensorboard_manager.log_metrics = record
  trainer.train()

  steps_per_epoch = 2
  # Epochs 2 and 4 are due; epoch 5 is the last epoch.
  assert [step for step, _ in logged] == [2 * 2, 4 * 2, 5 * 2]
  assert logged[0][1] == ["Eval/FID", "Eval/KID", "Eval/KID_std"]
  assert steps_per_epoch == len(trainer.dataloader)
  # FID is evaluated on the EMA model, which stays in eval mode.
  assert not trainer.ema.model.training
  assert trainer.model.training


def test_trainer_without_fid_rate_has_no_evaluator(tmp_path):
  trainer = _trainer(tmp_path, fid_rate=None)
  assert trainer.evaluator is None


def test_trainer_requires_an_evaluation_dataset(tmp_path):
  model = _model()
  with pytest.raises(ValueError, match="evaluation_dataset"):
    DiffusionTrainer(
      model=model,
      dataset=_dataset(8),
      optimizer=torch.optim.AdamW(model.parameters()),
      training_configuration=TrainingConfiguration(
        training_name="t",
        batch_size=4,
        learning_rate=1e-3,
        number_of_epochs=1,
        num_workers=0,
      ),
      device="cpu",
      output_directory=tmp_path,
      evaluation_configuration=EvaluationConfiguration(fid_rate=1),
    )
