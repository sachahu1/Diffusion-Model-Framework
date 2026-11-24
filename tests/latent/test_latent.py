import pytest
import torch
from torch.utils.data import TensorDataset

from diffusion_models.autoencoders import ConvVae
from diffusion_models.core.schemas import (
  EvaluationConfiguration,
  LogConfiguration,
  TrainingConfiguration,
)
from diffusion_models.diffusion.diffusers.beta_schedulers import (
  LinearBetaScheduler,
)
from diffusion_models.diffusion.diffusers.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.diffusion.models.SmallAttentionUnet import (
  SmallAttentionUnet,
)
from diffusion_models.diffusion.schemas import Checkpoint
from diffusion_models.latent.cache import (
  LatentCache,
  build_latent_cache,
  estimate_channel_statistics,
  estimate_scale_factor,
)
from diffusion_models.latent.dataset import LatentDataset
from diffusion_models.latent.normalization import LatentNormalization
from diffusion_models.latent.trainer import LatentDiffusionTrainer


@pytest.fixture
def autoencoder():
  torch.manual_seed(0)
  return ConvVae(
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    latent_channels=2,
  ).eval()


@pytest.fixture
def images():
  torch.manual_seed(1)
  return TensorDataset(
    torch.randn(10, 3, 16, 16).clamp(-1, 1), torch.arange(10)
  )


def _cache(autoencoder, images, tmp_path, **kwargs):
  cache = build_latent_cache(
    autoencoder, images, batch_size=4, device="cpu", num_workers=0, **kwargs
  )
  cache.to_file(tmp_path / "latents.pt")
  return cache


def test_cache_holds_both_views_and_matches_the_encoder(
  autoencoder, images, tmp_path
):
  cache = _cache(autoencoder, images, tmp_path)
  # (images, views, latent channels, h, w): the downsample factor is 2.
  assert cache.mean.shape == (10, 2, 2, 8, 8)
  assert cache.std.shape == cache.mean.shape
  assert torch.equal(cache.labels, torch.arange(10))
  mean, _ = autoencoder.encode_distribution(images.tensors[0][:1])
  flipped, _ = autoencoder.encode_distribution(images.tensors[0][:1].flip(-1))
  assert torch.allclose(cache.mean[0, 0].float(), mean[0], atol=1e-2)
  assert torch.allclose(cache.mean[0, 1].float(), flipped[0], atol=1e-2)


def test_no_flip_stores_a_single_view(autoencoder, images, tmp_path):
  assert _cache(autoencoder, images, tmp_path, flip=False).mean.shape[1] == 1


def test_global_normalization_makes_latents_unit_variance(
  autoencoder, images, tmp_path
):
  cache = _cache(autoencoder, images, tmp_path)
  # The factor is estimated on latents drawn from the posteriors.
  dataset = LatentDataset(
    tmp_path / "latents.pt", normalization="global", sample=True
  )
  assert dataset.normalization.scale_values() == pytest.approx(
    cache.scale_factor
  )
  std = torch.stack([dataset[i][0] for i in range(len(dataset))]).std()
  assert std == pytest.approx(1.0, rel=0.25)


def test_channel_normalization_standardises_every_channel(
  autoencoder, images, tmp_path
):
  _cache(autoencoder, images, tmp_path)
  dataset = LatentDataset(tmp_path / "latents.pt", sample=True)
  latents = torch.stack([dataset[i][0] for i in range(len(dataset))])
  assert latents.mean(dim=(0, 2, 3)).abs().max() < 0.25
  assert latents.std(dim=(0, 2, 3)) == pytest.approx(1.0, rel=0.25)


def test_channel_statistics_match_the_data():
  mean = torch.randn(256, 1, 2, 4, 4) * torch.tensor([1.0, 3.0]).view(
    1, 1, 2, 1, 1
  )
  mean += torch.tensor([5.0, -2.0]).view(1, 1, 2, 1, 1)
  channel_mean, channel_std = estimate_channel_statistics(
    mean, torch.zeros_like(mean)
  )
  assert channel_mean == pytest.approx([5.0, -2.0], abs=0.3)
  assert channel_std == pytest.approx([1.0, 3.0], rel=0.15)


def test_old_caches_without_channel_statistics_still_load(
  autoencoder, images, tmp_path
):
  cache = _cache(autoencoder, images, tmp_path)
  cache.channel_mean = cache.channel_std = None
  cache.to_file(tmp_path / "old.pt")
  dataset = LatentDataset(tmp_path / "old.pt")
  assert dataset.normalization.scale.numel() == 2


def test_normalization_round_trips_and_serialises():
  normalization = LatentNormalization(scale=[0.5, 2.0], shift=[1.0, -1.0])
  latents = torch.randn(3, 2, 4, 4)
  assert torch.allclose(
    normalization.denormalize(normalization.normalize(latents)),
    latents,
    atol=1e-6,
  )
  rebuilt = LatentNormalization(
    scale=normalization.scale_values(), shift=normalization.shift_values()
  )
  assert torch.equal(
    rebuilt.normalize(latents), normalization.normalize(latents)
  )
  # A global scale serialises to a plain float with no shift.
  global_scale = LatentNormalization(scale=0.7)
  assert global_scale.scale_values() == pytest.approx(0.7)
  assert global_scale.shift_values() is None


def test_estimate_scale_factor():
  mean = torch.randn(512, 1, 4, 4) * 3
  assert estimate_scale_factor(mean, torch.zeros_like(mean)) == pytest.approx(
    1 / 3, rel=0.1
  )


def test_dataset_yields_scaled_latent_and_label(autoencoder, images, tmp_path):
  _cache(autoencoder, images, tmp_path)
  dataset = LatentDataset(tmp_path / "latents.pt")
  assert len(dataset) == 10
  latent, label = dataset[3]
  assert latent.shape == (2, 8, 8)
  assert latent.dtype == torch.float32
  assert label == 3


def test_dataset_sampling_draws_from_the_posterior(
  autoencoder, images, tmp_path
):
  _cache(autoencoder, images, tmp_path, flip=False)
  mean_only = LatentDataset(tmp_path / "latents.pt", sample=False)
  assert torch.equal(mean_only[0][0], mean_only[0][0])
  sampled = LatentDataset(tmp_path / "latents.pt", sample=True)
  assert not torch.equal(sampled[0][0], sampled[0][0])


def test_unknown_cache_version_is_rejected(tmp_path):
  torch.save({"version": 99}, tmp_path / "old.pt")
  with pytest.raises(ValueError, match="version"):
    LatentCache.from_file(tmp_path / "old.pt")


def _trainer(tmp_path, autoencoder, latents_path, fid=False, **configuration):
  dataset = LatentDataset(latents_path)
  model = SmallAttentionUnet(
    diffuser=GaussianDiffuser(LinearBetaScheduler()),
    image_channels=autoencoder.latent_channels,
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
  )
  return LatentDiffusionTrainer(
    autoencoder=autoencoder,
    model=model,
    dataset=dataset,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    training_configuration=TrainingConfiguration(
      training_name="test",
      batch_size=4,
      learning_rate=1e-3,
      number_of_epochs=1,
      checkpoint_rate=1,
      num_workers=0,
      **configuration,
    ),
    log_configuration=LogConfiguration(number_of_images=2, image_rate=10**6),
    device="cpu",
    output_directory=tmp_path,
  )


def test_trainer_trains_on_latents_and_stores_the_scale_factor(
  autoencoder, images, tmp_path
):
  _cache(autoencoder, images, tmp_path)
  trainer = _trainer(tmp_path, autoencoder, tmp_path / "latents.pt")
  trainer.train()
  checkpoint = Checkpoint.from_file(
    str(tmp_path / "checkpoints" / "epoch_0.pt")
  )
  assert checkpoint.image_channels == 2
  assert checkpoint.latent_scale_factor == pytest.approx(
    trainer.normalization.scale_values()
  )
  assert checkpoint.latent_shift == pytest.approx(
    trainer.normalization.shift_values()
  )
  assert not any(p.requires_grad for p in trainer.autoencoder.parameters())


def test_trainer_decodes_samples_to_images(autoencoder, images, tmp_path):
  _cache(autoencoder, images, tmp_path)
  trainer = _trainer(tmp_path, autoencoder, tmp_path / "latents.pt")
  decoded = trainer.to_image(torch.randn(3, 2, 8, 8))
  assert decoded.shape == (3, 3, 16, 16)
  assert decoded.abs().max() <= 1


def test_trainer_checks_latent_channels(autoencoder, images, tmp_path):
  _cache(autoencoder, images, tmp_path)
  other = ConvVae(
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    latent_channels=3,
  )
  with pytest.raises(ValueError, match="latent channels"):
    _trainer(tmp_path, other, tmp_path / "latents.pt")


def test_evaluator_samples_latents_and_decodes_them(
  autoencoder, images, tmp_path
):
  pytest.importorskip("torchmetrics")
  from diffusion_models.core.evaluation.fid import ImageQualityEvaluator

  _cache(autoencoder, images, tmp_path)
  trainer = _trainer(tmp_path, autoencoder, tmp_path / "latents.pt")
  seen = {}

  class Recorder(torch.nn.Module):
    training = False

    def denoise(self, noise):
      seen["shape"] = tuple(noise.shape)
      return [noise]

  evaluator = ImageQualityEvaluator(
    real_dataset=images,
    number_of_samples=10,
    batch_size=5,
    device="cpu",
    compute_kid=False,
    num_workers=0,
  )
  evaluator.fid.inception = None  # not exercised: only the sampling path
  generated = []
  original = evaluator._update
  evaluator._update = lambda x, real: generated.append(x) if not real else None
  evaluator._prepare_real = lambda: setattr(evaluator, "_real_ready", True)
  evaluator.fid.compute = lambda: torch.tensor(0.0)
  evaluator.fid.reset = lambda: None
  evaluator.evaluate(
    Recorder(), noise_shape=(2, 8, 8), decode=trainer.to_image
  )
  assert seen["shape"] == (5, 2, 8, 8)
  assert generated[0].shape == (5, 3, 16, 16)
  del original


def test_diffuser_clamps_samples_unless_disabled():
  scheduler = LinearBetaScheduler()
  clamped = GaussianDiffuser(scheduler)
  unbounded = GaussianDiffuser(scheduler, clip_value=None)
  samples = torch.tensor([-3.0, 0.5, 2.5])
  assert clamped.clip(samples).tolist() == [-1.0, 0.5, 1.0]
  assert unbounded.clip(samples).tolist() == [-3.0, 0.5, 2.5]


def test_denoising_keeps_unbounded_samples_without_clipping():
  class Zero(torch.nn.Module):
    def forward(self, x, timestep, labels=None):
      return torch.zeros_like(x)

  for clip_value, bounded in ((1.0, True), (None, False)):
    diffuser = GaussianDiffuser(LinearBetaScheduler(), clip_value=clip_value)
    noise = torch.randn(4, 2, 8, 8) * 3
    final = diffuser.denoise_batch(noise, Zero())[-1]
    assert (final.abs().max() <= 1.0) == bounded


def test_inference_decodes_latent_samples(autoencoder):
  from diffusion_models.diffusion.diffusion_inference import DiffusionInference

  model = SmallAttentionUnet(
    diffuser=GaussianDiffuser(LinearBetaScheduler(), clip_value=None),
    image_channels=2,
    base_channels=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
  )
  inference = DiffusionInference(
    model=model,
    image_shape=(2, 8),
    device="cpu",
    decode=lambda latents: autoencoder.decode(latents).clamp(-1, 1),
    reverse_transforms=lambda images: images,
  )
  grid = inference._visualise_images(torch.randn(4, 2, 8, 8))
  # Four 16x16 decoded images in a 2x2 grid (with make_grid's 2 pixel padding).
  assert grid.size == (2 * 16 + 3 * 2, 2 * 16 + 3 * 2)


def test_whitening_round_trips_and_decorrelates():
  generator = torch.Generator().manual_seed(0)
  mixing = torch.randn(4, 4, generator=generator)
  latents = torch.einsum(
    "oc,nchw->nohw", mixing, torch.randn(64, 4, 8, 8, generator=generator)
  )
  flat = latents.permute(0, 2, 3, 1).reshape(-1, 4).double()
  mean = flat.mean(0)
  eigenvalues, eigenvectors = torch.linalg.eigh(torch.cov((flat - mean).T))
  normalization = LatentNormalization(
    scale=1.0,
    shift=mean.float(),
    whitening=(eigenvalues.rsqrt().unsqueeze(1) * eigenvectors.T).float(),
  )
  normalized = normalization.normalize(latents)
  covariance = torch.cov(normalized.permute(0, 2, 3, 1).reshape(-1, 4).T)
  assert torch.allclose(covariance, torch.eye(4), atol=1e-3)
  assert torch.allclose(
    normalization.denormalize(normalized), latents, atol=1e-4
  )
  assert normalization.whitening_values() is not None
