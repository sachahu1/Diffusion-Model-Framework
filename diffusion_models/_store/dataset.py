import torch
from hydra_zen import builds, store
from torchvision import datasets
from torchvision.transforms import v2

dataset_store = store(group="dataset")

# Normalize(0.5, 0.5) maps [0, 1] -> [-1, 1]; its inverse is mean=-1, std=2.
ImageTransforms = builds(
  v2.Compose,
  transforms=[
    builds(v2.ToImage),
    builds(v2.Resize, size=["${image_size}", "${image_size}"]),
    builds(v2.ToDtype, dtype=builds(torch.get_default_dtype), scale=True),
    builds(v2.Normalize, mean=[0.5] * 3, std=[0.5] * 3),
  ],
)

dataset_store(
  builds(
    datasets.CelebA,
    root="../data",
    download=False,
    split="train",
    transform=ImageTransforms,
  ),
  name="celeba",
)

# Random images, handy for smoke-testing the pipeline without any data.
dataset_store(
  builds(
    datasets.FakeData,
    size=64,
    image_size=["${image_channels}", "${image_size}", "${image_size}"],
    transform=ImageTransforms,
    populate_full_signature=True,
  ),
  name="fake",
)
