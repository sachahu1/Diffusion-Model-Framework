import torch
from torchvision import datasets
from transformers import (
  CLIPTokenizer,
  CLIPTextModel,
  CLIPModel,
  CLIPProcessor,
  AutoTokenizer,
)


class ClipTextEmbedder(object):
  def __init__(self, dataset: torch.utils.data.Dataset):
    self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    self.tokenizer = AutoTokenizer.from_pretrained(
      "openai/clip-vit-base-patch32"
    )

  def embed(self):
    inputs = self.tokenizer(sample[1], padding=True, return_tensors="pt")

    outputs = self.model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    pooled_output = outputs.pooler_output  # pooled (EOS token) states


if __name__ == "__main__":
  dataset = datasets.CocoCaptions(
    root="../data/coco/train2017",
    annFile="../data/coco/annotations/captions_train2017.json",
  )

  sample = dataset[0]

  model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

  tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

  inputs = tokenizer(sample[1], padding=True, return_tensors="pt")

  outputs = model(**inputs)

  last_hidden_state = outputs.last_hidden_state

  pooled_output = outputs.pooler_output  # pooled (EOS token) states

  print()
