import abc
from dataclasses import dataclass
from typing import Tuple, Optional, Any

import torch
from transformers import CLIPTextModel, AutoTokenizer


@dataclass
class BatchData:
  images: torch.Tensor
  labels: Optional[torch.Tensor] = None


class BaseBatchProcessor(abc.ABC):
  @abc.abstractmethod
  def __call__(self, batch) -> BatchData:
    raise NotImplementedError


class SimpleBatchProcessor(BaseBatchProcessor):
  def __init__(self):
    pass

  def __call__(self, batch) -> BatchData:
    return BatchData(
      images=batch[0],
    )


class ClipTextEmbedder(object):
  def __init__(self):
    self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    self.tokenizer = AutoTokenizer.from_pretrained(
      "openai/clip-vit-base-patch32"
    )

  def embed(self, text_labels):
    inputs = self.tokenizer(text_labels, padding=True, return_tensors="pt")

    outputs = self.model(**inputs)

    # last_hidden_state = outputs.last_hidden_state

    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    return pooled_output


class ClipBatchProcessor(BaseBatchProcessor):
  def __init__(self):
    self.clip_text_embedder = ClipTextEmbedder()

  def __call__(self, batch: Tuple[torch.Tensor, Any]) -> BatchData:
    idx = torch.randint(0, len(batch[0][1]), (len(batch),))

    images = []
    raw_labels = []

    for i in range(len(batch)):
      images.append(batch[i][0])
      raw_labels.append(batch[i][1][idx[i]])

    labels = self.clip_text_embedder.embed(raw_labels)

    return BatchData(
      images=torch.stack(images),
      labels=labels,
    )
