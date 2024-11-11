"""Running inference.

Below is a code example of an inference script. Simply plug in your own
checkpoint and start running inference. To extend this further, you might
want to look into compiling your model to ONNX, TensorRT, OpenVino or other
formats.

.. literalinclude:: /../../examples/infer.py
   :language: python
   :linenos:
   :lines: 16-50
"""

__all__ = []

from torchvision.transforms import v2

from diffusion_models.diffusion_inference import DiffusionInference
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import Checkpoint


if __name__ == "__main__":
  checkpoint_file_path = "your_checkpoint.pt"

  checkpoint = Checkpoint.from_file(checkpoint_file_path)
  gaussian_diffuser = GaussianDiffuser.from_checkpoint(checkpoint)  # Switch to DdimDiffuser for faster inference

  model = SimpleUnet(
    image_channels=checkpoint.image_channels, diffuser=gaussian_diffuser
  )
  model.load_state_dict(checkpoint.model_state_dict)
  # model = model.compile(mode="reduce-overhead", fullgraph=True)

  reverse_transforms = v2.Compose(
    [
      v2.Lambda(lambda x: (x + 1) / 2),
      v2.Resize((128, 128)),
    ]
  )

  inference = DiffusionInference(
    model=model,
    device="cuda",
    reverse_transforms=reverse_transforms,
  )
  inference.generate(number_of_images=25, save_gif=True)
