import os
import pathlib
import shutil
import tempfile

import torch
from torch.cuda.amp import GradScaler

from diffusion_models.utils.schemas import BetaSchedulerConfiguration, \
  Checkpoint


class TestCheckpoint:
    @classmethod
    def setup_class(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.checkpoint_save_path = os.path.join(cls.temp_dir, 'test_file_path')
        cls.checkpoint = Checkpoint(
            epoch=1,
            model_state_dict={},
            optimizer_state_dict={},
            beta_scheduler_config=BetaSchedulerConfiguration(
              steps=2,
              betas=torch.tensor([1, 2]),
              alpha_bars=torch.tensor([3, 4]),
            ),
            tensorboard_run_name="test_run",
            image_channels=3,
            loss=0.5
        )

    @classmethod
    def teardown_class(cls):
        # Class-wide teardown: remove the temporary directory and its contents
        print("Teardown class")
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_to_file(self):
        self.checkpoint.to_file(self.checkpoint_save_path)
        assert pathlib.Path(self.checkpoint_save_path).is_file(), "Checkpoint file was not saved."

    def test_from_file(self):
        checkpoint = Checkpoint.from_file(self.checkpoint_save_path)

        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.epoch == self.checkpoint.epoch
        assert checkpoint.model_state_dict == self.checkpoint.model_state_dict
        assert checkpoint.optimizer_state_dict == self.checkpoint.optimizer_state_dict

        assert checkpoint.beta_scheduler_config.steps == self.checkpoint.beta_scheduler_config.steps
        assert torch.all(checkpoint.beta_scheduler_config.betas == self.checkpoint.beta_scheduler_config.betas)
        assert torch.all(checkpoint.beta_scheduler_config.alpha_bars == self.checkpoint.beta_scheduler_config.alpha_bars)

        assert checkpoint.scaler is None
        assert checkpoint.tensorboard_run_name == self.checkpoint.tensorboard_run_name
        assert checkpoint.image_channels == self.checkpoint.image_channels
        assert checkpoint.loss == self.checkpoint.loss
