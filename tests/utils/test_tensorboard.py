import os
import shutil

import pytest
import torch

from diffusion_models.utils.tensorboard import TensorboardManager


class TestTensorboardManager: # todo: fix this race condition
    @classmethod
    def teardown_class(cls):
        # Class-wide teardown
        print("Teardown class")
        shutil.rmtree("../runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def clean_up(self):
        yield  # this is where the testing happens
        # clean up after each test
        shutil.rmtree('../runs/test_run', ignore_errors=True)

    def test_tensorboard_constructor(self):
        tensorboard = TensorboardManager("test_run")
        assert tensorboard.log_directory == "../runs/test_run"
        assert os.path.exists(tensorboard.log_directory)

    def test_tensorboard_no_log_name_constructor(self):
        tensorboard = TensorboardManager()
        assert tensorboard.log_directory == "../runs/fill this"
        assert os.path.exists(tensorboard.log_directory)

    def test_log_metrics(self):
        tensorboard = TensorboardManager("test_run")
        metrics = {"accuracy": 0.85, "loss": 0.15}
        global_step = 10
        tensorboard.log_metrics(metrics, global_step)
        # Check if the tensorboard log file is created
        assert os.listdir(tensorboard.log_directory) != []

    def test_log_images(self):
        tensorboard = TensorboardManager("test_run")
        tag = "test_images"
        images = torch.randn(10, 3, 244, 244)
        timestep = 1
        tensorboard.log_images(tag, images, timestep)
        # Check if the tensorboard log file is created
        assert os.listdir(tensorboard.log_directory) != []

    @pytest.mark.parametrize(
        "tag, images, timestep",
        [("test_images", torch.randn(10, 3, 244, 244), 1),
         ("more_images", torch.randn(5, 1, 244, 244), 5)]
    )
    def test_log_images_parametrised(self, tag, images, timestep):
        tensorboard = TensorboardManager("test_run")
        tensorboard.log_images(tag, images, timestep)
        # Check if the tensorboard log file is created
        assert os.listdir(tensorboard.log_directory) != []
