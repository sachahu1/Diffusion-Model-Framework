import importlib
import warnings

import pytest

OLD_TO_NEW = {
  "diffusion_models.gaussian_diffusion.ddimm_diffuser": "diffusion_models.diffusion.diffusers.ddimm_diffuser",
  "diffusion_models.gaussian_diffusion.beta_schedulers": "diffusion_models.diffusion.diffusers.beta_schedulers",
  "diffusion_models.models.SmallAttentionUnet": "diffusion_models.diffusion.models.SmallAttentionUnet",
  "diffusion_models.models.base_diffusion_model": "diffusion_models.diffusion.models.base_diffusion_model",
  "diffusion_models.diffusion_trainer": "diffusion_models.diffusion.trainer",
  "diffusion_models.training": "diffusion_models.diffusion.training",
  "diffusion_models.utils.ema": "diffusion_models.core.utils.ema",
  "diffusion_models.utils.tensorboard": "diffusion_models.core.utils.tensorboard",
  "diffusion_models.evaluation.fid": "diffusion_models.core.evaluation.fid",
  "diffusion_models.evaluation.checkpoint": "diffusion_models.diffusion.evaluation",
}


@pytest.mark.parametrize("old,new", OLD_TO_NEW.items())
def test_old_paths_return_the_same_module(old, new):
  with pytest.warns(DeprecationWarning, match="has moved"):
    module = importlib.import_module(old)
  assert module is importlib.import_module(new)
  # The real module keeps its own identity.
  assert module.__spec__.name == new
  assert module.__name__ == new


def test_old_from_import_works_and_keeps_class_identity():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from diffusion_models.gaussian_diffusion.ddimm_diffuser import DdimDiffuser
    from diffusion_models.models.SmallAttentionUnet import SmallAttentionUnet
  from diffusion_models.diffusion.diffusers.ddimm_diffuser import (
    DdimDiffuser as NewDdim,
  )

  assert DdimDiffuser is NewDdim
  assert SmallAttentionUnet.__module__ == (
    "diffusion_models.diffusion.models.SmallAttentionUnet"
  )


def test_old_schemas_module_exposes_names_from_both_new_modules():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from diffusion_models.utils import schemas
  from diffusion_models.core.schemas import TrainingConfiguration
  from diffusion_models.diffusion.schemas import Checkpoint, Timestep

  assert schemas.TrainingConfiguration is TrainingConfiguration
  assert schemas.Checkpoint is Checkpoint
  assert schemas.Timestep is Timestep


def test_old_target_paths_in_stored_configs_still_instantiate():
  from hydra_zen import instantiate
  from omegaconf import OmegaConf

  from diffusion_models.diffusion.diffusers.ddimm_diffuser import DdimDiffuser

  config = OmegaConf.create(
    {
      "_target_": "diffusion_models.gaussian_diffusion.ddimm_diffuser.DdimDiffuser",
      "beta_scheduler": {
        "_target_": "diffusion_models.gaussian_diffusion.beta_schedulers.LinearBetaScheduler"
      },
      "number_of_steps": 5,
    }
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    diffuser = instantiate(config)
  assert isinstance(diffuser, DdimDiffuser)
  assert diffuser.number_of_steps == 5


def test_unrelated_modules_are_untouched():
  with pytest.raises(ModuleNotFoundError):
    importlib.import_module("diffusion_models.does_not_exist")
