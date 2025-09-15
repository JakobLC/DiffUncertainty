import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import hydra.utils
import numpy as np
import torch
from argparse import Namespace, ArgumentParser
import random

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from uncertainty_modeling.lightning_experiment import LightningExperiment
import warnings
# warnings.filterwarnings("error")
warnings.filterwarnings("ignore", message=r".*upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*For seamless cloud uploads and versioning, try installing*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*PyTorch skipping the first value of the learning rate schedule.*", category=UserWarning)
torch.set_float32_matmul_precision('medium')

def set_seed(seed):
    print(f"SETTING GLOBAL SEED TO {seed}")
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@hydra.main(version_base=None, config_path="configs", config_name="softmax_config")
def main(cfg_hydra: DictConfig):
    """Uses the pl.Trainer to fit & test the model

    Args:
        hparams ([Namespace]): hparams
        nested_dict ([dict], optional): Subset of hparams for saving. Defaults to None.
    """
    config = cfg_hydra
    # Use Environment Variables if accessible
    if "DATASET_LOCATION" in os.environ.keys():
        config.data_input_dir = os.environ["DATASET_LOCATION"]
    if "EXPERIMENT_LOCATION" in os.environ.keys():
        config.save_dir = os.environ["EXPERIMENT_LOCATION"]
    if "LSB_JOBID" in os.environ.keys() and config.version is None:
        config.version = os.environ["LSB_JOBID"]
    if config.seed is not None:
        set_seed(config.seed)

    if "gradient_clip_val" in config.keys():
        gradient_clip_val = config.gradient_clip_val
        gradient_clip_algorithm = "norm"
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    logger = hydra.utils.instantiate(config.logger, version=config.version)
    progress_bar = hydra.utils.instantiate(config.progress_bar)
    trainer = pl.Trainer(
        **config.trainer,
        logger=logger,
        profiler="simple",
        callbacks=progress_bar,
        deterministic="warn",
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
    )
    dm = hydra.utils.instantiate(
        config.datamodule,
        data_input_dir=config.data_input_dir,
        seed=config.seed,
        _recursive_=False,
    )
    dm.prepare_data()
    model = LightningExperiment(config, **config)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
