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
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf, open_dict
from uncertainty_modeling.callbacks import ScheduledCheckpointCallback
from uncertainty_modeling.callbacks import GracefulShutdownCallback
from uncertainty_modeling.lightning_experiment import LightningExperiment
import uncertainty_modeling.data.torch_dataloader  # noqa: F401
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
            
def _extract_component_name(section, fallback_label):
    """Return a short token for config sections, preferring explicit nicknames."""
    def _maybe_token(value):
        if isinstance(value, str):
            token = value.strip()
            if token:
                return token.replace(" ", "_")
        return None

    if section is None:
        return fallback_label

    lookup_keys = ("nickname", "name", "target", "_target_")
    for key in lookup_keys:
        if isinstance(section, DictConfig):
            value = OmegaConf.select(section, key, default=None)
        else:
            value = getattr(section, key, None)
        token = _maybe_token(value)
        if token is not None:
            return token
    return fallback_label


def _export_hparams_to_yaml(config: DictConfig, destination: Path) -> None:
    """Persist the current Hydra config to disk for reproducibility."""
    try:
        yaml_text = OmegaConf.to_yaml(config, resolve=False)
        destination.write_text(yaml_text)
    except Exception as exc:
        warnings.warn(f"Failed to export hyperparameters to {destination}: {exc}")

@hydra.main(version_base=None, config_path="configs", config_name="standard")
def main(cfg_hydra: DictConfig):
    """Uses the pl.Trainer to fit & test the model

    Args:
        hparams ([Namespace]): hparams
        nested_dict ([dict], optional): Subset of hparams for saving. Defaults to None.
    """
    config = cfg_hydra
    # Use Environment Variables if accessible
    dataset_override = os.environ.get("DATASET_LOCATION")
    if dataset_override is not None:
        if "data" in config:
            with open_dict(config.data):
                config.data.data_input_dir = dataset_override
        else:
            config.data_input_dir = dataset_override
    if "EXPERIMENT_LOCATION" in os.environ.keys():
        config.save_dir = os.environ["EXPERIMENT_LOCATION"]
    if "LSB_JOBID" in os.environ.keys() and config.version is None:
        config.version = os.environ["LSB_JOBID"]

    exp_name = getattr(config, "exp_name", None)
    if OmegaConf.is_missing(config, "exp_name") or exp_name is None or (isinstance(exp_name, str) and exp_name.strip() == ""):
        data_section = config.get("data", None) or config.get("datamodule", None)
        network_section = config.get("network", None)
        model_section = config.get("model", None)
        eu_method_section = config.get("eu_method", None)
        auto_name = "-".join(
            [
                _extract_component_name(data_section, "data"),
                _extract_component_name(network_section, "network"),
                _extract_component_name(model_section, "model"),
                _extract_component_name(eu_method_section, "eu"),
            ]
        )
        with open_dict(config):
            config.exp_name = auto_name
        exp_name = auto_name
    if config.seed is not None:
        set_seed(config.seed)

    version_value = getattr(config, "version", None)
    if version_value is None or (isinstance(version_value, str) and version_value.strip() == ""):
        version_value = "version_0"
        with open_dict(config):
            config.version = version_value

    base_save_root = Path(getattr(config, "save_dir", Path.cwd())).expanduser()
    run_dir = (base_save_root / str(exp_name) / str(version_value)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(config):
        config.save_dir = run_dir.as_posix()
    _export_hparams_to_yaml(config, run_dir / "hparams.yaml")

    logger = hydra.utils.instantiate(config.logger, version=config.version)
    progress_bar = hydra.utils.instantiate(config.progress_bar)
    scheduled_ckpt_cb = None
    graceful_shutdown_cb = None
    if "ckpt_save_freq" in config and config.ckpt_save_freq is not None:
        ckpt_cfg = config.ckpt_save_freq
        use_linear = bool(getattr(ckpt_cfg, "use_linear_saving", False))
        use_exponential = bool(getattr(ckpt_cfg, "use_exponential_saving", False))
        if use_linear or use_exponential:
            scheduled_ckpt_cb = ScheduledCheckpointCallback(ckpt_cfg)
        # Graceful shutdown handling
        shutdown_timer = ckpt_cfg.get("shutdown_timer", None)
        do_shutdown = bool(ckpt_cfg.get("do_shutdown", False))
        full_last_ckpt = bool(ckpt_cfg.get("full_last_ckpt", False))
        if do_shutdown or (shutdown_timer is not None):
            graceful_shutdown_cb = GracefulShutdownCallback(shutdown_timer=shutdown_timer, do_shutdown=do_shutdown, full_last_ckpt=full_last_ckpt)
    # Custom checkpoint naming: remove '=' and '-' by using explicit fields
    # Configure ModelCheckpoint to control whether the final `last.ckpt` includes optimizer
    # state via `save_weights_only`. When `full_last_ckpt` is True, we want the full checkpoint
    # (weights + optimizer); otherwise save only weights to save space.
    checkpoint_dir = os.path.join(config.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        monitor="generation/val_loss",
        mode="min",
        filename="epoch{epoch}_step{step}",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=0,
        save_weights_only=not full_last_ckpt,
        dirpath=checkpoint_dir,
    )
    callbacks = [progress_bar, checkpoint_cb]

    if scheduled_ckpt_cb is not None:
        callbacks.append(scheduled_ckpt_cb)
    if graceful_shutdown_cb is not None:
        callbacks.append(graceful_shutdown_cb)
    trainer_kwargs = OmegaConf.to_container(config.trainer, resolve=True)
    if not isinstance(trainer_kwargs, dict):
        raise TypeError("trainer configuration must resolve to a dict")
    trainer_kwargs.setdefault("default_root_dir", config.save_dir)

    trainer = pl.Trainer(
        **trainer_kwargs,
        logger=logger,
        callbacks=callbacks,
    )
    data_cfg = config.get("data", None)
    datamodule_cfg = config.get("datamodule", None)
    if data_cfg is None and datamodule_cfg is None:
        raise ValueError("No data configuration available. Provide either 'data' or 'datamodule'.")

    if data_cfg is not None:
        dm = hydra.utils.instantiate(
            data_cfg,
            seed=config.seed,
            _recursive_=False,
        )
    else:
        instantiate_kwargs = {
            "data_input_dir": config.data_input_dir,
            "seed": config.seed,
        }
        if hasattr(config, "AUGMENTATIONS"):
            instantiate_kwargs["augmentations"] = config.AUGMENTATIONS

        dm = hydra.utils.instantiate(
            datamodule_cfg,
            _recursive_=False,
            **instantiate_kwargs,
        )
    dm.prepare_data()
    # If resuming, do not load pretrained backbone weights when instantiating the model
    with open_dict(config):
        try:
            if (("ckpt_path" in config and config.ckpt_path is not None and str(config.ckpt_path) != "null") or (hasattr(config, "resume_from_ckpt") and config.resume_from_ckpt not in [None, "", "null"])):
                if "MODEL" in config and "PRETRAINED" in config.MODEL:
                    config.MODEL.PRETRAINED = False
                if "model" in config and "cfg" in config.model and "MODEL" in config.model.cfg and "PRETRAINED" in config.model.cfg.MODEL:
                    config.model.cfg.MODEL.PRETRAINED = False
        except Exception:
            pass

    model = LightningExperiment(config, **config)

    # Resolve checkpoint path: prefer explicit ckpt_path, otherwise use
    # resume_from_ckpt (relative to the current version directory) if provided.
    ckpt_path = None
    try:
        if "ckpt_path" in config and config.ckpt_path is not None and str(config.ckpt_path) != "null":
            ckpt_path = str(config.ckpt_path)
    except Exception:
        ckpt_path = None

    # Minimal mapping from resume_from_ckpt (relative) to an absolute ckpt_path
    if ckpt_path is None and hasattr(config, "resume_from_ckpt"):
        rel = str(config.resume_from_ckpt or "").strip()
        if rel not in ["", "null"]:
            if config.version is None:
                raise ValueError("resume_from_ckpt is set but config.version is None. A version must be specified to resume.")

            # Use the logger's version directory as the base for relative checkpoints
            if hasattr(logger, "log_dir") and logger.log_dir is not None:
                version_dir = logger.log_dir
            else:
                # Fallback: construct from save_dir and version
                version_dir = getattr(config, "save_dir", os.getcwd())

            candidate = os.path.join(version_dir, rel)
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"resume_from_ckpt '{candidate}' does not exist.")
            ckpt_path = candidate

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
