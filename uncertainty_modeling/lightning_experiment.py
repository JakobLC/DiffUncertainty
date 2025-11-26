import math
import os
from random import randrange
from typing import Optional, Tuple, List, Mapping, Any
from argparse import Namespace, ArgumentParser

import hydra
import yaml
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributions as td
import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel

import torchvision
from omegaconf import DictConfig, OmegaConf, open_dict
#from evaluation.metrics.dice_old_torchmetrics import dice
from evaluation.metrics.dice_wrapped import dice
from evaluation.metrics.ged_fast import ged_binary_fast

from loss_modules import SoftDiceLoss

import uncertainty_modeling.data.cityscapes_labels as cs_labels
from uncertainty_modeling.unc_mod_utils.test_utils import calculate_ged


class SwagTracker:
    """Minimal diagonal SWAG accumulator tracking mean and variance of weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig | Mapping[str, Any] | None,
        trainer_max_epochs: int | None = None,
    ) -> None:
        self._raw_config = self._to_plain_dict(cfg)
        self.enabled: bool = bool(self._raw_config.get("enabled", False))
        self.snapshot_freq: int = max(1, int(self._raw_config.get("snapshot_frequency", 1)))
        self.max_snapshots: int = max(1, int(self._raw_config.get("max_snapshots", 20)))
        self.min_variance: float = float(self._raw_config.get("min_variance", 1e-30))
        self.storage_device = torch.device("cpu")
        self.dtype = torch.float32
        self.n_snapshots: int = 0
        self.trainer_max_epochs: int | None = trainer_max_epochs
        self.start_epoch: int = self._derive_start_epoch(self.trainer_max_epochs)
        if not self.enabled:
            self.param_count = 0
            self.mean = torch.tensor([], device=self.storage_device, dtype=self.dtype)
            self.sq_mean = torch.tensor([], device=self.storage_device, dtype=self.dtype)
            return
        sample_vec = self._vectorize(model)
        self.param_count = sample_vec.numel()
        self.mean = torch.zeros_like(sample_vec)
        self.sq_mean = torch.zeros_like(sample_vec)

    def _vectorize(self, model: torch.nn.Module) -> torch.Tensor:
        views = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            tensor = param.detach().to(self.storage_device, dtype=self.dtype)
            views.append(tensor.reshape(-1))
        if not views:
            raise RuntimeError("SWAG tracking requires at least one trainable parameter.")
        return torch.cat(views, dim=0)

    @staticmethod
    def _to_plain_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        if isinstance(cfg, Mapping):
            return dict(cfg)
        return {}

    def _derive_start_epoch(self, trainer_max_epochs: int | None) -> int:
        if trainer_max_epochs is None or trainer_max_epochs <= 0:
            return 0
        last_epoch_index = trainer_max_epochs - 1
        span = self.snapshot_freq * max(self.max_snapshots - 1, 0)
        start_epoch = last_epoch_index - span
        return max(start_epoch, 0)

    def set_trainer_max_epochs(self, trainer_max_epochs: int | None) -> None:
        """Update start epoch once the actual trainer schedule is known."""
        self.trainer_max_epochs = trainer_max_epochs
        self.start_epoch = self._derive_start_epoch(trainer_max_epochs)

    def maybe_collect(self, model: torch.nn.Module, epoch: int) -> bool:
        if not self.enabled:
            return False
        if self.n_snapshots >= self.max_snapshots:
            return False
        if epoch < self.start_epoch:
            return False
        if (epoch - self.start_epoch) % self.snapshot_freq != 0:
            return False
        vec = self._vectorize(model)
        total = self.n_snapshots + 1
        old_fac = self.n_snapshots / total if self.n_snapshots > 0 else 0.0
        new_fac = 1.0 / total
        self.mean.mul_(old_fac).add_(vec, alpha=new_fac)
        self.sq_mean.mul_(old_fac).add_(vec.square(), alpha=new_fac)
        self.n_snapshots += 1
        return True

    def diag_variance(self) -> torch.Tensor | None:
        if not self.enabled or self.n_snapshots < 2:
            return None
        variances = torch.clamp(self.sq_mean - self.mean.square(), min=self.min_variance)
        return variances

    def state_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "mean": self.mean.clone(),
            "sq_mean": self.sq_mean.clone(),
            "n_snapshots": self.n_snapshots,
            "config": self._raw_config,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not self.enabled or not state:
            return
        mean = state.get("mean")
        sq_mean = state.get("sq_mean")
        if mean is not None and mean.shape == self.mean.shape:
            self.mean.copy_(mean)
        if sq_mean is not None and sq_mean.shape == self.sq_mean.shape:
            self.sq_mean.copy_(sq_mean)
        self.n_snapshots = int(state.get("n_snapshots", self.n_snapshots))



class LightningExperiment(pl.LightningModule):

    def __init__(
        self,
        hparams: DictConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        nested_hparam_dict: Optional[dict] = None,
        aleatoric_loss: bool = False,
        n_aleatoric_samples: int = 10,
        *args,
        **kwargs
    ):
        """Experiment Class which handles the optimizer, training, validation & testing.
        Saves hparams as well as the nested_hparam_dict when instance is called in pl.Trainer.fit(model=unet_exp)

        Args:
            hparams ([dict/Namespace]): hparams
            learning_rate (float, optional): [learning rate for optimizer]. Defaults to 1e-4.
            weight_decay (float, optional): [weight decay on model]. Defaults to 1e-6.
            nested_hparam_dict (Optional[dict], optional): if dict -> saved in the experiment_directory. Defaults to None.
        """
        super(LightningExperiment, self).__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        dataset_override = os.environ.get("DATASET_LOCATION")
        if dataset_override is not None:
            if isinstance(hparams, DictConfig):
                if "data" in hparams:
                    with open_dict(hparams.data):
                        hparams.data.data_input_dir = dataset_override
                else:
                    hparams.data_input_dir = dataset_override
            else:
                data_section = getattr(hparams, "data", None)
                if isinstance(data_section, dict):
                    data_section["data_input_dir"] = dataset_override
                elif data_section is not None and hasattr(data_section, "__dict__"):
                    setattr(data_section, "data_input_dir", dataset_override)
                else:
                    setattr(hparams, "data_input_dir", dataset_override)

        def _get_section(cfg, key):
            if isinstance(cfg, DictConfig):
                return cfg.get(key, None)
            if isinstance(cfg, dict):
                return cfg.get(key, None)
            return getattr(cfg, key, None)

        def _get_value(cfg, key, default=None):
            if cfg is None:
                return default
            if isinstance(cfg, DictConfig):
                return cfg.get(key, default)
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        data_cfg = _get_section(hparams, "data")
        datamodule_cfg = _get_section(hparams, "datamodule")
        trainer_cfg = _get_section(hparams, "trainer")
        active_data_cfg = data_cfg if data_cfg is not None else datamodule_cfg
        if active_data_cfg is None:
            raise ValueError(
                "LightningExperiment requires either 'data' or legacy 'datamodule' configuration."
            )

        self._trainer_max_epochs = _get_value(trainer_cfg, "max_epochs", None)

        self.ignore_index = _get_value(active_data_cfg, "ignore_index", 0) or 0
        self.evaluate_all_raters = bool(
            _get_value(
                active_data_cfg,
                "evaluate_all_raters",
                getattr(hparams, "evaluate_all_raters", True),
            )
        )
        self._train_batch_size = _get_value(
            active_data_cfg, "batch_size", getattr(hparams, "batch_size", None)
        )
        self._dataset_name = _get_value(
            active_data_cfg, "name", getattr(hparams, "dataset", None)
        )

        self.save_hyperparameters(OmegaConf.to_container(hparams))
        self.nested_hparam_dict = nested_hparam_dict

        if not hasattr(self.hparams, "batch_size") and self._train_batch_size is not None:
            setattr(self.hparams, "batch_size", self._train_batch_size)

        if aleatoric_loss:
            self.model = hydra.utils.instantiate(
                hparams.model, aleatoric_loss=aleatoric_loss
            )
        else:
            self.model = hydra.utils.instantiate(hparams.model)
        self.AU_type = getattr(self.model, "AU_type", None)
        if self.AU_type not in {"softmax", "ssn", "diffusion"}:
            raise ValueError(f"Unsupported AU_type '{self.AU_type}'.")
        self.is_generative = bool(getattr(self.model, "is_generative", None))
        self.diffusion_num_steps = int(getattr(self.model, "diffusion_num_steps", 50))
        self.diffusion_sampler_type = getattr(self.model, "diffusion_sampler_type", "ddpm") or "ddpm"
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.ssn_pretrain_epochs = int(getattr(hparams, "pretrain_epochs", 5))
        print(self.ssn_pretrain_epochs)
        exit()
        ckpt_cfg = hparams.get("ckpt_save_freq", None)
        self.track_ema_weights = bool(getattr(ckpt_cfg, "track_ema_weights", False))
        self.ema_decay = float(getattr(ckpt_cfg, "ema_decay", 0.999))
        if self.track_ema_weights and not (0.0 < self.ema_decay <= 1.0):
            raise ValueError("ema_decay must lie in the interval (0, 1].")
        self.ema_model: Optional[AveragedModel] = None
        self._ema_initialized = False

        self.aleatoric_loss = aleatoric_loss
        if self.aleatoric_loss:
            raise ValueError("Aleatoric loss not updated/tested since repo refactor.")
        self.n_aleatoric_samples = n_aleatoric_samples
        self.dice_loss = SoftDiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.nll_loss = torch.nn.NLLLoss()

        self._val_metric_accumulators = {}
        # By default, only compute GED (and the standard per-image dice is logged separately).
        # Users can override this in hparams to include extra metrics like ["dice", "major_dice", ...].
        self._validation_additional_metrics = []

        if "optimizer" in hparams:
            self.optimizer_conf = hparams.optimizer
        else:
            self.optimizer_conf = None
        if "lr_scheduler" in hparams:
            self.lr_scheduler_conf = hparams.lr_scheduler
        else:
            self.lr_scheduler_conf = None

        swag_cfg = getattr(hparams, "swag", None)
        self.swag_tracker: SwagTracker | None = None
        if getattr(self.model, "swag_enabled", False):
            self.swag_tracker = SwagTracker(
                self.model,
                swag_cfg,
                trainer_max_epochs=self._trainer_max_epochs,
            )
            if not self.swag_tracker.enabled:
                warnings.warn(
                    "SWAG was enabled for the model, but config.swag.enabled is false; skipping SWAG tracking.",
                    RuntimeWarning,
                )
                self.swag_tracker = None

    def configure_optimizers(self) -> Tuple[List[optim.Adam], List[dict]]:
        """Define the optimizers and learning rate schedulers. Adam is used as optimizer.

        Returns:
            optimizer [List[optim.Adam]]: The optimizer which is used in training (Adam)
            scheduler [dict]: The learning rate scheduler
        """
        # Only optimize base model parameters; EMA weights are maintained separately.
        params = list(self.model.parameters())
        if self.optimizer_conf:
            optimizer = hydra.utils.instantiate(self.optimizer_conf, params)
        else:
            optimizer = optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        # scheduler dictionary which defines scheduler and how it is used in the training loop
        if self.lr_scheduler_conf:
            max_steps = self.trainer.datamodule.max_steps()
            scheduler = {
                "scheduler": hydra.utils.instantiate(
                    self.lr_scheduler_conf, optimizer=optimizer, total_iters=max_steps
                ),
                "monitor": "validation/val_loss",
                "interval": "step",
                "frequency": 1,
            }
            scheduler_list = [scheduler]
        else:
            scheduler_list = []
        return [optimizer],scheduler_list

    def _ema_avg_fn(
        self,
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: int,
    ) -> torch.Tensor:
        if num_averaged == 0:
            return model_parameter.detach()
        decay = self.ema_decay
        return averaged_model_parameter * decay + model_parameter.detach() * (1.0 - decay)

    def _ensure_ema_model(self) -> None:
        if not self.track_ema_weights:
            return
        if not self._ema_initialized:
            self.ema_model = AveragedModel(self.model, avg_fn=self._ema_avg_fn)
            self.ema_model.requires_grad_(False)
            self._ema_initialized = True
        if self.ema_model is not None and next(self.ema_model.parameters(), None) is not None:
            self.ema_model.to(self.device, non_blocking=True)  # type: ignore[arg-type]
            self.ema_model.eval()

    def _update_ema_weights(self) -> None:
        if not self.track_ema_weights:
            return
        self._ensure_ema_model()
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._ensure_ema_model()
        if self.swag_tracker is not None:
            trainer_max_epochs = getattr(self.trainer, "max_epochs", None) if self.trainer is not None else None
            self.swag_tracker.set_trainer_max_epochs(trainer_max_epochs)

    def optimizer_step(self, *args, **kwargs):  # type: ignore[override]
        output = super().optimizer_step(*args, **kwargs)
        self._update_ema_weights()
        return output
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.track_ema_weights:
            self._ensure_ema_model()
            if self.ema_model is not None and "ema_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])  # type: ignore[arg-type]
        swag_state = checkpoint.get("swag_state_dict")
        if self.swag_tracker is not None and swag_state is not None:
            self.swag_tracker.load_state_dict(swag_state)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.track_ema_weights and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()  # type: ignore[assignment]
        if self.swag_tracker is not None:
            checkpoint["swag_state_dict"] = self.swag_tracker.state_dict()

    def on_fit_start(self):
        """Called when fit begins
        Be careful: on_fit_start is executed before the train_loop as well as the test_loop v 1.0.3
        Logs the hyperparameters.
        """

        # set placeholders for the metrics according to the stage of the trainer
        if self.trainer.testing is False:
            metric_placeholder = {
                "validation/val_loss": 0.0,
                "validation/val_dice": 0.0,
            }
        else:
            metric_placeholder = {"test/test_loss": 0.0, "test/test_dice": 0.0}

        self.hparams.version = self.logger.version
        # Save nested_hparam_dict if available
        if self.nested_hparam_dict is not None:
            with open(
                os.path.join(self.logger.experiment.log_dir, "hparams_sub_nested.yml"),
                "w",
            ) as file:
                yaml.dump(self.nested_hparam_dict, file, default_flow_style=False)

            sub_hparams = dict()
            for subdict in self.nested_hparam_dict.values():
                sub_hparams.update(subdict)
            sub_hparams = Namespace(**sub_hparams)
            self.logger.log_hyperparams(sub_hparams, metrics=metric_placeholder)
        else:
            self.logger.log_hyperparams(
                Namespace(**self.hparams), metrics=metric_placeholder
            )

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[td.LowRankMultivariateNormal, bool]:
        """Forward pass through the network

        Args:
            x: The input batch

        Returns:
            torch.Tensor or (distribution, cov_failed_flag)
        """
        return self.model(x, **kwargs)

    def forward_ssn(self, batch: dict, target: torch.Tensor, val: bool = False):
        if self.current_epoch < self.ssn_pretrain_epochs:
            mean, cov_failed_flag = self.forward(batch["data"], mean_only=True)
            samples = mean.rsample([self.n_aleatoric_samples])
        else:
            distribution, cov_failed_flag = self.forward(batch["data"])
            samples = distribution.rsample([self.n_aleatoric_samples])
        samples = samples.view(
            [
                self.n_aleatoric_samples,
                batch["data"].size()[0],
                self.model.num_classes,
                *batch["data"].size()[2:],
            ]
        )
        if val:
            softmax_samples = torch.softmax(samples, dim=2)
            # one sample batch for visualization
            sample_idx = randrange(self.n_aleatoric_samples)
            output = samples[sample_idx]
        target = target.unsqueeze(1)
        target = target.expand((self.n_aleatoric_samples,) + target.shape)
        flat_size = self.n_aleatoric_samples * batch["data"].size()[0]
        samples = samples.view(flat_size, self.model.num_classes, -1)
        target = target.reshape(flat_size, -1)
        if self.ignore_index != 0:
            log_prob = -F.cross_entropy(
                samples, target, ignore_index=self.ignore_index, reduction="none"
            ).view((self.n_aleatoric_samples, batch["data"].size()[0], -1))
        else:
            log_prob = -F.cross_entropy(samples, target, reduction="none").view(
                (self.n_aleatoric_samples, batch["data"].size()[0], -1)
            )
        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0)
            - math.log(self.n_aleatoric_samples)
        )
        loss = -loglikelihood
        if val:
            return loss, output, softmax_samples
        return loss

    def _prepare_diffusion_target(
        self, target: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target = target.to(device)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        ignore_mask: torch.Tensor | None = None
        if isinstance(self.ignore_index, int) and self.ignore_index >= 0:
            ignore_mask = target == self.ignore_index
            if ignore_mask.any():
                target = target.masked_fill(ignore_mask, 0)
        one_hot = F.one_hot(target, num_classes=self.model.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        loss_mask: torch.Tensor | None = None
        if ignore_mask is not None:
            valid = (~ignore_mask).float()
            loss_mask = valid.unsqueeze(1).expand(-1, self.model.num_classes, -1, -1)
            if ignore_mask.any():
                one_hot = one_hot * loss_mask
        return one_hot, loss_mask

    def _diffusion_sample_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "diffusion_sample_loop"):
            raise RuntimeError("Model does not implement diffusion_sample_loop")
        batch_size = inputs.shape[0]
        spatial_shape = inputs.shape[2:]
        samples: List[torch.Tensor] = []
        for _ in range(self.n_aleatoric_samples):
            x_init = torch.randn(
                (batch_size, self.model.num_classes, *spatial_shape),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            sample_output = self.model.diffusion_sample_loop(
                x_init=x_init,
                im=inputs,
                num_steps=self.diffusion_num_steps,
                sampler_type=self.diffusion_sampler_type,
                clip_x=False,
                guidance_weight=0.0,
                progress_bar=False,
                self_cond=False,
            )
            samples.append(torch.softmax(sample_output, dim=1))
        return torch.stack(samples)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a training step, i.e. pass a batch to the network and calculate the loss.

        Args:
            batch (dict): The training batch
            batch_idx (int): The index of the current batch

        Returns:
            loss [torch.Tensor]: The computed loss
        """
        inputs = batch["data"]
        if isinstance(inputs, list):
            inputs = inputs[0]
        inputs = inputs.float()
        batch["data"] = inputs
        target = batch["seg"].long().squeeze(1).to(inputs.device)
        if self.AU_type == "ssn":
            loss = self.forward_ssn(batch, target)
        elif self.AU_type == "diffusion":
            if not hasattr(self.model, "diffusion_train_loss_step"):
                raise RuntimeError("Model does not implement diffusion training logic.")
            diffusion_target, diffusion_mask = self._prepare_diffusion_target(target, inputs.device)
            loss, _ = self.model.diffusion_train_loss_step(
                x=diffusion_target,
                im=inputs,
                loss_mask=diffusion_mask,
                eps=None,
                t=None,
                self_cond=False,
            )
        elif self.aleatoric_loss:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu, s = self.forward(inputs)
            sigma = torch.exp(s / 2)
            all_samples = torch.zeros(
                (self.n_aleatoric_samples, *mu.size()), device=device
            )
            for t in range(self.n_aleatoric_samples):
                epsilon = torch.randn(s.size(), device=device)
                sample = mu + sigma * epsilon
                log_sample_prob = F.log_softmax(sample)
                all_samples[t] = log_sample_prob
            log_sample_avg = torch.logsumexp(all_samples, 0) - torch.log(
                torch.tensor(self.n_aleatoric_samples)
            )
            loss = self.dice_loss(torch.exp(log_sample_avg), target) + self.nll_loss(
                log_sample_avg, target
            )
        else:
            output = self.forward(inputs)
            output_softmax = F.softmax(output, dim=1)

            if self.ignore_index != 0:
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
            else:
                loss = self.dice_loss(output_softmax, target) + self.ce_loss(
                    output, target
                )
        log_batch_size = (
            self._train_batch_size if self._train_batch_size is not None else target.shape[0]
        )
        self.log(
            "training/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=log_batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.swag_tracker is None:
            return
        collected = self.swag_tracker.maybe_collect(self.model, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self._val_metric_accumulators = {}

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Run a validation batch, log qualitative grids, and accumulate metrics."""

        if dataloader_idx == 0:
            split = "val"
        elif dataloader_idx == 1:
            split = "train"
        else:
            split = f"val_{dataloader_idx}"

        target_full = batch["seg"].long()
        dataset_name = self._dataset_name or getattr(self.hparams, "dataset", None)
        is_lidc_dataset = bool(dataset_name and "lidc" in str(dataset_name).lower())
        if target_full.ndim == 3:
            if is_lidc_dataset and self.evaluate_all_raters:
                raise ValueError(
                    "LIDC dataset should have multiple raters in validation when evaluate_all_raters is True."
                )
            target_full = target_full.unsqueeze(1)
        target = target_full[:, 0]
        batch_size = target.shape[0]
        multi_rater_available = self.evaluate_all_raters and target_full.shape[1] > 1

        inputs = batch["data"]
        if isinstance(inputs, list):
            inputs = inputs[0]
        inputs = inputs.float()

        softmax_stack: Optional[torch.Tensor] = None
        if self.AU_type == "ssn":
            eval_loss, output, softmax_stack = self.forward_ssn(
                batch, target, val=True
            )
            # Compute dice across individual samples (not mean prob) when not overridden by GED later
            per_sample_dice = []
            # softmax_stack: (S, B, C, H, W)
            for t in range(softmax_stack.shape[0]):
                d = dice(
                    softmax_stack[t],
                    target,
                    num_classes=self.model.num_classes,
                    ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                    binary_dice=self.model.num_classes == 2,
                    is_softmax=True,
                )
                per_sample_dice.append(d if isinstance(d, torch.Tensor) else torch.tensor(float(d), device=inputs.device))
            eval_dice = torch.stack(per_sample_dice).mean()
        elif self.AU_type == "diffusion":
            diffusion_target, diffusion_mask = self._prepare_diffusion_target(target, inputs.device)
            if not hasattr(self.model, "diffusion_train_loss_step"):
                raise RuntimeError("Model does not implement diffusion training logic.")
            with torch.no_grad():
                eval_loss, _ = self.model.diffusion_train_loss_step(
                    x=diffusion_target,
                    im=inputs,
                    loss_mask=diffusion_mask,
                    eps=None,
                    t=None,
                    self_cond=False,
                )
            softmax_stack = self._diffusion_sample_predictions(inputs)
            per_sample_dice = []
            for t in range(softmax_stack.shape[0]):
                d = dice(
                    softmax_stack[t],
                    target,
                    num_classes=self.model.num_classes,
                    ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                    binary_dice=self.model.num_classes == 2,
                    is_softmax=True,
                )
                per_sample_dice.append(
                    d if isinstance(d, torch.Tensor) else torch.tensor(float(d), device=inputs.device)
                )
            eval_dice = torch.stack(per_sample_dice).mean()
            output = softmax_stack.mean(dim=0)
        elif self.aleatoric_loss:
            mu, s = self.forward(inputs)
            sigma = torch.exp(s / 2)
            all_samples = torch.empty(
                (self.n_aleatoric_samples, *mu.size()), device=mu.device, dtype=mu.dtype
            )
            for t in range(self.n_aleatoric_samples):
                epsilon = torch.randn_like(s)
                sample = mu + sigma * epsilon
                log_sample_prob = F.log_softmax(sample, dim=1)
                all_samples[t] = log_sample_prob
            log_sample_avg = torch.logsumexp(all_samples, dim=0) - math.log(
                self.n_aleatoric_samples
            )
            prob_avg = torch.exp(log_sample_avg)
            eval_loss = self.dice_loss(prob_avg, target) + self.nll_loss(
                log_sample_avg, target
            )
            # For evaluation, prefer averaging dice across individual samples (not mean prob)
            # to match the requested behavior when not evaluating all raters.
            sample_probs = torch.exp(all_samples)
            per_sample_dice = []
            for t in range(self.n_aleatoric_samples):
                d = dice(
                    sample_probs[t],
                    target,
                    num_classes=self.model.num_classes,
                    ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                    binary_dice=self.model.num_classes == 2,
                    is_softmax=True,
                )
                per_sample_dice.append(d if isinstance(d, torch.Tensor) else torch.tensor(float(d), device=mu.device))
            eval_dice = torch.stack(per_sample_dice).mean()
            output = log_sample_avg
            softmax_stack = sample_probs
        else:
            output = self.forward(inputs)
            output_softmax = F.softmax(output, dim=1)
            if self.ignore_index != 0:
                eval_loss = F.cross_entropy(
                    output, target, ignore_index=self.ignore_index
                )
            else:
                eval_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
                    output, target
                )
            output_labels = torch.argmax(output_softmax, dim=1)
            eval_dice = dice(
                output_labels,
                target,
                num_classes=self.model.num_classes,
                ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                binary_dice=self.model.num_classes == 2,
            )
            softmax_stack = output_softmax.unsqueeze(0) if multi_rater_available else None

        ged_results: List[dict] = []
        if multi_rater_available and softmax_stack is not None:
            predictions = softmax_stack.detach()
            targets = target_full.detach()
            use_fast = (
                is_lidc_dataset
                and self.model.num_classes == 2
            )
            for idx in range(batch_size):
                pred_stack = predictions[:, idx]
                if pred_stack.ndim == 3:
                    pred_stack = pred_stack.unsqueeze(0)
                gt_stack = targets[idx]
                if gt_stack.ndim == 2:
                    gt_stack = gt_stack.unsqueeze(0)
                # Always request 'dice' for internal use, but only 'ged' will be logged unless enabled in settings.
                requested_metrics = list(set((self._validation_additional_metrics or []) + ["dice"]))
                if use_fast:
                    ged_result = ged_binary_fast(
                        pred_stack,
                        gt_stack,
                        ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                        additional_metrics=requested_metrics,
                    )
                else:
                    ged_result = calculate_ged(
                        pred_stack,
                        gt_stack,
                        ignore_index=self.ignore_index if self.ignore_index != 0 else None,
                        additional_metrics=requested_metrics,
                    )
                ged_results.append(ged_result)

            # Use GED-based mean dice (random pred vs random GT in expectation) for the logged dice when evaluating all raters
            ged_dices = [float(r.get("dice")) for r in ged_results if "dice" in r]
            if ged_dices:
                eval_dice = torch.tensor(sum(ged_dices) / len(ged_dices), device=inputs.device, dtype=torch.float32)

        # Visualization of Segmentations
        if batch_idx == 1 and target.ndim == 3:
            pred_seg_val = torch.argmax(output, dim=1, keepdim=True)
            pred_seg_val = torch.squeeze(pred_seg_val, 1)
            target_seg_val = target
            # transform the labels to color map for visualization
            pred_seg_val_color = torch.zeros((*pred_seg_val.shape, 3), dtype=torch.long)
            target_seg_val_color = torch.zeros((*target_seg_val.shape, 3), dtype=torch.long)
            for k, v in cs_labels.trainId2color.items():
                pred_seg_val_color[pred_seg_val == k] = torch.tensor(v)
                target_seg_val_color[target_seg_val == k] = torch.tensor(v)
            pred_seg_val_color = torch.swapaxes(pred_seg_val_color, 1, 3)
            target_seg_val_color = torch.swapaxes(target_seg_val_color, 1, 3)
            pred_seg_val_color = torch.swapaxes(pred_seg_val_color, 2, 3)
            target_seg_val_color = torch.swapaxes(target_seg_val_color, 2, 3)

            grid = torchvision.utils.make_grid(pred_seg_val_color)
            # TensorBoard expects images in [0,1] float. Data here may be in [0,255].
            if grid.dtype != torch.float32:
                grid = grid.float()
            # If values are in 0-255 range, scale to 0-1
            if grid.max() > 1.0:
                grid = grid / 255.0
            self.logger.experiment.add_image(f"validation/{split}_pred_seg", grid, self.current_epoch)
            grid = torchvision.utils.make_grid(target_seg_val_color)
            if grid.dtype != torch.float32:
                grid = grid.float()
            if grid.max() > 1.0:
                grid = grid / 255.0
            self.logger.experiment.add_image(f"validation/{split}_target_seg", grid, self.current_epoch)

        loss_value = (
            eval_loss.detach().float().mean().item()
            if isinstance(eval_loss, torch.Tensor)
            else float(eval_loss)
        )
        dice_value = (
            eval_dice.detach().float().mean().item()
            if isinstance(eval_dice, torch.Tensor)
            else float(eval_dice)
        )

        if split not in self._val_metric_accumulators:
            ged_keys = ["ged", *self._validation_additional_metrics] if self.evaluate_all_raters else []
            self._val_metric_accumulators[split] = {
                "loss_sum": 0.0,
                "dice_sum": 0.0,
                "count": 0,
                "ged_sums": {key: 0.0 for key in ged_keys},
                "ged_count": 0,
            }

        metrics = self._val_metric_accumulators[split]
        metrics["loss_sum"] += loss_value
        metrics["dice_sum"] += dice_value
        metrics["count"] += 1

        if ged_results:
            metrics["ged_count"] += len(ged_results)
            for result in ged_results:
                for key, value in result.items():
                    if key not in metrics["ged_sums"]:
                        continue
                    metrics["ged_sums"][key] += float(value)

        return eval_loss

    def on_validation_epoch_end(self) -> None:
        accumulators = getattr(self, "_val_metric_accumulators", None)
        if not accumulators:
            return

        device = self.device if hasattr(self, "device") else torch.device("cpu")

        # Log current optimizer LR once per epoch to help diagnose plateaus/scheduler effects
        try:
            if self.trainer is not None and len(self.trainer.optimizers) > 0:
                current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
                self.log(
                    "optimization/lr",
                    torch.tensor(current_lr, device=device),
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )
        except Exception:
            # Best-effort LR logging; ignore if optimizer not yet available
            pass
        for split, metrics in accumulators.items():
            count = metrics.get("count", 0)
            if count == 0:
                continue
            avg_loss = metrics["loss_sum"] / count
            avg_dice = metrics["dice_sum"] / count

            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            avg_dice_tensor = torch.tensor(avg_dice, device=device)

            self.log(
                f"validation/{split}_loss",
                avg_loss_tensor,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.log(
                f"validation/{split}_dice",
                avg_dice_tensor,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

            ged_count = metrics.get("ged_count", 0)
            if ged_count > 0:
                ged_sums = metrics.get("ged_sums", {})
                for metric_name, total in ged_sums.items():
                    avg_metric = total / ged_count
                    # Only prefix with 'ged' for the actual GED. Other metrics are logged plainly.
                    log_key = f"validation/{split}_{metric_name}"
                    metric_tensor = torch.tensor(avg_metric, device=device)
                    self.log(
                        log_key,
                        metric_tensor,
                        prog_bar=False,
                        logger=True,
                        on_epoch=True,
                        add_dataloader_idx=False,
                        sync_dist=True,
                    )

        self._val_metric_accumulators = {}

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a test step, i.e.pass a test batch through the network, calculate loss and dice score for logging
        and visualize the results in logging

        Args:
            batch (dict): The test batch
            batch_idx (int): The index of the current batch

        Returns:
            test_loss [torch.Tensor]: The computed loss
        """
        raise NotImplementedError(
            "Lightning test_step is unused for this project; please run dedicated testers for evaluation."
        )
        output = self.forward(batch["data"].float())
        output_softmax = F.softmax(output, dim=1)

        target = batch["seg"].long().squeeze()

        test_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
            output, target
        )
        test_dice = dice(output_softmax, target, 
                        num_classes        = self.model.num_classes,
                        ignore_index       = self.ignore_index,
                        binary_dice        = self.model.num_classes == 2,
                        is_softmax         = True)
        log = {"test/test_loss": test_loss, "test/test_dice": test_dice}
        self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        return test_loss

    @staticmethod
    def add_module_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments to parser that are specific for experiment module (learning rate, weight decay and seed)

        Args:
            parent_parser (ArgumentParser): The parser to add the arguments

        Returns:
            parser [ArgumentParser]: The parent parser with the appended module specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Learning rate.",
            default=1e-4,
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            help="Weight decay value for optimizer.",
            default=1e-6,
        )
        parser.add_argument(
            "--seed", type=int, help="Random seed for training", default=123
        )
        return parser


if __name__ == "__main__":
    trainer = pl.Trainer()
    trainer.test()
    pass
