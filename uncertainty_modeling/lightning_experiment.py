import math
import os
from random import randrange
from typing import Optional, Tuple, List
from argparse import Namespace, ArgumentParser

import hydra
import yaml

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributions as td
import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel

import torchvision
from omegaconf import DictConfig, OmegaConf
#from evaluation.metrics.dice_old_torchmetrics import dice
from evaluation.metrics.dice_wrapped import dice

import uncertainty_modeling.models.ssn_unet3D_module
from loss_modules import SoftDiceLoss
from data_carrier_3D import DataCarrier3D

from global_utils.checkpoint_format import format_checkpoint_subdir

import uncertainty_modeling.data.cityscapes_labels as cs_labels


class LightningExperiment(pl.LightningModule):
    def __init__(
        self,
        hparams: DictConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        nested_hparam_dict: Optional[dict] = None,
        aleatoric_loss: bool = False,
        n_aleatoric_samples: int = 10,
        pretrain_epochs: int = 5,
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
        if "DATASET_LOCATION" in os.environ.keys():
            hparams.data_input_dir = os.environ["DATASET_LOCATION"]
        self.save_hyperparameters(OmegaConf.to_container(hparams))
        self.nested_hparam_dict = nested_hparam_dict

        if "ignore_index" in hparams.datamodule:
            self.ignore_index = hparams.datamodule.ignore_index
        else:
            self.ignore_index = 0

        if aleatoric_loss is not None:
            self.model = hydra.utils.instantiate(
                hparams.model, aleatoric_loss=aleatoric_loss
            )
        else:
            self.model = hydra.utils.instantiate(hparams.model)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs

        self.track_ema_weights = bool(getattr(hparams, "track_ema_weights", False))
        self.ema_decay = float(getattr(hparams, "ema_decay", 0.999))
        if self.track_ema_weights and not (0.0 < self.ema_decay <= 1.0):
            raise ValueError("ema_decay must lie in the interval (0, 1].")
        self.ema_model: Optional[AveragedModel] = None
        self._ema_initialized = False

        self.aleatoric_loss = aleatoric_loss
        self.n_aleatoric_samples = n_aleatoric_samples
        self.dice_loss = SoftDiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.nll_loss = torch.nn.NLLLoss()

        self.test_datacarrier = DataCarrier3D()
        self._val_metric_accumulators = {}

        if "optimizer" in hparams:
            self.optimizer_conf = hparams.optimizer
        else:
            self.optimizer_conf = None
        if "lr_scheduler" in hparams:
            self.lr_scheduler_conf = hparams.lr_scheduler
        else:
            self.lr_scheduler_conf = None

    def configure_optimizers(self) -> Tuple[List[optim.Adam], List[dict]]:
        """Define the optimizers and learning rate schedulers. Adam is used as optimizer.

        Returns:
            optimizer [List[optim.Adam]]: The optimizer which is used in training (Adam)
            scheduler [dict]: The learning rate scheduler
        """
        if self.optimizer_conf:
            optimizer = hydra.utils.instantiate(self.optimizer_conf, self.parameters())
        else:
            optimizer = optim.Adam(
                self.parameters(),
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
        else:
            scheduler = {
                "scheduler": lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=10
                ),
                "monitor": "validation/val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        return [optimizer], [scheduler]

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

    def optimizer_step(self, *args, **kwargs):  # type: ignore[override]
        output = super().optimizer_step(*args, **kwargs)
        self._update_ema_weights()
        return output

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if not self.track_ema_weights:
            return
        self._ensure_ema_model()
        if self.ema_model is not None and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])  # type: ignore[arg-type]

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
    ) -> torch.Tensor | td.LowRankMultivariateNormal:
        """Forward pass through the network

        Args:
            x: The input batch

        Returns:
            [torch.Tensor]: The result of the V-Net
        """
        return self.model(x, **kwargs)

    def forward_ssn(self, batch: dict, target: torch.Tensor, val: bool = False):
        if self.current_epoch < self.pretrain_epochs:
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
            val_dice = torch.zeros([self.n_aleatoric_samples])
            sample_labels = torch.argmax(samples, dim=2)
            for idx, sample in enumerate(sample_labels):                
                val_dice[idx] = dice(sample, target, 
                                     num_classes        = self.model.num_classes,
                                     ignore_index       = self.ignore_index,
                                     include_background = self.model.num_classes > 2,
                                     average            = "micro")
            val_dice = torch.mean(val_dice)
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
            return loss, output, val_dice
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a training step, i.e. pass a batch to the network and calculate the loss.

        Args:
            batch (dict): The training batch
            batch_idx (int): The index of the current batch

        Returns:
            loss [torch.Tensor]: The computed loss
        """
        target = batch["seg"].long().squeeze(1)
        # TODO: check if this works with all models
        if type(
            self.model
        ) is uncertainty_modeling.models.ssn_unet3D_module.SsnUNet3D or (
            hasattr(self.model, "ssn") and self.model.ssn
        ):
            loss = self.forward_ssn(batch, target)
        elif self.aleatoric_loss:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu, s = self.forward(batch["data"])
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
            output = self.forward(batch["data"])
            output_softmax = F.softmax(output, dim=1)

            if self.ignore_index != 0:
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
            else:
                loss = self.dice_loss(output_softmax, target) + self.ce_loss(
                    output, target
                )
        self.log(
            "training/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_metric_accumulators = {}

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Perform a validation step, i.e.pass a validation batch through the network, visualize the results in logging
        and calculate loss and dice score for logging

        Args:
            batch (dict): The validation batch
            batch_idx (int): The index of the current batch

        Returns:
            val_loss [torch.Tensor]: The computed loss
        """
        if dataloader_idx == 0:
            split = "val"
        else:
            split = "train"
        target = batch["seg"].long().squeeze(1)
        if type(
            self.model
        ) is uncertainty_modeling.models.ssn_unet3D_module.SsnUNet3D or (
            hasattr(self.model, "ssn") and self.model.ssn
        ):
            eval_loss, output, eval_dice = self.forward_ssn(batch, target, val=True)
        elif self.aleatoric_loss:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu, s = self.forward(batch["data"])
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
            eval_loss = self.dice_loss(
                torch.exp(log_sample_avg), target
            ) + self.nll_loss(log_sample_avg, target)
            eval_dice = dice(
                torch.exp(log_sample_avg), target,
                num_classes        = self.model.num_classes,
                ignore_index       = self.ignore_index,
                include_background = self.model.num_classes > 2,
                average            = "micro"
            )
            # use averaged logits for visualization to keep pipeline uniform
            output = log_sample_avg
        else:
            output = self.forward(batch["data"].float())
            output_softmax = F.softmax(output, dim=1)
            output_labels = torch.argmax(output_softmax, dim=1)
            if self.ignore_index != 0:
                eval_loss = F.cross_entropy(
                    output, target, ignore_index=self.ignore_index
                )
            else:
                eval_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
                    output, target
                )

            eval_dice = dice(output_labels, target,
                            num_classes        = self.model.num_classes,
                            ignore_index       = self.ignore_index,
                            include_background = self.model.num_classes > 2,
                            average            = "micro")

        # Visualization of Segmentations
        if batch_idx == 1 and len(batch["seg"].shape) == 3:
            pred_seg_val = torch.argmax(output, dim=1, keepdim=True)
            pred_seg_val = torch.squeeze(pred_seg_val, 1)
            target_seg_val = batch["seg"].long()
            target_seg_val = torch.squeeze(target_seg_val, 1)
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

        if isinstance(eval_loss, torch.Tensor):
            loss_value = eval_loss.detach().float().mean().item()
        else:
            loss_value = float(eval_loss)
        if isinstance(eval_dice, torch.Tensor):
            dice_value = eval_dice.detach().float().mean().item()
        else:
            dice_value = float(eval_dice)

        metrics = self._val_metric_accumulators.setdefault(
            split, {"loss_sum": 0.0, "dice_sum": 0.0, "count": 0}
        )
        metrics["loss_sum"] += loss_value
        metrics["dice_sum"] += dice_value
        metrics["count"] += 1

        return eval_loss

    def on_validation_epoch_end(self) -> None:
        accumulators = getattr(self, "_val_metric_accumulators", None)
        if not accumulators:
            return

        device = self.device if hasattr(self, "device") else torch.device("cpu")
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
        output = self.forward(batch["data"].float())
        output_softmax = F.softmax(output, dim=1)

        target = batch["seg"].long().squeeze()

        test_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
            output, target
        )
        test_dice = dice(output_softmax, target, ignore_index=self.ignore_index)
        self.test_datacarrier.concat_data(batch=batch, softmax_pred=output_softmax)

        log = {"test/test_loss": test_loss, "test/test_dice": test_dice}
        self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        return test_loss

    def on_test_end(self) -> None:
        checkpoint_tag = format_checkpoint_subdir(
            getattr(self.hparams, "checkpoint_epoch", None),
            getattr(self.hparams, "ema", None),
        )
        self.test_datacarrier.save_data(
            root_dir=self.hparams.save_dir,
            exp_name=self.hparams.exp_name,
            version=self.logger.version,
            checkpoint_tag=checkpoint_tag,
        )

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
