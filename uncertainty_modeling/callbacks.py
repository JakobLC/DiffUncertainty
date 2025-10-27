from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Set

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn


class ScheduledCheckpointCallback(pl.Callback):
    """Save checkpoints at predetermined epochs using linear or exponential schedules.

    Args:
        ckpt_config: Configuration block under ``ckpt_save_freq``.
    """

    def __init__(self, ckpt_config: DictConfig):
        super().__init__()
        self.cfg = ckpt_config
        self.use_linear = bool(ckpt_config.get("use_linear_saving", False))
        self.use_exponential = bool(ckpt_config.get("use_exponential_saving", False))
        if self.use_linear and self.use_exponential:
            raise ValueError(
                "ckpt_save_freq cannot enable both linear and exponential checkpointing."
            )

        self.only_small_ckpts = bool(ckpt_config.get("only_small_ckpts", False))
        self.linear_freq = int(ckpt_config.get("linear_freq", 0)) if self.use_linear else None
        self.exponential_start = (
            int(ckpt_config.get("exponential_start", 0)) if self.use_exponential else None
        )
        self.exponent_base = (
            float(ckpt_config.get("exponent_base", 0.0)) if self.use_exponential else None
        )
        self.end_epoch_cfg = ckpt_config.get("end", None)

        self._scheduled_epochs: List[int] = []
        self._saved_epochs: Set[int] = set()
        self._dirpath: Optional[Path] = None
        self._max_epochs: Optional[int] = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if trainer.sanity_checking:
            return
        if not (self.use_linear or self.use_exponential):
            return
        if trainer.max_epochs is None or trainer.max_epochs <= 0:
            rank_zero_warn(
                "Scheduled checkpoint callback requires trainer.max_epochs to be set; disabling callback."
            )
            return
        self._max_epochs = int(trainer.max_epochs)
        end_epoch = self._resolve_end_epoch(self._max_epochs)
        schedule = self._build_schedule(end_epoch)
        schedule = [epoch for epoch in schedule if epoch <= end_epoch]
        if not schedule:
            rank_zero_warn("Scheduled checkpoint callback produced no epochs; disabling callback.")
            return
        self._scheduled_epochs = schedule
        self._saved_epochs.clear()

        if trainer.is_global_zero:
            self._dirpath = self._resolve_dirpath(trainer)
            self._dirpath.mkdir(parents=True, exist_ok=True)
            rank_zero_info(
                f"Scheduled checkpoints ({self._mode}) will be written to {self._dirpath} at epochs: {self._scheduled_epochs}"
            )

    @property
    def _mode(self) -> str:
        if self.use_linear:
            return "linear"
        if self.use_exponential:
            return "exponential"
        return "disabled"

    def _resolve_end_epoch(self, max_epochs: int) -> int:
        try:
            end_epoch = int(self.end_epoch_cfg) if self.end_epoch_cfg is not None else max_epochs
        except (TypeError, ValueError):
            end_epoch = max_epochs
        return max(1, min(end_epoch, max_epochs))

    def _build_schedule(self, end_epoch: int) -> List[int]:
        epochs: Set[int] = set()
        if self.use_linear:
            if self.linear_freq is None or self.linear_freq <= 0:
                raise ValueError("linear_freq must be a positive integer when linear saving is enabled.")
            current = self.linear_freq
            while current <= end_epoch:
                epochs.add(current)
                current += self.linear_freq
        elif self.use_exponential:
            if self.exponential_start is None or self.exponential_start <= 0:
                raise ValueError("exponential_start must be a positive integer when exponential saving is enabled.")
            if self.exponent_base is None or self.exponent_base <= 1.0:
                raise ValueError("exponent_base must be greater than 1.0 when exponential saving is enabled.")
            start = self.exponential_start
            end_value = max(start, end_epoch)
            if start == end_value:
                epochs.add(start)
            else:
                ratio = end_value / start
                approx_steps = max(2, int(math.ceil(math.log(ratio, self.exponent_base))) + 1)
                log_start = math.log(start)
                log_end = math.log(end_value)
                step = (log_end - log_start) / (approx_steps - 1)
                for idx in range(approx_steps):
                    epoch = int(round(math.exp(log_start + idx * step)))
                    if epoch >= start:
                        epochs.add(epoch)
        epochs = sorted(epoch for epoch in epochs if epoch >= 1)
        return epochs

    def _resolve_dirpath(self, trainer: pl.Trainer) -> Path:
        base: Optional[Path] = None
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        dirpath = getattr(checkpoint_cb, "dirpath", None)
        if dirpath is not None:
            base = Path(dirpath)
        elif trainer.log_dir is not None:
            base = Path(trainer.log_dir)
        else:
            base = Path(trainer.default_root_dir)
        return base / "scheduled_ckpts"

    def _format_filename(self, epoch: int) -> str:
        prefix = "lin" if self.use_linear else "exp" if self.use_exponential else "ckpt"
        return f"{prefix}-epoch={epoch:04d}.ckpt"

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._scheduled_epochs or trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        epoch_idx = trainer.current_epoch + 1
        if epoch_idx not in self._scheduled_epochs:
            return
        if epoch_idx in self._saved_epochs:
            return
        if self._dirpath is None:
            return
        filename = self._format_filename(epoch_idx)
        filepath = str(self._dirpath / filename)
        trainer.save_checkpoint(filepath, weights_only=self.only_small_ckpts)
        self._saved_epochs.add(epoch_idx)
        rank_zero_info(
            f"Saved scheduled checkpoint at epoch {epoch_idx} to {filepath} (weights_only={self.only_small_ckpts})."
        )

    def state_dict(self) -> dict:
        return {"saved_epochs": sorted(self._saved_epochs)}

    def load_state_dict(self, state_dict: dict) -> None:
        saved = state_dict.get("saved_epochs", [])
        if isinstance(saved, Sequence):
            self._saved_epochs = set(int(epoch) for epoch in saved)
        else:
            self._saved_epochs = set()