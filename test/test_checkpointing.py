import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from uncertainty_modeling.callbacks import ScheduledCheckpointCallback
from uncertainty_modeling.lightning_experiment import LightningExperiment


class ScheduledCheckpointCallbackTest(unittest.TestCase):
    def test_linear_schedule_generation(self):
        cfg = {
            "use_linear_saving": True,
            "linear_freq": 10,
            "only_small_ckpts": True,
        }
        callback = ScheduledCheckpointCallback(cfg)
        schedule = callback._build_schedule(50)
        self.assertEqual(schedule, [10, 20, 30, 40, 50])

    def test_exponential_schedule_generation(self):
        cfg = {
            "use_exponential_saving": True,
            "exponential_start": 10,
            "exponent_base": 2.0,
            "only_small_ckpts": False,
        }
        callback = ScheduledCheckpointCallback(cfg)
        schedule = callback._build_schedule(160)
        self.assertEqual(schedule, [10, 20, 40, 80, 160])

    def test_callback_raises_on_conflict(self):
        cfg = {
            "use_linear_saving": True,
            "use_exponential_saving": True,
        }
        with self.assertRaises(ValueError):
            ScheduledCheckpointCallback(cfg)

    def test_callback_saves_on_expected_epochs(self):
        cfg = {
            "use_linear_saving": True,
            "linear_freq": 5,
            "only_small_ckpts": True,
            "end": 20,
        }
        callback = ScheduledCheckpointCallback(cfg)
        trainer = SimpleNamespace()
        trainer.sanity_checking = False
        trainer.max_epochs = 20
        trainer.is_global_zero = True
        trainer.log_dir = None
        trainer.default_root_dir = None
        trainer.checkpoint_callback = SimpleNamespace(dirpath=None)
        saved = []

        def save_checkpoint(path, weights_only=False):
            saved.append((Path(path), weights_only))

        trainer.save_checkpoint = save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.log_dir = tmpdir
            callback.setup(trainer, MagicMock())
            trainer.current_epoch = 4
            callback.on_train_epoch_end(trainer, MagicMock())
            trainer.current_epoch = 19
            callback.on_train_epoch_end(trainer, MagicMock())

        self.assertEqual(len(saved), 1)
        path, weights_only = saved[0]
        self.assertIn("scheduled_ckpts", str(path))
        self.assertTrue(weights_only)
        self.assertTrue(path.name.startswith("lin-epoch=0005"))


class LightningExperimentEMATest(unittest.TestCase):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self.num_classes = 2
            self.ssn = False

        def forward(self, x, **kwargs):
            return self.layer(x)

    def test_state_dict_contains_ema_weights(self):
        dummy_model = self.DummyModel()
        cfg = OmegaConf.create(
            {
                "datamodule": {"ignore_index": 0},
                "model": {"_target_": "tests.DummyModel"},
                "batch_size": 2,
                "track_ema_weights": True,
                "ema_decay": 0.9,
            }
        )
        with patch(
            "uncertainty_modeling.lightning_experiment.hydra.utils.instantiate",
            return_value=dummy_model,
        ):
            module = LightningExperiment(cfg)
        module._ensure_ema_model()
        state_keys = module.state_dict().keys()
        self.assertTrue(
            any(key.startswith("ema_model.") for key in state_keys),
            "EMA parameters not found in module.state_dict()",
        )


if __name__ == "__main__":
    unittest.main()
