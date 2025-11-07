"""
------------------------------------------------------------------------------
Code slightly adapted and mainly from:
https://github.com/MIC-DKFZ/semantic_segmentation/blob/public/datasets/DataModules.py
------------------------------------------------------------------------------
"""

import hydra
from omegaconf import DictConfig
import numpy as np

from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
import random

import uncertainty_modeling.augmentations as custom_augmentations

# set number of Threads to 0 for opencv and albumentations
cv2.setNumThreads(0)


def seed_worker(worker_id):
    """
    from: https://github.com/MIC-DKFZ/image_classification/blob/master/base_model.py
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_max_steps(
    size_dataset,
    batch_size,
    num_devices,
    accumulate_grad_batches,
    num_epochs,
    drop_last=True,
) -> int:
    """
    Computing the number of  steps, needed for polynomial lr scheduler
    Considering the number of gpus and if accumulate_grad_batches is used

    Returns
    -------
    int:
        total number of steps
    int:
        number of steps per epoch
    """
    # How many steps per epoch in total
    if drop_last:
        steps_per_epoch = size_dataset // batch_size  # round off if drop_last=False
    else:
        steps_per_epoch = np.ceil(
            size_dataset / batch_size
        )  # round up if drop_last=False

    # For ddp and multiple gpus the effective batch sizes doubles
    steps_per_gpu = int(np.ceil(steps_per_epoch / num_devices))
    # Include accumulate_grad_batches
    steps_per_epoch = int(np.ceil(steps_per_gpu / accumulate_grad_batches))
    max_steps = num_epochs * steps_per_epoch

    return max_steps, steps_per_epoch

from omegaconf import ListConfig


def apply_augment_mult(augmentations: DictConfig) -> DictConfig:
    """Finds and applies the augmentation multiplier to the augmentations config."""
    augment_mult = augmentations.get("augment_mult", 1)
    if augment_mult == 1:
        return augmentations

    apply_mult_keys = augmentations.get("apply_mult_keys", [])
    transforms_root = augmentations.TRAIN[0].Compose.transforms

    for key in apply_mult_keys:
        current = transforms_root
        parent_ref = None
        parent_key = None

        for subkey in key.split("."):
            if isinstance(current, (list, ListConfig)):
                matched = False
                for item in current:
                    if not isinstance(item, (dict, DictConfig)):
                        continue
                    if subkey in item:
                        parent_ref = item
                        parent_key = subkey
                        current = item[subkey]
                        matched = True
                        break
                if not matched:
                    raise ValueError(
                        f"Could not find subkey {subkey} from key {key} in list: {current}"
                    )
            elif isinstance(current, (DictConfig, dict)):
                if subkey not in current:
                    raise ValueError(
                        f"Subkey {subkey} from key {key} not found in dict: {current}"
                    )
                parent_ref = current
                parent_key = subkey
                current = current[subkey]
            else:
                raise ValueError(
                    f"Unexpected type {type(current)} when traversing augmentation config."
                )

        if parent_ref is None or parent_key is None:
            raise ValueError(f"Failed to resolve key path {key} in augmentation config.")

        value = current
        if isinstance(value, (int, float)):
            parent_ref[parent_key] = value * augment_mult
        elif isinstance(value, (list, ListConfig)):
            if isinstance(value, ListConfig):
                for idx in range(len(value)):
                    value[idx] = value[idx] * augment_mult
            else:
                parent_ref[parent_key] = [v * augment_mult for v in value]
        else:
            raise ValueError(
                f"Unexpected type {type(value)} for augmentation parameter when applying augment_mult."
            )

    return augmentations

def get_augmentations_from_config(augmentations: DictConfig) -> list:
    """
    Build an Albumentations augmentation pipeline from the input config

    Parameters
    ----------
    augmentations : DictConfig
        config of the Augmentation

    Returns
    -------
    list :
        list of Albumentations transforms
    """
    # otherwise recursively build the transformations
    trans = []
    for augmentation in augmentations:
        transforms = list(augmentation.keys())

        for transform in transforms:
            parameters = getattr(augmentation, transform)
            if parameters is None:
                parameters = {}
            if hasattr(A, transform):
                if "transforms" in list(parameters.keys()):
                    # "transforms" indicates a transformation which takes a list of other transformations
                    # as input ,e.g. A.Compose -> recursively build these transforms
                    transforms = get_augmentations_from_config(parameters.transforms)
                    del parameters["transforms"]
                    func = getattr(A, transform)
                    trans.append(func(transforms=transforms, **parameters))
                else:
                    # load transformation form Albumentations
                    parameters = {k: list(v) if isinstance(v, ListConfig) else v for k, v in parameters.items()}
                    func = getattr(A, transform)
                    trans.append(func(**parameters))
            elif hasattr(A.pytorch, transform):
                # ToTensorV2 transformation is located in A.pytorch
                func = getattr(A.pytorch, transform)
                trans.append(func(**parameters))
            elif hasattr(custom_augmentations, transform):
                func = getattr(custom_augmentations, transform)
                trans.append(func(**parameters))
            else:
                raise ValueError(f"Augmentation {transform} not found in Albumentations or custom augmentations.")
    return trans


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_input_dir: str,
        dataset,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        augmentations: DictConfig,
        evaluate_training_data: bool = False,
        evaluate_all_raters: bool = True,
        tta: bool = False,
        **kwargs,
    ) -> None:
        """
        __init__ the LightningModule
        save parameters

        Parameters
        ----------
        dataset : DictConfig
            config of the dataset, is called by hydra.src.instantiate(dataset,split=.., transforms=..)
        batch_size : int
            batch size for train dataloader
        val_batch_size : int
            batch size for val and test dataloader
        num_workers : int
            number of workers for all dataloaders
        augmentations : DictConfig
            config containing the augmentations for Train, Test and Validation
        """
        super().__init__()

        # parameters for dataloader
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        # self.augmentations = get_augmentations()
        self.augmentations = augmentations
        self.data_input_dir = data_input_dir
        # dataset which is defined in the config
        self.dataset = dataset
        self.test_split = kwargs.get("test_split", None)
        self.tta = tta
        # whether to return an additional validation dataloader sampling from training set
        self.evaluate_training_data = evaluate_training_data
        self.evaluate_all_raters = evaluate_all_raters
        # placeholder for optional evaluation dataset based on training data
        self.DS_train_eval = None

    def _is_lidc2d_dataset(self) -> bool:
        target = ""
        if isinstance(self.dataset, DictConfig):
            target = self.dataset.get("_target_", "")
        elif isinstance(self.dataset, dict):
            target = self.dataset.get("_target_", "")
        return "LIDC2DDataset" in str(target)

    def setup(self, stage: str = None) -> None:
        """
        Setting up the Datasets by initializing the augmentation and the dataloader

        Parameters
        ----------
        stage: str
            current stage which is given by Pytorch Lightning
        """
        is_lidc2d = self._is_lidc2d_dataset()

        if stage in (None, "fit"):
            self.augmentations = apply_augment_mult(self.augmentations)
            transforms_train = get_augmentations_from_config(self.augmentations.TRAIN)[0]
            train_kwargs = dict(
                base_dir=self.data_input_dir,
                split="train",
                transforms=transforms_train,
            )
            if is_lidc2d:
                train_kwargs["return_all_raters"] = False
            self.DS_train = hydra.utils.instantiate(self.dataset, **train_kwargs)
        if stage in (None, "fit", "validate"):
            transforms_val = get_augmentations_from_config(self.augmentations.VALIDATION)[0]
            val_kwargs = dict(
                base_dir=self.data_input_dir,
                split="val",
                transforms=transforms_val,
                tta=self.tta,
            )
            if is_lidc2d:
                val_kwargs["return_all_raters"] = self.evaluate_all_raters
            self.DS_val = hydra.utils.instantiate(self.dataset, **val_kwargs)
            # Optionally build a validation-view of the training set (independent of train dataloader)
            if self.evaluate_training_data:
                # create a training dataset instance but with validation transforms
                train_eval_kwargs = dict(
                    base_dir=self.data_input_dir,
                    split="train",
                    transforms=transforms_val,
                    tta=self.tta,
                )
                if is_lidc2d:
                    train_eval_kwargs["return_all_raters"] = self.evaluate_all_raters
                DS_train_full_for_eval = hydra.utils.instantiate(
                    self.dataset,
                    **train_eval_kwargs,
                )
                # sample a random subset of the training set equal in size to the validation set
                target_size = min(len(DS_train_full_for_eval), len(self.DS_val))
                if target_size > 0:
                    indices = np.random.permutation(len(DS_train_full_for_eval))[:target_size]
                    self.DS_train_eval = Subset(DS_train_full_for_eval, indices.tolist())
                else:
                    self.DS_train_eval = Subset(DS_train_full_for_eval, [])
        if stage in (None, "test"):
            transforms_test = get_augmentations_from_config(self.augmentations.TEST)[0]
            test_split = (
                self.test_split
                if self.test_split == "unlabeled" or self.test_split == "val"
                else f"{self.test_split}_test"
            )
            test_kwargs = dict(
                base_dir=self.data_input_dir,
                split=test_split,
                transforms=transforms_test,
                tta=self.tta,
            )
            if is_lidc2d:
                test_kwargs["return_all_raters"] = True
            self.DS_test = hydra.utils.instantiate(self.dataset, **test_kwargs)

    def max_steps(self) -> int:
        """
        Computing and Logging the number of training steps, needed for polynomial lr scheduler
        Considering the number of gpus and if accumulate_grad_batches is used

        Returns
        -------
        int:
            number of training steps
        """
        # computing the maximal number of steps for training
        max_steps, max_steps_epoch = get_max_steps(
            size_dataset=len(self.DS_train),
            batch_size=self.batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            num_epochs=self.trainer.max_epochs,
            drop_last=True,
        )

        print(
            "Number of Training steps: {}  ({} steps per epoch)".format(
                max_steps, max_steps_epoch
            )
        )

        max_steps_val, max_steps_epoch_val = get_max_steps(
            size_dataset=len(self.DS_val),
            batch_size=self.val_batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=1,
            num_epochs=self.trainer.max_epochs,
            drop_last=False,
        )

        print(
            "Number of Validation steps: {}  ({} steps per epoch)".format(
                max_steps_val, max_steps_epoch_val
            )
        )
        
        if self.evaluate_training_data and self.DS_train_eval is not None:
            max_steps_train_eval, max_steps_epoch_train_eval = get_max_steps(
                size_dataset=len(self.DS_train_eval),
                batch_size=self.val_batch_size,
                num_devices=self.trainer.num_devices,
                accumulate_grad_batches=1,
                num_epochs=self.trainer.max_epochs,
                drop_last=False,
            )

            print(
                "Number of Train-Eval steps: {}  ({} steps per epoch)".format(
                    max_steps_train_eval, max_steps_epoch_train_eval
                )
            )
        return max_steps

    def train_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            train dataloader
        """
        return DataLoader(
            self.DS_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            validation dataloader
        """
        val_loader = DataLoader(
            self.DS_val,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        if self.evaluate_training_data and self.DS_train_eval is not None:
            train_eval_loader = DataLoader(
                self.DS_train_eval,
                pin_memory=True,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )
            return [val_loader, train_eval_loader]

        return val_loader

    def test_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            test dataloader
        """
        return DataLoader(
            self.DS_test,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
