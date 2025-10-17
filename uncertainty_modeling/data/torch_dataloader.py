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

from typing import List, Union
from omegaconf import DictConfig, ListConfig, open_dict

def _set_by_path(cfg: Union[DictConfig, ListConfig], path: List[str], value) -> None:
    
    #Set cfg[path[0]][path[1]]...[path[-1]] = value.
    #- Raises KeyError/IndexError if any segment is missing.
    #- Supports ListConfig via integer-like segments (e.g., "0").
    
    if not path:
        raise KeyError("Empty path")

    cur = cfg
    for seg in path[:-1]:
        if isinstance(cur, ListConfig):
            try:
                idx = int(seg)
            except ValueError:
                raise KeyError(f"Expected list index, got '{seg}' in path {path}")
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"Index {idx} out of range for segment '{seg}' in path {path}")
            cur = cur[idx]
        else:
            if seg not in cur:
                raise KeyError(f"Missing key '{seg}' in path {path}")
            cur = cur[seg]

    last = path[-1]
    with open_dict(cfg):
        if isinstance(cur, ListConfig):
            try:
                idx = int(last)
            except ValueError:
                raise KeyError(f"Expected list index, got '{last}' in path {path}")
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"Index {idx} out of range for last segment '{last}' in path {path}")
            cur[idx] = value
        else:
            if last not in cur:
                raise KeyError(f"Missing final key '{last}' in path {path}")
            cur[last] = value

def apply_augment_mult2(augmentations: DictConfig) -> DictConfig:
    if "augment_mult" in augmentations.TRAIN and "apply_mult" in augmentations.TRAIN:
        mult = augmentations.TRAIN.augment_mult
        for key_seq in augmentations.TRAIN.apply_mult:
            path = ["TRAIN","Compose"] + key_seq.split(".")
            d = augmentations
            for k in path:
                d = d[k]
            if isinstance(d, list):
                for i in range(len(d)):
                    d[i] = d[i] * mult
            else:
                d = d * mult
            print("Setting", path, "to", d)
            _set_by_path(augmentations, path, d)
        del augmentations.TRAIN.augment_mult
        del augmentations.TRAIN.apply_mult
    return augmentations

def apply_augment_mult(augmentations: DictConfig) -> DictConfig:
    return augmentations
    print("augment_mult" in augmentations.TRAIN and "apply_mult" in augmentations.TRAIN)
    if "augment_mult" in augmentations.TRAIN and "apply_mult" in augmentations.TRAIN:
        mult = augmentations.TRAIN.augment_mult
        for key_seq in augmentations.TRAIN.apply_mult:
            keys = key_seq.split(".")
            d = augmentations.TRAIN.Compose.transforms
            for k in keys[:-1]:
                d = d[k]
            last_key = keys[-1]
            if last_key in d:
                if isinstance(d[last_key], list):
                    d[last_key] = [v * mult for v in d[last_key]]
                elif isinstance(d[last_key], (int, float)):
                    d[last_key] = d[last_key] * mult
                else:
                    print(f"Warning: augment_mult not applied to {key_seq} with value {d[last_key]}")
            else:
                print(f"Warning: key {last_key} not found in augmentation {key_seq}")
        del augmentations.TRAIN.augment_mult
        del augmentations.TRAIN.apply_mult
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
                print("No Operation Found: %s", transform)
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
        # placeholder for optional evaluation dataset based on training data
        self.DS_train_eval = None

    def setup(self, stage: str = None) -> None:
        """
        Setting up the Datasets by initializing the augmentation and the dataloader

        Parameters
        ----------
        stage: str
            current stage which is given by Pytorch Lightning
        """
        if stage in (None, "fit"):
            #print(self.augmentations.TRAIN.Compose.transforms.RandomScale.scale_limit)
            #self.augmentations = apply_augment_mult(self.augmentations)
            #print(self.augmentations.TRAIN.Compose.transforms.RandomScale.scale_limit)
            #exit()
            transforms_train = get_augmentations_from_config(self.augmentations.TRAIN)[0]
            self.DS_train = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split="train",
                transforms=transforms_train,
            )
        if stage in (None, "fit", "validate"):
            transforms_val = get_augmentations_from_config(self.augmentations.VALIDATION)[0]
            self.DS_val = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split="val",
                transforms=transforms_val,
                tta=self.tta,
            )
            # Optionally build a validation-view of the training set (independent of train dataloader)
            if self.evaluate_training_data:
                # create a training dataset instance but with validation transforms
                DS_train_full_for_eval = hydra.utils.instantiate(
                    self.dataset,
                    base_dir=self.data_input_dir,
                    split="train",
                    transforms=transforms_val,
                    tta=self.tta,
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
            self.DS_test = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split=test_split,
                transforms=transforms_test,
                tta=self.tta,
            )

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
