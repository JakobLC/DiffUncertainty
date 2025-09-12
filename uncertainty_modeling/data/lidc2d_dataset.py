import os
import pickle
import fnmatch
from typing import List

import albumentations
import numpy as np
import torch

class LIDC2DDataset(torch.utils.data.Dataset):
    """LIDC 2D (cropped nodule) dataset for 2D segmentation.

    This mirrors the interface of the existing Cityscapes dataset so it can be
    plugged into the generic BaseDataModule without additional changes.

    Assumptions:
      * "base_dir" points to the root folder of either lidc_2d or lidc_2d_small.
      * A preprocessed/ folder exists with subfolders images/ and labels/.
      * A splits pickle (splits.pkl) containing keys: train, val, id_test, ood_test
        stored at ``splits_path`` (same structure as for Cityscapes & 3D LIDC).
      * Images are single–channel numpy arrays of shape (H, W); we replicate
        them to 3 channels so that pretrained RGB backbones (HRNet) can be used
        unchanged. (If later desired, model INPUT_CHANNELS can be set to 1 and
        replication disabled.)
      * For each image file ``<id>.npy`` there are rater masks named
        ``<id>_00_mask.npy``, ``<id>_01_mask.npy`` ... (up to 03). We currently
        load the first rater (``_00_mask``). A majority vote or random-rater
        strategy can be added easily if needed.

    TTA behaviour replicates that of the Cityscapes dataset: we create a list of
    augmented image variants plus the base transformed mask (mask is not
    TTA-augmented except for the normalization & tensor conversion applied to
    the original sample). This matches how uncertainty aggregation expects the
    structure for GTA/Cityscapes.
    """

    def __init__(
        self,
        splits_path: str,
        base_dir: str,
        split: str = "train",
        file_pattern: str = "*.npy",
        transforms=None,
        data_fold_id: int = 0,  # kept for API compatibility (unused here)
        tta: bool = False,
        replicate_channels: bool = True,
    ):
        self.splits_path = splits_path
        self.split = split
        self.file_pattern = file_pattern
        self.transforms = transforms
        self.tta = tta
        self.replicate_channels = replicate_channels

        # load splits (same structure as cityscapes & 3D LIDC datamodule)
        with open(self.splits_path, "rb") as f:
            splits = pickle.load(f)
        if split == "train":
            subject_ids = splits[data_fold_id]["train"]
        elif split == "val":
            subject_ids = splits[data_fold_id]["val"]
        elif split == "id_test":
            subject_ids = splits[data_fold_id]["id_test"]
        elif split == "ood_test":
            subject_ids = splits[data_fold_id]["ood_test"]
        else:
            raise ValueError(f"Unknown split '{split}'")
        assert "lidc" in base_dir, f"Did not find expected string 'licd' in {base_dir}"
        # Directory that contains the preprocessed images/labels folders
        proc_dir = os.path.join(base_dir, "preprocessed")
        image_dir = os.path.join(proc_dir, "images")
        label_dir = os.path.join(proc_dir, "labels")

        self.samples = get_lidc2d_samples(
            image_dir=image_dir,
            label_dir=label_dir,
            pattern=file_pattern,
            subject_ids=subject_ids,
        )

        self.imgs = [sample["image_path"] for sample in self.samples]
        self.masks = [sample["label_path"] for sample in self.samples]
        self.image_ids = [sample["image_id"] for sample in self.samples]

        print(
            f"Dataset: LIDC2D {split} - {len(self.imgs)} images - {len(self.masks)} masks",
        )

    def __len__(self):
        return len(self.imgs)

    def _load_image(self, path: str) -> np.ndarray:
        img = np.load(path)  # shape (H, W)
        if img.ndim == 2:
            img = img.astype(np.float32)
            if self.replicate_channels:
                # replicate to 3 channels for RGB backbones
                img = np.repeat(img[..., None], 3, axis=2)
            else:
                img = img[..., None]
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        return np.load(path)  # (H, W) binary (0 background / 1 nodule)

    def __getitem__(self, idx: int):
        img = self._load_image(self.imgs[idx])
        mask = self._load_mask(self.masks[idx])
        if self.tta:
            images: List[np.ndarray] = [img]
            transforms_applied: List[List[str]] = [[]]

            # Simple TTA variants consistent with Cityscapes implementation
            flip_transform = albumentations.HorizontalFlip(p=1.0)
            noise_transform = albumentations.GaussNoise(p=1.0)

            flipped = flip_transform(image=img)
            images.append(flipped["image"])
            transforms_applied.append(["HorizontalFlip"])

            noise = noise_transform(image=img)
            images.append(noise["image"])
            transforms_applied.append(["GaussNoise"])

            flipped_noise = noise_transform(image=flipped["image"])
            images.append(flipped_noise["image"])
            transforms_applied.append(["HorizontalFlip", "GaussNoise"])

            # Apply albumentations pipeline (normalization + tensor) to images
            images = [self.transforms(image=image)["image"].float() for image in images]
            transformed = self.transforms(image=img, mask=mask)
            mask = transformed["mask"]
            return {
                "data": images,
                "seg": mask,
                "image_id": self.image_ids[idx],
                "dataset": "lidc2d",
                "transforms": transforms_applied,
            }
        else:
            transformed = self.transforms(image=img, mask=mask)
            img_t = transformed["image"].float()
            mask_t = transformed["mask"]
            return {
                "data": img_t,
                "seg": mask_t,
                "image_id": self.image_ids[idx],
                "dataset": "lidc2d",
            }


def get_lidc2d_samples(
    image_dir: str,
    label_dir: str,
    pattern: str = "*.npy",
    subject_ids=None,
):
    samples = []
    (_, _, image_filenames) = next(os.walk(image_dir))
    (_, _, label_filenames) = next(os.walk(label_dir))
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        if subject_ids is not None and image_filename not in subject_ids:
            continue
        image_path = os.path.join(image_dir, image_filename)
        # choose first rater mask (00) by default
        base_id = image_filename.split(".")[0]
        mask_candidate = f"{base_id}_00_mask.npy"
        if mask_candidate in label_filenames:
            label_path = os.path.join(label_dir, mask_candidate)
        else:
            # fallback: if only single mask with same name
            label_path = os.path.join(label_dir, image_filename)
        samples.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "image_id": base_id,
            }
        )
    return samples
