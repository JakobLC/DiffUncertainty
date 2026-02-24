import os
import pickle
import fnmatch
import random
from typing import List

import albumentations
import numpy as np
import torch

AUGMENTED_SPLITS = {"ood_noise", "ood_blur", "ood_contrast", "ood_jpeg"}


class MultiRater2DDataset(torch.utils.data.Dataset):
    """Multi-rater 2D segmentation dataset (supports LIDC & Chaksu).

    Mirrors how the Cityscapes dataset plugs into ``BaseDataModule`` while
    reading from ``preprocessed/images`` & ``preprocessed/labels`` plus split
    definitions stored in ``splits.pkl``. Images can be grayscale (LIDC) or RGB
    (Chaksu) and every base image has multiple rater masks following a format
    such as ``{base_id}_{rater:02d}_mask.npy``. ``_meta`` inside ``splits.pkl``
    can optionally define ``dataset_name``, ``num_raters`` or ``rater_pattern``
    so configs only need to override the pieces that differ from LIDC defaults.
    """

    def __init__(
        self,
        splits_path: str,
        base_dir: str,
        split: str = "train",
        file_pattern: str = "*.npy",
        transforms=None,
        data_fold_id: int = 0,
        tta: bool = False,
        replicate_channels: bool = True,
        return_all_raters: bool = True,
        num_raters: int = None,
        rater_pattern: str = None,
        dataset_label: str = None,
    ):
        self.splits_path = splits_path
        self.split = split
        self.file_pattern = file_pattern
        self.transforms = transforms
        self.tta = tta
        self.replicate_channels = replicate_channels
        self.return_all_raters = return_all_raters
        self.dataset_label = dataset_label
        self.num_raters = num_raters
        self.rater_pattern = rater_pattern

        with open(self.splits_path, "rb") as f:
            splits = pickle.load(f)
        if not isinstance(splits, (list, tuple)) or not splits:
            raise ValueError("Expected splits.pkl to contain a non-empty list of fold dicts.")
        fold_entry = splits[data_fold_id]
        if not isinstance(fold_entry, dict):
            raise ValueError("Each fold entry inside splits.pkl must be a dictionary.")
        self.split_metadata = fold_entry.get("_meta", {})
        self.split_schema = self.split_metadata.get("schema")
        self.dataset_label = self.dataset_label or self.split_metadata.get("dataset_name", "lidc2d") #TODO: instead of this, use the last part of base_dir as the name
        meta_num_raters = self.split_metadata.get("num_raters")
        if self.num_raters is None:
            self.num_raters = int(meta_num_raters) if meta_num_raters is not None else 4
        if self.num_raters <= 0:
            raise ValueError("num_raters must be a positive integer.")
        meta_pattern = self.split_metadata.get("rater_pattern")
        self.rater_pattern = self.rater_pattern or meta_pattern or "{base_id}_{rater:02d}_mask.npy"
        subject_ids = self._resolve_subject_ids(fold_entry, split)
        # Directory that contains the preprocessed images/labels folders
        proc_dir = os.path.join(base_dir, "preprocessed")
        image_dir = self._resolve_image_dir(proc_dir, split)
        label_dir = os.path.join(proc_dir, "labels")

        self.samples = load_multirater_samples(
            image_dir=image_dir,
            label_dir=label_dir,
            pattern=file_pattern,
            subject_ids=subject_ids,
            num_raters=self.num_raters,
            rater_pattern=self.rater_pattern,
        )

        self.imgs = [sample["image_path"] for sample in self.samples]
        self.mask_paths = [sample["label_paths"] for sample in self.samples]
        self.image_ids = [sample["image_id"] for sample in self.samples]

        print(
            f"Dataset: {self.dataset_label} {split} - {len(self.imgs)} images",
        )

    def __len__(self):
        return len(self.imgs)

    def _resolve_subject_ids(self, fold_entry, split):
        if split == "unlabeled":
            id_pool = np.asarray(fold_entry.get("id_unlabeled_pool", []), dtype=object)
            ood_pool = np.asarray(fold_entry.get("ood_unlabeled_pool", []), dtype=object)
            return np.concatenate((id_pool, ood_pool)).tolist()
        key = "id_test" if split == "id" else split
        if key not in fold_entry:
            if split in {"ood", "ood_test"} and any(name in fold_entry for name in AUGMENTED_SPLITS):
                available = ", ".join(sorted(name for name in AUGMENTED_SPLITS if name in fold_entry)) or "<none>"
                raise ValueError(
                    "Requested split '" + split + "' is not available for schema '"
                    + str(self.split_schema or "unknown")
                    + "'. Pick one of: "
                    + available
                )
            available = sorted(k for k in fold_entry.keys() if not k.startswith("_"))
            raise ValueError(
                f"Unknown split '{split}'. Available options: {', '.join(available)}"
            )
        values = fold_entry[key]
        if isinstance(values, np.ndarray):
            return values.tolist()
        return list(values)

    def _resolve_image_dir(self, proc_dir: str, split: str) -> str:
        if split in AUGMENTED_SPLITS:
            candidate = os.path.join(proc_dir, "augmented", split, "images")
        else:
            candidate = os.path.join(proc_dir, "images")
        if not os.path.isdir(candidate):
            raise FileNotFoundError(
                f"Expected directory '{candidate}' for split '{split}', but it does not exist."
            )
        return candidate

    def _load_image(self, path: str) -> np.ndarray:
        img = np.load(path)
        img = img.astype(np.float32)
        if img.ndim == 2:
            if self.replicate_channels:
                img = np.repeat(img[..., None], 3, axis=2)
            else:
                img = img[..., None]
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        return np.load(path)

    def _load_masks(self, paths: List[str]) -> List[np.ndarray]:
        return [self._load_mask(p) for p in paths]

    def _select_mask(self, idx: int, image_shape: tuple) -> np.ndarray:
        mask_paths: List[str] = self.mask_paths[idx]
        if self.return_all_raters:
            # Evaluation requires all raters for metrics like GED.
            masks = self._load_masks(mask_paths)
            return masks
        chosen_path = random.choice(mask_paths)
        return self._load_mask(chosen_path)

    def __getitem__(self, idx: int):
        img = self._load_image(self.imgs[idx])
        mask = self._select_mask(idx, img.shape)
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
                "dataset": self.dataset_label,
                "transforms": transforms_applied,
            }
        else:
            if self.return_all_raters:
                transformed = self.transforms(image=img, masks=mask)
                mask_t = torch.stack(transformed["masks"], dim=0)    
            else:
                transformed = self.transforms(image=img, mask=mask)
                mask_t = transformed["mask"]
            img_t = transformed["image"].float()
            return {
                "data": img_t,
                "seg": mask_t,
                "image_id": self.image_ids[idx],
                "dataset": self.dataset_label,
            }

LIDC2DDataset = MultiRater2DDataset  # Backward compatibility alias

def load_multirater_samples(
    image_dir: str,
    label_dir: str,
    pattern: str = "*.npy",
    subject_ids=None,
    num_raters: int = 4,
    rater_pattern: str = "{base_id}_{rater:02d}_mask.npy",
):
    samples = []
    (_, _, image_filenames) = next(os.walk(image_dir))
    subject_filter = None
    if subject_ids is not None:
        subject_filter = set()
        for sid in subject_ids:
            sid_str = str(sid)
            if not sid_str:
                continue
            subject_filter.add(sid_str)
            basename = os.path.basename(sid_str)
            if basename:
                subject_filter.add(basename)
                subject_filter.add(os.path.splitext(basename)[0])
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        base_id = os.path.splitext(image_filename)[0]
        if subject_filter is not None and image_filename not in subject_filter and base_id not in subject_filter:
            continue
        image_path = os.path.join(image_dir, image_filename)
        label_paths = []
        for rater in range(num_raters):
            label_name = rater_pattern.format(
                base_id=base_id,
                filename=image_filename,
                rater=rater,
            )
            label_paths.append(os.path.join(label_dir, label_name))
        missing = [path for path in label_paths if not os.path.exists(path)]
        if missing:
            missing_rel = ", ".join(os.path.basename(p) for p in missing)
            raise FileNotFoundError(
                f"Missing expected rater masks [{missing_rel}] for image '{image_filename}'"
            )
        samples.append(
            {
                "image_path": image_path,
                "label_paths": label_paths,
                "image_id": base_id,
            }
        )
    return samples
