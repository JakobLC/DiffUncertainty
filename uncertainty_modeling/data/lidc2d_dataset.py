import os
import pickle
import fnmatch
import random
import hashlib
from typing import List

import numpy as np
import torch

AUGMENTED_SPLITS = {
    "ood_noise", "ood_blur", "ood_contrast", "ood_jpeg",  # LIDC augmented OOD splits
    "ood_fov",  # Retina augmented OOD split (FOV-based augmentation)
}

NUM_RATERS_TO_DATASET = {
    4: ["lidc64", "lidc128", "origlidc64", "origlidc128"],
    5: ["chaksu64", "chaksu128"],
    6: ["riga64", "riga128"],
    7: ["refuge64", "refuge128"],
}
DATASET_TO_NUM_RATERS = {ds: num for num, dss in NUM_RATERS_TO_DATASET.items() for ds in dss}


def infer_num_raters_from_dataset_name(dataset_name: str) -> int:
    dataset_key = str(dataset_name).strip().lower()
    try:
        return DATASET_TO_NUM_RATERS[dataset_key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_TO_NUM_RATERS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Cannot infer num_raters. Known datasets: {available}"
        ) from exc


def collate_multirater_batch(batch: List[dict]) -> dict:
    """Custom collate function for multi-rater datasets with variable rater counts.
    
    Handles batches where different samples have different numbers of raters (e.g., 5, 6, 7).
    Pads segmentation masks to the maximum number of raters in the batch.
    
    Parameters:
    -----------
    batch : List[dict]
        List of sample dicts, each with 'data', 'seg', 'image_id', etc.
    
    Returns:
    --------
    dict : Collated batch with padded 'seg' tensors
        - 'data': [batch_size, 3, H, W] or [batch_size, 1, H, W]
        - 'seg': [batch_size, max_raters, H, W] (padded with zeros)
        - 'image_id': [batch_size] list of image IDs
        - 'dataset': [batch_size] list of dataset names (if available)
        - 'selected_rater_idx': [batch_size] list of rater indices
    """
    if not batch:
        return {}
    
    # Determine max number of raters in this batch
    max_raters = max(sample['seg'].shape[0] for sample in batch)
    
    # Pad and stack
    padded_segs = []
    for sample in batch:
        seg = sample['seg']  # Shape: [num_raters, H, W]
        if seg.shape[0] < max_raters:
            # Pad with zeros to max_raters
            pad_shape = list(seg.shape)
            pad_shape[0] = max_raters - seg.shape[0]
            padding = torch.zeros(pad_shape, dtype=seg.dtype, device=seg.device)
            seg = torch.cat([seg, padding], dim=0)
        padded_segs.append(seg)
    
    # Stack all tensors
    collated = {
        'data': torch.stack([s['data'] for s in batch]),
        'seg': torch.stack(padded_segs),
        'image_id': [s['image_id'] for s in batch],
    }
    
    # Optional fields
    if 'dataset' in batch[0]:
        collated['dataset'] = [s['dataset'] for s in batch]
    if 'selected_rater_idx' in batch[0]:
        collated['selected_rater_idx'] = torch.stack([
            torch.tensor(s['selected_rater_idx']) if s['selected_rater_idx'] is not None 
            else torch.tensor(-1)
            for s in batch
        ])
    
    return collated



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
        single_rater: bool = False,
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
        self.single_rater = bool(single_rater)
        self._single_rater_seed = 13
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
        
        # Check if this is a combined dataset
        is_combined = "combined" in str(self.split_schema or "").lower()
        
        if is_combined:
            # Combined dataset: load from multiple dataset directories with variable GT counts
            self._init_combined_dataset(fold_entry, base_dir, split)
        else:
            # Single dataset: original behavior
            self._init_single_dataset(fold_entry, base_dir, split)

    def _init_single_dataset(self, fold_entry, base_dir, split):
        """Initialize for a single dataset (original behavior)."""
        inferred_dataset_label = self.dataset_label or self.split_metadata.get("dataset_name")
        if inferred_dataset_label is None:
            inferred_dataset_label = os.path.basename(os.path.normpath(base_dir))
        self.dataset_label = str(inferred_dataset_label)
        self.num_raters = infer_num_raters_from_dataset_name(self.dataset_label)
        if self.num_raters is not None and int(self.num_raters) != self.num_raters:
            raise ValueError(
                f"num_raters={self.num_raters} does not match the inferred value {self.num_raters} for dataset '{self.dataset_label}'."
            )
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
            pattern=self.file_pattern,
            subject_ids=subject_ids,
            num_raters=self.num_raters,
            rater_pattern=self.rater_pattern,
        )

        self.imgs = [sample["image_path"] for sample in self.samples]
        self.mask_paths = [sample["label_paths"] for sample in self.samples]
        self.image_ids = [sample["image_id"] for sample in self.samples]
        self._fixed_rater_indices = None
        if self.single_rater and self.num_raters > 0:
            self._fixed_rater_indices = [
                self._stable_rater_index(image_id) for image_id in self.image_ids
            ]

        print(
            f"Dataset: {self.dataset_label} {split} - {len(self.imgs)} images",
        )

    def _init_combined_dataset(self, fold_entry, base_dir, split):
        """Initialize for a combined dataset with prefixed paths and variable GT counts."""
        self.dataset_label = "combined_retina"
        
        # Get dataset configs from metadata
        dataset_configs = self.split_metadata.get("dataset_configs", {})
        if not dataset_configs:
            raise ValueError(
                "Combined dataset metadata must include 'dataset_configs' with num_raters info"
            )
        
        # Build mapping of dataset name -> config
        self.dataset_num_raters = {}
        for dataset_name, config in dataset_configs.items():
            self.dataset_num_raters[dataset_name] = config.get("num_raters")
        
        # Get parent directory of the combined dataset
        parent_dir = os.path.dirname(os.path.normpath(base_dir))
        
        # Determine if this is an augmented split and map to the actual split in metadata
        actual_split_for_meta = split
        if split in AUGMENTED_SPLITS and split not in fold_entry:
            # For augmented splits like ood_fov, ood_noise, etc., they reference id_test in metadata
            actual_split_for_meta = "id_test"
        
        subject_ids = self._resolve_subject_ids(fold_entry, actual_split_for_meta)
        
        # Load samples from multiple source directories, passing the split for augmented path resolution
        samples = load_combined_multirater_samples(
            parent_dir=parent_dir,
            subject_ids=subject_ids,
            dataset_num_raters=self.dataset_num_raters,
            pattern=self.file_pattern,
            split=split,  # Pass the original split name for augmented directory resolution
        )
        
        self.samples = samples
        self.imgs = [sample["image_path"] for sample in samples]
        self.mask_paths = [sample["label_paths"] for sample in samples]
        self.image_ids = [sample["image_id"] for sample in samples]
        self.dataset_prefixes = [sample["dataset_prefix"] for sample in samples]
        self.sample_num_raters = [sample["num_raters"] for sample in samples]
        
        self._fixed_rater_indices = None
        
        print(
            f"Dataset: {self.dataset_label} {split} - {len(self.imgs)} images "
            f"from {len(set(self.dataset_prefixes))} source datasets",
        )

    def __len__(self):
        return len(self.imgs)

    def _resolve_subject_ids(self, fold_entry, split):
        if split == "unlabeled":
            id_pool = np.asarray(fold_entry.get("id_unlabeled_pool", []), dtype=object)
            ood_pool = np.asarray(fold_entry.get("ood_unlabeled_pool", []), dtype=object)
            return np.concatenate((id_pool, ood_pool)).tolist()
        
        # Map split aliases to actual keys in fold_entry
        key = split
        if split == "id":
            key = "id_test"
        elif split == "ood":
            # Try "ood_test" first (old format), then "ood_fov" (new format for augmentation-based OOD)
            if "ood_test" in fold_entry:
                key = "ood_test"
            elif "ood_fov" in fold_entry:
                key = "ood_fov"
        elif split == "ood_test":
            # If explicitly requesting "ood_test", but it doesn't exist, try "ood_fov"
            if split not in fold_entry and "ood_fov" in fold_entry:
                key = "ood_fov"
        
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
        # divide by 255 if dtype==uint8
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
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

    def _get_num_raters(self, idx: int) -> int:
        """Get the number of raters for a specific sample."""
        if hasattr(self, 'sample_num_raters'):
            # Combined dataset: variable rater counts
            return self.sample_num_raters[idx]
        else:
            # Single dataset: same rater count for all
            return self.num_raters

    def _stable_rater_index(self, image_id: str, num_raters: int = None) -> int:
        if num_raters is None:
            num_raters = self.num_raters
        seed_key = f"{self._single_rater_seed}:{self.dataset_label}:{self.split}:{image_id}"
        digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], byteorder="big", signed=False)
        return value % num_raters

    def _get_selected_rater_index(self, idx: int) -> int:
        if self._fixed_rater_indices is not None:
            return int(self._fixed_rater_indices[idx])
        num_raters = self._get_num_raters(idx)
        return self._stable_rater_index(self.image_ids[idx], num_raters)

    def _select_mask(self, idx: int, image_shape: tuple) -> np.ndarray:
        mask_paths: List[str] = self.mask_paths[idx]
        if self.return_all_raters:
            # Evaluation requires all raters for metrics like GED.
            masks = self._load_masks(mask_paths)
            return masks
        if self.single_rater:
            selected_idx = self._get_selected_rater_index(idx)
            chosen_path = mask_paths[selected_idx]
        else:
            chosen_path = random.choice(mask_paths)
        return self._load_mask(chosen_path)

    def __getitem__(self, idx: int):
        img = self._load_image(self.imgs[idx])
        mask = self._select_mask(idx, img.shape)
        selected_rater_idx = self._get_selected_rater_index(idx) if self.single_rater else None
        
        # For combined datasets, use the actual dataset prefix; for single datasets, use dataset_label
        if hasattr(self, 'dataset_prefixes'):
            dataset_label = self.dataset_prefixes[idx]
        else:
            dataset_label = self.dataset_label
        
        if self.tta:
            # Option A: return raw image values under 'data' and keep model-side TTA/inversion in test_2D.py.
            if self.return_all_raters:
                mask_t = torch.from_numpy(np.stack(mask, axis=0)).long()
            else:
                mask_t = torch.from_numpy(mask).long()
            img_t = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
            sample = {
                "data": img_t,
                "seg": mask_t,
                "image_id": self.image_ids[idx],
                "dataset": dataset_label,
            }
            if selected_rater_idx is not None:
                sample["selected_rater_idx"] = selected_rater_idx
            return sample
        else:
            if self.transforms is not None:
                if self.return_all_raters:
                    transformed = self.transforms(image=img, masks=mask)
                    mask_t = torch.stack(transformed["masks"], dim=0)    
                else:
                    transformed = self.transforms(image=img, mask=mask)
                    mask_t = transformed["mask"]
                img_t = transformed["image"].float()
            else:
                # No transforms applied
                if self.return_all_raters:
                    mask_t = torch.from_numpy(np.stack(mask, axis=0)).long()
                else:
                    mask_t = torch.from_numpy(mask).long()
                img_t = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
            
            sample = {
                "data": img_t,
                "seg": mask_t,
                "image_id": self.image_ids[idx],
                "dataset": dataset_label,
            }
            if selected_rater_idx is not None:
                sample["selected_rater_idx"] = selected_rater_idx
            return sample

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


def load_combined_multirater_samples(
    parent_dir: str,
    subject_ids=None,
    dataset_num_raters: dict = None,
    pattern: str = "*.npy",
    rater_pattern: str = "{base_id}_{rater:02d}_mask.npy",
    split: str = None,
):
    """Load samples from multiple datasets with prefixed paths and variable rater counts.
    
    This function handles combined datasets where samples have prefixed paths like
    "chaksu64/t_000000.npy" and different datasets have different numbers of raters.
    Supports augmented splits (e.g., ood_fov, ood_noise) which reference the same
    samples as id_test but from an augmented image directory.
    
    Parameters:
    -----------
    parent_dir : str
        Parent directory containing source datasets (e.g., /path/to/values_datasets)
    subject_ids : list, optional
        List of subject IDs with dataset prefixes (e.g., ["chaksu64/t_000000.npy", ...])
    dataset_num_raters : dict
        Mapping of dataset name -> number of raters (e.g., {"chaksu64": 5, "refuge64": 7})
    pattern : str
        File pattern to match (default: "*.npy")
    rater_pattern : str
        Pattern for rater mask filenames
    split : str, optional
        The split name (e.g., "train", "ood_fov", "ood_noise"). If an augmented split,
        the image directory will be looked up in augmented/<split>/images instead of images.
    
    Returns:
    --------
    list of dicts with keys: image_path, label_paths, image_id, dataset_prefix, num_raters
    """
    samples = []
    
    if subject_ids is None:
        subject_ids = []
    
    # Determine if this is an augmented split
    is_augmented_split = split in AUGMENTED_SPLITS
    
    # Build a set of (dataset_name, image_filename) tuples from subject_ids
    subject_filter = {}
    for sid in subject_ids:
        sid_str = str(sid)
        if not sid_str:
            continue
        
        # Parse the prefixed path: "dataset_name/image_file.npy"
        if "/" in sid_str:
            parts = sid_str.split("/", 1)
            dataset_name = parts[0]
            image_spec = parts[1]
        else:
            continue
        
        if dataset_name not in subject_filter:
            subject_filter[dataset_name] = set()
        
        subject_filter[dataset_name].add(image_spec)
        # Also add basename and base without extension
        basename = os.path.basename(image_spec)
        subject_filter[dataset_name].add(basename)
        subject_filter[dataset_name].add(os.path.splitext(basename)[0])
    
    # Process each dataset
    for dataset_name in sorted(dataset_num_raters.keys()):
        num_raters = dataset_num_raters[dataset_name]
        dataset_root = os.path.join(parent_dir, dataset_name)
        
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        
        # Resolve image directory (may be in augmented/ subdirectory for augmented splits)
        if is_augmented_split:
            image_dir = os.path.join(dataset_root, "preprocessed", "augmented", split, "images")
        else:
            image_dir = os.path.join(dataset_root, "preprocessed", "images")
        
        # Labels are always in the main labels directory
        label_dir = os.path.join(dataset_root, "preprocessed", "labels")
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        dataset_filter = subject_filter.get(dataset_name)
        
        try:
            (_, _, image_filenames) = next(os.walk(image_dir))
        except StopIteration:
            continue
        
        for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
            base_id = os.path.splitext(image_filename)[0]
            
            # Check if this image should be included
            if dataset_filter is not None:
                if image_filename not in dataset_filter and base_id not in dataset_filter:
                    continue
            
            image_path = os.path.join(image_dir, image_filename)
            
            # Build label paths
            label_paths = []
            for rater in range(num_raters):
                label_name = rater_pattern.format(
                    base_id=base_id,
                    filename=image_filename,
                    rater=rater,
                )
                label_path = os.path.join(label_dir, label_name)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(
                        f"Missing rater mask: {label_path} for dataset {dataset_name}"
                    )
                label_paths.append(label_path)
            
            samples.append(
                {
                    "image_path": image_path,
                    "label_paths": label_paths,
                    "image_id": base_id,
                    "dataset_prefix": dataset_name,
                    "num_raters": num_raters,
                }
            )
    
    return samples
