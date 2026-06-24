import os
import pickle
import fnmatch
import random
import hashlib
from typing import List

import numpy as np
import torch

NUM_RATERS_TO_DATASET = {
    4: ["lidc64", "lidc128", "origlidc64", "origlidc128","npc64", "npc128"],
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
    """Multi-rater 2D segmentation dataset (supports LIDC, Chaksu, NPC, Retina).

    Reads from ``preprocessed/images`` & ``preprocessed/labels`` plus split
    definitions stored in ``splits.pkl``. Images can be grayscale (LIDC/NPC) or RGB
    (Chaksu/Retina) and every base image has multiple rater masks following a format
    such as ``{base_id}_{rater:02d}_mask.npy``. Supports augmented OOD splits where
    image paths are prefixed with the augmented directory (e.g., ``augmented/ood_noise/images/file.npy``).
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
        label_dir = os.path.join(proc_dir, "labels")

        # With the new format, subject_ids contain full paths relative to proc_dir
        # (e.g., "images/file.npy" or "augmented/ood_noise/images/file.npy")
        self.samples = load_multirater_samples(
            proc_dir=proc_dir,
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
        
        # With new format, subject_ids contain full paths with explicit augmented directory info
        subject_ids = self._resolve_subject_ids(fold_entry, split)
        
        # Load samples from multiple source directories
        samples = load_combined_multirater_samples(
            parent_dir=parent_dir,
            subject_ids=subject_ids,
            dataset_num_raters=self.dataset_num_raters,
            pattern=self.file_pattern,
            split=split,
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
        # The split should now be directly in fold_entry (renamed from id_test -> id, ood_test -> ood)
        key = split
        
        if key not in fold_entry:
            available = sorted(k for k in fold_entry.keys() if not k.startswith("_"))
            raise ValueError(
                f"Unknown split '{split}'. Available options: {', '.join(available)}"
            )
        values = fold_entry[key]
        if isinstance(values, np.ndarray):
            return values.tolist()
        return list(values)

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
    proc_dir: str,
    label_dir: str,
    pattern: str = "*.npy",
    subject_ids=None,
    num_raters: int = 4,
    rater_pattern: str = "{base_id}_{rater:02d}_mask.npy",
):
    """Load multi-rater samples with paths relative to proc_dir.
    
    With the new standardized format:
    - subject_ids contain full paths relative to proc_dir (e.g., "images/file.npy" or "augmented/ood_noise/images/file.npy")
    - Labels are always in label_dir with the standard rater pattern
    """
    samples = []
    
    if subject_ids is None:
        subject_ids = []
    
    for image_relpath in subject_ids:
        image_relpath_str = str(image_relpath).strip()
        if not image_relpath_str:
            continue
        
        # Full path to the image file
        image_path = os.path.join(proc_dir, image_relpath_str)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Expected image file not found: {image_path}")
        
        # Extract base_id from the filename for constructing label paths
        image_filename = os.path.basename(image_relpath_str)
        base_id = os.path.splitext(image_filename)[0]
        
        # Construct label paths
        label_paths = []
        for rater in range(num_raters):
            label_name = rater_pattern.format(
                base_id=base_id,
                filename=image_filename,
                rater=rater,
            )
            label_path = os.path.join(label_dir, label_name)
            label_paths.append(label_path)
        
        # Verify all labels exist
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
    "dataset_name/images/file.npy" or "dataset_name/augmented/ood_noise/images/file.npy"
    and different datasets have different numbers of raters.
    
    Parameters:
    -----------
    parent_dir : str
        Parent directory containing source datasets (e.g., /path/to/values_datasets)
    subject_ids : list, optional
        List of subject IDs with full paths relative to each dataset (e.g., ["chaksu64/images/file.npy", ...])
    dataset_num_raters : dict
        Mapping of dataset name -> number of raters
    pattern : str
        File pattern to match (default: "*.npy")
    rater_pattern : str
        Pattern for rater mask filenames
    split : str, optional
        The split name (for informational purposes)
    
    Returns:
    --------
    list of dicts with keys: image_path, label_paths, image_id, dataset_prefix, num_raters
    """
    samples = []
    
    if subject_ids is None:
        subject_ids = []
    
    for subject_id in subject_ids:
        subject_id_str = str(subject_id).strip()
        if not subject_id_str:
            continue
        
        # Parse the prefixed path: "dataset_name/path/to/image.npy"
        if "/" not in subject_id_str:
            continue
        
        parts = subject_id_str.split("/", 1)
        dataset_name = parts[0]
        image_relpath = parts[1]  # e.g., "images/file.npy" or "augmented/ood_noise/images/file.npy"
        
        if dataset_name not in dataset_num_raters:
            continue
        
        num_raters = dataset_num_raters[dataset_name]
        dataset_root = os.path.join(parent_dir, dataset_name)
        
        # Full path to the image
        image_path = os.path.join(dataset_root, "preprocessed", image_relpath)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Expected image file not found: {image_path}")
        
        # Extract base_id from filename for constructing label paths
        image_filename = os.path.basename(image_relpath)
        base_id = os.path.splitext(image_filename)[0]
        
        # Labels are always in the main labels directory
        label_dir = os.path.join(dataset_root, "preprocessed", "labels")
        
        # Build label paths
        label_paths = []
        for rater in range(num_raters):
            label_name = rater_pattern.format(
                base_id=base_id,
                filename=image_filename,
                rater=rater,
            )
            label_path = os.path.join(label_dir, label_name)
            label_paths.append(label_path)
        
        # Verify all labels exist
        missing = [path for path in label_paths if not os.path.exists(path)]
        if missing:
            missing_rel = ", ".join(os.path.basename(p) for p in missing)
            raise FileNotFoundError(
                f"Missing expected rater masks [{missing_rel}] for image '{image_filename}' in dataset '{dataset_name}'"
            )
        
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
