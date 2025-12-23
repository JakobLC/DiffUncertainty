#!/usr/bin/env python3
"""Generate LIDC-2D patient-level splits and offline OOD augmentations."""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from vis_ood_aug import (
    AugmentationSpec,
    DEFAULT_ROOT,
    apply_spec,
    build_augmentation_specs,
    build_shared_parser,
    finalize_shared_args,
    read_metadata,
)

SCHEMA_NAME = "lidc_patient_aug_cv_v1"
OOD_SPLITS = ("ood_noise", "ood_blur", "ood_contrast", "ood_jpeg")

def parse_args() -> argparse.Namespace:
    parser = build_shared_parser(__doc__, default_root=DEFAULT_ROOT, default_seed=7)
    parser.add_argument(
        "--split-name",
        type=str,
        default="ood_aug",
        help="Subfolder inside base-dir/splits/ where the pickle will be stored.",
    )
    parser.add_argument(
        "--cycle-name",
        type=str,
        default="firstCycle",
        help="Cycle name (e.g., firstCycle) that mirrors existing split conventions.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds stored inside the pickle file.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of patients assigned to the held-out test split (0-1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-create augmented files even if they already exist.",
    )
    parser.add_argument(
        "--only-splits",
        action="store_true",
        help="Skip augmentation generation and only emit the splits pickle.",
    )
    args = parser.parse_args()
    args = finalize_shared_args(args)
    if not (0.0 < args.test_ratio < 1.0):
        parser.error("--test-ratio must be within (0, 1).")
    if args.num_splits < 2:
        parser.error("--num-splits must be at least 2.")
    return args

def split_patients(
    patient_to_images: Dict[str, List[str]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    patients = sorted(patient_to_images.keys())
    if len(patients) < 2:
        raise ValueError("Need at least two patients to create train/val/test splits.")
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_total = len(patients)
    n_test = max(1, int(round(n_total * test_ratio)))
    if n_test >= n_total:
        n_test = n_total - 1
    test_patients = set(patients[:n_test])
    remaining_patients = [p for p in patients if p not in test_patients]
    if not remaining_patients:
        raise ValueError("Test ratio leaves no patients for train/val splits.")

    def _collect(selected: Sequence[str]) -> List[str]:
        images: List[str] = []
        for patient in selected:
            images.extend(patient_to_images[patient])
        return images

    test_images = sorted(_collect(test_patients))
    pool_images = sorted(_collect(remaining_patients))
    if len(pool_images) < 2:
        raise ValueError("Not enough remaining samples to build validation folds.")
    return test_images, pool_images


def build_train_val_pairs(
    samples: List[str],
    num_splits: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    if num_splits > len(samples):
        raise ValueError(
            "Cannot create more splits than available samples after removing the test patients."
        )
    rng = np.random.default_rng(seed)
    samples_array = np.array(samples)
    indices = np.arange(len(samples_array))
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_splits)
    train_val_pairs: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(num_splits):
        val_idx = chunks[fold_idx]
        train_idx = np.concatenate([chunks[i] for i in range(num_splits) if i != fold_idx])
        train_ids = samples_array[train_idx].tolist()
        val_ids = samples_array[val_idx].tolist()
        train_val_pairs.append((train_ids, val_ids))
    return train_val_pairs


def build_fold_entries(
    train_val_pairs: Sequence[Tuple[List[str], List[str]]],
    test_images: List[str],
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    test_array = np.array(sorted(test_images), dtype=object)
    folds: List[Dict[str, np.ndarray]] = []
    for fold_id, (train_ids, val_ids) in enumerate(train_val_pairs):
        fold_entry: Dict[str, np.ndarray] = {
            "train": np.array(train_ids, dtype=object),
            "val": np.array(val_ids, dtype=object),
            "id_test": test_array,
            "id_unlabeled_pool": np.array([], dtype=object),
            "ood_unlabeled_pool": np.array([], dtype=object),
            "_meta": {
                "schema": SCHEMA_NAME,
                "seed": seed,
                "fold_id": fold_id,
                "ood_variants": list(OOD_SPLITS),
            },
        }
        for split_name in OOD_SPLITS:
            fold_entry[split_name] = test_array
        folds.append(fold_entry)
    return folds


def save_splits(
    base_dir: Path,
    split_name: str,
    cycle_name: str,
    fold_entries: Sequence[Dict[str, np.ndarray]],
) -> Path:
    split_path = base_dir / "splits" / split_name / cycle_name / "splits.pkl"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("wb") as handle:
        pickle.dump(list(fold_entries), handle)
    print(
        f"Saved {len(fold_entries)} fold(s) to {split_path} for cycle '{cycle_name}' (schema={SCHEMA_NAME})."
    )
    return split_path


def generate_augmentations(
    image_names: Iterable[str],
    base_dir: Path,
    specs: Dict[str, AugmentationSpec],
    overwrite: bool,
) -> None:
    proc_dir = base_dir / "preprocessed"
    src_dir = proc_dir / "images"
    aug_root = proc_dir / "augmented"
    for split_name in specs.keys():
        (aug_root / split_name / "images").mkdir(parents=True, exist_ok=True)
    missing = [name for name in image_names if not (src_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Could not find {len(missing)} source images under {src_dir}. Example: {missing[:3]}"
        )
    for image_name in tqdm(list(image_names), desc="Augmenting test images"):
        src_path = src_dir / image_name
        image = np.load(src_path)
        for split_name, spec in specs.items():
            target = aug_root / split_name / "images" / image_name
            if target.exists() and not overwrite:
                continue
            augmented = apply_spec(image, spec)
            np.save(target, augmented)
    print(f"Stored augmented files under {aug_root} for splits: {', '.join(specs.keys())}.")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    patient_map = read_metadata(args.metadata_csv)
    test_images, pool_images = split_patients(patient_map, args.test_ratio, args.seed)
    train_val_pairs = build_train_val_pairs(pool_images, args.num_splits, args.seed)
    fold_entries = build_fold_entries(train_val_pairs, test_images, args.seed)
    save_splits(args.base_dir, args.split_name, args.cycle_name, fold_entries)

    if args.only_splits:
        return

    specs = build_augmentation_specs(args)
    if not test_images:
        raise RuntimeError("Empty test split; cannot create OOD augmentations.")
    generate_augmentations(test_images, args.base_dir, specs, args.overwrite)


if __name__ == "__main__":
    main()
