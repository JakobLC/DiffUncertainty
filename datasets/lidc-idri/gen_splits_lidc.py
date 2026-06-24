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

OOD_SPLITS = ("ood_noise", "ood_blur", "ood_contrast", "ood_jpeg")

def parse_args() -> argparse.Namespace:
    parser = build_shared_parser(__doc__, default_root=DEFAULT_ROOT, default_seed=7)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional dataset name to overwrite the inhereted base_dir last name (e.g., 'origlidc128').",
    )
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
        default=4,
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
        "--overwrite-splits",
        action="store_true",
        help="Overwrite splits pickle file if it already exists. If False, show comparison of old vs new.",
    )
    parser.add_argument(
        "--no-only-splits",
        action="store_false",
        dest="only_splits",
        help="If set, only create the splits pickle file and skip generating augmented OOD images.",
    )
    args = parser.parse_args()
    args = finalize_shared_args(args)
    if args.dataset_name:
        args.base_dir = args.base_dir.parent / args.dataset_name
    if not (0.0 < args.test_ratio < 1.0):
        parser.error("--test-ratio must be within (0, 1).")
    if args.num_splits < 2:
        parser.error("--num-splits must be at least 2.")
    return args

def _collect_patient_images(
    patient_to_images: Dict[str, List[str]],
    patient_ids: Sequence[str],
) -> List[str]:
    images: List[str] = []
    for patient in patient_ids:
        if patient not in patient_to_images:
            raise KeyError(f"Patient '{patient}' missing from metadata map.")
        images.extend(patient_to_images[patient])
    # Add images/ prefix to paths (relative to preprocessed/)
    return sorted([f"images/{img}" for img in images])

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

    test_patient_ids = sorted(test_patients)
    remaining_sorted = sorted(remaining_patients)
    if len(remaining_sorted) < 2:
        raise ValueError("Not enough patients left to build validation folds.")
    return test_patient_ids, remaining_sorted


def build_train_val_pairs(
    patient_ids: List[str],
    patient_to_images: Dict[str, List[str]],
    num_splits: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    if num_splits > len(patient_ids):
        raise ValueError(
            "Cannot create more splits than available patients after removing the test patients."
        )
    rng = np.random.default_rng(seed)
    patients_array = np.array(patient_ids)
    indices = np.arange(len(patients_array))
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_splits)
    train_val_pairs: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(num_splits):
        val_idx = chunks[fold_idx]
        other_chunks = [chunks[i] for i in range(num_splits) if i != fold_idx]
        if not other_chunks:
            raise ValueError("Training fold would be empty; increase number of patients or reduce splits.")
        train_idx = np.concatenate(other_chunks)
        train_patients = patients_array[train_idx].tolist()
        val_patients = patients_array[val_idx].tolist()
        train_ids = _collect_patient_images(patient_to_images, train_patients)
        val_ids = _collect_patient_images(patient_to_images, val_patients)
        train_val_pairs.append((train_ids, val_ids))
    return train_val_pairs


def build_fold_entries(
    train_val_pairs: Sequence[Tuple[List[str], List[str]]],
    test_images: List[str],
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    id_array = np.array(sorted(test_images), dtype=object)
    folds: List[Dict[str, np.ndarray]] = []
    for fold_id, (train_ids, val_ids) in enumerate(train_val_pairs):
        fold_entry: Dict[str, np.ndarray] = {
            "train": np.array(train_ids, dtype=object),
            "val": np.array(val_ids, dtype=object),
            "id": id_array,
        }
        # Add augmented OOD splits - prefix with augmented/[split_name]/
        for split_name in OOD_SPLITS:
            augmented_paths = np.array(
                [f"augmented/{split_name}/{path}" for path in id_array],
                dtype=object,
            )
            fold_entry[split_name] = augmented_paths
        folds.append(fold_entry)
    return folds


def _format_split_preview(fold_entries: Sequence[Dict[str, np.ndarray]]) -> str:
    """Format first/last 2 entries from each key in the first fold for comparison."""
    if not fold_entries:
        return "<empty>"
    fold = fold_entries[0]
    lines = []
    for key in sorted(fold.keys()):
        if key.startswith("_"):
            continue
        values = fold[key]
        if len(values) == 0:
            lines.append(f"{key}: []")
        elif len(values) <= 4:
            lines.append(f"{key}: {list(values)}")
        else:
            first_two = list(values[:2])
            last_two = list(values[-2:])
            lines.append(f"{key}: [{first_two[0]}, {first_two[1]}, {last_two[0]}, {last_two[1]}]")
    return "\n  ".join(lines)


def save_splits(
    base_dir: Path,
    split_name: str,
    cycle_name: str,
    fold_entries: Sequence[Dict[str, np.ndarray]],
    overwrite_splits: bool = False,
) -> Path:
    split_path = base_dir / "splits" / split_name / cycle_name / "splits.pkl"
    
    # Check if file exists and overwrite is False
    if split_path.exists() and not overwrite_splits:
        print(f"\n[SKIPPED] Splits file already exists: {split_path}")
        print("  To proceed, rerun with --overwrite-splits\n")
        
        # Load old splits
        with split_path.open("rb") as handle:
            old_splits = pickle.load(handle)
        
        print("OLD splits (first 2 and last 2 entries per key):")
        print(f"  {_format_split_preview(old_splits)}\n")
        
        print("NEW splits (first 2 and last 2 entries per key):")
        print(f"  {_format_split_preview(fold_entries)}\n")
        
        return split_path
    
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("wb") as handle:
        pickle.dump(list(fold_entries), handle)
    print(
        f"Saved {len(fold_entries)} fold(s) to {split_path} for cycle '{cycle_name}'."
    )
    return split_path


def generate_augmentations(
    image_paths: Iterable[str],
    base_dir: Path,
    specs: Dict[str, AugmentationSpec],
    overwrite: bool,
) -> None:
    proc_dir = base_dir / "preprocessed"
    src_dir = proc_dir / "images"
    aug_root = proc_dir / "augmented"
    for split_name in specs.keys():
        (aug_root / split_name / "images").mkdir(parents=True, exist_ok=True)
    
    # Strip "images/" prefix from paths for loading
    image_names = [p.replace("images/", "") for p in image_paths]
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
    test_patients, pool_patients = split_patients(patient_map, args.test_ratio, args.seed)
    test_images = _collect_patient_images(patient_map, test_patients)
    train_val_pairs = build_train_val_pairs(pool_patients, patient_map, args.num_splits, args.seed)
    fold_entries = build_fold_entries(train_val_pairs, test_images, args.seed)
    save_splits(args.base_dir, args.split_name, args.cycle_name, fold_entries, args.overwrite_splits)

    if args.only_splits:
        return

    specs = build_augmentation_specs(args)
    if not test_images:
        raise RuntimeError("Empty test split; cannot create OOD augmentations.")
    generate_augmentations(test_images, args.base_dir, specs, args.overwrite)


if __name__ == "__main__":
    main()
