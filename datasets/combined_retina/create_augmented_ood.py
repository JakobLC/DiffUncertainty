"""Create augmented OOD samples for combined retina datasets.

This script generates augmented out-of-distribution samples by applying
three custom augmentations (FOV, Flash, Blur) to id_test images:
- ood_fov: Field of view circular mask
- ood_flash: Flash artifact simulation
- ood_blur: Gaussian blur

The augmented images are saved to individual dataset folders:
  values_datasets/chaksu{size}/preprocessed/augmented/ood_fov/
  values_datasets/chaksu{size}/preprocessed/augmented/ood_flash/
  values_datasets/chaksu{size}/preprocessed/augmented/ood_blur/
  (similarly for refuge{size}, riga{size}, etc.)

Augmentation hyperparameters are hardcoded based on:
  values/datasets/combined_retina/visualize_retina_ood_augs.py

Blur kernel sizes are scaled proportionally with image resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import albumentations as A

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uncertainty_modeling.data.lidc2d_dataset import MultiRater2DDataset
from uncertainty_modeling.augmentations import FieldOfViewCircularMask, FlashArtifact

DATASETS_ROOT = Path("/home/jloch/Desktop/diff/luzern/values_datasets")


# ============================================================================
# AUGMENTATION PARAMETERS (from visualize_retina_ood_augs.py)
# ============================================================================

# These are the hardcoded parameters from the visualization script
# Adjust these if you change the parameters in visualize_retina_ood_augs.py

def get_augmentation_params(image_size: int) -> Dict:
    """Get augmentation parameters for a given image size.
    
    Blur kernel sizes are scaled proportionally with resolution.
    For 128x128: [3.0, 7.0]
    For 64x64:   [1.5, 3.5]
    """
    blur_base_128 = (3.0, 7.0)
    blur_size_64 = (1.5, 3.5)
    
    # Linear interpolation of blur kernel size
    if image_size == 128:
        blur_sigma = blur_base_128
    elif image_size == 64:
        blur_sigma = blur_size_64
    else:
        # Interpolate for other sizes
        scale = image_size / 128.0
        blur_sigma = (blur_base_128[0] * scale, blur_base_128[1] * scale)
    
    return {
        "fov": {
            "radius": [1.0, 1.5],
            "edge_blur": 0.01,  # scalar, not a list
            "circle_dist": [0.1, 0.3],
        },
        "flash": {
            "additive": False,
            "additive_range": (-0.3, 1.0),
            "multiplicative_range": (0.2, 1.5),
            "size": 0.5,
            "sharpness": 4,
            "eccentricity": [0.2, 0.5],
            "center_shift": (0, 0.3),
        },
        "blur": {
            "sigma_limit": blur_sigma,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented OOD samples for combined retina datasets",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image resolution (default: 128, e.g., 64, 128)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of combined dataset (default: retina{image_size})",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Number of augmentations per image (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )


def load_splits(dataset_root: Path, fold_id: int = 0) -> dict:
    """Load splits from the split file.
    
    Handles numpy version compatibility issues by patching numpy._core if needed.
    """
    splits_path = dataset_root / "splits" / "random" / "firstCycle" / "splits.pkl"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    try:
        with splits_path.open("rb") as f:
            splits = pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            # Handle numpy version compatibility: patch numpy._core
            import sys
            sys.modules["numpy._core"] = np
            with splits_path.open("rb") as f:
                splits = pickle.load(f)
        else:
            raise
    
    if fold_id >= len(splits):
        raise IndexError(f"Fold {fold_id} not found. Available folds: {len(splits)}")
    
    return splits[fold_id]


def load_image(image_path: Path) -> np.ndarray:
    """Load a single image from disk."""
    img = np.load(image_path)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img.astype(np.uint8)


def ensure_augmented_dirs(dataset_root: Path) -> Dict[str, Path]:
    """Create augmented subdirectories and return paths."""
    augmented_root = dataset_root / "preprocessed" / "augmented"
    
    aug_types = ["ood_fov", "ood_flash", "ood_blur"]
    aug_dirs = {}
    
    for aug_type in aug_types:
        aug_dir = augmented_root / aug_type / "images"
        aug_dir.mkdir(parents=True, exist_ok=True)
        aug_dirs[aug_type] = aug_dir
    
    return aug_dirs


def create_augmentation_transforms(params: Dict) -> Dict:
    """Create albumentations transforms from parameters."""
    return {
        "fov": FieldOfViewCircularMask(
            radius=params["fov"]["radius"],
            edge_blur=params["fov"]["edge_blur"],
            circle_dist=params["fov"]["circle_dist"],
            always_apply=True,
            p=1.0,
        ),
        "flash": FlashArtifact(
            additive=params["flash"]["additive"],
            additive_range=params["flash"]["additive_range"],
            multiplicative_range=params["flash"]["multiplicative_range"],
            size=params["flash"]["size"],
            sharpness=params["flash"]["sharpness"],
            eccentricity=params["flash"]["eccentricity"],
            center_shift=params["flash"]["center_shift"],
            always_apply=True,
            p=1.0,
        ),
        "blur": A.GaussianBlur(
            sigma_limit=params["blur"]["sigma_limit"],
            always_apply=True,
            p=1.0,
        ),
    }


def cleanup_existing_augmented_dirs(splits: dict) -> None:
    """Remove existing augmented directories to start fresh."""
    id_test_paths = splits["id_test"]
    
    # Collect unique dataset prefixes
    datasets_to_clean = set()
    for rel_path in id_test_paths:
        parts = Path(rel_path).parts
        if len(parts) >= 1:
            dataset_prefix = parts[0]
            datasets_to_clean.add(dataset_prefix)
    
    # Remove augmented folders from each dataset
    for dataset_prefix in datasets_to_clean:
        dataset_root = DATASETS_ROOT / dataset_prefix
        augmented_dir = dataset_root / "preprocessed" / "augmented"
        
        if augmented_dir.exists():
            logging.info(f"Removing existing augmented folder: {augmented_dir}")
            shutil.rmtree(augmented_dir)


def process_dataset(
    splits: dict,
    transforms: Dict,
    num_augmentations: int = 1,
) -> None:
    """Process all id_test images and create augmented versions.
    
    For each image path in splits, extract the dataset prefix (e.g., "chaksu128")
    and save augmented images to that dataset's folder.
    """
    id_test_paths = splits["id_test"]
    
    total_images = len(id_test_paths)
    processed = 0
    skipped = 0
    
    # Group paths by dataset to create directories once
    paths_by_dataset = {}
    aug_dirs_by_dataset = {}
    
    for rel_path in id_test_paths:
        # Parse path: "chaksu128/t_000000.npy" -> dataset="chaksu128", filename="t_000000.npy"
        parts = Path(rel_path).parts
        if len(parts) >= 2:
            dataset_prefix = parts[0]  # e.g., "chaksu128"
            filename = parts[-1]        # e.g., "t_000000.npy"
        else:
            logging.warning(f"Unexpected path format: {rel_path}")
            skipped += 1
            continue
        
        if dataset_prefix not in paths_by_dataset:
            paths_by_dataset[dataset_prefix] = []
            # Create augmented directories for this dataset
            dataset_root = DATASETS_ROOT / dataset_prefix
            aug_dirs = ensure_augmented_dirs(dataset_root)
            aug_dirs_by_dataset[dataset_prefix] = aug_dirs
        
        paths_by_dataset[dataset_prefix].append((rel_path, filename))
    
    # Process images by dataset
    for dataset_prefix, paths_list in paths_by_dataset.items():
        dataset_root = DATASETS_ROOT / dataset_prefix
        image_dir = dataset_root / "preprocessed" / "images"
        aug_dirs = aug_dirs_by_dataset[dataset_prefix]
        
        logging.info(f"Processing {dataset_prefix}: {len(paths_list)} images")
        
        for rel_path, filename in paths_list:
            image_path = image_dir / filename
            
            if not image_path.exists():
                logging.warning(f"Image not found: {image_path}")
                skipped += 1
                continue
            
            try:
                # Load image
                img = load_image(image_path)
                
                # Generate augmentations
                for aug_idx in range(num_augmentations):
                    # FOV augmentation
                    fov_img = transforms["fov"](image=img)["image"]
                    fov_path = aug_dirs["ood_fov"] / filename
                    np.save(fov_path, fov_img)
                    
                    # Flash augmentation
                    flash_img = transforms["flash"](image=img)["image"]
                    flash_path = aug_dirs["ood_flash"] / filename
                    np.save(flash_path, flash_img)
                    
                    # Blur augmentation
                    blur_img = transforms["blur"](image=img)["image"]
                    blur_path = aug_dirs["ood_blur"] / filename
                    np.save(blur_path, blur_img)
                
                processed += 1
                if (processed + skipped) % 50 == 0:
                    logging.info(f"  Processed {processed + skipped}/{total_images} images overall...")
            
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                skipped += 1
    
    logging.info(
        f"Completed: {processed} processed, {skipped} skipped out of {total_images} total"
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    
    # Determine dataset name
    dataset_name = args.dataset_name or f"retina{args.image_size}"
    dataset_root = (DATASETS_ROOT / dataset_name).expanduser().resolve()
    
    logging.info(f"Processing combined retina dataset: {dataset_name}")
    
    # Validate dataset exists
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    
    # Load splits (splits file is in retina128, not in individual dataset folders)
    logging.info("Loading splits...")
    splits = load_splits(dataset_root)
    id_test_count = len(splits["id_test"])
    logging.info(f"Found {id_test_count} id_test images across all datasets")
    
    # Get augmentation parameters
    logging.info("Preparing augmentation parameters...")
    params = get_augmentation_params(args.image_size)
    
    # Create transforms
    logging.info("Creating augmentation transforms...")
    transforms = create_augmentation_transforms(params)
    
    # Clean up existing augmented directories
    logging.info("Cleaning up existing augmented directories...")
    cleanup_existing_augmented_dirs(splits)
    
    # Process dataset
    logging.info("Processing images...")
    process_dataset(
        splits,
        transforms,
        num_augmentations=args.num_augmentations,
    )
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
