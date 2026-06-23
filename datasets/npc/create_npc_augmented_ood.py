"""Create augmented OOD samples for NPC dataset using MONAI transforms.

This script generates augmented out-of-distribution samples by applying
three MONAI augmentations (Rician noise, Histogram shift, Gibbs noise) to id_test images:
- ood_noise: RandRicianNoise
- ood_hist: RandHistogramShift
- ood_gibbs: RandGibbsNoise

The augmented images are saved to individual dataset folders:
  values_datasets/npc{size}/preprocessed/augmented/ood_noise/
  values_datasets/npc{size}/preprocessed/augmented/ood_hist/
  values_datasets/npc{size}/preprocessed/augmented/ood_gibbs/

Augmentation hyperparameters are hardcoded based on:
  values/datasets/npc/visualize_npc_ood_augs_v2.py

Augmentations are applied per-channel but without normalization after applying transforms.
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
from monai.transforms import (
	RandRicianNoise,
	RandHistogramShift,
	RandGibbsNoise,
)

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASETS_ROOT = Path("/home/jloch/Desktop/diff/luzern/values_datasets")


# ============================================================================
# AUGMENTATION PARAMETERS (from visualize_npc_ood_augs_v2.py)
# ============================================================================

RICIAN_NOISE_PARAMS = {
	"std": 0.2,
	"prob": 1.0,
}

HISTOGRAM_SHIFT_PARAMS = {
	"num_control_points": 10,
	"prob": 1.0,
}

GIBBS_NOISE_PARAMS = {
	"alpha": (0.6, 0.7),
	"prob": 1.0,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create augmented OOD samples for NPC dataset",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=128,
		help="Image resolution (default: 128, e.g., 64, 128)",
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
	splits_path = dataset_root / "splits" / "ood_aug" / "firstCycle" / "splits.pkl"
	
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
	return img.astype(np.float32)


def ensure_augmented_dirs(dataset_root: Path) -> Dict[str, Path]:
	"""Create augmented subdirectories and return paths."""
	augmented_root = dataset_root / "preprocessed" / "augmented"
	
	aug_types = ["ood_noise", "ood_hist", "ood_gibbs"]
	aug_dirs = {}
	
	for aug_type in aug_types:
		aug_dir = augmented_root / aug_type / "images"
		aug_dir.mkdir(parents=True, exist_ok=True)
		aug_dirs[aug_type] = aug_dir
	
	return aug_dirs


def apply_transform_per_channel(img_chw: torch.Tensor, transform) -> torch.Tensor:
	"""Apply transform to each channel separately.
	
	Args:
		img_chw: Image tensor with shape (C, H, W)
		transform: MONAI transform to apply
	
	Returns:
		Transformed image with shape (C, H, W)
	"""
	channels = []
	for c in range(img_chw.shape[0]):
		channel = img_chw[c:c+1, :, :]  # (1, H, W)
		transformed = transform(channel)
		channels.append(transformed)
	
	result = torch.cat(channels, dim=0)  # (C, H, W)
	return result


def create_augmentation_transforms() -> Dict:
	"""Create MONAI transforms from parameters."""
	return {
		"noise": RandRicianNoise(**RICIAN_NOISE_PARAMS),
		"hist": RandHistogramShift(**HISTOGRAM_SHIFT_PARAMS),
		"gibbs": RandGibbsNoise(**GIBBS_NOISE_PARAMS),
	}


def cleanup_existing_augmented_dirs(dataset_root: Path) -> None:
	"""Remove existing augmented directories to start fresh."""
	augmented_dir = dataset_root / "preprocessed" / "augmented"
	
	if augmented_dir.exists():
		logging.info(f"Removing existing augmented folder: {augmented_dir}")
		shutil.rmtree(augmented_dir)


def process_dataset(
	dataset_root: Path,
	splits: dict,
	transforms: Dict,
	num_augmentations: int = 1,
) -> None:
	"""Process all id_test images and create augmented versions.
	
	For each image path in id_test split, apply augmentations and save to augmented folders.
	"""
	image_dir = dataset_root / "preprocessed" / "images"
	aug_dirs = ensure_augmented_dirs(dataset_root)
	
	id_test_paths = splits["id_test"]
	total_images = len(id_test_paths)
	processed = 0
	skipped = 0
	
	logging.info(f"Processing {total_images} id_test images")
	
	for rel_path in id_test_paths:
		# Parse path: "images/filename.npy"
		parts = Path(rel_path).parts
		if len(parts) >= 2:
			filename = parts[-1]
		else:
			logging.warning(f"Unexpected path format: {rel_path}")
			skipped += 1
			continue
		
		image_path = image_dir / filename
		
		if not image_path.exists():
			logging.warning(f"Image not found: {image_path}")
			skipped += 1
			continue
		
		try:
			# Load image and convert to torch tensor
			img = load_image(image_path)
			img_chw = torch.from_numpy(img).permute(2, 0, 1).float()  # (C, H, W)
			
			# Generate augmentations (without normalization)
			for aug_idx in range(num_augmentations):
				# Rician noise
				noise_result = apply_transform_per_channel(img_chw, transforms["noise"])
				noise_img = noise_result.permute(1, 2, 0).numpy().astype(np.float32)
				noise_path = aug_dirs["ood_noise"] / filename
				np.save(noise_path, noise_img)
				
				# Histogram shift
				hist_result = apply_transform_per_channel(img_chw, transforms["hist"])
				hist_img = hist_result.permute(1, 2, 0).numpy().astype(np.float32)
				hist_path = aug_dirs["ood_hist"] / filename
				np.save(hist_path, hist_img)
				
				# Gibbs noise
				gibbs_result = apply_transform_per_channel(img_chw, transforms["gibbs"])
				gibbs_img = gibbs_result.permute(1, 2, 0).numpy().astype(np.float32)
				gibbs_path = aug_dirs["ood_gibbs"] / filename
				np.save(gibbs_path, gibbs_img)
			
			processed += 1
			if (processed + skipped) % 50 == 0:
				logging.info(f"  Processed {processed + skipped}/{total_images} images...")
		
		except Exception as e:
			logging.error(f"Error processing {image_path}: {e}")
			skipped += 1
	
	logging.info(
		f"Completed: {processed} processed, {skipped} skipped out of {total_images} total"
	)


def main() -> None:
	args = parse_args()
	setup_logging(args.verbose)
	
	# Setup dataset paths
	dataset_name = f"npc{args.image_size}"
	dataset_root = (DATASETS_ROOT / dataset_name).expanduser().resolve()
	
	logging.info(f"Processing NPC dataset: {dataset_name}")
	
	# Validate dataset exists
	if not dataset_root.is_dir():
		raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
	
	# Load splits
	logging.info("Loading splits...")
	splits = load_splits(dataset_root)
	id_test_count = len(splits["id_test"])
	logging.info(f"Found {id_test_count} id_test images")
	
	# Create transforms
	logging.info("Creating augmentation transforms...")
	transforms = create_augmentation_transforms()
	
	# Clean up existing augmented directories
	logging.info("Cleaning up existing augmented directories...")
	cleanup_existing_augmented_dirs(dataset_root)
	
	# Process dataset
	logging.info("Processing images...")
	process_dataset(
		dataset_root,
		splits,
		transforms,
		num_augmentations=args.num_augmentations,
	)
	
	logging.info("Done!")


if __name__ == "__main__":
	main()
