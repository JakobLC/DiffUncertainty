"""Preprocess the NPC dataset into the lidc_2d-style layout.

Steps implemented here:
1. Load each 3D volume from H5 files (t1, t1c, t2 channels + 4 label masks).
2. Normalize each channel independently per-volume to [0, 1] based on min/max.
3. Extract 2D slices along the first axis of the 3D volume.
4. For each 2D slice:
   - Perform the largest central square crop (e.g., 130x170 → 130x130).
   - Resize both image and masks to a fixed size (default 128x128).
   - Save as .npy files with metadata tracking.

The output structure mirrors ``values_datasets/lidc_2d``:

<save_path>/
	images/
	labels/
	metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SplitSpec:
	"""Configuration describing a single split (train/val)."""

	name: str
	split_dir: Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Crop and save the NPC dataset into a lidc_2d style format",
	)
	parser.add_argument(
		"--train-dir",
		type=Path,
		default=Path("/home/jloch/Desktop/diff/luzern/values_datasets/npc/MMIS2024TASK1/training"),
		help="Directory holding training 3D volumes",
	)
	parser.add_argument(
		"--val-dir",
		type=Path,
		default=Path("/home/jloch/Desktop/diff/luzern/values_datasets/npc/MMIS2024TASK1/validation"),
		help="Directory holding validation 3D volumes",
	)
	parser.add_argument(
		"--save-path",
		type=str,
		default="/home/jloch/Desktop/diff/luzern/values_datasets/npc{image_size}/preprocessed",
		help="Where to store the processed dataset",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=128,
		help="Final square size for saved crops",
	)
	parser.add_argument(
		"--save-empty",
		action="store_true",
		default=False,
		help="Skip saving 2D slices with no tumor (all labels 0)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow writing into a non-empty save directory",
	)
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Skip samples whose outputs already exist instead of overwriting",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Increase logging verbosity",
	)
	return parser.parse_args()


def configure_logging(verbose: bool) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
	)


def get_largest_square_crop_size(height: int, width: int) -> int:
	"""Get the largest square size that fits in a height x width image."""
	return min(height, width)


def crop_largest_square(array: np.ndarray, crop_size: int) -> np.ndarray:
	"""Crop the largest square from the center of a 2D or 3D array.
	
	For 2D: array.shape = (H, W, ...) → crop to (crop_size, crop_size, ...)
	For 3D volumes with slices on axis 0: array.shape = (H, W)
	"""
	if array.ndim == 2:
		height, width = array.shape
	else:
		raise ValueError(f"Expected 2D array, got shape {array.shape}")
	
	if crop_size > height or crop_size > width:
		raise ValueError(
			f"Crop size {crop_size} exceeds array dimensions {height}x{width}"
		)
	
	center_y = height // 2
	center_x = width // 2
	half = crop_size // 2
	
	y_start = center_y - half
	x_start = center_x - half
	y_end = y_start + crop_size
	x_end = x_start + crop_size
	
	return array[y_start:y_end, x_start:x_end]


def resize_array(arr: np.ndarray, size: int, order: str) -> np.ndarray:
	"""Resize a 2D or 3D array to size x size using scipy.ndimage.zoom.
	
	Args:
		arr: 2D or 3D array (for 3D, channels are on axis 2)
		size: target square dimension
		order: 'bilinear' or 'nearest' (maps to scipy order 1 or 0)
	
	Works directly with floats, no conversion needed.
	"""
	scipy_order = 1 if order == "bilinear" else 0
	current_height, current_width = arr.shape[:2]
	zoom_factor_h = size / current_height
	zoom_factor_w = size / current_width
	
	if arr.ndim == 2:
		zoom_factors = (zoom_factor_h, zoom_factor_w)
	elif arr.ndim == 3:
		# Keep channels unchanged
		zoom_factors = (zoom_factor_h, zoom_factor_w, 1)
	else:
		raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
	
	resized = ndimage.zoom(arr, zoom_factors, order=scipy_order)
	return resized


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
	images_dir = root / "images"
	labels_dir = root / "labels"
	images_dir.mkdir(parents=True, exist_ok=True)
	labels_dir.mkdir(parents=True, exist_ok=True)
	return images_dir, labels_dir


def load_h5_volume(h5_path: Path) -> Dict[str, np.ndarray]:
	"""Load all data from an H5 file."""
	data = {}
	with h5py.File(h5_path, 'r') as f:
		for key in f.keys():
			data[key] = f[key][:]
	return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
	"""Normalize a 3D volume to [0, 1] based on min/max across entire volume."""
	v_min = volume.min()
	v_max = volume.max()
	if v_max == v_min:
		return np.zeros_like(volume, dtype=np.float32)
	normalized = (volume.astype(np.float32) - v_min) / (v_max - v_min)
	return normalized


def has_any_label(label_slices: List[np.ndarray]) -> bool:
	"""Check if any of the label slices contains foreground (non-zero)."""
	for label in label_slices:
		if label.any():
			return True
	return False


def process_volume(
	h5_path: Path,
	split_name: str,
	sample_idx: int,
	out_image_dir: Path,
	out_label_dir: Path,
	image_size: int,
	save_empty: bool,
	skip_existing: bool,
) -> Tuple[int, List[dict]]:
	"""Process a single 3D volume, extracting 2D slices.
	
	Returns:
		(next_sample_idx, list of metadata rows)
	"""
	try:
		data = load_h5_volume(h5_path)
	except Exception as exc:
		logging.error("Failed to load H5 file %s: %s", h5_path, exc)
		return sample_idx, []
	
	# Extract and validate 3D volumes
	try:
		t1 = data['t1']
		t1c = data['t1c']
		t2 = data['t2']
		label_a1 = data['label_a1']
		label_a2 = data['label_a2']
		label_a3 = data['label_a3']
		label_a4 = data['label_a4']
	except KeyError as exc:
		logging.error("Missing required key in %s: %s", h5_path, exc)
		return sample_idx, []
	
	# Verify all volumes have the same shape
	expected_shape = t1.shape
	for name, vol in [('t1c', t1c), ('t2', t2), ('label_a1', label_a1),
	                    ('label_a2', label_a2), ('label_a3', label_a3), ('label_a4', label_a4)]:
		if vol.shape != expected_shape:
			logging.error(
				"Shape mismatch in %s for key %s: expected %s, got %s",
				h5_path, name, expected_shape, vol.shape
			)
			return sample_idx, []
	
	# Normalize each channel independently
	t1_norm = normalize_volume(t1)
	t1c_norm = normalize_volume(t1c)
	t2_norm = normalize_volume(t2)
	
	# Determine crop size from the 2D slice dimensions (second and third axes)
	_, height, width = expected_shape
	crop_size = get_largest_square_crop_size(height, width)
	
	rows: List[dict] = []
	current_sample_idx = sample_idx
	num_slices = expected_shape[0]
	
	for slice_idx in range(num_slices):
		# Extract 2D slices along first axis
		image_slice_t1 = t1_norm[slice_idx]
		image_slice_t1c = t1c_norm[slice_idx]
		image_slice_t2 = t2_norm[slice_idx]
		
		label_a1_slice = label_a1[slice_idx]
		label_a2_slice = label_a2[slice_idx]
		label_a3_slice = label_a3[slice_idx]
		label_a4_slice = label_a4[slice_idx]
		
		# Check if we should skip empty slices
		label_slices = [label_a1_slice, label_a2_slice, label_a3_slice, label_a4_slice]
		if not save_empty and not has_any_label(label_slices):
			logging.debug("Skipping empty slice %d from %s", slice_idx, h5_path.name)
			continue
		
		# Build sample ID
		sample_id = f"{split_name}{h5_path.stem[7:]}_slice{slice_idx:03d}"
		image_file = f"{sample_id}.npy"
		label_files = [
			f"{sample_id}_{i:02d}_mask.npy" for i in range(4)
		]
		
		if skip_existing:
			targets = [out_image_dir / image_file] + [
				out_label_dir / fname for fname in label_files
			]
			if all(path.exists() for path in targets):
				logging.debug("Skipping %s because outputs already exist", sample_id)
				current_sample_idx += 1
				continue
			if any(path.exists() for path in targets):
				logging.warning(
					"Partial outputs exist for %s; skipping to avoid overwrite",
					sample_id,
				)
				current_sample_idx += 1
				continue
		
		# Crop largest square from center
		image_t1_cropped = crop_largest_square(image_slice_t1, crop_size)
		image_t1c_cropped = crop_largest_square(image_slice_t1c, crop_size)
		image_t2_cropped = crop_largest_square(image_slice_t2, crop_size)
		
		# Stack into 3-channel image and resize
		image_3ch = np.stack(
			[image_t1_cropped, image_t1c_cropped, image_t2_cropped],
			axis=2
		)
		# Resize while preserving float [0, 1] range
		image_resized = resize_array(image_3ch, image_size, order="bilinear")
		
		# Process and save labels
		label_resized_list = []
		for label_idx, (label_slice, label_file) in enumerate(
			zip(label_slices, label_files)
		):
			label_cropped = crop_largest_square(label_slice.astype(np.uint8), crop_size)
			label_resized = resize_array(label_cropped, image_size, order="nearest")
			label_resized_list.append(label_resized)
		
		# Check if any label has positive values after resizing
		if not save_empty and not has_any_label(label_resized_list):
			logging.debug("Skipping slice with no labels after resizing: %s", sample_id)
			continue
		
		# Save image as float32 to preserve normalized values
		np.save(out_image_dir / image_file, image_resized.astype(np.float32))
		
		# Save labels
		for label_resized, label_file in zip(label_resized_list, label_files):
			np.save(out_label_dir / label_file, label_resized.astype(np.uint8))
		
		rows.append({
			"sample_id": sample_id,
			"split": split_name,
			"source_volume": str(h5_path),
			"slice_index": slice_idx,
			"image_file": image_file,
			"label_files": ",".join(label_files),
			"crop_size": crop_size,
		})
		
		current_sample_idx += 1
	
	return current_sample_idx, rows


def process_split(
	split: SplitSpec,
	split_prefix: str,
	start_index: int,
	image_size: int,
	images_dir: Path,
	labels_dir: Path,
	save_empty: bool,
	skip_existing: bool,
) -> Tuple[int, List[dict]]:
	"""Process all volumes in a split."""
	if not split.split_dir.exists():
		raise FileNotFoundError(
			f"Split directory missing: {split.split_dir}"
		)
	
	# Find all H5 files in the split directory
	h5_files = sorted(split.split_dir.glob("*.h5"))
	if not h5_files:
		logging.warning("No H5 files found in %s", split.split_dir)
		return start_index, []
	
	logging.info("Found %d H5 files in %s", len(h5_files), split.split_dir)
	
	next_index = start_index
	all_rows: List[dict] = []
	
	for h5_path in tqdm(h5_files, desc=split.name, unit="volume"):
		_, rows = process_volume(
			h5_path=h5_path,
			split_name=split_prefix,
			sample_idx=next_index,
			out_image_dir=images_dir,
			out_label_dir=labels_dir,
			image_size=image_size,
			save_empty=save_empty,
			skip_existing=skip_existing,
		)
		next_index += len(rows)
		all_rows.extend(rows)
	
	return next_index, all_rows


def write_metadata(save_path: Path, rows: Sequence[dict]) -> None:
	"""Write metadata.csv file."""
	if not rows:
		logging.warning("No samples were processed, metadata will not be written")
		return
	metadata_path = save_path / "metadata.csv"
	with metadata_path.open("w", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=sorted(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)
	logging.info("Metadata saved to %s", metadata_path)


def main() -> None:
	args = parse_args()
	configure_logging(args.verbose)
	
	save_path = Path(args.save_path.format(image_size=args.image_size)).expanduser().resolve()
	if save_path.exists() and not args.overwrite:
		non_empty = any(save_path.glob("*"))
		if non_empty:
			logging.error(
				"Save path %s already has content. Use --overwrite to continue.",
				save_path,
			)
			sys.exit(1)
	save_path.mkdir(parents=True, exist_ok=True)
	images_dir, labels_dir = ensure_output_dirs(save_path)
	
	splits = [
		SplitSpec(
			name="train",
			split_dir=args.train_dir.expanduser().resolve(),
		),
		SplitSpec(
			name="val",
			split_dir=args.val_dir.expanduser().resolve(),
		),
	]
	
	sample_index = 0
	metadata_rows: List[dict] = []
	
	for split in splits:
		split_prefix = split.name  # 'train' or 'val'
		sample_index, rows = process_split(
			split=split,
			split_prefix=split_prefix,
			start_index=sample_index,
			image_size=args.image_size,
			images_dir=images_dir,
			labels_dir=labels_dir,
			save_empty=args.save_empty,
			skip_existing=args.skip_existing,
		)
		metadata_rows.extend(rows)
	
	write_metadata(save_path, metadata_rows)
	logging.info("Processing complete. Processed %d slices.", len(metadata_rows))


if __name__ == "__main__":
	main()
