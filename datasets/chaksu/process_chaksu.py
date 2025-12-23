"""Preprocess the Chaksu dataset into the lidc_2d-style layout.

Steps implemented here follow the user specification:
1. Load each fundus image, together with the five Disc/Cup annotations.
2. Clean every mask by keeping only the largest connected component.
3. Combine masks into a multi-class label (0 background, 1 Disc-only, 2 Cup).
4. Use the union of Disc masks to determine a crop centered on the optic disc.
5. Crop and resize both the image (bilinear) and masks (nearest) to a fixed size.
6. Save the arrays as .npy files and keep track of provenance in metadata.csv.

The output structure mirrors ``values_datasets/lidc_2d``:

<save_path>/
	images/
	labels/
	metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm.auto import tqdm

#DATA_TRAIN_PATH = Path("/data/chaksu/Train")
#DATA_TEST_PATH = Path("/data/chaksu/Test")
# The circle widths are used to derive crop sizes per machine.
CIRCLE_WIDTHS: Dict[str, int] = {"Bosch": 1440, "Forus": 1900, "Remidio": 2200}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
 

@dataclass(frozen=True)
class SplitSpec:
	"""Configuration describing a single image/label split."""

	name: str
	image_dir: Path
	label_dir: Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Crop and save the Chaksu dataset into a lidc_2d style format",
	)
	parser.add_argument(
		"--train-image-dir",
		type=Path,
		default=Path("/data/chaksu/Train/1.0_Original_Fundus_Images"),
		help="Directory holding train images grouped by machine",
	)
	parser.add_argument(
		"--test-image-dir",
		type=Path,
		default=Path("/data/chaksu/Test/1.0_Original_Fundus_Images"),
		help="Directory holding test images grouped by machine",
	)
	parser.add_argument(
		"--train-label-dir",
		type=Path,
		default=Path("/data/chaksu/Train/3.0_Doctors_Annotations_Binary_OD_OC"),
		help="Directory that stores Expert folders for the training split",
	)
	parser.add_argument(
		"--test-label-dir",
		type=Path,
		default=Path("/data/chaksu/Test/3.0_Doctors_Annotations_Binary_OD_OC"),
		help="Directory that stores Expert folders for the test split if available",
	)
	parser.add_argument(
		"--save-path",
		type=str,
		default="/home/jloch/Desktop/diff/luzern/values_datasets/chaksu{image_size}/preprocessed",
		help="Where to store the processed dataset",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=128,
		help="Final square size for saved crops",
	)
	parser.add_argument(
		"--rel-sidelength",
		type=float,
		default=0.30,
		help="Side length multiplier applied to circle_widths[machine]",
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


def list_experts(label_dir: Path) -> List[str]:
	experts = [p.name for p in sorted(label_dir.iterdir()) if p.is_dir()]
	return [exp for exp in experts if exp.lower().startswith("expert")]


def iter_machine_images(image_root: Path, machine: str) -> Iterable[Path]:
	machine_dir = image_root / machine
	if not machine_dir.is_dir():
		return []
	yield from (
		path
		for path in machine_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)


def find_mask_file(root: Path, stem: str) -> Path:
	exact = root / f"{stem}.png"
	if exact.exists():
		return exact
	matches = sorted(root.glob(f"{stem}.*"))
	if matches:
		return matches[0]
	raise FileNotFoundError(f"Missing mask for {stem} in {root}")


def load_mask(mask_root: Path, stem: str) -> np.ndarray:
	path = find_mask_file(mask_root, stem)
	mask = Image.open(path).convert("L")
	arr = np.array(mask) > 0
	return arr.astype(np.uint8)


def load_image(image_path: Path) -> np.ndarray:
	return np.array(Image.open(image_path).convert("RGB"))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
	binary = mask.astype(bool)
	if not binary.any():
		return binary
	labeled, num = ndimage.label(binary)
	if num <= 1:
		return binary
	counts = np.bincount(labeled.ravel())
	counts[0] = 0
	largest = counts.argmax()
	return labeled == largest


def build_label(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
	label = np.zeros_like(disc_mask, dtype=np.uint8)
	label[disc_mask] = 1
	label[cup_mask] = 2
	return label


def get_center_from_mask(mask: np.ndarray) -> Tuple[float, float]:
	coords = np.argwhere(mask)
	if coords.size == 0:
		raise ValueError("Cannot compute center from empty mask")
	y_min, x_min = coords.min(axis=0)
	y_max, x_max = coords.max(axis=0)
	center_y = (y_min + y_max) / 2.0
	center_x = (x_min + x_max) / 2.0
	return center_y, center_x


def crop_square(array: np.ndarray, center: Tuple[float, float], size: int) -> np.ndarray:
	if size <= 0:
		raise ValueError("Crop size must be positive")
	half = size / 2.0
	center_y, center_x = center
	top = int(round(center_y - half))
	left = int(round(center_x - half))
	bottom = top + size
	right = left + size

	pad_top = max(0, -top)
	pad_left = max(0, -left)
	pad_bottom = max(0, bottom - array.shape[0])
	pad_right = max(0, right - array.shape[1])

	if any((pad_top, pad_bottom, pad_left, pad_right)):
		pad_width = (
			(pad_top, pad_bottom),
			(pad_left, pad_right),
		)
		if array.ndim == 3:
			pad_width += ((0, 0),)
		array = np.pad(array, pad_width, mode="constant", constant_values=0)
		top += pad_top
		bottom += pad_top
		left += pad_left
		right += pad_left

	return array[top:bottom, left:right]


def resize_array(arr: np.ndarray, size: int, order: str) -> np.ndarray:
	data = arr.astype(np.uint8)
	pil_image = Image.fromarray(data)
	resample = Image.BILINEAR if order == "bilinear" else Image.NEAREST
	resized = pil_image.resize((size, size), resample=resample)
	return np.array(resized)


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
	images_dir = root / "images"
	labels_dir = root / "labels"
	images_dir.mkdir(parents=True, exist_ok=True)
	labels_dir.mkdir(parents=True, exist_ok=True)
	return images_dir, labels_dir


def process_sample(
	image_path: Path,
	machine: str,
	experts: Sequence[str],
	label_dir: Path,
	out_image_dir: Path,
	out_label_dir: Path,
	sample_id: str,
	image_file: str,
	label_files: Sequence[str],
	image_size: int,
	rel_sidelength: float,
	min_crop_size: int,
) -> Optional[dict]:
	try:
		image = load_image(image_path)
	except Exception as exc:  # noqa: BLE001 - we want to continue processing
		logging.error("Failed to load image %s: %s", image_path, exc)
		return None

	disc_masks: List[np.ndarray] = []
	label_arrays: List[np.ndarray] = []
	stem = image_path.stem

	for expert in experts:
		cup_root = label_dir / expert / machine / "Cup"
		disc_root = label_dir / expert / machine / "Disc"
		if not cup_root.exists() or not disc_root.exists():
			logging.warning("Missing cup/disc directory for %s/%s", expert, machine)
			return None
		try:
			cup_mask = load_mask(cup_root, stem)
			disc_mask = load_mask(disc_root, stem)
		except FileNotFoundError as exc:
			logging.warning("%s", exc)
			return None

		cup_mask = keep_largest_component(cup_mask)
		disc_mask = keep_largest_component(disc_mask)
		disc_mask = np.logical_or(disc_mask, cup_mask)
		cup_mask = np.logical_and(cup_mask, disc_mask)

		disc_masks.append(disc_mask)
		label_arrays.append(build_label(disc_mask, cup_mask))

	union_disc = np.any(np.stack(disc_masks, axis=0), axis=0)
	if not union_disc.any():
		logging.warning("Union disc mask empty for %s", image_path)
		return None

	try:
		center = get_center_from_mask(union_disc)
	except ValueError as exc:
		logging.warning("%s", exc)
		return None

	circle_width = CIRCLE_WIDTHS[machine]
	crop_size = int(round(rel_sidelength * circle_width))
	crop_size = max(crop_size, min_crop_size)
	crop_size = max(2, crop_size)

	cropped_image = crop_square(image, center, crop_size)
	resized_image = resize_array(cropped_image, image_size, order="bilinear")

	np.save(out_image_dir / image_file, resized_image.astype(np.uint8))

	for idx, label_array in enumerate(label_arrays):
		cropped_label = crop_square(label_array.astype(np.uint8), center, crop_size)
		resized_label = resize_array(cropped_label, image_size, order="nearest")
		label_file = label_files[idx]
		np.save(out_label_dir / label_file, resized_label.astype(np.uint8))

	return {
		"sample_id": sample_id,
		"machine": machine,
		"source_image": str(image_path),
		"image_file": image_file,
		"label_files": json.dumps(label_files),
		"crop_size": crop_size,
	}


def process_split(
	split: SplitSpec,
	split_prefix: str,
	start_index: int,
	image_size: int,
	rel_sidelength: float,
	min_crop_size: int,
	images_dir: Path,
	labels_dir: Path,
	skip_existing: bool,
) -> Tuple[int, List[dict]]:
	if not split.image_dir.exists():
		raise FileNotFoundError(
			f"Image directory missing for split {split.name}: {split.image_dir}",
		)
	if not split.label_dir.exists():
		raise FileNotFoundError(
			f"Label directory missing for split {split.name}: {split.label_dir}",
		)

	experts = list_experts(split.label_dir)
	if not experts:
		logging.warning("No expert folders found in %s", split.label_dir)
		return start_index, []

	next_index = start_index
	rows: List[dict] = []

	for machine in CIRCLE_WIDTHS:
		machine_images = sorted(iter_machine_images(split.image_dir, machine))
		if not machine_images:
			logging.info("No %s images for split %s", machine, split.name)
			continue

		logging.info(
			"Processing %d %s images for split %s",
			len(machine_images),
			machine,
			split.name,
		)

		for image_path in tqdm(
			machine_images,
			desc=f"{split.name}-{machine}",
			unit="img",
		):
			sample_id = f"{split_prefix}_{next_index:06d}"
			image_file = f"{sample_id}.npy"
			label_files = [
				f"{sample_id}_{idx:02d}_mask.npy" for idx in range(len(experts))
			]

			if skip_existing:
				targets = [images_dir / image_file] + [
					labels_dir / fname for fname in label_files
				]
				if all(path.exists() for path in targets):
					logging.info("Skipping %s because outputs already exist", sample_id)
					next_index += 1
					continue
				if any(path.exists() for path in targets):
					logging.warning(
						"Partial outputs exist for %s; skipping to avoid overwrite",
						sample_id,
					)
					next_index += 1
					continue

			sample_row = process_sample(
				image_path=image_path,
				machine=machine,
				experts=experts,
				label_dir=split.label_dir,
				out_image_dir=images_dir,
				out_label_dir=labels_dir,
				sample_id=sample_id,
				image_file=image_file,
				label_files=label_files,
				image_size=image_size,
				rel_sidelength=rel_sidelength,
				min_crop_size=min_crop_size,
			)

			if sample_row is None:
				continue

			next_index += 1
			rows.append({**sample_row, "split": split.name})

	return next_index, rows


def write_metadata(save_path: Path, rows: Sequence[dict]) -> None:
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
			image_dir=args.train_image_dir.expanduser().resolve(),
			label_dir=args.train_label_dir.expanduser().resolve(),
		),
		SplitSpec(
			name="test",
			image_dir=args.test_image_dir.expanduser().resolve(),
			label_dir=args.test_label_dir.expanduser().resolve(),
		),
	]

	sample_index = 0
	metadata_rows: List[dict] = []
	for split in splits:
		split_prefix = split.name[0]
		sample_index, rows = process_split(
			split=split,
			split_prefix=split_prefix,
			start_index=sample_index,
			image_size=args.image_size,
			rel_sidelength=args.rel_sidelength,
			min_crop_size=args.image_size // 2,
			images_dir=images_dir,
			labels_dir=labels_dir,
			skip_existing=args.skip_existing,
		)
		metadata_rows.extend(rows)

	write_metadata(save_path, metadata_rows)


if __name__ == "__main__":
	main()