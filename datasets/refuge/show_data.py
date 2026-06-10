"""Visualize processed REFUGE samples from a size-specific output directory.

The script expects the output layout produced by ``process_refuge.py``:

<save_path>/
	images/
	labels/

Images are RGB ``.npy`` arrays. Each sample has seven label files named like
``<sample_id>_00_mask.npy`` through ``<sample_id>_06_mask.npy``. Each mask is
encoded as 0 = background, 1 = disc, 2 = cup.

Example:
	python show_data.py --image-size 128 --n-rows 3 --n-cols 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Show processed REFUGE samples as an image grid",
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=Path("/home/jloch/Desktop/diff/luzern/values_datasets/refuge{image_size}/preprocessed"),
		help="Processed REFUGE root produced by process_refuge.py",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=128,
		help="Resolution suffix used in the processed dataset path",
	)
	parser.add_argument(
		"--n-rows",
		type=int,
		default=3,
		help="Number of rows in the display grid",
	)
	parser.add_argument(
		"--n-cols",
		type=int,
		default=4,
		help="Number of columns in the display grid",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed used to choose samples",
	)
	return parser.parse_args()


def resolve_data_root(template: Path, image_size: int) -> Path:
	return Path(str(template).format(image_size=image_size)).expanduser().resolve()


def list_sample_stems(images_dir: Path) -> List[str]:
	return sorted(path.stem for path in images_dir.glob("*.npy"))


def list_split_stems(stems: Sequence[str], split: str) -> List[str]:
	return [stem for stem in stems if stem.startswith(f"{split}_")]


def load_sample(images_dir: Path, labels_dir: Path, stem: str) -> np.ndarray:
	return np.load(images_dir / f"{stem}.npy")


def load_label(labels_dir: Path, stem: str, annotator_index: int) -> np.ndarray:
	return np.load(labels_dir / f"{stem}_{annotator_index:02d}_mask.npy")


def overlay_mask(ax: plt.Axes, mask: np.ndarray, linewidth: float) -> None:
	disc_mask = (mask >= 1).astype(float)
	cup_mask = (mask == 2).astype(float)
	if disc_mask.any():
		ax.contour(disc_mask, levels=[0.5], colors="C0", linewidths=linewidth)
	if cup_mask.any():
		ax.contour(cup_mask, levels=[0.5], colors="C6", linewidths=linewidth)


def overlay_all_segmentations(ax: plt.Axes, labels_dir: Path, stem: str) -> None:
	colors = [
		"#1f77b4",
		"#ff7f0e",
		"#2ca02c",
		"#d62728",
		"#9467bd",
		"#8c564b",
		"#e377c2",
	]
	for annotator_index, color in enumerate(colors):
		mask = load_label(labels_dir, stem, annotator_index)
		overlay_mask(ax, mask, linewidth=1.0)


def choose_stems(stems: Sequence[str], count: int) -> List[str]:
	if len(stems) >= count:
		return random.sample(list(stems), count)
	return [random.choice(list(stems)) for _ in range(count)]


def main() -> None:
	args = parse_args()
	random.seed(args.seed)

	data_root = resolve_data_root(args.data_root, args.image_size)
	images_dir = data_root / "images"
	labels_dir = data_root / "labels"

	if not images_dir.exists() or not labels_dir.exists():
		raise FileNotFoundError(f"Missing processed data directories under {data_root}")

	stems = list_sample_stems(images_dir)
	if not stems:
		raise FileNotFoundError(f"No .npy samples found in {images_dir}")

	if args.n_rows % 3 != 0:
		raise ValueError("--n-rows must be divisible by 3 so rows can be split evenly across train, val, and test")

	rows_per_split = args.n_rows // 3
	cols_per_row = args.n_cols * 3
	split_to_stems = {split: list_split_stems(stems, split) for split in SPLIT_ORDER}
	for split, split_stems in split_to_stems.items():
		if not split_stems:
			raise FileNotFoundError(f"No samples found for split '{split}' in {images_dir}")

	row_specs: List[Tuple[str, List[str]]] = []
	for split in SPLIT_ORDER:
		split_stems = split_to_stems[split]
		for _ in range(rows_per_split):
			row_specs.append((split, choose_stems(split_stems, cols_per_row)))

	fig, axes = plt.subplots(args.n_rows, cols_per_row, figsize=(4 * cols_per_row, 4 * args.n_rows))
	axes_2d = np.atleast_2d(axes)

	for row_index, (split, row_stems) in enumerate(row_specs):
		for col_index, stem in enumerate(row_stems):
			ax = axes_2d[row_index, col_index]
			image = load_sample(images_dir, labels_dir, stem)
			ax.imshow(image)
			overlay_all_segmentations(ax, labels_dir, stem)
			ax.set_title(f"{split}: {stem}", fontsize=8)
			ax.axis("off")

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
