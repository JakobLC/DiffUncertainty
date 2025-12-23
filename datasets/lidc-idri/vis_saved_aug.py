#!/usr/bin/env python3
"""Quickly visualize offline LIDC-2D augmentations stored on disk."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from vis_ood_aug import DEFAULT_ROOT, ROW_ORDER, load_id_image, load_label_sum


#DEFAULT_ROOT = Path("/home/jloch/Desktop/diff/luzern/values_datasets/lidc_small")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root that contains preprocessed/{images,labels,augmented}.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of random image columns to plot (default: 8).",
    )
    return parser.parse_args()


def _collect_aug_splits(base_dir: Path) -> List[str]:
    aug_root = base_dir / "preprocessed" / "augmented"
    if not aug_root.is_dir():
        raise FileNotFoundError(
            f"Expected augmented directory at {aug_root}. Run gen_ood_aug.py first."
        )
    splits = []
    for sub in sorted(aug_root.iterdir()):
        images_dir = sub / "images"
        if images_dir.is_dir():
            splits.append(sub.name)
    if not splits:
        raise RuntimeError(f"No augmented splits with images found under {aug_root}.")
    return splits


def _collect_common_augmented_names(base_dir: Path, splits: Sequence[str]) -> List[str]:
    aug_root = base_dir / "preprocessed" / "augmented"
    common: set[str] | None = None
    for split in splits:
        images_dir = aug_root / split / "images"
        files = {path.name for path in images_dir.glob("*.npy")}
        if not files:
            raise RuntimeError(f"Split '{split}' under {images_dir} has no .npy files.")
        common = files if common is None else common & files
    if not common:
        raise RuntimeError(
            "No overlapping augmented images found across splits; cannot render grid."
        )
    return sorted(common)


def _choose_images(names: Sequence[str], k: int) -> List[str]:
    if k >= len(names):
        return list(names)
    rng = random.SystemRandom()
    return sorted(rng.sample(list(names), k))


def _determine_rows(splits: Sequence[str]) -> List[Tuple[str, str]]:
    preferred = []
    seen = set()
    for key, label in ROW_ORDER:
        if key in {"label", "id"} or key in splits:
            preferred.append((key, label))
            seen.add(key)
    for split in splits:
        if split not in seen:
            formatted = split.replace("_", " ").title()
            preferred.append((split, formatted))
    return preferred


def _load_augmented(base_dir: Path, split_name: str, image_name: str) -> np.ndarray:
    aug_path = base_dir / "preprocessed" / "augmented" / split_name / "images" / image_name
    if not aug_path.exists():
        raise FileNotFoundError(f"Missing augmented file {aug_path}.")
    return np.load(aug_path).astype(np.float32)


def render_grid(base_dir: Path, num_images: int) -> None:
    splits = _collect_aug_splits(base_dir)
    names = _collect_common_augmented_names(base_dir, splits)
    rows = _determine_rows(splits)
    selection = _choose_images(names, num_images)
    fig, axes = plt.subplots(len(rows), len(selection), figsize=(20, 10))
    if len(selection) == 1:
        axes = np.expand_dims(axes, axis=1)
    for col, image_name in enumerate(selection):
        label_sum = load_label_sum(base_dir, image_name)
        id_image = load_id_image(base_dir, image_name)
        aug_cache = {}
        for split in splits:
            try:
                aug_cache[split] = _load_augmented(base_dir, split, image_name)
            except FileNotFoundError:
                continue
        id_vis = (id_image - id_image.mean()) / (id_image.std() + 1e-8)
        for row_idx, (key, label) in enumerate(rows):
            ax = axes[row_idx][col]
            if key == "label":
                ax.imshow(label_sum, cmap="inferno")
            elif key == "id":
                ax.imshow(id_vis, cmap="gray", vmin=-2, vmax=2)
            else:
                aug_img = aug_cache.get(key)
                if aug_img is None:
                    ax.axis("off")
                else:
                    mean = float(aug_img.mean())
                    std = float(aug_img.std() + 1e-8)
                    ax.imshow((aug_img - mean) / std, cmap="gray", vmin=-2, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=14,
                    labelpad=40,
                )
        axes[0][col].set_title(image_name)
    plt.show()


def main() -> None:
    args = parse_args()
    render_grid(args.base_dir, args.num_images)


if __name__ == "__main__":
    main()
