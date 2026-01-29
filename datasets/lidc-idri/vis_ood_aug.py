#!/usr/bin/env python3
"""Visualize ID vs. OOD augmentations for random LIDC-2D samples."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ROOT = Path("/home/jloch/Desktop/diff/luzern/values_datasets/lidc128")

ROW_ORDER = [
    ("label", "Label sum"),
    ("id", "ID image"),
    ("ood_noise", "OOD noise"),
    ("ood_blur", "OOD blur"),
    ("ood_contrast", "OOD contrast"),
    ("ood_jpeg", "OOD JPEG"),
]


@dataclass(frozen=True)
class AugmentationSpec:
    """Holds metadata about a deterministic augmentation."""

    name: str
    transform: A.BasicTransform | None = None
    transform_factory: Callable[[Tuple[int, int]], A.BasicTransform] | None = None
    requires_uint8: bool = False
    requires_rgb: bool = False

    def make_transform(self, image_shape: Tuple[int, int]) -> A.BasicTransform:
        if self.transform_factory is not None:
            return self.transform_factory(image_shape)
        if self.transform is None:
            raise ValueError(f"AugmentationSpec '{self.name}' did not define a transform.")
        return self.transform


def _ordered_pair(first: float, second: float) -> Tuple[float, float]:
    return (first, second) if first <= second else (second, first)


class GaussianNoiseNoClip(A.ImageOnlyTransform):
    """Additive Gaussian noise without any clipping."""

    def __init__(
        self,
        *,
        mean: float = 0.0,
        std_limit: Tuple[float, float] = (0.0, 0.05),
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        low, high = _ordered_pair(std_limit[0], std_limit[1])
        self.mean = mean
        self.std_limit = (low, high)

    def apply(self, image: np.ndarray, std: float = 0.0, **params) -> np.ndarray:
        img = image.astype(np.float32, copy=False)
        if std <= 0.0:
            return img
        noise = np.random.normal(loc=self.mean, scale=std, size=img.shape).astype(np.float32)
        return img + noise

    def get_params(self) -> Dict[str, float]:
        if self.std_limit[0] == self.std_limit[1]:
            std = self.std_limit[0]
        else:
            std = random.uniform(self.std_limit[0], self.std_limit[1])
        return {"std": std}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("mean", "std_limit")


def build_shared_parser(
    description: str,
    *,
    default_root: Path = DEFAULT_ROOT,
    default_seed: int = 7,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_root,
        help="Root directory that contains preprocessed/{images,labels}.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="CSV file with patient/image metadata (must contain 'Patient ID').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Random seed used for sampling and albumentations RNGs.",
    )
    parser.add_argument(
        "--noise-std-min",
        type=float,
        default=0.2,
        help="Lower bound for Gaussian noise standard deviation (relative to image scale).",
    )
    parser.add_argument(
        "--noise-std-max",
        type=float,
        default=1.0,
        help="Upper bound for Gaussian noise standard deviation (relative to image scale).",
    )
    parser.add_argument(
        "--blur-sigma-min",
        type=float,
        default=0.05,
        help="Minimum Gaussian blur sigma expressed as a fraction of min(image height, width).",
    )
    parser.add_argument(
        "--blur-sigma-max",
        type=float,
        default=0.2,
        help="Maximum Gaussian blur sigma expressed as a fraction of min(image height, width).",
    )
    parser.add_argument(
        "--contrast-clip-limit",
        type=float,
        default=5.0,
        help="Clip limit for CLAHE contrast enhancement.",
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=10,
        help="Lower JPEG quality bound for compression-based OOD samples.",
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=20,
        help="Upper JPEG quality bound for compression-based OOD samples.",
    )
    return parser


def finalize_shared_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.metadata_csv is None:
        args.metadata_csv = args.base_dir / "metadata.csv"
    return args


def read_metadata(metadata_csv: Path) -> Dict[str, List[str]]:
    with metadata_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        patient_to_images: Dict[str, List[str]] = defaultdict(list)
        for row in reader:
            patient_id = row["Patient ID"].strip()
            image_path = Path(row["Image Save Path"]).name
            patient_to_images[patient_id].append(image_path)
    if not patient_to_images:
        raise RuntimeError(f"No entries found in metadata file {metadata_csv}.")
    return patient_to_images


def _normalize_uint8(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    min_val = float(image.min())
    max_val = float(image.max())
    scale = max(max_val - min_val, 1e-8)
    normalized = ((image - min_val) / scale * 255.0).clip(0, 255).astype(np.uint8)
    return normalized, min_val, scale


def _restore_from_uint8(image: np.ndarray, min_val: float, scale: float) -> np.ndarray:
    return (image.astype(np.float32) / 255.0 * scale + min_val).astype(np.float32)


def build_augmentation_specs(args: argparse.Namespace) -> Dict[str, AugmentationSpec]:
    noise_min, noise_max = _ordered_pair(args.noise_std_min, args.noise_std_max)
    noise_min = max(0.0, noise_min)
    noise_max = max(noise_min, noise_max)
    blur_min, blur_max = _ordered_pair(args.blur_sigma_min, args.blur_sigma_max)
    jpeg_min, jpeg_max = _ordered_pair(args.jpeg_quality_min, args.jpeg_quality_max)

    def _blur_factory(image_shape: Tuple[int, int]) -> A.BasicTransform:
        height, width = image_shape[:2]
        ref = max(1, min(height, width))
        sigma_low = max(1e-3, blur_min * ref)
        sigma_high = max(sigma_low, blur_max * ref)
        return A.GaussianBlur(
            sigma_limit=(sigma_low, sigma_high),
            always_apply=True,
        )

    noise_desc = f"Gauss noise (std in [{noise_min:.2f}, {noise_max:.2f}])"
    blur_desc = f"Gauss blur (sigma in\n[{blur_min:.2f}, {blur_max:.2f}])"
    jpeg_desc = f"JPEG quality in [{jpeg_min}, {jpeg_max}]"

    return {
        "ood_noise": AugmentationSpec(
            name=noise_desc,
            transform=GaussianNoiseNoClip(
                mean=0.0,
                std_limit=(noise_min, noise_max),
                always_apply=True,
            ),
        ),
        "ood_blur": AugmentationSpec(
            name=blur_desc,
            transform_factory=_blur_factory,
        ),
        "ood_contrast": AugmentationSpec(
            name=f"CLAHE (clip={args.contrast_clip_limit})",
            transform=A.CLAHE(
                clip_limit=args.contrast_clip_limit,
                tile_grid_size=(8, 8),
                always_apply=True,
            ),
            requires_uint8=True,
        ),
        "ood_jpeg": AugmentationSpec(
            name=jpeg_desc,
            transform=A.ImageCompression(
                quality_lower=jpeg_min,
                quality_upper=jpeg_max,
                always_apply=True,
            ),
            requires_uint8=True,
            requires_rgb=True,
        ),
    }


def apply_spec(image: np.ndarray, spec: AugmentationSpec) -> np.ndarray:
    img = image
    if img.ndim == 2:
        img = img[..., None]
    transform = spec.make_transform(img.shape[:2])
    if spec.requires_uint8:
        img_uint8, min_val, scale = _normalize_uint8(img)
        if spec.requires_rgb and img_uint8.shape[2] == 1:
            img_uint8 = np.repeat(img_uint8, 3, axis=2)
        augmented = transform(image=img_uint8)["image"]
        if spec.requires_rgb and augmented.shape[2] > 1:
            augmented = augmented[..., :1]
        restored = _restore_from_uint8(augmented, min_val, scale)
        return restored.squeeze().astype(np.float32)
    augmented = transform(image=img)["image"]
    return augmented.squeeze().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = build_shared_parser(__doc__, default_seed=11)
    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
        help="Number of random columns to render (default: 12).",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save the figure as PNG (in addition to plt.show()).",
    )
    return parser.parse_args()


def sample_images(patient_map: Dict[str, List[str]], num_images: int, seed: int) -> List[str]:
    pool = [image_name for images in patient_map.values() for image_name in images]
    if not pool:
        raise RuntimeError("Metadata CSV did not contain any image entries.")
    rng = random.Random(seed)
    rng.shuffle(pool)
    if num_images > len(pool):
        num_images = len(pool)
    return sorted(pool[:num_images])


def load_label_sum(base_dir: Path, image_name: str) -> np.ndarray:
    label_dir = base_dir / "preprocessed" / "labels"
    stem = Path(image_name).stem
    masks = []
    for idx in range(4):
        candidate = label_dir / f"{stem}_{idx:02d}_mask.npy"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing label file {candidate}")
        masks.append(np.load(candidate))
    stacked = np.stack(masks, axis=0)
    return stacked.sum(axis=0).astype(np.float32)


def load_id_image(base_dir: Path, image_name: str) -> np.ndarray:
    image_path = base_dir / "preprocessed" / "images" / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing ID image {image_path}")
    return np.load(image_path).astype(np.float32)


def build_row_labels(specs: Dict[str, AugmentationSpec]) -> Dict[str, str]:
    labels = {
        "label": "Label sum",
        "id": "ID image",
    }
    for key, base in [
        ("ood_noise", "OOD noise"),
        ("ood_blur", "OOD blur"),
        ("ood_contrast", "OOD contrast"),
        ("ood_jpeg", "OOD JPEG"),
    ]:
        labels[key] = f"{base}\n{specs[key].name}" if key in specs else base
    return labels


def render_grid(
    images: List[str],
    base_dir: Path,
    specs: Dict[str, AugmentationSpec],
    args: argparse.Namespace,
) -> None:
    row_labels = build_row_labels(specs)
    fig, axes = plt.subplots(len(ROW_ORDER), len(images), figsize=(20, 10))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=1)
    for col, image_name in enumerate(images):
        label_sum = load_label_sum(base_dir, image_name)
        id_image = load_id_image(base_dir, image_name)
        aug_cache = {}
        for key, _ in ROW_ORDER:
            if key in {"label", "id"}:
                continue
            if key in specs:
                aug_cache[key] = apply_spec(id_image, specs[key])
        id_vis = (id_image - id_image.mean()) / (id_image.std() + 1e-8)
        for row_idx, (key, _) in enumerate(ROW_ORDER):
            ax = axes[row_idx][col]
            if key == "label":
                ax.imshow(label_sum, cmap="inferno")
            elif key == "id":
                ax.imshow(id_vis, cmap="gray", vmin=-2, vmax=2)
            else:
                aug_img = aug_cache.get(key)
                if aug_img is None:
                    ax.axis("off")
                    continue
                mean = float(aug_img.mean())
                std = float(aug_img.std() + 1e-8)
                ax.imshow((aug_img - mean) / std, cmap="gray", vmin=-2, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(
                    row_labels.get(key, key),
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=14,
                    labelpad=40,
                )
        axes[0][col].set_title(image_name)
    # plt.tight_layout()
    if args.save_plot:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_plot, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    args = parse_args()
    args = finalize_shared_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    patient_map = read_metadata(args.metadata_csv)
    selected = sample_images(patient_map, args.num_images, args.seed)
    specs = build_augmentation_specs(args)
    render_grid(selected, args.base_dir, specs, args)


if __name__ == "__main__":
    main()
