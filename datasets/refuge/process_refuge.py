"""Preprocess the REFUGE multi-rater optic disc/cup dataset.

The script mirrors the Chaksu preprocessing flow, but REFUGE has a simpler
layout: each case lives in a single folder containing one fundus image,
`*_disc.bmp`, `*_cup.bmp`, and seven `seg_disc`/`seg_cup` annotations.

Processing strategy:
1. Gather every case across Training-400, Validation-400, and Test-400.
2. Build dataset-wide and split-specific mean disc diameters from individual
    annotator discs (not unions), computing the mean across all 7 annotators
    and all cases per split. Uses a combined val/test split for valtest.
3. Crop each image around the center of the disc union bounding box using a
    square with side length equal to twice the selected mean diameter.
4. Pad outside-image regions with black.
5. Save the resized image and a single class mask with labels:
   0 = background, 1 = disc, 2 = cup.
   Disc pixels are assigned first and cup pixels are written on top.

No metadata file is written because REFUGE does not expose any meaningful
dataset-level provenance worth preserving here.

NOTE: Mean diameters are computed from individual annotator discs to match
Chaksu's approach, not from unions. This produces consistent crop sizing
across both datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CASE_IMAGE_NAME = "{stem}.jpg"
CASE_DISC_NAME = "{stem}_disc.bmp"
CASE_CUP_NAME = "{stem}_cup.bmp"
CASE_DISC_ANNOTATION = "{stem}_seg_disc_{idx}.png"
CASE_CUP_ANNOTATION = "{stem}_seg_cup_{idx}.png"


@dataclass(frozen=True)
class SplitSpec:
    """REFUGE split rooted at a folder containing per-case subdirectories."""

    name: str
    root: Path


@dataclass(frozen=True)
class CaseSpec:
    """A single REFUGE case folder."""

    split: str
    folder: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop and save the REFUGE dataset into a lidc_2d style format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/jloch/Desktop/diff/luzern/values_datasets/refuge/REFUGE-Multirater"),
        help="Directory containing Training-400, Validation-400, and Test-400",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/home/jloch/Desktop/diff/luzern/values_datasets/refuge{image_size}/preprocessed",
        help="Where to store the processed dataset",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Final square size for saved crops",
    )
    parser.add_argument(
        "--crop-multiplier",
        type=float,
        default=2.0,
        help="Square crop side length as a multiple of the dataset-wide mean disc diameter",
    )
    parser.add_argument(
        "--use-all-normalization",
        action="store_true",
        help="Use the all-split mean diameter for every sample instead of train/valtest normalization",
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


def list_case_dirs(split_root: Path) -> List[Path]:
    if not split_root.exists():
        return []
    return [p for p in sorted(split_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def iter_all_cases(splits: Sequence[SplitSpec]) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    for split in splits:
        for case_dir in list_case_dirs(split.root):
            cases.append(CaseSpec(split=split.name, folder=case_dir))
    return cases


def find_case_file(folder: Path, filename: str) -> Path:
    path = folder / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing expected file: {path}")


def load_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def load_binary_mask(mask_path: Path) -> np.ndarray:
    return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return binary
    labeled, num = ndimage.label(binary)
    if num <= 1:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0, 0, 0, 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(x_max), int(y_min), int(y_max)


def get_bbox_center(mask: np.ndarray) -> Tuple[float, float]:
    x_min, x_max, y_min, y_max = get_bbox(mask)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return center_y, center_x


def get_bbox_diameter(mask: np.ndarray) -> float:
    x_min, x_max, y_min, y_max = get_bbox(mask)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return float(max(width, height))


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
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        if array.ndim == 3:
            pad_width += ((0, 0),)
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
        top += pad_top
        bottom += pad_top
        left += pad_left
        right += pad_left

    return array[top:bottom, left:right]


def resize_array(arr: np.ndarray, size: int, order: str) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.uint8))
    resample = Image.BILINEAR if order == "bilinear" else Image.NEAREST
    resized = image.resize((size, size), resample=resample)
    return np.array(resized)


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def load_annotation_sets(case: CaseSpec) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    stem = case.folder.name
    disc_masks: List[np.ndarray] = []
    cup_masks: List[np.ndarray] = []

    for idx in range(1, 8):
        disc_mask = keep_largest_component(
            load_binary_mask(find_case_file(case.folder, CASE_DISC_ANNOTATION.format(stem=stem, idx=idx)))
        )
        cup_mask = keep_largest_component(
            load_binary_mask(find_case_file(case.folder, CASE_CUP_ANNOTATION.format(stem=stem, idx=idx)))
        )
        # Ensure disc includes all cup pixels, matching Chaksu's approach
        disc_mask = np.logical_or(disc_mask, cup_mask)
        cup_mask = np.logical_and(cup_mask, disc_mask)
        
        disc_masks.append(disc_mask.astype(bool))
        cup_masks.append(cup_mask.astype(bool))

    if not disc_masks or not cup_masks:
        raise ValueError(f"Missing annotations for case {case.folder}")

    return disc_masks, cup_masks


def collect_case_statistics(case: CaseSpec) -> Tuple[List[float], Path, str]:
    """Compute individual disc diameters for all 7 annotators.
    
    Returns a list of 7 diameters (one per annotator), not the union.
    This matches Chaksu's approach of averaging individual measurements.
    """
    disc_masks, _ = load_annotation_sets(case)
    
    # Compute diameter for each individual annotator
    diameters: List[float] = []
    for disc_mask in disc_masks:
        if disc_mask.any():
            diameter = get_bbox_diameter(disc_mask)
            diameters.append(diameter)
    
    if not diameters:
        raise ValueError(f"No disc measurements for case {case.folder}")
    
    stem = case.folder.name
    image_path = find_case_file(case.folder, CASE_IMAGE_NAME.format(stem=stem))
    return diameters, image_path, stem


def build_labels_from_annotations(case: CaseSpec) -> List[np.ndarray]:
    disc_masks, cup_masks = load_annotation_sets(case)
    labels: List[np.ndarray] = []

    for disc_mask, cup_mask in zip(disc_masks, cup_masks):
        label = np.zeros_like(disc_mask, dtype=np.uint8)
        label[disc_mask] = 1
        label[cup_mask] = 2
        labels.append(label)

    return labels


def load_or_compute_mean_diameters(cases: Sequence[CaseSpec], cache_path: Path) -> dict:
    """Compute mean disc diameters from individual annotator measurements.
    
    For each case, we get 7 measurements (one per annotator). We then flatten
    all measurements across all cases and compute the mean per split.
    """
    required_keys = {"all", "train", "valtest"}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if required_keys.issubset(payload):
            return {key: float(payload[key]) for key in required_keys}
        logging.info("Existing mean diameter cache has an older schema; recomputing %s", cache_path)

    diameters_by_split: dict[str, List[float]] = {"all": [], "train": [], "valtest": []}
    for case in tqdm(cases, desc="Measuring discs", unit="case"):
        case_diameters, _, _ = collect_case_statistics(case)
        split_key = "train" if case.split == "train" else "valtest"
        # Flatten: add all 7 individual measurements
        diameters_by_split[split_key].extend(case_diameters)
        diameters_by_split["all"].extend(case_diameters)

    if not diameters_by_split["all"]:
        raise ValueError("No disc diameters could be computed")

    payload = {
        split: float(np.mean(values)) if values else float("nan")
        for split, values in diameters_by_split.items()
    }
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def process_case(
    case: CaseSpec,
    sample_id: str,
    image_size: int,
    crop_size: int,
    images_dir: Path,
    labels_dir: Path,
    skip_existing: bool,
) -> Optional[Tuple[str, str]]:
    stem = case.folder.name
    image_path = find_case_file(case.folder, CASE_IMAGE_NAME.format(stem=stem))
    image_file = f"{sample_id}.npy"
    label_files = [f"{sample_id}_{idx:02d}_mask.npy" for idx in range(7)]

    image_target = images_dir / image_file
    label_targets = [labels_dir / label_file for label_file in label_files]

    if skip_existing and image_target.exists() and all(path.exists() for path in label_targets):
        logging.info("Skipping %s because outputs already exist", sample_id)
        return image_file, label_files[0]
    if skip_existing and (image_target.exists() or any(path.exists() for path in label_targets)):
        logging.warning("Partial outputs exist for %s; skipping to avoid overwrite", sample_id)
        return None

    image = load_image(image_path)
    labels = build_labels_from_annotations(case)

    disc_masks, _ = load_annotation_sets(case)
    disc_union = np.any(np.stack(disc_masks, axis=0), axis=0)
    if not disc_union.any():
        logging.warning("Union disc mask empty for %s", case.folder)
        return None

    center = get_bbox_center(disc_union)

    cropped_image = crop_square(image, center, crop_size)

    resized_image = resize_array(cropped_image, image_size, order="bilinear")

    np.save(image_target, resized_image.astype(np.uint8))
    for label, label_target in zip(labels, label_targets):
        cropped_label = crop_square(label, center, crop_size)
        resized_label = resize_array(cropped_label, image_size, order="nearest")
        np.save(label_target, resized_label.astype(np.uint8))
    return image_file, label_files[0]


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_root = args.data_root.expanduser().resolve()
    splits = [
        SplitSpec(name="train", root=data_root / "Training-400"),
        SplitSpec(name="val", root=data_root / "Validation-400"),
        SplitSpec(name="test", root=data_root / "Test-400"),
    ]

    cases = iter_all_cases(splits)
    if not cases:
        logging.error("No REFUGE cases found under %s", data_root)
        sys.exit(1)

    mean_diam_path = Path(__file__).with_name("mean_diam.json")
    mean_diameters = load_or_compute_mean_diameters(cases, mean_diam_path)
    logging.info(
        "Mean disc diameters: all=%.3f train=%.3f valtest=%.3f",
        mean_diameters["all"],
        mean_diameters["train"],
        mean_diameters["valtest"],
    )

    if args.use_all_normalization:
        normalization_key = "all"
        logging.info("Using all-split normalization for every sample")
    else:
        normalization_key = None
        logging.info("Using split-specific normalization")

    save_path = Path(args.save_path.format(image_size=args.image_size)).expanduser().resolve()
    if save_path.exists() and not args.overwrite:
        non_empty = any(save_path.iterdir())
        if non_empty:
            logging.error(
                "Save path %s already has content. Use --overwrite to continue.",
                save_path,
            )
            sys.exit(1)
    save_path.mkdir(parents=True, exist_ok=True)
    images_dir, labels_dir = ensure_output_dirs(save_path)

    sample_index = 0
    for case in tqdm(cases, desc="Saving cases", unit="case"):
        sample_id = f"{case.split}_{sample_index:06d}"
        split_key = normalization_key or ("train" if case.split == "train" else "valtest")
        crop_size = max(1, int(round(mean_diameters[split_key] * args.crop_multiplier)))
        result = process_case(
            case=case,
            sample_id=sample_id,
            image_size=args.image_size,
            crop_size=crop_size,
            images_dir=images_dir,
            labels_dir=labels_dir,
            skip_existing=args.skip_existing,
        )
        if result is not None:
            sample_index += 1


if __name__ == "__main__":
    main()


