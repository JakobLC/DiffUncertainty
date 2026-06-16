"""Preprocess the RIGA retinal optic disc/cup dataset.

The script processes cleaned RIGA data (output from clean_riga.py) following the
same approach as Chaksu and REFUGE:

Processing strategy:
1. Load cleaned RIGA images (prime.png) and their 6 ground truth label variants (gt*.png).
2. Extract individual disc diameters from each GT variant, grouping by clinic
   (BinRushed, MESSIDOR, or Magrabi).
3. Compute mean disc diameters per clinic using individual annotator measurements.
4. Cache the computed mean diameters to skip recomputation.
5. Crop each image around the center of the disc union using 2× mean diameter as crop size.
6. Resize to fixed output size and save as .npy files with individual label masks per variant.

Output structure mirrors REFUGE/Chaksu:

<save_path>/
	images/
	labels/

Each image has one .npy file, and each corresponding label has 6 .npy files
(one per GT variant, with encoding: 0=background, 1=donut/disc-ring, 2=cup).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class CaseSpec:
    """A single RIGA case with prime image and GT variants."""

    clinic: str
    image_file: Path  # prime.png
    gt_files: List[Path]  # [gt1.png, ..., gt6.png]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop and save the RIGA dataset in a lidc_2d style format",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/jloch/Desktop/diff/luzern/values_datasets/riga/cleaned"),
        help="Directory containing cleaned RIGA images (output from clean_riga.py)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/home/jloch/Desktop/diff/luzern/values_datasets/riga{image_size}/preprocessed",
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
        help=(
            "Square crop side length as a multiple of the mean disc diameter. "
            "Default 2.0 gives crop_size = 2 * mean_diameter."
        ),
    )
    parser.add_argument(
        "--min-raters",
        type=int,
        default=6,
        help=(
            "Minimum number of successfully loaded raters/GT variants required to process an image. "
            "Must be in range [1, 6]. Default 6 requires all variants."
        ),
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
    args = parser.parse_args()
    
    # Validate min_raters
    if not (1 <= args.min_raters <= 6):
        parser.error(f"--min-raters must be in range [1, 6], got {args.min_raters}")
    
    return args


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )


def extract_clinic(filename: str) -> str:
    """Extract clinic name from filename.
    
    Filename should contain exactly one of: BinRushed, MESSIDOR, Magrabi
    
    Raises
    ------
    ValueError
        If filename doesn't contain exactly one clinic identifier.
    """
    clinics = ["BinRushed", "MESSIDOR", "Magrabi"]
    found = [clinic for clinic in clinics if clinic in filename]
    
    if len(found) != 1:
        raise ValueError(
            f"Filename '{filename}' must contain exactly one of {clinics}, but found {found}"
        )
    
    return found[0]


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


def get_bbox_diameter(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return float(max(width, height))


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


def load_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def load_label(label_path: Path) -> np.ndarray:
    """Load a RIGA GT label file as uint8 array with values 0, 1, 2.
    
    For palette images, np.array() returns the palette indices directly,
    which is what we want (0=background, 1=donut, 2=cup).
    """
    img = Image.open(label_path)
    # For palette mode images, np.array() gives palette indices directly
    arr = np.array(img)
    return arr.astype(np.uint8)


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


def list_cases(data_path: Path, min_raters: int = 6) -> List[CaseSpec]:
    """List all RIGA cases from cleaned directory.
    
    Expects files named like:
    - {id}_{name}_prime.png
    - {id}_{name}_gt1.png
    - {id}_{name}_gt2.png
    - ...
    - {id}_{name}_gt6.png
    
    Parameters
    ----------
    data_path : Path
        Path to cleaned RIGA directory
    min_raters : int
        Minimum number of GT variants required (default 6)
    """
    cases: List[CaseSpec] = []
    prime_files = sorted(data_path.glob("*_prime.png"))
    
    for prime_file in prime_files:
        stem = prime_file.stem.replace("_prime", "")  # Remove _prime suffix
        
        # Extract clinic from the full filename
        try:
            clinic = extract_clinic(prime_file.name)
        except ValueError as e:
            logging.warning(f"Skipping {prime_file.name}: {e}")
            continue
        
        # Find corresponding GT files
        gt_files = []
        for gt_idx in range(1, 7):
            gt_file = data_path / f"{stem}_gt{gt_idx}.png"
            if not gt_file.exists():
                logging.warning(f"Missing GT variant {gt_idx} for {prime_file.name}")
                break
            gt_files.append(gt_file)
        
        if len(gt_files) >= min_raters:
            cases.append(CaseSpec(clinic=clinic, image_file=prime_file, gt_files=gt_files[:6]))
        elif len(gt_files) > 0:
            logging.info(
                f"Skipping {prime_file.name}: only {len(gt_files)} GT variants found, "
                f"but min_raters={min_raters} required"
            )
    
    return cases


def extract_disc_mask_from_gt(gt_label: np.ndarray) -> np.ndarray:
    """Extract disc mask from RIGA GT label.
    
    RIGA GT encoding:
    - 0: outside
    - 1: donut (disc ring, excluding cup)
    - 2: center (cup)
    
    Disc = donut + center (i.e., any value >= 1)
    """
    return (gt_label >= 1).astype(bool)


def collect_case_statistics(case: CaseSpec) -> Tuple[List[float], Path, str]:
    """Extract individual disc diameters from all 6 GT variants.
    
    Returns
    -------
    diameters : List[float]
        List of 6 disc diameters (one per GT variant)
    image_path : Path
        Path to the prime image
    stem : str
        Base filename stem
    """
    diameters: List[float] = []
    
    for gt_file in case.gt_files:
        gt_label = load_label(gt_file)
        disc_mask = keep_largest_component(extract_disc_mask_from_gt(gt_label))
        
        if disc_mask.any():
            diameter = get_bbox_diameter(disc_mask)
            diameters.append(diameter)
    
    if not diameters:
        raise ValueError(f"No disc measurements for case {case.image_file}")
    
    stem = case.image_file.stem.replace("_prime", "")
    return diameters, case.image_file, stem


def load_or_compute_mean_diameters(cases: Sequence[CaseSpec], cache_path: Path) -> dict:
    """Compute mean disc diameters per clinic.
    
    For each clinic, flattens all individual measurements across all cases
    and computes the mean.
    """
    required_keys = {"BinRushed", "MESSIDOR", "Magrabi"}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if required_keys.issubset(payload):
            return {key: float(payload[key]) for key in required_keys}
        logging.info("Existing mean diameter cache has an older schema; recomputing %s", cache_path)

    diameters_by_clinic: dict[str, List[float]] = {
        "BinRushed": [],
        "MESSIDOR": [],
        "Magrabi": [],
    }
    
    for case in tqdm(cases, desc="Measuring discs", unit="case"):
        try:
            case_diameters, _, _ = collect_case_statistics(case)
            diameters_by_clinic[case.clinic].extend(case_diameters)
        except Exception as e:
            logging.warning(f"Failed to measure case {case.image_file}: {e}")
            continue

    if not any(diameters_by_clinic.values()):
        raise ValueError("No disc diameters could be computed")

    payload = {
        clinic: float(np.mean(values)) if values else float("nan")
        for clinic, values in diameters_by_clinic.items()
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
    min_raters: int = 6,
) -> Optional[Tuple[str, str]]:
    """Process a single RIGA case.
    
    Parameters
    ----------
    min_raters : int
        Minimum number of GT variants that must load successfully.
        If fewer load, the case is skipped.
    
    Returns
    -------
    result : Tuple[str, str]
        (sample_id, original_filename) if successful, None otherwise.
    """
    stem = case.image_file.stem.replace("_prime", "")
    image_file = f"{sample_id}.npy"
    # Save one mask file per GT variant (matching Chaksu/REFUGE structure)
    label_files = [f"{sample_id}_{idx:02d}_mask.npy" for idx in range(len(case.gt_files))]

    image_target = images_dir / image_file
    label_targets = [labels_dir / label_file for label_file in label_files]

    if skip_existing and image_target.exists() and all(path.exists() for path in label_targets):
        logging.info("Skipping %s because outputs already exist", sample_id)
        return (sample_id, case.image_file.name)
    if skip_existing and (image_target.exists() or any(path.exists() for path in label_targets)):
        logging.warning("Partial outputs exist for %s; skipping to avoid overwrite", sample_id)
        return None

    # Load image
    try:
        image = load_image(case.image_file)
    except Exception as e:
        logging.error(f"Failed to load image {case.image_file}: {e}")
        return None

    # Load all GT labels and compute disc union for center calculation
    gt_labels: List[np.ndarray] = []
    disc_masks: List[np.ndarray] = []
    
    for gt_file in case.gt_files:
        try:
            gt_label = load_label(gt_file)
            disc_mask = keep_largest_component(extract_disc_mask_from_gt(gt_label))
            gt_labels.append(gt_label)
            disc_masks.append(disc_mask)
        except Exception as e:
            logging.warning(f"Failed to load label {gt_file}: {e}")
            # Continue instead of failing, to allow partial rater success

    # Check if we have enough successfully loaded labels
    if len(gt_labels) < min_raters:
        logging.warning(
            "Case %s: only %d out of %d GT variants loaded successfully, "
            "but min_raters=%d required. Skipping.",
            case.image_file,
            len(gt_labels),
            len(case.gt_files),
            min_raters,
        )
        return None

    if not disc_masks:
        logging.warning("No disc masks for case %s", case.image_file)
        return None

    # Compute union of all disc masks for center calculation
    disc_union = np.any(np.stack(disc_masks, axis=0), axis=0)
    if not disc_union.any():
        logging.warning("Empty disc union for case %s", case.image_file)
        return None

    center = get_bbox_center(disc_union)

    # Crop and resize image
    cropped_image = crop_square(image, center, crop_size)
    resized_image = resize_array(cropped_image, image_size, order="bilinear")
    np.save(image_target, resized_image.astype(np.uint8))

    # Crop and resize each GT label individually (matching Chaksu/REFUGE)
    for gt_label, label_target in zip(gt_labels, label_targets):
        cropped_label = crop_square(gt_label, center, crop_size)
        resized_label = resize_array(cropped_label, image_size, order="nearest")
        np.save(label_target, resized_label.astype(np.uint8))

    return (sample_id, case.image_file.name)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        logging.error("Data path does not exist: %s", data_path)
        sys.exit(1)

    cases = list_cases(data_path, min_raters=args.min_raters)
    if not cases:
        logging.error("No RIGA cases found in %s", data_path)
        sys.exit(1)

    logging.info(f"Found {len(cases)} RIGA cases")
    for clinic in ["BinRushed", "MESSIDOR", "Magrabi"]:
        count = sum(1 for c in cases if c.clinic == clinic)
        logging.info(f"  {clinic}: {count}")

    mean_diam_path = Path(__file__).with_name("mean_diam.json")
    mean_diameters = load_or_compute_mean_diameters(cases, mean_diam_path)
    logging.info("Mean disc diameters (mm): %s", json.dumps(mean_diameters, indent=2))

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
    metadata_rows: List[dict] = []
    
    for case in tqdm(cases, desc="Saving cases", unit="case"):
        sample_id = f"{case.clinic[0:3].lower()}_{sample_index:06d}"
        crop_size = max(1, int(round(mean_diameters[case.clinic] * args.crop_multiplier)))
        
        result = process_case(
            case=case,
            sample_id=sample_id,
            image_size=args.image_size,
            crop_size=crop_size,
            images_dir=images_dir,
            labels_dir=labels_dir,
            skip_existing=args.skip_existing,
            min_raters=args.min_raters,
        )
        
        if result is not None:
            sample_id, original_filename = result
            metadata_rows.append({
                "sample_id": sample_id,
                "original_filename": original_filename,
            })
            sample_index += 1
    
    # Write metadata CSV
    if metadata_rows:
        metadata_path = save_path / "metadata.csv"
        with metadata_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["sample_id", "original_filename"])
            writer.writeheader()
            writer.writerows(metadata_rows)
        logging.info("Metadata saved to %s", metadata_path)


if __name__ == "__main__":
    main()
