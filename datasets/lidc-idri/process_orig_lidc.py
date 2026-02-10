#!/usr/bin/env python3
"""Convert the aggregated LIDC pickle into origlidc-style image/mask folders."""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_IMAGE_SIZE = 128
RATER_COUNT = 4


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_pickle = repo_root / "values_datasets" / "lidc_orig" / "data_lidc.pickle"
    default_metadata = repo_root / "values_datasets" / "lidc_orig" / "LIDC-IDRI_MetaData.csv"
    default_output_root = repo_root / "values_datasets"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=default_pickle,
        help="Path to data_lidc.pickle containing image/mask entries.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=default_metadata,
        help="Path to LIDC-IDRI_MetaData.csv for mapping series to patients.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Root directory where origlidc folders live (defaults to values_datasets).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Target spatial size for saved slices (must divide the source size).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset folder to create (defaults to origlidc{image_size}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    args = parser.parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if BASE_IMAGE_SIZE % args.image_size != 0:
        raise ValueError(
            f"image_size={args.image_size} must divide source size {BASE_IMAGE_SIZE}."
        )
    if args.dataset_name is None:
        args.dataset_name = f"origlidc{args.image_size}"
    return args


def load_series_to_subject_map(metadata_csv: Path) -> Dict[str, str]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    df = pd.read_csv(metadata_csv, usecols=["Subject ID", "Series ID"], dtype=str)
    if df.empty:
        raise ValueError("Metadata CSV is empty or missing required columns.")
    mapping: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        subject = str(row[0]).strip()
        study = str(row[1]).strip()
        if not subject or not study:
            continue
        mapping.setdefault(study, subject)
    if not mapping:
        raise ValueError("Failed to build series-to-subject mapping from metadata.")
    return mapping


def reshape_reduce(array: np.ndarray, target_size: int, reduce: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}.")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square array, got shape {array.shape}.")
    src_size = array.shape[0]
    if src_size == target_size:
        return np.array(array, copy=True)
    factor = src_size // target_size
    if src_size % target_size != 0:
        raise ValueError(
            f"Cannot evenly downsample from {src_size} to {target_size}."
        )
    reshaped = np.ascontiguousarray(array).reshape(target_size, factor, target_size, factor)
    if reduce == "mean":
        reduced = reshaped.mean(axis=(1, 3))
    elif reduce == "max":
        reduced = reshaped.max(axis=(1, 3))
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")
    return reduced


def format_patient_code(subject_id: str) -> str:
    parts = subject_id.strip().split("-")
    candidate = parts[-1] if parts else subject_id
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if not digits:
        digits = "".join(ch for ch in subject_id if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract numeric code from subject_id={subject_id}.")
    return digits.zfill(4)


def save_entry(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
    patient_code: str,
    patient_label: str,
    idx: int,
    image_size: int,
    images_dir: Path,
    labels_dir: Path,
    overwrite: bool,
) -> Tuple[str, int]:
    image_proc = reshape_reduce(image, image_size, reduce="mean")
    processed_masks = [
        reshape_reduce(mask, image_size, reduce="max").astype(np.uint8)
        for mask in masks
    ]
    stem = f"{patient_code}_{idx:03d}"
    image_path = images_dir / f"{stem}.npy"
    if image_path.exists() and not overwrite:
        return stem, 0
    np.save(image_path, image_proc)
    for rater_idx, mask in enumerate(processed_masks):
        mask_path = labels_dir / f"{stem}_{rater_idx:02d}_mask.npy"
        if mask_path.exists() and not overwrite:
            continue
        np.save(mask_path, mask)
    return stem, 1


def main() -> None:
    args = parse_args()
    dataset_root = args.output_root / args.dataset_name
    images_dir = dataset_root / "preprocessed" / "images"
    labels_dir = dataset_root / "preprocessed" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    series_to_subject = load_series_to_subject_map(args.metadata_csv)
    if not args.pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {args.pickle_path}")
    with args.pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Expected pickle to contain a mapping of key -> sample dict.")

    metadata_rows = []
    patient_counts: MutableMapping[str, int] = defaultdict(int)
    skipped_existing = 0

    items = payload.items()
    for key, sample in tqdm(items, total=len(payload), desc="Processing entries"):
        if not isinstance(sample, MutableMapping):
            raise TypeError(f"Sample {key} is not a mapping.")
        series_uid = str(sample.get("series_uid", "")).strip()
        if not series_uid:
            raise KeyError(f"Sample {key} missing 'series_uid'.")
        subject_id = series_to_subject.get(series_uid)
        if subject_id is None:
            raise KeyError(f"Series UID {series_uid} not found in metadata.")
        patient_code = format_patient_code(subject_id)
        image = np.asarray(sample.get("image"))
        if image.shape != (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE):
            raise ValueError(
                f"Sample {key} image has shape {image.shape}, expected {(BASE_IMAGE_SIZE, BASE_IMAGE_SIZE)}."
            )
        masks = sample.get("masks")
        if not isinstance(masks, Sequence):
            raise TypeError(f"Sample {key} masks entry must be a sequence.")
        if len(masks) != RATER_COUNT:
            raise ValueError(f"Sample {key} expected {RATER_COUNT} masks, found {len(masks)}.")
        mask_arrays = [np.asarray(m, dtype=np.uint8) for m in masks]
        mask_shapes = {mask.shape for mask in mask_arrays}
        if mask_shapes != {(BASE_IMAGE_SIZE, BASE_IMAGE_SIZE)}:
            raise ValueError(f"Sample {key} mask shapes inconsistent: {mask_shapes}.")

        entry_idx = patient_counts[patient_code]
        stem, wrote = save_entry(
            image=image,
            masks=mask_arrays,
            patient_code=patient_code,
            patient_label=subject_id,
            idx=entry_idx,
            image_size=args.image_size,
            images_dir=images_dir,
            labels_dir=labels_dir,
            overwrite=args.overwrite,
        )
        patient_counts[patient_code] += 1
        if wrote:
            metadata_rows.append({
                "image_name": f"{stem}.npy",
                "patient_id": subject_id,
            })
        else:
            skipped_existing += 1

    if not metadata_rows:
        raise RuntimeError("No new samples were written; nothing to store in metadata.")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = dataset_root / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(
        f"Wrote {len(metadata_rows)} samples to {dataset_root} (skipped {skipped_existing} existing files)."
    )
    print(f"Metadata saved to {metadata_path}.")


if __name__ == "__main__":
    main()
