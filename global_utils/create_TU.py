#!/usr/bin/env python3
"""Create TU folders (AU + EU) wherever AU and EU exist but TU is missing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

UNC_FOLDER_NAMES = ("AU", "EU", "TU")


def iter_split_dirs(root: Path):
    pattern = root.glob("*/test_results/*/*/*")
    for split_dir in pattern:
        if not split_dir.is_dir():
            continue
        au_dir = split_dir / "AU"
        eu_dir = split_dir / "EU"
        tu_dir = split_dir / "TU"
        if au_dir.is_dir() and eu_dir.is_dir() and not tu_dir.exists():
            yield split_dir


def list_tiff_files(directory: Path) -> list[str]:
    files = sorted([p.name for p in directory.glob("*.tif") if p.is_file()])
    if not files:
        raise ValueError(f"No .tif files found in {directory}")
    return files


def ensure_matching_files(au_dir: Path, eu_dir: Path) -> list[str]:
    au_files = list_tiff_files(au_dir)
    eu_files = list_tiff_files(eu_dir)
    if au_files != eu_files:
        mismatch = set(au_files).symmetric_difference(eu_files)
        raise ValueError(
            f"File name mismatch between {au_dir} and {eu_dir}. Differing entries: {sorted(mismatch)}"
        )
    return au_files


def sum_images(au_path: Path, eu_path: Path) -> np.ndarray:
    au_img = tifffile.imread(au_path)
    eu_img = tifffile.imread(eu_path)
    if au_img.shape != eu_img.shape:
        raise ValueError(f"Image shape mismatch: {au_path} vs {eu_path}")
    result = au_img.astype(np.float32) + eu_img.astype(np.float32)
    if np.issubdtype(au_img.dtype, np.integer):
        info = np.iinfo(au_img.dtype)
        result = np.clip(result, info.min, info.max)
    return result.astype(au_img.dtype, copy=False)


def process_split(split_dir: Path, dry_run: bool) -> None:
    au_dir = split_dir / "AU"
    eu_dir = split_dir / "EU"
    tu_dir = split_dir / "TU"
    files = ensure_matching_files(au_dir, eu_dir)
    if dry_run:
        print(f"[DRY] Would create {tu_dir} ({len(files)} files)")
        return

    try:
        tu_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Destination already exists: {tu_dir}") from None

    progress = tqdm(files, desc=f"Creating TU for {split_dir}", unit="file")
    for filename in progress:
        au_file = au_dir / filename
        eu_file = eu_dir / filename
        tu_file = tu_dir / filename
        tu_img = sum_images(au_file, eu_file)
        tifffile.imwrite(tu_file, tu_img)
    progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TU folders by summing AU and EU tif files")
    default_root = Path(__file__).resolve().parent.parent / "saves"
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory containing the saves folders (default: %(default)s)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Preview matching folders without creating TU files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    matches = list(iter_split_dirs(root))
    if not matches:
        print("No splits with AU/EU but missing TU were found.")
        return

    for split_dir in matches:
        try:
            process_split(split_dir, args.dry)
        except Exception as exc:
            print(f"Error while processing {split_dir}: {exc}", file=sys.stderr)
            if not args.dry:
                raise


if __name__ == "__main__":
    main()
