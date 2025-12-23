#!/usr/bin/env python3
"""
Recursively convert .tif/.tiff to .png *in place* (delete the TIFF) iff:
- Data are integer-valued (no floats, no fractional values)
- All values are within [0, 255]
- Output is written as 8-bit PNG (L, LA, RGB, RGBA) depending on channels

Requires: tifffile (given), plus Pillow (PIL) for PNG writing.
If Pillow is missing, install it or swap out the writer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import tqdm
import tifffile as tiff

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "This script needs Pillow for saving PNGs. "
        "Install it (pip install pillow) or replace the save_png() implementation."
    ) from e

import numpy as np


def _is_integer_valued(a: np.ndarray) -> bool:
    """True if array values are all integers (handles floats that are actually integral)."""
    if np.issubdtype(a.dtype, np.integer):
        return True
    if np.issubdtype(a.dtype, np.floating):
        # Fast rejection: any NaN/Inf => not safe.
        if not np.isfinite(a).all():
            return False
        # Check integral-ness without huge memory.
        # If any value differs from its nearest integer -> fractional.
        return np.equal(a, np.rint(a)).all()
    return False


def _in_uint8_range(a: np.ndarray) -> bool:
    """True if min/max are within 0..255."""
    # Use nanmin/nanmax for safety, but NaNs are rejected earlier.
    mn = float(np.min(a))
    mx = float(np.max(a))
    return (mn >= 0.0) and (mx <= 255.0)


def _squeeze_tiff_shape(a: np.ndarray) -> np.ndarray:
    """
    Normalize common TIFF shapes to something PIL understands:
      - (H, W) -> grayscale
      - (H, W, C) where C in {1,2,3,4} -> L/LA/RGB/RGBA
    Rejects other shapes (volumes, time series, etc.).
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (1, 2, 3, 4):
        return a
    # Some TIFFs come as (1, H, W) or (H, W, 1) already handled above.
    if a.ndim == 3 and a.shape[0] == 1:
        a2 = a[0]
        if a2.ndim == 2:
            return a2
    raise ValueError(f"Unsupported TIFF shape for PNG conversion: {a.shape}")


def _to_uint8(a: np.ndarray) -> np.ndarray:
    """Convert integer-valued array to uint8 safely (assumes range checked)."""
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.uint8, copy=False)
    # float but integral-valued -> round to int then cast
    return np.rint(a).astype(np.uint8)


def _save_png_uint8(arr_uint8: np.ndarray, out_path: Path) -> None:
    """
    Save uint8 array as PNG via Pillow.
    - (H,W) -> L
    - (H,W,2) -> LA
    - (H,W,3) -> RGB
    - (H,W,4) -> RGBA
    """
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3:
        c = arr_uint8.shape[2]
        if c == 1:
            img = Image.fromarray(arr_uint8[:, :, 0], mode="L")
        elif c == 2:
            img = Image.fromarray(arr_uint8, mode="LA")
        elif c == 3:
            img = Image.fromarray(arr_uint8, mode="RGB")
        elif c == 4:
            img = Image.fromarray(arr_uint8, mode="RGBA")
        else:
            raise ValueError(f"Unsupported channel count for PNG: {c}")
    else:
        raise ValueError(f"Unsupported array ndim for PNG: {arr_uint8.ndim}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Optimize is safe; compress_level is 0..9 (Pillow), but keep default.
    img.save(out_path, format="PNG", optimize=True)


def convert_tiffs_to_png_inplace(root: Path, *, delete_original: bool = True, dry_run: bool = False) -> dict:
    """
    Walk root recursively, and for each .tif/.tiff:
      - Load via tifffile
      - If values are integer-valued and within [0,255] AND shape is PNG-compatible,
        write sibling .png and (optionally) delete original tiff.

    Returns a summary dict.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    summary = {
        "root": str(root),
        "seen": 0,
        "converted": 0,
        "skipped_not_uint8_range_or_integral": 0,
        "skipped_unsupported_shape": 0,
        "skipped_read_error": 0,
        "skipped_existing_png": 0,
        "errors_delete": 0,
    }
    files = list(root.rglob("*.tif"))+list(root.rglob("*.tiff"))
    for p in tqdm.tqdm(files):
        summary["seen"] += 1
        out_png = p.with_suffix(".png")
        if out_png.exists():
            summary["skipped_existing_png"] += 1
            continue

        try:
            arr = tiff.imread(str(p))
        except Exception:
            summary["skipped_read_error"] += 1
            continue

        try:
            arr = _squeeze_tiff_shape(arr)
        except Exception:
            summary["skipped_unsupported_shape"] += 1
            continue

        if not _is_integer_valued(arr) or not _in_uint8_range(arr):
            summary["skipped_not_uint8_range_or_integral"] += 1
            continue

        arr_u8 = _to_uint8(arr)

        if dry_run:
            summary["converted"] += 1
            continue

        _save_png_uint8(arr_u8, out_png)
        summary["converted"] += 1

        if delete_original:
            try:
                p.unlink()
            except Exception:
                summary["errors_delete"] += 1

    return summary


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Convert eligible TIFFs to PNG recursively (in place).")
    ap.add_argument("--root", type=str, help="Root folder to scan recursively", default="/data/chaksu")
    ap.add_argument("--keep-tif", action="store_true", help="Do not delete original TIFF after writing PNG")
    ap.add_argument("--dry-run", action="store_true", help="Scan and count conversions without writing files")
    args = ap.parse_args(argv)

    summary = convert_tiffs_to_png_inplace(
        Path(args.root),
        delete_original=not args.keep_tif,
        dry_run=args.dry_run,
    )

    # Minimal, machine-friendly summary
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
