#!/usr/bin/env python3
"""Rename uncertainty folders under values/saves/*/*/test_results/*/*/*/[unc].

Example usage:
    python values/global_utils/rename_unc.py --dry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

UNC_RENAMES = {
    "aleatoric_uncertainty": "AU",
    "epistemic_uncertainty": "EU",
    "pred_entropy": "TU",
}

def iter_unc_dirs(saves_root: Path) -> Iterable[Path]:
    """Yield directories matching the required pattern with rename candidates."""
    pattern = saves_root.glob("*/test_results/*/*/*/*")
    for path in pattern:
        if path.is_dir() and path.name in UNC_RENAMES:
            yield path


def rename_dirs(saves_root: Path, dry_run: bool) -> None:
    any_matches = False
    for src in iter_unc_dirs(saves_root):
        any_matches = True
        dst = src.with_name(UNC_RENAMES[src.name])
        action = "Would rename" if dry_run else "Renaming"
        print(f"{action}: {src} -> {dst}")
        if dry_run:
            continue
        if dst.exists():
            print(f"  Skipping because destination already exists: {dst}", file=sys.stderr)
            continue
        src.rename(dst)
    if not any_matches:
        print("No matching uncertainty folders found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename uncertainty folders to AU/EU/TU")
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
        help="Show which folders would be renamed without making changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")
    rename_dirs(root, args.dry)


if __name__ == "__main__":
    main()
