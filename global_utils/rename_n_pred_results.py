"""Rename evaluation result folders to include n_pred in the version name.

Example:
  test_results1/prob_unet_ensemble_0/e500_ema
  -> test_results1/prob_unet_ensemble_0_n1/e500_ema
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_N_PRED = ["1", "2", "3", "6", "18", "32", "56"]


def _parse_n_pred(value: str) -> list[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return tokens or list(DEFAULT_N_PRED)


def _rename_for_n_pred(exp_path: Path, n_pred_values: list[str], dry_run: bool) -> None:
    for n_pred in n_pred_values:
        root = exp_path / f"test_results{n_pred}"
        if not root.is_dir():
            print(f"[skip] Missing directory: {root}")
            continue
        for e500_dir in sorted(root.glob("*/e500_ema")):
            version_dir = e500_dir.parent
            if f"_n{n_pred}" in version_dir.name:
                continue
            new_dir = version_dir.with_name(f"{version_dir.name}_n{n_pred}")
            if new_dir.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing directory: {new_dir}"
                )
            if dry_run:
                print(f"[dry-run] {version_dir} -> {new_dir}")
            else:
                shutil.move(version_dir, new_dir)
                print(f"[moved] {version_dir} -> {new_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename evaluation result folders to include n_pred in the version name."
    )
    parser.add_argument(
        "--exp-path",
        type=Path,
        default=Path("/home/jloch/Desktop/diff/luzern/values/saves/chaksu128"),
        help="Path to the experiment root containing test_results* folders.",
    )
    parser.add_argument(
        "--n-pred",
        type=_parse_n_pred,
        default=list(DEFAULT_N_PRED),
        help="Comma-separated n_pred values to process (default: 1,2,3,6,18,32,56).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying anything.",
    )
    args = parser.parse_args()
    exp_path = args.exp_path.expanduser().resolve()
    if not exp_path.is_dir():
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")
    _rename_for_n_pred(exp_path, args.n_pred, args.dry_run)


if __name__ == "__main__":
    main()
