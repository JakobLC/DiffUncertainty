from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


LIDC_SPLITS = ["id", "val", "ood_noise", "ood_blur", "ood_jpeg", "ood_contrast"]
CHAKSU_SPLITS = ["id", "val", "ood"]


def infer_dataset_splits(exp_name: str) -> List[str] | None:
    exp = exp_name.lower()
    if "lidc" in exp:
        return list(LIDC_SPLITS)
    if "chaksu" in exp:
        return list(CHAKSU_SPLITS)
    return None


def infer_required_unc_dirs(version_name: str) -> List[str]:
    # Softmax runs only produce predictive uncertainty (TU).
    if "softmax" in version_name.lower():
        return ["TU"]
    return ["TU", "AU", "EU"]


def list_results_roots(exp_dir: Path) -> List[Path]:
    roots = [p for p in sorted(exp_dir.iterdir()) if p.is_dir() and p.name.startswith("test_results")]
    preferred = [p for p in roots if p.name == "test_results"]
    return preferred + [p for p in roots if p.name != "test_results"]


def list_versions(results_root: Path) -> List[Tuple[str, Path]]:
    versions: List[Tuple[str, Path]] = []
    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        for version_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            versions.append((f"{model_dir.name}/{version_dir.name}", version_dir))
    return versions


def expected_split_dirs(version_dir: Path, exp_name: str) -> List[Path]:
    split_names = infer_dataset_splits(exp_name)
    if split_names is None:
        # Fallback for unknown datasets: use discovered split folders.
        return [p for p in sorted(version_dir.iterdir()) if p.is_dir()]
    return [version_dir / split for split in split_names]


def is_missing(version_name: str, version_dir: Path, exp_name: str) -> bool:
    required_unc = infer_required_unc_dirs(version_name)
    required_folders = ["pred_seg"] + required_unc
    for split_dir in expected_split_dirs(version_dir, exp_name):
        for folder in required_folders:
            if not (split_dir / folder).is_dir():
                return True
    return False


def required_eval_jsons(version_name: str, version_dir: Path, exp_name: str) -> List[Path]:
    required_unc = infer_required_unc_dirs(version_name)
    paths: List[Path] = [
        version_dir / "quantile_analysis.json",
        version_dir / "threshold_analysis.json",
        version_dir / "ood_detection.json",
    ]

    split_names = infer_dataset_splits(exp_name)
    if split_names is None:
        split_names = [p.name for p in expected_split_dirs(version_dir, exp_name)]

    for split in split_names:
        split_dir = version_dir / split
        paths.append(split_dir / "area.json")
        for unc in required_unc:
            paths.append(split_dir / f"aggregated_{unc}.json")

    # Calibration + ambiguity are expected on non-val splits.
    for split in split_names:
        if split == "val":
            continue
        split_dir = version_dir / split
        paths.append(split_dir / "calibration.json")
        paths.append(split_dir / "ambiguity_modeling.json")

    return paths


def is_finished(version_name: str, version_dir: Path, exp_name: str) -> bool:
    for path in required_eval_jsons(version_name, version_dir, exp_name):
        if not path.is_file():
            return False
    return True


def classify_versions(exp_dir: Path, exp_name: str) -> Dict[str, Dict[str, object]]:
    statuses: Dict[str, Dict[str, object]] = {}
    for results_root in list_results_roots(exp_dir):
        for version_name, version_dir in list_versions(results_root):
            missing = is_missing(version_name, version_dir, exp_name)
            finished = is_finished(version_name, version_dir, exp_name)
            key = f"{results_root.name}/{version_name}"
            model_name = version_name.split("/")[0]
            statuses[key] = {
                "missing": missing,
                "finished": finished,
                "model_name": model_name,
            }
    return statuses


def print_matrix(statuses: Dict[str, Dict[str, object]]) -> None:
    matrix = Counter(
        (bool(s["finished"]), bool(s["missing"])) for s in statuses.values()
    )
    print("Matrix (rows=finished, cols=missing)")
    print("                 missing=False  missing=True")
    print(
        "finished=False"
        f"      {matrix[(False, False)]:>6}"
        f"         {matrix[(False, True)]:>6}"
    )
    print(
        "finished=True "
        f"      {matrix[(True, False)]:>6}"
        f"         {matrix[(True, True)]:>6}"
    )


def print_verbose(statuses: Dict[str, Dict[str, object]]) -> None:
    grouped: Dict[Tuple[bool, bool], set[str]] = defaultdict(set)
    for status in statuses.values():
        grouped[(bool(status["missing"]), bool(status["finished"]))].add(
            str(status["model_name"])
        )

    order = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    for key in order:
        missing, finished = key
        print()
        print(f"Group missing={missing}, finished={finished}")
        for model_name in sorted(grouped.get(key, set())):
            print(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify evaluated versions into missing/finished groups and print a matrix."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment folder under values/saves, e.g. origlidc128 or chaksu128.",
    )
    parser.add_argument(
        "--base-path",
        default="/home/jloch/Desktop/diff/luzern/values/saves",
        help="Base path containing experiment folders.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sorted version names (one per line) for each matrix group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.base_path).expanduser() / args.exp
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    statuses = classify_versions(exp_dir, args.exp)
    if not statuses:
        print(f"No versions found under {exp_dir} (expected test_results*/<model>/<version>).")
        return

    print_matrix(statuses)
    if args.verbose:
        print_verbose(statuses)


if __name__ == "__main__":
    main()
