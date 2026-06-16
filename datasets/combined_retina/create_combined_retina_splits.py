"""Create random splits for combined retina datasets (Chaksu, Refuge, Riga).

This script combines three retina datasets (chaksu64, refuge64, riga64) into a 
single unified dataset for cross-dataset training and evaluation. Each dataset 
has different numbers of ground truth annotations:
- Chaksu: 5 annotators per image
- Refuge: 7 annotators per image
- Riga: 6 annotators per image

The combined dataset uses prefixed paths to disambiguate between datasets:
- "chaksu64/t_000000.npy" for a Chaksu image
- "refuge64/test_000800.npy" for a Refuge image
- "riga64/bin_000000.npy" for a Riga image

Splits are created uniformly at random (not scanner-based):
- train: 64%
- val: 16%
- id_test: 20%
- ood_fov: 20% (identical to id_test, designed for FOV augmentation-based OOD)
- ood_flash: 20% (identical to id_test, designed for flash augmentation-based OOD)
- ood_blur: 20% (identical to id_test, designed for blur augmentation-based OOD)
- id_unlabeled_pool: empty
- ood_unlabeled_pool: empty

The resulting splits.pkl is a 1-element list (no cross-validation) saved to:
values_datasets/retina64/splits/random/firstCycle/splits.pkl

Processing requirements:
- All three datasets must be preprocessed (preprocessed/images and preprocessed/labels)
- Each image must have the correct number of GT masks for its dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Hardcoded seed for reproducibility
SEED = 42

# Base dataset configurations with their rater counts and naming patterns
BASE_DATASET_CONFIGS = {
    "chaksu": {
        "num_raters": 5,
        "rater_pattern": "{base_id}_{rater:02d}_mask.npy",
    },
    "refuge": {
        "num_raters": 7,
        "rater_pattern": "{base_id}_{rater:02d}_mask.npy",
    },
    "riga": {
        "num_raters": 6,
        "rater_pattern": "{base_id}_{rater:02d}_mask.npy",
    },
}

DATASETS_ROOT = Path("/home/jloch/Desktop/diff/luzern/values_datasets")


def get_dataset_configs(image_size: int) -> dict:
    """Build dataset configurations for a given image size.
    
    Parameters:
    -----------
    image_size : int
        Image resolution (e.g., 64, 128)
    
    Returns:
    --------
    dict : Dataset configurations with versioned dataset names (e.g., chaksu64, chaksu128)
    """
    return {
        f"{name}{image_size}": config
        for name, config in BASE_DATASET_CONFIGS.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random splits for combined retina datasets",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Image resolution (default: 64, e.g., 64, 128)",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default=None,
        help="Name of the output combined dataset (default: retina{image_size})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )


def validate_dataset_preprocessing(dataset_configs: dict) -> Dict[str, Path]:
    """Validate that all source datasets are preprocessed.
    
    Parameters:
    -----------
    dataset_configs : dict
        Dataset configurations (e.g., from get_dataset_configs())
    
    Returns:
    --------
    dict : Mapping of dataset_name -> dataset_root
    """
    validated = {}
    for dataset_name in dataset_configs.keys():
        dataset_root = (DATASETS_ROOT / dataset_name).expanduser().resolve()
        processed_root = dataset_root / "preprocessed"
        if not processed_root.is_dir():
            base_name = dataset_name.rstrip('0123456789')
            raise FileNotFoundError(
                f"Preprocessed directory not found for {dataset_name}: {processed_root}. "
                f"Did you run process_{base_name}.py?"
            )
        validated[dataset_name] = dataset_root
    return validated


def collect_samples_from_dataset(
    dataset_name: str,
    dataset_root: Path,
    dataset_configs: dict,
) -> List[tuple[str, str, int]]:
    """Collect (prefixed_path, base_id, num_raters) for all samples in a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., "chaksu64")
    dataset_root : Path
        Root directory of the dataset
    dataset_configs : dict
        Dataset configurations
    
    Returns:
    --------
    List[tuple] : List of (prefixed_path, base_id, num_raters) tuples
        - prefixed_path: e.g., "chaksu64/t_000000.npy"
        - base_id: e.g., "t_000000"
        - num_raters: e.g., 5
    """
    image_dir = dataset_root / "preprocessed" / "images"
    label_dir = dataset_root / "preprocessed" / "labels"
    config = dataset_configs[dataset_name]
    num_raters = config["num_raters"]
    rater_pattern = config["rater_pattern"]
    
    samples: List[tuple[str, str, int]] = []
    
    for image_file in sorted(os.listdir(image_dir)):
        if not image_file.endswith(".npy"):
            continue
        
        base_id = os.path.splitext(image_file)[0]
        
        # Verify all rater masks exist
        label_paths = []
        for rater in range(num_raters):
            label_name = rater_pattern.format(base_id=base_id, rater=rater)
            label_path = label_dir / label_name
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing rater mask: {label_path}"
                )
            label_paths.append(label_path)
        
        # Create prefixed path for this dataset
        prefixed_path = f"{dataset_name}/{image_file}"
        samples.append((prefixed_path, base_id, num_raters))
    
    if not samples:
        raise RuntimeError(f"No samples found in {dataset_name}")
    
    logging.info(f"Collected {len(samples)} samples from {dataset_name}")
    return samples


def create_random_split(
    all_samples: List[tuple[str, str, int]],
    dataset_configs: dict,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    seed: int = SEED,
) -> dict:
    """Create random train/val/id_test/ood_fov splits from all samples."""
    np.random.seed(seed)
    
    total = len(all_samples)
    indices = np.random.permutation(total)
    
    train_count = int(np.ceil(total * train_ratio))
    val_count = int(np.ceil(total * val_ratio))
    test_count = int(np.ceil(total * test_ratio))
    
    # Adjust counts to ensure they sum to total
    test_count = total - train_count - val_count
    
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]
    
    # Extract paths from samples
    train_paths = np.array([all_samples[i][0] for i in train_indices])
    val_paths = np.array([all_samples[i][0] for i in val_indices])
    test_paths = np.array([all_samples[i][0] for i in test_indices])
    
    # ood_fov is identical to id_test (for FOV augmentation-based OOD generation)
    ood_fov_paths = test_paths.copy()
    # ood_flash is identical to id_test (for flash augmentation-based OOD generation)
    ood_flash_paths = test_paths.copy()
    # ood_blur is identical to id_test (for blur augmentation-based OOD generation)
    ood_blur_paths = test_paths.copy()
    
    # Empty pools
    empty = np.array([], dtype=object)
    
    fold_dict = {
        "train": train_paths,
        "val": val_paths,
        "id_test": test_paths,
        "ood_fov": ood_fov_paths,
        "ood_flash": ood_flash_paths,
        "ood_blur": ood_blur_paths,
        "id_unlabeled_pool": empty,
        "ood_unlabeled_pool": empty,
        "_meta": {
            "schema": "combined_retina_random",
            "datasets": list(dataset_configs.keys()),
            "dataset_configs": {
                name: {
                    "num_raters": config["num_raters"],
                }
                for name, config in dataset_configs.items()
            },
            "split_ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "id_test": test_ratio,
                "ood_fov": test_ratio,
                "ood_flash": test_ratio,
                "ood_blur": test_ratio,
            },
            "seed": seed,
        },
    }
    
    logging.info(
        f"Created split: {len(train_paths)} train, {len(val_paths)} val, "
        f"{len(test_paths)} id_test, {len(ood_fov_paths)} ood_fov, "
        f"{len(ood_flash_paths)} ood_flash, {len(ood_blur_paths)} ood_blur"
    )
    
    return fold_dict


def ensure_output_path(output_path: Path) -> Path:
    """Ensure output path ends with splits.pkl and create parent directories."""
    if output_path.suffix != ".pkl":
        output_path = output_path / "splits.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    
    # Get dataset configurations for the specified image size
    dataset_configs = get_dataset_configs(args.image_size)
    
    # Determine output dataset name
    output_dataset = args.output_dataset or f"retina{args.image_size}"
    
    logging.info(f"Processing datasets at resolution {args.image_size}x{args.image_size}")
    logging.info(f"Output dataset: {output_dataset}")
    
    # Validate source datasets
    logging.info("Validating source datasets...")
    validated_datasets = validate_dataset_preprocessing(dataset_configs)
    
    # Collect samples from all datasets
    logging.info("Collecting samples from all datasets...")
    all_samples: List[tuple[str, str, int]] = []
    for dataset_name in sorted(dataset_configs.keys()):
        samples = collect_samples_from_dataset(
            dataset_name, validated_datasets[dataset_name], dataset_configs
        )
        all_samples.extend(samples)
    
    total_samples = len(all_samples)
    logging.info(f"Total combined samples: {total_samples}")
    
    # Create splits
    logging.info("Creating random splits...")
    fold_dict = create_random_split(all_samples, dataset_configs, seed=SEED)
    
    # Prepare output
    output_root = (DATASETS_ROOT / output_dataset).expanduser().resolve()
    output_path = output_root / "splits" / "random" / "firstCycle" / "splits.pkl"
    output_path = ensure_output_path(output_path)
    
    # Save as 1-element list (no cross-validation)
    splits = [fold_dict]
    
    with output_path.open("wb") as handle:
        pickle.dump(splits, handle)
    
    logging.info(f"Saved splits to {output_path}")
    logging.info(
        f"Split configuration:\n"
        f"  train:   {len(fold_dict['train'])} ({100*len(fold_dict['train'])/total_samples:.1f}%)\n"
        f"  val:     {len(fold_dict['val'])} ({100*len(fold_dict['val'])/total_samples:.1f}%)\n"
        f"  id_test: {len(fold_dict['id_test'])} ({100*len(fold_dict['id_test'])/total_samples:.1f}%)\n"
        f"  ood_fov: {len(fold_dict['ood_fov'])} ({100*len(fold_dict['ood_fov'])/total_samples:.1f}%)\n"
        f"  ood_flash: {len(fold_dict['ood_flash'])} ({100*len(fold_dict['ood_flash'])/total_samples:.1f}%)\n"
        f"  ood_blur: {len(fold_dict['ood_blur'])} ({100*len(fold_dict['ood_blur'])/total_samples:.1f}%)"
    )


if __name__ == "__main__":
    main()
