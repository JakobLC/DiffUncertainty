"""
File to visualize the ood augmentations for the retina dataset.
Loads random retina128/64 images and applies custom albumentations augmentations,
displaying them in a grid with rows for: no augment, fov, flash, blur.

Augmentation parameters are hardcoded at the top for easy experimentation.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A

# Add paths for imports
values_dir = Path(__file__).resolve().parents[2]  # /home/jloch/Desktop/diff/luzern/values
sys.path.insert(0, str(values_dir))

from uncertainty_modeling.data.lidc2d_dataset import MultiRater2DDataset
from uncertainty_modeling.augmentations import FieldOfViewCircularMask, FlashArtifact


# ============================================================================
# HARDCODED PARAMETERS - Change these to experiment with augmentations
# ============================================================================

# Dataset parameters
IMAGE_SIZE = 64  # 64 or 128
NUM_IMAGES = 10  # Number of images to display
SEED = 42

# FOV augmentation parameters
FOV_RADIUS = [1.0,1.5]  # fraction of image side length, or (min, max) tuple
FOV_EDGE_BLUR = 0.01  # fraction of image side length, or (min, max) tuple
FOV_CIRCLE_DIST = [0.1,0.3]  # distance from image center to circle perimeter, or (min, max) tuple

# Flash augmentation parameters
FLASH_ADDITIVE = False  # True = additive, False = multiplicative
FLASH_ADDITIVE_RANGE = (-0.3, 1.0)  # if FLASH_ADDITIVE=True
FLASH_MULTIPLICATIVE_RANGE = (0.2, 1.5)  # if FLASH_ADDITIVE=False
FLASH_SIZE = 0.5  # (a+b)/2 as fraction of image
FLASH_SHARPNESS = 4  # Higher = sharper edge
FLASH_ECCENTRICITY = [0.2,0.5]  # 0 to 1, or (min, max) tuple
FLASH_CENTER_SHIFT = (0, 0.3)  # distance from center, or (min, max) tuple

# Blur augmentation parameters (Gaussian blur from albumentations)
BLUR_SIGMA_LIMIT = (3.0, 7.0)  # Blur strength range


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_img_for_plot(img_hwc: np.ndarray) -> np.ndarray:
    """Normalize image for plotting to [0, 1] range."""
    img = np.asarray(img_hwc, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    if img.shape[-1] == 1:
        channel = img[..., 0]
        mn, mx = float(np.min(channel)), float(np.max(channel))
        if mx > mn:
            channel = (channel - mn) / (mx - mn)
        return np.clip(channel, 0.0, 1.0)
    
    img3 = img[..., :3]
    mn, mx = float(np.min(img3)), float(np.max(img3))
    if mx > mn:
        img3 = (img3 - mn) / (mx - mn)
    return np.clip(img3, 0.0, 1.0)


def _build_dataset(data_input_dir: str, image_size: int, split: str = "id_test"):
    """Build the retina dataset."""
    splits_path = Path(data_input_dir) / "splits" / "random" / "firstCycle" / "splits.pkl"
    
    dataset = MultiRater2DDataset(
        splits_path=str(splits_path),
        base_dir=data_input_dir,
        dataset_label=f"retina{image_size}",
        num_raters=None,
    )
    
    # Filter to the desired split
    split_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_split = sample.get("split", None)
        if sample_split == split:
            split_indices.append(idx)
    
    return dataset, split_indices


def _sample_indices(indices_list: list, batch_size: int, seed: int):
    """Sample random indices from available split."""
    n = min(max(1, batch_size), len(indices_list))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(indices_list), size=n, replace=False)
    return [indices_list[int(i)] for i in sampled]


def main():
    """Main visualization function."""
    
    # Setup dataset path
    if IMAGE_SIZE == 64:
        data_input_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/retina64"
    elif IMAGE_SIZE == 128:
        data_input_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/retina128"
    else:
        raise ValueError(f"IMAGE_SIZE must be 64 or 128, got {IMAGE_SIZE}")
    
    print(f"Loading retina{IMAGE_SIZE} dataset from {data_input_dir}...")
    dataset, split_indices = _build_dataset(data_input_dir, IMAGE_SIZE, split="id_test")
    
    if len(split_indices) == 0:
        print("No images found in id_test split, using all available images")
        split_indices = list(range(len(dataset)))
    
    print(f"Found {len(split_indices)} images in dataset")
    
    # Sample random images
    sampled_indices = _sample_indices(split_indices, NUM_IMAGES, SEED)
    print(f"Sampled {len(sampled_indices)} images")
    
    # Create augmentation transforms
    fov_transform = FieldOfViewCircularMask(
        radius=FOV_RADIUS,
        edge_blur=FOV_EDGE_BLUR,
        circle_dist=FOV_CIRCLE_DIST,
        always_apply=True,
        p=1.0,
    )
    
    flash_transform = FlashArtifact(
        additive=FLASH_ADDITIVE,
        additive_range=FLASH_ADDITIVE_RANGE,
        multiplicative_range=FLASH_MULTIPLICATIVE_RANGE,
        size=FLASH_SIZE,
        sharpness=FLASH_SHARPNESS,
        eccentricity=FLASH_ECCENTRICITY,
        center_shift=FLASH_CENTER_SHIFT,
        always_apply=True,
        p=1.0,
    )
    
    blur_transform = A.GaussianBlur(
        sigma_limit=BLUR_SIGMA_LIMIT,
        always_apply=True,
        p=1.0,
    )
    
    # Load and transform images
    original_images = []
    fov_images = []
    flash_images = []
    blur_images = []
    image_ids = []
    
    for idx in sampled_indices:
        sample = dataset[idx]
        img = sample["data"]
        
        # Convert from CHW tensor to HWC numpy
        if isinstance(img, torch.Tensor):
            img_hwc = img.permute(1, 2, 0).numpy().astype(np.float32)
        else:
            img_hwc = np.asarray(img, dtype=np.float32)
        
        # For augmentations, convert to uint8 [0, 255] if needed
        if img_hwc.max() <= 1.0:
            # Image is in [0, 1] range, scale to [0, 255]
            img_uint8 = (img_hwc * 255.0).astype(np.uint8)
        else:
            # Already in [0, 255] range
            img_uint8 = img_hwc.astype(np.uint8)
        
        image_ids.append(str(sample.get("image_id", idx)))
        
        # Store original (normalized for plotting)
        original_images.append(_normalize_img_for_plot(img_hwc))
        
        # Apply augmentations to uint8 version
        fov_img = fov_transform(image=img_uint8)["image"]
        fov_images.append(_normalize_img_for_plot(fov_img))
        
        flash_img = flash_transform(image=img_uint8)["image"]
        flash_images.append(_normalize_img_for_plot(flash_img))
        
        blur_img = blur_transform(image=img_uint8)["image"]
        blur_images.append(_normalize_img_for_plot(blur_img))
    
    # Create visualization grid
    n_cols = len(sampled_indices)
    n_rows = 4  # original, fov, flash, blur
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    
    row_data = [
        ("Original", original_images),
        ("FOV Mask", fov_images),
        ("Flash Artifact", flash_images),
        ("Gaussian Blur", blur_images),
    ]
    
    for r, (row_name, row_images) in enumerate(row_data):
        for c in range(n_cols):
            ax = axes[r, c]
            img = row_images[c]
            
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img, vmin=0.0, vmax=1.0)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if c == 0:
                ax.set_ylabel(row_name, fontsize=11, fontweight="bold")
            if r == 0:
                ax.set_title(f"{image_ids[c]}", fontsize=9)
    
    fig.tight_layout()
    print("Displaying visualization...")
    plt.show()


if __name__ == "__main__":
    main()