"""
File to visualize the OOD augmentations for the NPC dataset.
Loads random NPC preprocessed images and applies custom albumentations augmentations,
displaying them in a grid with rows for: no augment, blur, flash, pixelwise noise, filtered noise.

Augmentation parameters are hardcoded at the top for easy experimentation.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# Add paths for imports
values_dir = Path(__file__).resolve().parents[2]  # /home/jloch/Desktop/diff/luzern/values
sys.path.insert(0, str(values_dir))

from uncertainty_modeling.augmentations import FlashArtifact, FilteredImageNoise


# ============================================================================
# HARDCODED PARAMETERS - Change these to experiment with augmentations
# ============================================================================

# Dataset parameters
IMAGE_SIZE = 128  # 64 or 128
NUM_IMAGES = 5  # Number of images to display
SEED = 42

# Blur augmentation parameters (Gaussian blur from albumentations)
BLUR_SIGMA_LIMIT_BASE = (3.0, 7.0)  # For 128x128; will be halved for 64x64

# Flash augmentation parameters
FLASH_ADDITIVE = False  # True = additive, False = multiplicative
FLASH_ADDITIVE_RANGE = (-0.3, 1.0)  # if FLASH_ADDITIVE=True
FLASH_MULTIPLICATIVE_RANGE = (0.2, 1.5)  # if FLASH_ADDITIVE=False
FLASH_SIZE = 0.5  # (a+b)/2 as fraction of image
FLASH_SHARPNESS = 4  # Higher = sharper edge
FLASH_ECCENTRICITY = (0.2, 0.5)  # 0 to 1, or (min, max) tuple
FLASH_CENTER_SHIFT = (0, 0.3)  # distance from center, or (min, max) tuple

# Pixelwise noise parameters
PIXELWISE_NOISE_SCALE = 0.125  # Multiplied by random normal noise

# Filtered noise parameters
FILTERED_NOISE_SCALE = 0.125  # Multiplied by filtered noise
FILTERED_NOISE_SIGMA_BASE = 2.3  # For 128x128; will be halved for 64x64


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


def _load_npc_images(base_dir: str, image_size: int, num_samples: int, seed: int):
	"""Load random NPC preprocessed images."""
	images_dir = Path(base_dir) / "images"
	
	if not images_dir.exists():
		raise FileNotFoundError(f"Images directory not found: {images_dir}")
	
	# Find all image files
	image_files = sorted(images_dir.glob("*.npy"))
	if len(image_files) == 0:
		raise FileNotFoundError(f"No .npy images found in {images_dir}")
	
	# Sample random images
	n = min(max(1, num_samples), len(image_files))
	rng = np.random.default_rng(seed)
	sampled_indices = rng.choice(len(image_files), size=n, replace=False)
	
	images = []
	image_ids = []
	for idx in sampled_indices:
		img_path = image_files[int(idx)]
		img = np.load(img_path).astype(np.float32)
		
		# Ensure it's in [0, 1] range for float images
		if img.max() > 1.0:
			img = np.clip(img / 255.0, 0.0, 1.0)
		
		# Convert to uint8 for augmentations
		img_uint8 = (img * 255.0).astype(np.uint8)
		
		images.append(img_uint8)
		image_ids.append(img_path.stem)
	
	return images, image_ids


class PixelwiseNoise(A.ImageOnlyTransform):
	"""Apply pixelwise Gaussian noise scaled by image intensities."""
	
	def __init__(self, noise_scale: float = 0.125, p: float = 1.0):
		super().__init__(p=p)
		self.noise_scale = float(noise_scale)
	
	def apply(self, img, **params):
		orig_dtype = img.dtype
		img = img.astype(np.float32, copy=True) / 255.0  # Normalize to [0, 1]
		
		# Generate pixelwise Gaussian noise
		noise = np.random.normal(0.0, 1.0, size=img.shape).astype(np.float32)
		
		# Scale noise
		noise_scaled = self.noise_scale * noise
		
		# Multiply by image intensities
		noise_modulated = noise_scaled * img
		
		# Add to image
		result = img + noise_modulated
		
		# Clip and convert back to uint8
		result = np.clip(result, 0.0, 1.0) * 255.0
		return result.astype(orig_dtype)
	
	def get_transform_init_args_names(self):
		return ("noise_scale",)


def main():
	"""Main visualization function."""
	
	# Setup dataset path
	if IMAGE_SIZE == 64:
		data_input_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/npc64/preprocessed"
	elif IMAGE_SIZE == 128:
		data_input_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/npc128/preprocessed"
	else:
		raise ValueError(f"IMAGE_SIZE must be 64 or 128, got {IMAGE_SIZE}")
	
	print(f"Loading NPC{IMAGE_SIZE} dataset from {data_input_dir}...")
	images, image_ids = _load_npc_images(data_input_dir, IMAGE_SIZE, NUM_IMAGES, SEED)
	print(f"Loaded {len(images)} images")
	
	# Scale parameters based on image size (64x64 uses half the values)
	scale_factor = 0.5 if IMAGE_SIZE == 64 else 1.0
	blur_sigma_limit = tuple(s * scale_factor for s in BLUR_SIGMA_LIMIT_BASE)
	filtered_noise_sigma = FILTERED_NOISE_SIGMA_BASE * scale_factor
	
	# Create augmentation transforms
	blur_transform = A.GaussianBlur(
		sigma_limit=blur_sigma_limit,
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
		p=1.0,
	)
	
	pixelwise_noise_transform = PixelwiseNoise(
		noise_scale=PIXELWISE_NOISE_SCALE,
		p=1.0,
	)
	
	filtered_noise_transform = FilteredImageNoise(
		noise_scale=FILTERED_NOISE_SCALE,
		sigma=filtered_noise_sigma,
		p=1.0,
	)
	
	# Load and transform images
	original_images = []
	blur_images = []
	flash_images = []
	pixelwise_noise_images = []
	filtered_noise_images = []
	
	for img_uint8 in images:
		# Store original (convert back to float for plotting)
		img_float = img_uint8.astype(np.float32) / 255.0
		original_images.append(_normalize_img_for_plot(img_float))
		
		# Apply augmentations to uint8 version
		blur_img = blur_transform(image=img_uint8)["image"]
		blur_images.append(_normalize_img_for_plot(blur_img.astype(np.float32) / 255.0))
		
		flash_img = flash_transform(image=img_uint8)["image"]
		flash_images.append(_normalize_img_for_plot(flash_img.astype(np.float32) / 255.0))
		
		pixelwise_img = pixelwise_noise_transform(image=img_uint8)["image"]
		pixelwise_noise_images.append(_normalize_img_for_plot(pixelwise_img.astype(np.float32) / 255.0))
		
		# FilteredImageNoise expects float [0, 1]
		filtered_img = filtered_noise_transform(image=img_float)["image"]
		filtered_noise_images.append(_normalize_img_for_plot(filtered_img))
	
	# Create visualization grid
	n_cols = len(images)
	n_rows = 5  # original, blur, flash, pixelwise noise, filtered noise
	
	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(3.5 * n_cols, 3.5 * n_rows),
		squeeze=False,
	)
	
	row_data = [
		("Original", original_images),
		("Gaussian Blur", blur_images),
		("Flash Artifact", flash_images),
		("Pixelwise Noise", pixelwise_noise_images),
		("Filtered Noise", filtered_noise_images),
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
