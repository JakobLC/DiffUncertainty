"""
Visualize MONAI augmentations for the NPC dataset.
Loads random NPC preprocessed images and applies MONAI transforms,
displaying them in a grid with rows for: original + 6 augmentation types.

Augmentation parameters are hardcoded at the top for easy experimentation.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import (
	RandBiasField,
	RandAdjustContrast,
	RandHistogramShift,
	RandGibbsNoise,
	RandKSpaceSpikeNoise,
	RandRicianNoise,
)


# ============================================================================
# HARDCODED PARAMETERS - Change these to experiment with augmentations
# ============================================================================

# Dataset parameters
IMAGE_SIZE = 128  # 64 or 128
NUM_IMAGES = 5  # Number of images to display
SEED = 42

# MONAI augmentation parameters (all with prob=1.0 to always apply)
BIAS_FIELD_PARAMS = {
	"coeff_range": (-0.4, 0.4),
	"degree": 3,
	"prob": 1.0,
}

CONTRAST_PARAMS = {
	"prob": 1.0,
	"gamma": (1.8, 2.2),
}

HISTOGRAM_SHIFT_PARAMS = {
	"num_control_points": 10,
	"prob": 1.0,
}

GIBBS_NOISE_PARAMS = {
	"alpha": (0.75, 0.75),
	"prob": 1.0,
}

KSPACE_SPIKE_PARAMS = {
	"prob": 1.0,
	"intensity_range": (6, 8),
}

RICIAN_NOISE_PARAMS = {
	"std": 0.2,
	"prob": 1.0,
}


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


def _normalize_with_quantiles(channel: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
	"""Normalize channel using smart quantile-based clipping with mean±4std bounds.
	
	Computes mean and std of the channel, calculates mean-4std and mean+4std as bounds.
	Uses max(q_low_quantile, mean-4std) as lower bound and min(q_high_quantile, mean+4std) as upper bound.
	
	Args:
		channel: 2D array
		q_low: Lower quantile (default 1%)
		q_high: Upper quantile (default 99%)
	
	Returns:
		Normalized channel in [0, 1]
	"""
	channel_float = channel.astype(np.float32)
	
	# Compute mean and std
	mean = np.mean(channel_float)
	std = np.std(channel_float)
	
	# Compute quantiles
	q_low_val = np.percentile(channel_float, q_low)
	q_high_val = np.percentile(channel_float, q_high)
	
	# Compute mean±4std bounds
	bound_low = mean - 4 * std
	bound_high = mean + 4 * std
	
	# Use the value that's more conservative (closer to mean for lower, closer to mean for upper)
	v_low = max(q_low_val, bound_low)
	v_high = min(q_high_val, bound_high)
	
	if v_high == v_low:
		return np.zeros_like(channel_float, dtype=np.float32)
	
	normalized = (channel_float - v_low) / (v_high - v_low)
	return np.clip(normalized, 0.0, 1.0)


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
		
		images.append(img)
		image_ids.append(img_path.stem)
	
	return images, image_ids


def apply_transform_per_channel(img_chw: torch.Tensor, transform) -> torch.Tensor:
	"""Apply transform to each channel separately.
	
	Args:
		img_chw: Image tensor with shape (C, H, W)
		transform: MONAI transform to apply
	
	Returns:
		Transformed image with shape (C, H, W)
	"""
	channels = []
	for c in range(img_chw.shape[0]):
		channel = img_chw[c:c+1, :, :]  # (1, H, W)
		transformed = transform(channel)
		channels.append(transformed)
	
	result = torch.cat(channels, dim=0)  # (C, H, W)
	return result


def _normalize_channels_individually(img_hwc: np.ndarray) -> np.ndarray:
	"""Normalize each channel of an image individually using smart quantile normalization.
	
	Args:
		img_hwc: Image array with shape (H, W, C)
	
	Returns:
		Image with each channel normalized to [0, 1]
	"""
	img = np.asarray(img_hwc, dtype=np.float32)
	
	if img.ndim == 2:
		# Single channel image
		return _normalize_with_quantiles(img)
	
	# Multi-channel image: normalize each channel individually
	normalized = np.zeros_like(img, dtype=np.float32)
	for c in range(img.shape[2]):
		channel = img[..., c]
		normalized[..., c] = _normalize_with_quantiles(channel)
	
	return normalized


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
	
	# Create MONAI transforms
	bias_field_transform = RandBiasField(**BIAS_FIELD_PARAMS)
	contrast_transform = RandAdjustContrast(**CONTRAST_PARAMS)
	histogram_shift_transform = RandHistogramShift(**HISTOGRAM_SHIFT_PARAMS)
	gibbs_noise_transform = RandGibbsNoise(**GIBBS_NOISE_PARAMS)
	kspace_spike_transform = RandKSpaceSpikeNoise(**KSPACE_SPIKE_PARAMS)
	rician_noise_transform = RandRicianNoise(**RICIAN_NOISE_PARAMS)
	
	# Load and transform images
	original_images = []
	bias_field_images = []
	contrast_images = []
	histogram_shift_images = []
	gibbs_noise_images = []
	kspace_spike_images = []
	rician_noise_images = []
	
	for img_float in images:
		# img_float is (H, W, C) from numpy
		# Convert to (C, H, W) torch tensor
		img_chw = torch.from_numpy(img_float).permute(2, 0, 1).float()  # (C, H, W)
		
		# Store original with per-channel normalization
		original_images.append(_normalize_channels_individually(img_float))
		
		# Apply transforms per channel
		try:
			bias_field_result = apply_transform_per_channel(img_chw, bias_field_transform)
			bias_field_images.append(_normalize_channels_individually(bias_field_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in bias field: {e}")
			bias_field_images.append(original_images[-1])
		
		try:
			contrast_result = apply_transform_per_channel(img_chw, contrast_transform)
			contrast_images.append(_normalize_channels_individually(contrast_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in contrast: {e}")
			contrast_images.append(original_images[-1])
		
		try:
			histogram_shift_result = apply_transform_per_channel(img_chw, histogram_shift_transform)
			histogram_shift_images.append(_normalize_channels_individually(histogram_shift_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in histogram shift: {e}")
			histogram_shift_images.append(original_images[-1])
		
		try:
			gibbs_result = apply_transform_per_channel(img_chw, gibbs_noise_transform)
			gibbs_noise_images.append(_normalize_channels_individually(gibbs_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in Gibbs noise: {e}")
			gibbs_noise_images.append(original_images[-1])
		
		try:
			kspace_result = apply_transform_per_channel(img_chw, kspace_spike_transform)
			kspace_spike_images.append(_normalize_channels_individually(kspace_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in k-space spike: {e}")
			kspace_spike_images.append(original_images[-1])
		
		try:
			rician_result = apply_transform_per_channel(img_chw, rician_noise_transform)
			rician_noise_images.append(_normalize_channels_individually(rician_result.permute(1, 2, 0).numpy()))
		except Exception as e:
			print(f"Error in Rician noise: {e}")
			rician_noise_images.append(original_images[-1])
	
	# Create visualization grid
	n_cols = len(images) * 2  # Double columns: RGB + T1c grayscale
	n_rows = 7  # original + 6 augmentations
	
	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(3.5 * n_cols, 3.5 * n_rows),
		squeeze=False,
	)
	
	row_data = [
		("Original", original_images),
		("RandBiasField", bias_field_images),
		("RandAdjustContrast", contrast_images),
		("RandHistogramShift", histogram_shift_images),
		("RandGibbsNoise", gibbs_noise_images),
		("RandKSpaceSpikeNoise", kspace_spike_images),
		("RandRicianNoise", rician_noise_images),
	]
	
	for r, (row_name, row_images) in enumerate(row_data):
		for c_img, img in enumerate(row_images):
			# RGB image in even columns
			c_rgb = c_img * 2
			ax_rgb = axes[r, c_rgb]
			
			if img.ndim == 2:
				ax_rgb.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
			else:
				ax_rgb.imshow(img, vmin=0.0, vmax=1.0)
			
			ax_rgb.set_xticks([])
			ax_rgb.set_yticks([])
			
			if c_img == 0:
				ax_rgb.set_ylabel(row_name, fontsize=11, fontweight="bold")
			if r == 0:
				ax_rgb.set_title(f"{image_ids[c_img]}", fontsize=9)
			
			# T1c channel as grayscale in odd columns
			c_t1c = c_img * 2 + 1
			ax_t1c = axes[r, c_t1c]
			
			# Extract T1c channel (index 1) and normalize
			if img.ndim == 3 and img.shape[2] >= 2:
				t1c_channel = img[..., 1]
				t1c_normalized = _normalize_with_quantiles(t1c_channel, q_low=1.0, q_high=99.0)
				ax_t1c.imshow(t1c_normalized, cmap="gray", vmin=0.0, vmax=1.0)
			else:
				ax_t1c.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
			
			ax_t1c.set_xticks([])
			ax_t1c.set_yticks([])
			
			if r == 0:
				ax_t1c.set_title(f"{image_ids[c_img]} (T1c)", fontsize=9)
	
	fig.tight_layout()
	print("Displaying visualization...")
	plt.show()


if __name__ == "__main__":
	main()
