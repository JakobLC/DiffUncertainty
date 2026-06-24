"""
Visualize 3D Gibbs noise augmentation on NPC volumes.

Loads full 3D volumes from H5 files, applies RandGibbsNoise (3D),
and displays 2D slices (axial) from both original and augmented volumes.

Normalization uses smart quantile-based clipping with mean±4std bounds.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from scipy import ndimage
from monai.transforms import RandGibbsNoise


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

# Dataset parameters
NUM_VOLUMES = 1  # Number of volumes to visualize
SLICES_PER_VOLUME = 3  # Number of slices to show per volume
SEED = 42

# Gibbs noise parameters
GIBBS_NOISE_PARAMS = {
	"alpha": (0.9, 0.9),
	"prob": 1.0,
}

# Upscaling parameters (applied before Gibbs, downscaled after)
UPSCALE_MULTIPLIER = 2  # 1 = no upscaling, 2 = 2x upscaling, etc.

# Normalization parameters
Q_LOW = 1.0  # Lower quantile (%)
Q_HIGH = 99.0  # Upper quantile (%)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_with_quantiles(
	channel: np.ndarray,
	q_low: float = 1.0,
	q_high: float = 99.0,
) -> np.ndarray:
	"""Normalize channel using smart quantile-based clipping with mean±4std bounds.
	
	Computes mean and std of the channel, calculates mean-4std and mean+4std as bounds.
	Uses max(q_low_quantile, mean-4std) as lower bound and min(q_high_quantile, mean+4std) as upper bound.
	
	Args:
		channel: 1D, 2D, or 3D array
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
	
	# Use the value that's more conservative
	v_low = max(q_low_val, bound_low)
	v_high = min(q_high_val, bound_high)
	
	if v_high == v_low:
		return np.zeros_like(channel_float, dtype=np.float32)
	
	normalized = (channel_float - v_low) / (v_high - v_low)
	return np.clip(normalized, 0.0, 1.0)


def load_h5_volume(h5_path: Path) -> dict:
	"""Load all data from an H5 file."""
	data = {}
	with h5py.File(h5_path, 'r') as f:
		for key in f.keys():
			data[key] = f[key][:]
	return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
	"""Normalize a 3D volume to [0, 1] based on min/max across entire volume."""
	v_min = volume.min()
	v_max = volume.max()
	if v_max == v_min:
		return np.zeros_like(volume, dtype=np.float32)
	normalized = (volume.astype(np.float32) - v_min) / (v_max - v_min)
	return normalized


def load_volumes(
	base_dir: str,
	num_samples: int,
	seed: int,
) -> list:
	"""Load random NPC 3D volumes from train directory."""
	split_dir = Path(base_dir) / "MMIS2024TASK1" / "training"
	
	if not split_dir.exists():
		raise FileNotFoundError(f"Split directory not found: {split_dir}")
	
	# Find all H5 files
	h5_files = sorted(split_dir.glob("*.h5"))
	if len(h5_files) == 0:
		raise FileNotFoundError(f"No .h5 volumes found in {split_dir}")
	
	# Sample random volumes
	n = min(max(1, num_samples), len(h5_files))
	rng = np.random.default_rng(seed)
	sampled_indices = rng.choice(len(h5_files), size=n, replace=False)
	
	volumes = []
	volume_ids = []
	for idx in sampled_indices:
		h5_path = h5_files[int(idx)]
		data = load_h5_volume(h5_path)
		
		# Extract and normalize the three channels
		t1 = normalize_volume(data['t1'])
		t1c = normalize_volume(data['t1c'])
		t2 = normalize_volume(data['t2'])
		
		# Stack into (D, H, W, C) format for processing
		volume_3ch = np.stack([t1, t1c, t2], axis=-1)  # (D, H, W, 3)
		
		volumes.append(volume_3ch)
		volume_ids.append(h5_path.stem)
	
	return volumes, volume_ids


def apply_gibbs_3d(volume_dhwc: np.ndarray, transform, upscale_multiplier: int = 1) -> np.ndarray:
	"""Apply 3D Gibbs noise to a volume with optional upscaling/downscaling.
	
	Args:
		volume_dhwc: Volume with shape (D, H, W, C)
		transform: MONAI RandGibbsNoise transform
		upscale_multiplier: Integer multiplier for upscaling before transform.
							If > 1, volume is upscaled, transform applied, then downscaled.
	
	Returns:
		Transformed volume with shape (D, H, W, C)
	"""
	# Convert to (C, D, H, W) for MONAI (channel-first)
	volume_cdhw = volume_dhwc.transpose(3, 0, 1, 2)
	
	# Upscale if needed
	if upscale_multiplier > 1:
		zoom_factors = (1, upscale_multiplier, upscale_multiplier, upscale_multiplier)
		volume_cdhw = ndimage.zoom(volume_cdhw, zoom_factors, order=1)
	
	volume_torch = torch.from_numpy(volume_cdhw).float()
	
	# Apply transform
	transformed = transform(volume_torch)
	
	# Downscale if needed
	if upscale_multiplier > 1:
		volume_cdhw_out = transformed.numpy()
		zoom_factors = (1, 1.0 / upscale_multiplier, 1.0 / upscale_multiplier, 1.0 / upscale_multiplier)
		volume_cdhw_out = ndimage.zoom(volume_cdhw_out, zoom_factors, order=1)
		transformed = torch.from_numpy(volume_cdhw_out)
	else:
		transformed = transformed
	
	# Convert back to (D, H, W, C)
	result = transformed.numpy().transpose(1, 2, 3, 0)
	return result


def extract_slices(volume_dhwc: np.ndarray, num_slices: int, seed: int) -> list:
	"""Extract random 2D axial slices from a 3D volume.
	
	Args:
		volume_dhwc: Volume with shape (D, H, W, C)
		num_slices: Number of slices to extract
		seed: Random seed
	
	Returns:
		List of 2D slices (H, W, C)
	"""
	depth = volume_dhwc.shape[0]
	n = min(max(1, num_slices), depth)
	
	rng = np.random.default_rng(seed)
	slice_indices = rng.choice(depth, size=n, replace=False)
	
	slices = []
	for idx in sorted(slice_indices):
		slices.append(volume_dhwc[int(idx)])
	
	return slices


def main():
	"""Main visualization function."""
	
	# Setup dataset path
	data_base_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/npc"
	
	print(f"Loading NPC volumes from {data_base_dir}...")
	volumes, volume_ids = load_volumes(data_base_dir, NUM_VOLUMES, SEED)
	print(f"Loaded {len(volumes)} volumes")
	
	# Create Gibbs transform
	gibbs_transform = RandGibbsNoise(**GIBBS_NOISE_PARAMS)
	
	# Process each volume
	all_slices_original = []
	all_slices_gibbs = []
	all_slice_ids = []
	
	for vol_idx, (volume, vol_id) in enumerate(zip(volumes, volume_ids)):
		print(f"Processing volume {vol_idx + 1}/{len(volumes)}: {vol_id}")
		
		# Extract slices from original volume
		slices_original = extract_slices(volume, SLICES_PER_VOLUME, SEED + vol_idx)
		
		# Apply Gibbs noise to full volume
		try:
			volume_gibbs = apply_gibbs_3d(volume, gibbs_transform, UPSCALE_MULTIPLIER)
		except Exception as e:
			print(f"Error applying Gibbs noise: {e}")
			volume_gibbs = volume
		
		# Extract same slices from augmented volume
		slices_gibbs = extract_slices(volume_gibbs, SLICES_PER_VOLUME, SEED + vol_idx)
		
		# Store slices
		all_slices_original.extend(slices_original)
		all_slices_gibbs.extend(slices_gibbs)
		
		# Create IDs for each slice
		for slice_idx in range(len(slices_original)):
			all_slice_ids.append(f"{vol_id}_slice{slice_idx}")
	
	# Visualization: create grid with columns: original RGB, original T1c, gibbs RGB, gibbs T1c
	n_slices = len(all_slices_original)
	n_rows = 2  # original and gibbs
	n_cols = n_slices * 2  # RGB + T1c for each slice
	
	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(4 * n_cols, 6 * n_rows),
		squeeze=False,
	)
	
	row_data = [
		("Original", all_slices_original),
		("Gibbs Noise", all_slices_gibbs),
	]
	
	for r, (row_name, row_slices) in enumerate(row_data):
		for c_slice, slice_hwc in enumerate(row_slices):
			# RGB image in even columns
			c_rgb = c_slice * 2
			ax_rgb = axes[r, c_rgb]
			
			# Normalize RGB channels
			rgb_normalized = slice_hwc[..., :3].copy()
			for c in range(3):
				rgb_normalized[..., c] = _normalize_with_quantiles(
					rgb_normalized[..., c], Q_LOW, Q_HIGH
				)
			
			ax_rgb.imshow(rgb_normalized, vmin=0.0, vmax=1.0)
			ax_rgb.set_xticks([])
			ax_rgb.set_yticks([])
			
			if c_slice == 0:
				ax_rgb.set_ylabel(row_name, fontsize=12, fontweight="bold")
			if r == 0:
				ax_rgb.set_title(f"{all_slice_ids[c_slice]} (RGB)", fontsize=10)
			
			# T1c channel as grayscale in odd columns
			c_t1c = c_slice * 2 + 1
			ax_t1c = axes[r, c_t1c]
			
			# Extract T1c channel (index 1) and normalize
			t1c_channel = slice_hwc[..., 1]
			t1c_normalized = _normalize_with_quantiles(t1c_channel, Q_LOW, Q_HIGH)
			ax_t1c.imshow(t1c_normalized, cmap="gray", vmin=0.0, vmax=1.0)
			ax_t1c.set_xticks([])
			ax_t1c.set_yticks([])
			
			if r == 0:
				ax_t1c.set_title(f"{all_slice_ids[c_slice]} (T1c)", fontsize=10)
	
	fig.tight_layout()
	print("Displaying visualization...")
	plt.show()


if __name__ == "__main__":
	main()
