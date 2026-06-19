from typing import Dict, Any

import albumentations as A
import numpy as np
import scipy.ndimage as nd
import uncertainty_modeling.data.cityscapes_labels as cs_labels

class StochasticLabelSwitches(A.BasicTransform):
    def __init__(self, always_apply=False, p=0.5, n_reference_samples: int = 1):
        super(StochasticLabelSwitches, self).__init__(always_apply, p)
        self._name2id = cs_labels.name2trainId
        self._label_switches = {
            "sidewalk": 8.0 / 17.0,
            "person": 7.0 / 17.0,
            "car": 6.0 / 17.0,
            "vegetation": 5.0 / 17.0,
            "road": 4.0 / 17.0,
        }
        self.n_reference_samples = n_reference_samples

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        masks = []
        for reference in range(self.n_reference_samples):
            mask_copy = mask.copy()
            for c, p in self._label_switches.items():
                init_id = self._name2id[c]
                final_id = self._name2id[c + "_2"]
                switch_instances = np.random.binomial(1, p, 1)

                if switch_instances[0]:
                    mask_copy[mask_copy == init_id] = final_id
            masks.append(mask_copy)
        if len(masks) > 1:
            return np.array(masks)
        else:
            return masks[0]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_transform_init_args_names(self):
        return ()

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}


class SampleNormalize(A.ImageOnlyTransform):
    """Normalize each sample to zero mean and unit std across all channels."""

    def __init__(self, eps: float = 1e-6, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.eps = float(eps)

    def apply(self, img, **params):
        img = img.astype(np.float32, copy=False)
        mean = float(np.mean(img))
        std = float(np.std(img))
        if std < self.eps:
            std = 1.0
        return (img - mean) / std

    def get_transform_init_args_names(self):
        return ("eps",)

class IntensityGuidedSDFDeform(A.DualTransform):
    """
    Image unchanged. Mask is deformed by modifying class-wise signed distance fields.

    Assumes integer label mask:
        0 = background, 1..K = nested / concentric labels.
    """

    def __init__(
        self,
        std_brightness_deform=10.0,
        std_random_deform=20.0,
        brightness_deform_weight=5.0,
        random_deform_weight=5.0,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.std_brightness_deform = std_brightness_deform
        self.std_random_deform = std_random_deform
        self.brightness_deform_weight = brightness_deform_weight
        self.random_deform_weight = random_deform_weight

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def apply(self, img, **params):
        # image itself is unchanged
        return img

    def apply_to_mask(self, mask, new_mask=None, **params):
        return new_mask.astype(mask.dtype)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        mask = params["mask"]

        new_mask = self._deform_mask(image, mask)
        return {"new_mask": new_mask}

    def _brightness(self, image):
        image = image.astype(np.float32)

        if image.ndim == 3 and image.shape[-1] == 3:
            return (
                0.299 * image[..., 0]
                + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]
            )

        if image.ndim == 3 and image.shape[-1] == 1:
            return image[..., 0]

        return image

    def _binary_sdf(self, binary):
        binary = binary.astype(bool)

        return (
            np.clip(nd.distance_transform_edt(binary) - 0.5, 0, None)
            - np.clip(nd.distance_transform_edt(~binary) - 0.5, 0, None)
        )

    def _make_delta(self, brightness, binary):
        brightness_blur = nd.gaussian_filter(
            brightness.astype(np.float32),
            sigma=self.std_brightness_deform,
        )

        inside = binary.astype(bool)
        outside = ~inside

        if inside.sum() == 0 or outside.sum() == 0:
            delta_brightness = np.zeros_like(brightness, dtype=np.float32)
        else:
            sdf = self._binary_sdf(binary)
            mean_inside = brightness[np.logical_and(sdf > 0, sdf < self.std_brightness_deform)].mean()
            mean_outside = brightness[np.logical_and(sdf < 0, sdf > -self.std_brightness_deform)].mean()
            delta_brightness = 2.0 * (
                (brightness_blur - mean_outside) / (mean_inside - mean_outside)
            ) - 1.0

        noise = np.random.normal(size=brightness.shape).astype(np.float32)
        delta_random = nd.gaussian_filter(noise, sigma=self.std_random_deform)

        mean_abs = np.mean(np.abs(delta_random))
        if mean_abs > 1e-6:
            delta_random = delta_random / mean_abs
        else:
            delta_random = np.zeros_like(delta_random)

        return (
            self.brightness_deform_weight * delta_brightness
            + self.random_deform_weight * delta_random
        )

    def _deform_binary(self, brightness, binary):
        sdf = self._binary_sdf(binary)
        delta = self._make_delta(brightness, binary)

        return (sdf + delta) >= 0.0

    def _deform_mask(self, image, mask):
        brightness = self._brightness(image)

        # Albumentations usually uses HxW integer label masks.
        # If one-hot is accidentally passed, convert to labels.
        if mask.ndim == 3:
            mask_labels = np.argmax(mask, axis=-1).astype(np.int32)
        else:
            mask_labels = mask.astype(np.int32)

        max_label = int(mask_labels.max())
        out = np.zeros_like(mask_labels, dtype=np.int32)

        # Compose lower labels first, then overwrite with higher labels.
        for label_idx in range(1, max_label + 1):
            binary = mask_labels >= label_idx
            deformed_binary = self._deform_binary(brightness, binary)
            out[deformed_binary] = label_idx

        return out

    def get_transform_init_args_names(self):
        return (
            "std_brightness_deform",
            "std_random_deform",
            "brightness_deform_weight",
            "random_deform_weight",
        )


class MaskOnlyElasticTransform(A.DualTransform):
    """Apply ElasticTransform to masks only while keeping images unchanged."""

    def __init__(self, always_apply=False, p=1.0, **elastic_kwargs):
        super().__init__(always_apply=always_apply, p=p)
        self._elastic = A.ElasticTransform(always_apply=True, p=1.0, **elastic_kwargs)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        return self._elastic.apply_to_mask(mask, **params)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self._elastic, "get_params_dependent_on_targets"):
            try:
                return self._elastic.get_params_dependent_on_targets(params)
            except NotImplementedError:
                pass
        return self._elastic.get_params()

    def get_transform_init_args_names(self):
        return self._elastic.get_transform_init_args_names()


class FieldOfViewCircularMask(A.ImageOnlyTransform):
    """
    Simulates the field-of-view of a retina camera by applying a circular mask with edge blurring.
    
    Parameters:
    - radius: radius of the circular mask as a fraction of sidelength. 
              If tuple/list, random uniform value from interval. Default: 0.5
    - edge_blur: width of edge blurring as a fraction of sidelength.
                 If tuple/list, random uniform value from interval. Default: 0.02
    - circle_dist: distance from image center to the circle perimeter as a fraction of sidelength.
                   Positive = circle boundary is inside image, Negative = image center is inside circle.
                   If tuple/list, random uniform value from interval. Default: 0.2
    """
    
    def __init__(
        self,
        radius=0.5,
        edge_blur=0.02,
        circle_dist=0.2,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.radius = radius
        self.edge_blur = edge_blur
        self.circle_dist = circle_dist
    
    def _sample_param(self, param):
        """Sample a parameter which can be a scalar or a (min, max) tuple."""
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        return param
    
    def _get_interval(self, param):
        """Get min/max interval from parameter (scalar or tuple)."""
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param[0], param[1]
        return param, param
    
    def apply(self, img, **params):
        orig_dtype = img.dtype
        img = img.astype(np.float32, copy=True)
        height, width = img.shape[:2]
        
        # Sample parameters
        radius_px = self._sample_param(self.radius) * height
        edge_blur_px = self._sample_param(self.edge_blur) * height
        circle_dist_px = self._sample_param(self.circle_dist) * height
        
        # Calculate the distance of circle center from image center
        # circle_dist = radius - shift_dist, so shift_dist = radius - circle_dist
        shift_dist = radius_px - circle_dist_px
        
        # Create center with random shift
        center_y = height / 2.0
        center_x = width / 2.0
        
        # Random angle for center shift
        angle = np.random.uniform(0, 2 * np.pi)
        center_y += shift_dist * np.sin(angle)
        center_x += shift_dist * np.cos(angle)
        
        # Create coordinate grids normalized to [-0.5, 0.5]
        y = np.arange(height, dtype=np.float32) / height - 0.5
        x = np.arange(width, dtype=np.float32) / width - 0.5
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # Normalize center coordinates
        center_y_norm = center_y / height - 0.5
        center_x_norm = center_x / width - 0.5
        
        # Distance from center
        dist = np.sqrt((yy - center_y_norm)**2 + (xx - center_x_norm)**2)
        
        # Create cone-shaped mask using clipped linear function
        radius_norm = radius_px / height
        edge_blur_norm = edge_blur_px / height
        
        # Mask starts falling off at (radius - edge_blur) and reaches 0 at (radius + edge_blur)
        mask = np.ones_like(dist)
        
        # Linear falloff in edge region
        edge_start = radius_norm - edge_blur_norm
        edge_end = radius_norm + edge_blur_norm
        
        # Pixels beyond the edge_end become 0, linearly interpolate in between
        mask = np.clip((edge_end - dist) / (2 * edge_blur_norm), 0, 1)
        
        # Expand mask to match image channels
        if img.ndim == 3:
            mask = mask[..., np.newaxis]
        
        result = img * mask
        result = np.clip(result, 0, 255.0)
        return result.astype(orig_dtype)
    
    def get_transform_init_args_names(self):
        return ("radius", "edge_blur", "circle_dist")


class FlashArtifact(A.ImageOnlyTransform):
    """
    Simulates a flash artifact in retina images by applying a soft oval-shaped bright spot.
    
    Parameters:
    - additive: whether to add flash on top of image (True) or multiply (False). Default: False
    - additive_range: tuple of (min, max) values to add when additive=True. Default: (-0.3, 1)
    - multiplicative_range: tuple of (min, max) values to multiply when additive=False. Default: (0.2, 2)
    - size: (a+b)/2 value of the ellipse shape. Default: 0.3
    - sharpness: sharpness of the flash (c parameter). Default: 5
    - eccentricity: eccentricity of ellipse, between 0 and 1.
                    If tuple/list, random uniform value from interval. Default: 0.7
    - center_shift: distance from center to shift flash center as a fraction of sidelength.
                    If tuple/list, random uniform value from interval. Default: [0, 0.25]
    """
    
    def __init__(
        self,
        additive=False,
        additive_range=(-0.3, 1.0),
        multiplicative_range=(0.2, 2.0),
        size=0.3,
        sharpness=5,
        eccentricity=0.7,
        center_shift=(0, 0.25),
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.additive = additive
        self.additive_range = additive_range
        self.multiplicative_range = multiplicative_range
        self.size = size
        self.sharpness = sharpness
        self.eccentricity = eccentricity
        self.center_shift = center_shift
    
    def _sample_param(self, param):
        """Sample a parameter which can be a scalar or a (min, max) tuple."""
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        return param
    
    def _get_interval(self, param):
        """Get min/max interval from parameter (scalar or tuple)."""
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param[0], param[1]
        return param, param
    
    def apply(self, img, **params):
        orig_dtype = img.dtype
        img = img.astype(np.float32, copy=True)
        height, width = img.shape[:2]
        
        # Sample parameters
        eccentricity = self._sample_param(self.eccentricity)
        
        # Sample distance from center shift interval
        shift_min, shift_max = self._get_interval(self.center_shift)
        shift_dist = np.random.uniform(shift_min * height, shift_max * height)
        
        # Random rotation angle for ellipse
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        
        # Ellipse semi-axes
        size_px = self.size * height
        a = size_px
        b = size_px * (1 - eccentricity)  # b <= a
        
        # Create center with random shift
        center_y = height / 2.0
        center_x = width / 2.0
        
        # Random angle for center shift
        shift_angle = np.random.uniform(0, 2 * np.pi)
        center_y += shift_dist * np.sin(shift_angle)
        center_x += shift_dist * np.cos(shift_angle)
        
        # Create coordinate grids normalized to [-0.5, 0.5]
        y = np.arange(height, dtype=np.float32) / height - 0.5
        x = np.arange(width, dtype=np.float32) / width - 0.5
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # Normalize center coordinates
        center_y_norm = center_y / height - 0.5
        center_x_norm = center_x / width - 0.5
        
        # Translate to center
        dx = xx - center_x_norm
        dy = yy - center_y_norm
        
        # Rotate coordinates by rotation_angle
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a
        
        # Normalize axes
        a_norm = a / height
        b_norm = b / height
        
        # Compute rotated ellipse levelset
        ellipse_level = self.sharpness * (
            (dx_rot / a_norm)**2 +
            (dy_rot / b_norm)**2 -
            1.0
        )
        
        # Clip to avoid overflow in exp
        ellipse_level = np.clip(ellipse_level, -50, 50)
        
        # Apply sigmoid to get smooth falloff
        sigmoid_mask = 1.0 / (1.0 + np.exp(ellipse_level))
        
        # Scale mask to appropriate range (working in [0, 255] space for uint8 images)
        if self.additive:
            val_min, val_max = self.additive_range
            # Convert from [0, 1] range to [0, 255] range for uint8 images
            intensity_mask = val_min + sigmoid_mask * (val_max - val_min)
            intensity_mask = intensity_mask * 255.0
        else:
            val_min, val_max = self.multiplicative_range
            # Multiplicative range stays as-is for scaling
            intensity_mask = val_min + sigmoid_mask * (val_max - val_min)
        
        # Expand mask to match image channels
        if img.ndim == 3:
            intensity_mask = intensity_mask[..., np.newaxis]
        
        # Apply the flash
        if self.additive:
            result = img + intensity_mask
        else:
            result = img * intensity_mask
        
        # Clip to valid range [0, 255]
        result = np.clip(result, 0, 255.0)
        
        return result.astype(orig_dtype)
    
    def get_transform_init_args_names(self):
        return ("additive", "additive_range", "multiplicative_range", "size", 
                "sharpness", "eccentricity", "center_shift")


class FilteredImageNoise(A.ImageOnlyTransform):
	"""
	Apply spatially-filtered Gaussian noise to an image, modulated by image intensities.
	
	Parameters:
	- noise_scale: multiplicative factor applied to the filtered noise. Default: 0.125
	- sigma: standard deviation of the Gaussian filter applied spatially. Default: 2.3
	"""
	
	def __init__(
		self,
		noise_scale: float = 0.125,
		sigma: float = 2.3,
		p: float = 1.0,
	):
		super().__init__(p=p)
		self.noise_scale = float(noise_scale)
		self.sigma = float(sigma)
	
	def apply(self, img, **params):
		orig_dtype = img.dtype
		img = img.astype(np.float32, copy=True)
		height, width = img.shape[:2]
		n_channels = img.shape[2] if img.ndim == 3 else 1
		
		# Generate Gaussian noise in spatial dimensions only (H, W)
		noise_spatial = np.random.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
		
		# Apply Gaussian filter in spatial dimensions only
		noise_filtered = nd.gaussian_filter(noise_spatial, sigma=self.sigma)
		
		# Normalize: (noise - mean) / std
		noise_mean = float(np.mean(noise_filtered))
		noise_std = float(np.std(noise_filtered))
		if noise_std > 1e-6:
			noise_normalized = (noise_filtered - noise_mean) / noise_std
		else:
			noise_normalized = noise_filtered
		
		# Scale by noise_scale factor
		noise_scaled = self.noise_scale * noise_normalized
		
		# Expand to match image channels (H, W) -> (H, W, C)
		# Each channel gets the same spatial noise pattern
		if img.ndim == 3:
			noise_expanded = noise_scaled[..., np.newaxis]  # (H, W, 1)
			noise_expanded = np.repeat(noise_expanded, n_channels, axis=2)  # (H, W, C)
		else:
			noise_expanded = noise_scaled
		
		# Multiply noise by image intensities
		noise_modulated = noise_expanded * img
		
		# Add to image
		result = img + noise_modulated
		
		# Clip to valid range [0, 1] for float images
		result = np.clip(result, 0.0, 1.0)
		
		return result.astype(orig_dtype)
	
	def get_transform_init_args_names(self):
		return ("noise_scale", "sigma")