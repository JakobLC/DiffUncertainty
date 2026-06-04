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