import argparse
import copy
from pathlib import Path
import sys
#print(str(Path(__file__).resolve().parents[1]))
#exit()
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1]/"uncertainty_modeling"))
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from uncertainty_modeling.data.torch_dataloader import apply_augment_mult
from uncertainty_modeling.test_2D import AlbumentationsTTABackend


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize test-time augmentation effects on images and labels. "
            "Columns are samples and rows are: image pre, image post, label pre, label post, label inverted."
        )
    )
    parser.add_argument("--tta_yaml", 
                        default="/home/jloch/Desktop/diff/luzern/values/uncertainty_modeling/configs/data/TTA_chaksu128_strong.yaml",
                        type=str, help="Path to TTA YAML config.")
    parser.add_argument(
        "--split",
        type=str,
        default="id_test",
        help="Dataset split to sample from (e.g. id_test, ood_noise, train, val).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size / number of examples (columns).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for random sample selection.",
    )
    parser.add_argument(
        "--rater_index",
        type=int,
        default=0,
        help="Which rater mask to visualize for multi-rater datasets.",
    )
    return parser.parse_args()


def _load_yaml_cfg(path: str):
    cfg = OmegaConf.load(path)
    if "data" not in cfg:
        raise ValueError("Expected TTA YAML to contain a top-level 'data' section.")
    if "dataset" not in cfg.data:
        raise ValueError("Expected TTA YAML to contain 'data.dataset'.")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    aug_cfg = None
    if "data" in cfg and "augmentations" in cfg.data:
        aug_cfg = cfg.data.augmentations
    elif "augmentations" in cfg:
        aug_cfg = cfg.augmentations
    else:
        raise ValueError("Expected TTA YAML to include either data.augmentations or augmentations.")

    aug_cfg_copy = OmegaConf.create(copy.deepcopy(aug_cfg))
    aug_cfg_copy = apply_augment_mult(aug_cfg_copy)
    aug_cfg_resolved = OmegaConf.to_container(aug_cfg_copy, resolve=True)
    return data_cfg, aug_cfg_resolved


def _build_dataset(data_cfg: dict, split: str):
    if "data_input_dir" not in data_cfg:
        raise ValueError("Expected TTA YAML to define data.data_input_dir.")
    if "dataset" not in data_cfg:
        raise ValueError("Expected TTA YAML to define data.dataset.")

    dataset_cfg = data_cfg["dataset"]
    return hydra.utils.instantiate(
        dataset_cfg,
        base_dir=data_cfg["data_input_dir"],
        split=split,
        transforms=None,
        data_fold_id=int(data_cfg.get("data_fold_id", 0)),
        tta=True,
        return_all_raters=True,
    )


def _to_one_hot(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    mask_int = np.asarray(mask_hw, dtype=np.int64)
    if mask_int.ndim != 2:
        raise ValueError(f"Expected 2D integer mask for one-hot conversion, got shape {mask_int.shape}.")
    mask_int = np.clip(mask_int, 0, max(num_classes - 1, 0))
    one_hot = np.eye(num_classes, dtype=np.float32)[mask_int]
    return one_hot


def _forward_replay(label_hwc: np.ndarray, replay: dict) -> np.ndarray:
    transformed = label_hwc
    replay_transforms = replay.get("transforms", []) if isinstance(replay, dict) else []
    for transform in replay_transforms:
        if not transform.get("applied", False):
            continue
        fullname = str(transform.get("__class_fullname__", ""))
        transform_name = fullname.split(".")[-1]
        params = transform.get("params", {}) or {}

        if transform_name == "HorizontalFlip":
            transformed = cv2.flip(transformed, 1)
            continue

        if transform_name == "Rotate":
            angle = None
            for key in ("angle", "x"):
                value = params.get(key)
                if isinstance(value, (int, float)):
                    angle = float(value)
                    break
            if angle is None:
                raise RuntimeError(f"Rotate replay did not contain an angle: {params}")
            transformed = AlbumentationsTTABackend._warp_inverse_affine(
                transformed,
                angle_deg=angle,
                scale=1.0,
            )
            continue

        if transform_name == "RandomScale":
            scale = None
            for key in ("scale", "x"):
                value = params.get(key)
                if isinstance(value, (int, float)):
                    scale = float(value)
                    break
            if scale is None:
                raise RuntimeError(f"RandomScale replay did not contain a scale value: {params}")
            transformed = AlbumentationsTTABackend._warp_inverse_affine(
                transformed,
                angle_deg=0.0,
                scale=scale,
            )
            continue

        if transform_name == "Affine":
            forward_matrix = AlbumentationsTTABackend._extract_affine_matrix_from_params(params)
            transformed = cv2.warpAffine(
                transformed,
                forward_matrix,
                (transformed.shape[1], transformed.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if transformed.ndim == 2:
                transformed = transformed[..., None]
    return transformed


def _renormalize_hwc_channels(arr_hwc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    channel_sum = np.sum(arr_hwc, axis=-1, keepdims=True)
    safe = np.maximum(channel_sum, eps)
    renorm = arr_hwc / safe
    return np.where(channel_sum > eps, renorm, arr_hwc)


def _normalize_img_for_plot(img_hwc: np.ndarray) -> np.ndarray:
    img = np.asarray(img_hwc, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
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


def _label_to_plot(label_hwc: np.ndarray) -> np.ndarray:
    label = np.asarray(label_hwc, dtype=np.float32)
    if label.ndim == 2:
        return label
    if label.shape[-1] == 3:
        return np.clip(label, 0.0, 1.0)

    # Rule requested by user: non-3-class labels use channel index 1 when available.
    channel_idx = 1 if label.shape[-1] > 1 else 0
    return np.clip(label[..., channel_idx], 0.0, 1.0)


def _sample_indices(dataset_len: int, batch_size: int, seed: int):
    n = min(max(1, batch_size), dataset_len)
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_len, size=n, replace=False)
    return [int(i) for i in indices]


def main():
    args = _parse_args()
    yaml_path = Path(args.tta_yaml).expanduser().resolve()

    data_cfg, augmentations_cfg = _load_yaml_cfg(yaml_path.as_posix())
    dataset = _build_dataset(data_cfg, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{args.split}' is empty.")

    num_classes = int(data_cfg.get("num_classes", 2))
    tta_backend = AlbumentationsTTABackend(augmentations_cfg)

    indices = _sample_indices(len(dataset), args.batch_size, args.seed)

    pre_images = []
    post_images = []
    pre_labels = []
    post_labels = []
    inv_labels = []
    diff_labels = []
    image_ids = []

    batch_images = []
    masks_for_samples = []
    for idx in indices:
        sample = dataset[idx]
        img = sample["data"]
        seg = sample["seg"]
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            raise ValueError(f"Expected sample['data'] to be CHW tensor, got {type(img)} with shape {getattr(img, 'shape', None)}")
        if not isinstance(seg, torch.Tensor) or seg.ndim != 3:
            raise ValueError(f"Expected sample['seg'] to be RHW tensor, got {type(seg)} with shape {getattr(seg, 'shape', None)}")
        if args.rater_index < 0 or args.rater_index >= seg.shape[0]:
            raise IndexError(
                f"rater_index={args.rater_index} is out of range for sample with {seg.shape[0]} raters."
            )
        batch_images.append(img)
        masks_for_samples.append(seg[args.rater_index].detach().cpu().numpy())
        image_ids.append(str(sample.get("image_id", idx)))

    batch = torch.stack(batch_images, dim=0)
    aug_batch, replays = tta_backend.sample_batch(batch)

    for i in range(batch.shape[0]):
        img_pre_hwc = batch[i].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        img_post_hwc = aug_batch[i].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

        label_pre_hwc = _to_one_hot(masks_for_samples[i], num_classes=num_classes)
        label_post_hwc = _forward_replay(label_pre_hwc, replays[i])
        label_inv_hwc = tta_backend._invert_replay(label_post_hwc, replays[i])
        label_inv_hwc = _renormalize_hwc_channels(label_inv_hwc)

        pre_images.append(_normalize_img_for_plot(img_pre_hwc))
        post_images.append(_normalize_img_for_plot(img_post_hwc))
        pre_labels.append(_label_to_plot(label_pre_hwc))
        post_labels.append(_label_to_plot(label_post_hwc))
        inv_labels.append(_label_to_plot(label_inv_hwc))
        label_diff_hwc = np.abs(label_pre_hwc - label_inv_hwc)
        diff_labels.append(_label_to_plot(label_diff_hwc))

    rows = [
        ("Image pre-augmentation", pre_images, None),
        ("Image post-augmentation", post_images, None),
        ("Label pre-augmentation", pre_labels, "gray"),
        ("Label post-augmentation", post_labels, "gray"),
        ("Label post + inverted", inv_labels, "gray"),
        ("Label |original - reconstructed|", diff_labels, "inferno"),
    ]

    n_cols = len(indices)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 3.2 * len(rows)), squeeze=False)

    for r, (row_name, row_images, default_cmap) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            img = row_images[c]
            if img.ndim == 2:
                ax.imshow(img, cmap=default_cmap or "gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_name, fontsize=10)
            if r == 0:
                ax.set_title(f"{image_ids[c]}", fontsize=9)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
