import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader


def _compute_area(mask: np.ndarray) -> float:
    mask_arr = np.asarray(mask)
    return float(np.count_nonzero(mask_arr > 0))


def _compute_border(mask: np.ndarray) -> float:
    mask_arr = np.asarray(mask)
    if mask_arr.size == 0:
        return 0.0
    total_border = 0
    for axis, axis_len in enumerate(mask_arr.shape):
        if axis_len < 2:
            continue
        slicer_a = [slice(None)] * mask_arr.ndim
        slicer_b = [slice(None)] * mask_arr.ndim
        slicer_a[axis] = slice(0, -1)
        slicer_b[axis] = slice(1, None)
        neighbors_a = mask_arr[tuple(slicer_a)]
        neighbors_b = mask_arr[tuple(slicer_b)]
        total_border += int(np.count_nonzero(neighbors_a != neighbors_b))
    return float(total_border)


def _compute_stats_from_mask(mask: np.ndarray) -> tuple[float, float]:
    mask_arr = np.asarray(mask)
    return _compute_area(mask_arr), _compute_border(mask_arr)


def _stack_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(pred) for pred in predictions]
    if not arrays:
        raise ValueError("No prediction segmentations available to compute statistics.")
    return np.stack(arrays, axis=0)


def _majority_mask(pred_stack: np.ndarray, threshold: float) -> np.ndarray:
    binary_stack = pred_stack > 0
    fraction = binary_stack.mean(axis=0)
    return (fraction >= threshold).astype(np.uint8)


def _resolve_mean_mask(
    exp_dataloader: ExperimentDataloader,
    image_id: str,
    pred_stack: np.ndarray,
    threshold: float,
) -> np.ndarray:
    try:
        mean_seg = exp_dataloader.get_mean_pred_seg(image_id)
        if mean_seg is not None:
            return mean_seg
    except FileNotFoundError:
        pass
    except Exception:
        pass
    if pred_stack is None:
        raise ValueError("Prediction stack required when mean prediction is unavailable.")
    return _majority_mask(pred_stack, threshold)


def compute_prediction_shape_stats(
    exp_dataloader: ExperimentDataloader,
    mean_pred: bool = True,
    stats_filename: str = "area.json",
    majority_threshold: float = 0.5,
):
    if not 0.0 < majority_threshold <= 1.0:
        raise ValueError("majority_threshold must be within (0, 1].")
    if exp_dataloader.dataset_path is None:
        raise ValueError("Area/border statistics require a single dataset split (no paired splits).")
    dataset_path = exp_dataloader.dataset_path
    stats = {}
    for image_id in tqdm(exp_dataloader.image_ids, desc="Computing area/border stats"):
        pred_segs = exp_dataloader.get_pred_segs(image_id)
        pred_stack = _stack_predictions(pred_segs)
        if mean_pred:
            mask = _resolve_mean_mask(exp_dataloader, image_id, pred_stack, majority_threshold)
            area, border = _compute_stats_from_mask(mask)
        else:
            areas = []
            borders = []
            for pred in pred_stack:
                area_val, border_val = _compute_stats_from_mask(pred)
                areas.append(area_val)
                borders.append(border_val)
            area = float(np.mean(areas))
            border = float(np.mean(borders))
        stats[str(image_id)] = {"area": float(area), "border": float(border)}
    stats_path = Path(dataset_path) / stats_filename
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved prediction shape stats to {stats_path}")
    return stats
