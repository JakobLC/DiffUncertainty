from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _pairwise_iou(binary_masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise IoU matrix for binary masks of shape (num_masks, num_entries)."""
    inter = binary_masks @ binary_masks.transpose(0, 1)
    sums = binary_masks.sum(dim=1, keepdim=True)
    union = sums + sums.transpose(0, 1) - inter
    return inter / (union + float(eps))


def compute_subensemble_mask_stats(ckpt_filename: str | Path) -> dict[str, Any]:
    """Load a sub-ensemble checkpoint and summarize hard row-mask statistics.

    The function expects a checkpoint with a serialized `subensemble_masks` payload
    created by extraction code. Only rows-only models are supported; otherwise a
    NotImplementedError is raised.

    Args:
        ckpt_filename: Path to a checkpoint file.

    Returns:
        Dictionary with per-layer and network-wide active-row ratios, plus the
        hard overlap IoU matrix and its mean over off-diagonal pairs.
    """
    ckpt_path = Path(ckpt_filename).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must deserialize to a dictionary.")

    payload = checkpoint.get("subensemble_masks")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is missing a 'subensemble_masks' dictionary payload.")

    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("'subensemble_masks.layers' must be a non-empty list.")

    if payload.get("format") not in {None, "binary_channel_masks_v1"}:
        raise ValueError(
            "Unsupported mask payload format: "
            f"{payload.get('format')!r}. Expected 'binary_channel_masks_v1'."
        )

    non_rows_only_layers = [
        str(layer.get("name", "<unnamed>")) for layer in layers if not bool(layer.get("rows_only", False))
    ]
    if non_rows_only_layers:
        raise NotImplementedError(
            "Only rows-only masks are supported by this helper. "
            f"Found non-rows-only layers: {non_rows_only_layers}"
        )

    layer_names: list[str] = []
    layer_types: list[str] = []
    per_layer_total_rows: list[int] = []
    per_layer_active_rows_by_mask: list[list[int]] = []
    per_layer_active_ratio_by_mask: list[list[float]] = []
    concatenated_rows: list[torch.Tensor] = []
    total_active_rows: torch.Tensor | None = None
    total_rows = 0
    num_submodels: int | None = None

    for layer in layers:
        layer_name = str(layer.get("name", ""))
        if not layer_name:
            raise ValueError("Each layer entry in 'subensemble_masks.layers' must include a non-empty 'name'.")

        output_masks_raw = layer.get("output_masks")
        if output_masks_raw is None:
            raise ValueError(f"Layer '{layer_name}' is missing 'output_masks'.")

        output_masks = torch.as_tensor(output_masks_raw)
        if output_masks.ndim != 2:
            raise ValueError(
                f"Layer '{layer_name}' output_masks must have shape (num_submodels, out_channels). "
                f"Got shape {tuple(output_masks.shape)}."
            )

        output_masks = (output_masks > 0).to(torch.float32)
        layer_num_submodels = int(output_masks.shape[0])
        layer_num_rows = int(output_masks.shape[1])

        if num_submodels is None:
            num_submodels = layer_num_submodels
            total_active_rows = torch.zeros(num_submodels, dtype=torch.float32)
        elif layer_num_submodels != num_submodels:
            raise ValueError(
                "Inconsistent number of masks across layers: "
                f"expected {num_submodels}, got {layer_num_submodels} for layer '{layer_name}'."
            )

        assert total_active_rows is not None
        active_rows = output_masks.sum(dim=1)
        active_ratio = active_rows / max(layer_num_rows, 1)

        layer_names.append(layer_name)
        layer_types.append(str(layer.get("type", "unknown")))
        per_layer_total_rows.append(layer_num_rows)
        per_layer_active_rows_by_mask.append([int(v) for v in active_rows.tolist()])
        per_layer_active_ratio_by_mask.append([float(v) for v in active_ratio.tolist()])

        total_active_rows += active_rows
        total_rows += layer_num_rows
        concatenated_rows.append(output_masks)

    if num_submodels is None or total_active_rows is None or total_rows <= 0:
        raise ValueError("No valid row masks found in checkpoint payload.")

    # Reorganize to [mask_idx][layer_idx] so structure mirrors network lists per mask.
    active_rows_per_mask_per_layer = [
        [per_layer_active_rows_by_mask[layer_idx][mask_idx] for layer_idx in range(len(layer_names))]
        for mask_idx in range(num_submodels)
    ]
    active_ratio_per_mask_per_layer = [
        [per_layer_active_ratio_by_mask[layer_idx][mask_idx] for layer_idx in range(len(layer_names))]
        for mask_idx in range(num_submodels)
    ]

    all_rows = torch.cat(concatenated_rows, dim=1)
    iou_matrix = _pairwise_iou(all_rows)

    if num_submodels > 1:
        i, j = torch.triu_indices(num_submodels, num_submodels, offset=1)
        overlap_hard_iou_mean = float(iou_matrix[i, j].mean().item())
    else:
        overlap_hard_iou_mean = 0.0

    network_ratio = total_active_rows / float(total_rows)
    network_ratio_list = [float(v) for v in network_ratio.tolist()]
    active_all = float(sum(network_ratio_list) / max(len(network_ratio_list), 1))

    return {
        "checkpoint_path": ckpt_path.as_posix(),
        "num_submodels": num_submodels,
        "num_mask_layers": len(layer_names),
        "rows_only": True,
        "active_per_layer": {
            "total_rows": per_layer_total_rows,
            "active_rows_per_mask": active_rows_per_mask_per_layer,
            "active_row_ratio_per_mask": active_ratio_per_mask_per_layer,
            "layer_names": layer_names,
            "layer_types": layer_types,
        },
        "active_per_network": {
            "total_rows": int(total_rows),
            "active_rows_per_mask": [int(v) for v in total_active_rows.tolist()],
            "active_row_ratio_per_mask": network_ratio_list,
        },
        "active_all": active_all,
        "overlap_hard_iou_matrix": iou_matrix.tolist(),
        "overlap_hard_iou_mean": overlap_hard_iou_mean,
    }
