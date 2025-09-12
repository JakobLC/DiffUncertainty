import torch
from torchmetrics.segmentation import DiceScore

@torch.no_grad()
def dice(
    preds_idx: torch.Tensor,           # (N,H,W) int indices
    target_idx: torch.Tensor,          # (N,H,W) int indices
    num_classes: int,                  # original classes including class 0
    ignore_index: int = 255,           # void label to ignore entirely
    average: str = "micro",
    aggregation_level: str = "global"
):
    if preds_idx.shape != target_idx.shape or preds_idx.ndim != 3:
        raise ValueError("Expected preds/target to be (N,H,W) with identical shapes.")
    
    if preds_idx.min() < 0 or preds_idx.max() >= num_classes:
        raise ValueError(f"preds_idx has values outside [0,{num_classes-1}], got {preds_idx.min()}..{preds_idx.max()}")


    # ensure tensors & integer dtype
    preds_idx  = preds_idx.to(dtype=torch.long)
    target_idx = target_idx.to(dtype=torch.long)

    if ignore_index is None:
        ignore_index = -1  # use -1 internally for convenience
    # build a proper *tensor* mask on the same device
    ignore_mask = target_idx.eq(ignore_index)  # Tensor[bool] shape (N,H,W)

    if target_idx[~ignore_mask].min() < 0 or target_idx[~ignore_mask].max() >= num_classes:
        raise ValueError(f"target_idx has values outside [0,{num_classes-1}] (ignoring {ignore_index}), got {target_idx[~ignore_mask].min()}..{target_idx[~ignore_mask].max()}")
    target_idx = torch.where(ignore_mask, target_idx, target_idx.clamp(0, num_classes - 1))

    # Shift by +1 so original classes {0..C-1} -> {1..C}
    # Then set ignored pixels to 0 in BOTH preds & target (background channel).
    preds_shifted  = preds_idx.add(1)
    target_shifted = target_idx.add(1)

    # Important: condition must be Tensor; values can be Numbers
    preds_shifted  = torch.where(ignore_mask, 0, preds_shifted)
    target_shifted = torch.where(ignore_mask, 0, target_shifted)

    metric = DiceScore(
        num_classes=num_classes + 1,    # +1 for background(0)
        average=average,
        aggregation_level=aggregation_level,
        input_format="index",
        include_background=False,       # drop background => ignores void pixels
    )
    return metric(preds_shifted, target_shifted)