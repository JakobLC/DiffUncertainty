import torch
from torchmetrics.segmentation import DiceScore

@torch.no_grad()
def dice(
    preds_idx: torch.Tensor,           # (N,H,W) int indices
    target_idx: torch.Tensor,          # (N,H,W) int indices
    num_classes: int = None,                  # original classes including class 0
    ignore_index: int = 255,           # void label to ignore entirely
    include_background: bool = True,   # whether to ignore class 0
    average: str = "micro",
    aggregation_level: str = "global",
    is_softmax=False,                  # if True, preds are softmax probabilities (N,C,H,W)
):
    if is_softmax:
        assert preds_idx.ndim == 4, f"Expected (N,C,H,W) for preds when is_softmax=True, got {preds_idx.shape}"
        if num_classes is None:
            num_classes = preds_idx.shape[1]
        else:
            assert num_classes == preds_idx.shape[1], f"num_classes={num_classes} inconsistent with preds.shape[1]={preds_idx.shape[1]}"
        preds_idx = preds_idx.argmax(1)  # (N,H,W)

    assert num_classes is not None, "num_classes must be specified"
    if preds_idx.shape != target_idx.shape or preds_idx.ndim != 3:
        raise ValueError(f"Expected preds/target to be (N,H,W) with identical shapes. got shapes {preds_idx.shape} and {target_idx.shape}")
    
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
    if include_background:
        preds_shifted = preds_idx.add(1)
        target_shifted = target_idx.add(1)
    else:
        preds_shifted = preds_idx
        target_shifted = target_idx
    # Important: condition must be Tensor; values can be Numbers
    preds_shifted = torch.where(ignore_mask, 0, preds_shifted)
    target_shifted = torch.where(ignore_mask, 0, target_shifted)

    metric = DiceScore(
        num_classes=num_classes + int(include_background), # +1 for ignore class placeholder
        average=average,
        aggregation_level=aggregation_level,
        input_format="index",
        include_background=False,       # drop background => ignores void pixels
    )
    out = metric(preds_shifted, target_shifted)
    return out