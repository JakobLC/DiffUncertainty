import torch
from typing import Dict, List, Optional


def ged_binary_fast(
    output_softmax: torch.Tensor,
    ground_truth: torch.Tensor,
    ignore_index: Optional[int] = None,
    additional_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Fast Generalized Energy Distance (GED) for binary segmentation with multiple
    predictions and raters, computed fully on-device.

    Args:
        output_softmax: Tensor of shape (P, 2, H, W) with softmax probabilities.
        ground_truth:   Tensor of shape (G, H, W) with integer labels (0/1) and
                         optional ignore_index values.
        ignore_index:   Label value in ground_truth to ignore (set to None to disable).
        additional_metrics: Optional list of extra metrics to compute. Supported:
            - "dice": mean Dice across (P,G)
            - "max_dice_pred": mean over P of max dice vs any GT
            - "max_dice_gt": mean over G of max dice vs any prediction
            - "major_dice": dice between majority prediction and majority GT

    Returns:
        Dict with at least {"ged": float}. If requested/available, also returns the
        additional metrics listed above.
    """
    if additional_metrics is None:
        additional_metrics = ["dice"]

    if output_softmax.ndim != 4 or output_softmax.shape[1] != 2:
        raise ValueError("ged_binary_fast expects (P, 2, H, W) softmax input for binary segmentation")

    device = output_softmax.device
    P, C, H, W = output_softmax.shape
    gt = ground_truth.to(device)
    if gt.ndim != 3:
        raise ValueError("ged_binary_fast expects ground_truth of shape (G, H, W)")
    G = gt.shape[0]

    # Argmax predictions once
    pred_idx = output_softmax.argmax(dim=1)  # (P, H, W) with values in {0,1}

    # Valid mask from GT; ignore mask applied per GT slice
    if ignore_index is None:
        gt_valid = torch.ones_like(gt, dtype=torch.bool)
    else:
        gt_valid = (gt != ignore_index)

    # Build (P,G) dice matrix versus GT using only valid pixels in each GT slice
    pred_rep = pred_idx.unsqueeze(1)  # (P,1,H,W)
    gt_rep = gt.unsqueeze(0)          # (1,G,H,W)
    valid = gt_valid.unsqueeze(0)     # (1,G,H,W)

    pred_pos = (pred_rep == 1) & valid        # (P,G,H,W)
    gt_pos = (gt_rep == 1) & valid            # (P,G,H,W)

    tp = (pred_pos & gt_pos).sum(dim=(2, 3)).to(torch.float32)  # (P,G)
    pred_sum = pred_pos.sum(dim=(2, 3)).to(torch.float32)       # (P,G)
    gt_sum = gt_pos.sum(dim=(2, 3)).to(torch.float32)           # (P,G)

    denom = 2 * tp + (pred_sum - tp) + (gt_sum - tp)            # 2TP+FP+FN

    both_empty = (pred_sum == 0) & (gt_sum == 0)
    one_empty = (pred_sum == 0) ^ (gt_sum == 0)
    dice_pg = torch.zeros_like(denom, dtype=torch.float32)
    dice_pg[both_empty] = 1.0
    dice_pg[one_empty] = 0.0
    regular = ~(both_empty | one_empty)
    safe = denom > 0
    idx = regular & safe
    dice_pg[idx] = (2 * tp[idx]) / denom[idx]                  # (P,G)

    # Dist terms
    dist_gt_pred_2 = (1.0 - dice_pg).mean().item()

    # pred-pred distance using binary label masks
    pred_bin = (pred_idx == 1)                                 # (P,H,W)
    F = pred_bin.reshape(P, -1).to(torch.float32)              # (P,HW)
    tp_mat = F @ F.T                                           # (P,P)
    pos = F.sum(dim=1)                                         # (P,)
    denom_pp = pos[:, None] + pos[None, :]
    dice_pp = torch.ones_like(denom_pp, dtype=torch.float32)
    mask_pp = denom_pp > 0
    dice_pp[mask_pp] = (2.0 * tp_mat[mask_pp]) / denom_pp[mask_pp]
    dist_pred_pred_2 = float((1.0 - dice_pp).mean().item())

    # gt-gt distance (respect ignore mask of target j)
    gt_bin = (gt == 1)                                         # (G,H,W)
    dist_list = []
    for j in range(G):
        valid_j = gt_valid[j]                                  # (H,W)
        gtj_bin = gt_bin[j] & valid_j
        gtj_sum = gtj_bin.sum().to(torch.float32)
        gi_bin = gt_bin & valid_j.unsqueeze(0)                  # (G,H,W)
        tp_g = (gi_bin & gtj_bin.unsqueeze(0)).sum(dim=(1, 2)).to(torch.float32)  # (G,)
        gi_sum = gi_bin.sum(dim=(1, 2)).to(torch.float32)      # (G,)
        denom_g = gi_sum + gtj_sum
        dice_g = torch.ones_like(denom_g, dtype=torch.float32)
        mask_g = denom_g > 0
        dice_g[mask_g] = (2.0 * tp_g[mask_g]) / denom_g[mask_g]
        dist_list.append(1.0 - dice_g.mean())
    dist_gt_gt_2 = float(torch.stack(dist_list).mean().item()) if dist_list else 0.0

    ged = 2 * dist_gt_pred_2 - dist_pred_pred_2 - dist_gt_gt_2
    results: Dict[str, float] = {"ged": float(ged)}

    if "dice" in additional_metrics:
        results["dice"] = float(dice_pg.mean().item())
    if "max_dice_pred" in additional_metrics:
        results["max_dice_pred"] = float(dice_pg.max(dim=1).values.mean().item())
    if "max_dice_gt" in additional_metrics:
        results["max_dice_gt"] = float(dice_pg.max(dim=0).values.mean().item())
    if "major_dice" in additional_metrics:
        # Majority prediction from probabilities, and majority GT from raters
        majority_pred = output_softmax.mean(dim=0).argmax(dim=0)  # (H,W) in {0,1}
        # For GT, threshold fraction of positive raters (>= 0.5)
        # Respect ignore mask by excluding those pixels from the dice.
        pos_frac = (gt == 1).to(torch.float32).mean(dim=0)        # (H,W)
        majority_gt = (pos_frac >= 0.5).to(torch.long)            # (H,W) in {0,1}
        if ignore_index is not None:
            valid = (gt != ignore_index).all(dim=0)               # (H,W) True if no rater ignored
        else:
            valid = torch.ones_like(majority_gt, dtype=torch.bool)
        # Binary dice for a single pair with mask
        pred_pos = (majority_pred == 1) & valid
        gt_pos = (majority_gt == 1) & valid
        tp_m = (pred_pos & gt_pos).sum().to(torch.float32)
        pred_sum_m = pred_pos.sum().to(torch.float32)
        gt_sum_m = gt_pos.sum().to(torch.float32)
        if pred_sum_m == 0 and gt_sum_m == 0:
            major_dice = 1.0
        elif pred_sum_m == 0 or gt_sum_m == 0:
            major_dice = 0.0
        else:
            denom_m = pred_sum_m + gt_sum_m
            major_dice = float((2.0 * tp_m / denom_m).item())
        results["major_dice"] = major_dice

    return results
