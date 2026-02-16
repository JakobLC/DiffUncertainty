import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product
from pathlib import Path
import matplotlib.patheffects as pe
import torchvision
import sys
import torch
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
sys.path.append("/home/jloch/Desktop/diff/luzern/values")
sys.path.append("/home/jloch/Desktop/diff/luzern/values/uncertainty_modeling/")
from uncertainty_modeling.test_2D import test_cli, Tester
from unittest.mock import patch
with patch("sys.argv", ["notebook"]):
    args = test_cli()

#AU, EU, TU = "aleatoric_uncertainty", "epistemic_uncertainty", "predictive_uncertainty"
AU, EU, TU = "AU", "EU", "TU" # new
def load_result_table(epoch=320,
                      ema = "_ema",
                loop_params = {
                    "AU": ["softmax", "ssn", "diffusion" ],
                    "EU": [ "swag_diag", "swag", "dropout", "ensemble"],
                    "network": [ "unet-s"],
                },
                is_ood_aug = True,
                formatter = "{EU}_{AU}_{network}_lidc_2d_small",
                save_path = "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/test_results/",
                aggregation_type="patch_level",
                split_as_dict = True,
                swap_AU_EU = False):
    table = pd.DataFrame()
    first_aggr_check = True
    for values in product(*loop_params.values()):
        add_dict = dict(zip(loop_params.keys(),values))
        version = formatter.format(**add_dict)#.replace("ensemble","ens12[1,2,3,4,5]")
        if not (Path(f"{save_path}")/version).exists():
            print("Skipping missing version:", version)
            continue
        add_dict["version"] = version
        p = f"{save_path}{version}/e{epoch}{ema}/val/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["val_dice"] = loaded["mean"]["metrics"]["dice"]
        if add_dict["EU"] == "none":# swap
            add_dict["val_ged"] = loaded["mean"]["metrics"]["ged_bma"]
            add_dict["val_ged_bma"] = loaded["mean"]["metrics"]["ged"]
        else:
            add_dict["val_ged"] = loaded["mean"]["metrics"]["ged"]
            add_dict["val_ged_bma"] = loaded["mean"]["metrics"]["ged_bma"]
        
        p = f"{save_path}{version}/e{epoch}{ema}/id/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["id_dice"] = loaded["mean"]["metrics"]["dice"]
        if add_dict["EU"] == "none":# swap
            add_dict["id_ged"] = loaded["mean"]["metrics"]["ged_bma"]
            add_dict["id_ged_bma"] = loaded["mean"]["metrics"]["ged"]
        else:
            add_dict["id_ged"] = loaded["mean"]["metrics"]["ged"]
            add_dict["id_ged_bma"] = loaded["mean"]["metrics"]["ged_bma"]
        
        p = f"{save_path}{version}/e{epoch}{ema}/id/ambiguity_modeling.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["EU_ncc_id"] = loaded["mean"][EU]["metrics"]["ncc"]
        add_dict["AU_ncc_id"] = loaded["mean"][AU]["metrics"]["ncc"]
        add_dict["TU_ncc_id"] = loaded["mean"][TU]["metrics"]["ncc"]
        p = f"{save_path}{version}/e{epoch}{ema}/id/calibration.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["EU_ace_id"] = loaded["mean"][EU]["metrics"]["ace"]
        add_dict["AU_ace_id"] = loaded["mean"][AU]["metrics"]["ace"]
        add_dict["TU_ace_id"] = loaded["mean"][TU]["metrics"]["ace"]
        p = f"{save_path}{version}/e{epoch}{ema}/ood_detection.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        if is_ood_aug:
            for k in loaded.keys():
                k2 = k.replace("id&","")
                if first_aggr_check:
                    avail_aggr_types = loaded[k]["mean"][EU].keys()
                    assert aggregation_type in avail_aggr_types, f"{aggregation_type} Invalid aggregation type, available: {avail_aggr_types}"
                    first_aggr_check = False
                add_dict[f"EU_auc_{k2}"] = loaded[k]["mean"][EU][aggregation_type]["metrics"]["auroc"]
                add_dict[f"AU_auc_{k2}"] = loaded[k]["mean"][AU][aggregation_type]["metrics"]["auroc"]
                add_dict[f"TU_auc_{k2}"] = loaded[k]["mean"][TU][aggregation_type]["metrics"]["auroc"]

            for p in Path(f"{save_path}{version}/e{epoch}{ema}").glob("ood*/ambiguity_modeling.json"):
                k2 = p.parts[-2]
                with open(p, "r") as f:
                    loaded = json.load(f)
                add_dict[f"EU_ncc_{k2}"] = loaded["mean"][EU]["metrics"]["ncc"]
                add_dict[f"AU_ncc_{k2}"] = loaded["mean"][AU]["metrics"]["ncc"]
                add_dict[f"TU_ncc_{k2}"] = loaded["mean"][TU]["metrics"]["ncc"]
            for p in Path(f"{save_path}{version}/e{epoch}{ema}").glob("ood*/calibration.json"):
                k2 = p.parts[-2]
                with open(p, "r") as f:
                    loaded = json.load(f)
                add_dict[f"EU_ace_{k2}"] = loaded["mean"][EU]["metrics"]["ace"]
                add_dict[f"AU_ace_{k2}"] = loaded["mean"][AU]["metrics"]["ace"]
                add_dict[f"TU_ace_{k2}"] = loaded["mean"][TU]["metrics"]["ace"]
        else:
            add_dict["EU_auc"] = loaded["mean"][EU][aggregation_type]["metrics"]["auroc"]
            add_dict["AU_auc"] = loaded["mean"][AU][aggregation_type]["metrics"]["auroc"]
            add_dict["TU_auc"] = loaded["mean"][TU][aggregation_type]["metrics"]["auroc"]
            p = f"{save_path}{version}/e{epoch}{ema}/ood/ambiguity_modeling.json"
            with open(p, "r") as f:
                loaded = json.load(f)
            add_dict["EU_ncc_ood"] = loaded["mean"][EU]["metrics"]["ncc"]
            add_dict["AU_ncc_ood"] = loaded["mean"][AU]["metrics"]["ncc"]
            add_dict["TU_ncc_ood"] = loaded["mean"][TU]["metrics"]["ncc"]
            p = f"{save_path}{version}/e{epoch}{ema}/ood/calibration.json"
            with open(p, "r") as f:
                loaded = json.load(f)
            add_dict["EU_ace_ood"] = loaded["mean"][EU]["metrics"]["ace"]
            add_dict["AU_ace_ood"] = loaded["mean"][AU]["metrics"]["ace"]
            add_dict["TU_ace_ood"] = loaded["mean"][TU]["metrics"]["ace"]
        table = pd.concat([table, pd.DataFrame([add_dict])], ignore_index=True)
    #raise error incase table is empty
    if table.empty:
        raise ValueError(f"Result table is empty. Check if the specified epoch and save_path are correct. Current save_path: {save_path}")
    valid_ood_keys = []
    for k in table.columns:
        if k.startswith("EU_auc_ood"):
            valid_ood_keys.append(k.replace("EU_auc_",""))
    if swap_AU_EU:
        # swap AU and EU columns
        mapper = {}
        for col in table.columns:
            if "AU_" in col or "EU_" in col:
                new_col = col.replace("AU", "TEMP").replace("EU", "AU").replace("TEMP", "EU")
                mapper[col] = new_col
        table = table.rename(columns=mapper)
    for k2 in valid_ood_keys:
        table[f"(AU-EU)_ncc_{k2}"] = (table[f"AU_ncc_{k2}"] - table[f"EU_ncc_{k2}"])/table[f"AU_ncc_{k2}"]
        table[f"(EU-AU)_auc_{k2}"] = (table[f"EU_auc_{k2}"] - table[f"AU_auc_{k2}"])/table[f"EU_auc_{k2}"]
        table[f"min(AU,EU)_ace_{k2}"] = table[[f"AU_ace_{k2}", f"EU_ace_{k2}"]].min(axis=1)
        table[f"(TU-min(AU,EU))_ace_{k2}"] = (table[f"TU_ace_{k2}"]-table[f"min(AU,EU)_ace_{k2}"])/table[f"TU_ace_{k2}"]
    #same as above but for non-ood keys
    table[f"(AU-EU)_ncc_id"] = (table["AU_ncc_id"] - table["EU_ncc_id"])/table["AU_ncc_id"]
    table[f"min(AU,EU)_ace_id"] = table[["AU_ace_id", "EU_ace_id"]].min(axis=1)
    table[f"(TU-min(AU,EU))_ace_id"] = (table["TU_ace_id"]-table[f"min(AU,EU)_ace_id"])/table["TU_ace_id"]
    if split_as_dict:
        # look for column names sharing the same prefix before a valid split key, i.e.
        # [id, val, ood, ood_[something]]
        # merge into a dict as table[prefix] = {split_key: value, ...}
        split_keys = ["id", "val"] + valid_ood_keys
        new_table = pd.DataFrame()
        for col in table.columns:
            matched = False
            for sk in split_keys:
                suffix = f"_{sk}"
                if col.endswith(suffix):
                    prefix = col[:-len(suffix)]
                    if prefix not in new_table.columns:
                        new_table[prefix] = [{} for _ in range(len(table))]
                    for i in range(len(table)):
                        new_table.at[i, prefix][sk] = table.at[i, col]
                    matched = True
                    break
            if not matched:
                new_table[col] = table[col]
        table = new_table

    return table

def entropy(probs, dim=1, eps=1e-8):
    return -torch.sum(probs * torch.log(probs + eps), dim=dim)


def _minmax_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] while handling degenerate ranges."""
    min_val = tensor.amin()
    max_val = tensor.amax()
    denom = max_val - min_val
    if denom < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (denom + 1e-8)


def _to_three_channel_grayscale(img: torch.Tensor) -> torch.Tensor:
    """Expand a 2D map to 3-channel grayscale for visualization."""
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.dim() != 2:
        raise ValueError("Expected a single-channel 2D tensor for grayscale conversion.")
    img = img.clamp(0, 1)
    return img.unsqueeze(0).repeat(3, 1, 1)


def apply_colormap_tensor(
    img: torch.Tensor,
    cmap_name: str = "CMRmap",
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    """Apply a matplotlib colormap to a 2D tensor with optional range control."""
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.dim() != 2:
        raise ValueError("Colormap expects a single-channel tensor.")
    device = img.device
    img_cpu = img.detach().float().cpu()
    vmin = float(img_cpu.min()) if vmin is None else float(vmin)
    vmax = float(img_cpu.max()) if vmax is None else float(vmax)
    denom = max(vmax - vmin, 1e-8)
    img_norm = ((img_cpu - vmin) / denom).clamp(0, 1)
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(img_norm.numpy())[..., :3]
    colored_tensor = torch.from_numpy(colored).permute(2, 0, 1).to(device=device)
    return colored_tensor.clamp(0, 1)


def _get_colors(num_classes: int, device, dtype) -> torch.Tensor:
    # repeats the same bright saturated colors
    base_cmap = ["#000000", "#ff00a2", "#34ff30","#f24b4d", "#81c2f7", "#7300ff", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
    colors = []
    for i in range(num_classes):
        color_hex = base_cmap[i % len(base_cmap)]
        color_rgb = tuple(int(color_hex[j : j + 2], 16) for j in (1, 3, 5))
        colors.append(color_rgb)
    return torch.tensor(colors, dtype=dtype, device=device)/255.0


def onehot_to_rgb(onehot: torch.Tensor) -> torch.Tensor:
    """Render a (C, H, W)/(H, W, C) one-hot/probability map with tab10 colors."""
    if onehot.dim() != 3:
        raise ValueError("Expected a 3D tensor for one-hot visualization.")
    if 1 <= onehot.shape[0] <= 64:
        tensor = onehot
    elif 1 <= onehot.shape[-1] <= 64:
        tensor = onehot.permute(2, 0, 1)
    else:
        raise ValueError("Cannot infer channel dimension for one-hot tensor.")
    tensor = tensor.clamp_min(0).float()
    num_classes = tensor.shape[0]
    probs = tensor / tensor.sum(dim=0, keepdim=True).clamp_min(1e-8)
    colors = _get_colors(num_classes, probs.device, probs.dtype)
    rgb = torch.matmul(colors.T, probs.view(num_classes, -1)).view(3, probs.shape[1], probs.shape[2])
    return rgb.clamp(0, 1)


def prediction_to_display(prob_map: torch.Tensor) -> torch.Tensor:
    """Convert a probability map (C,H,W) into a 3-channel visualization."""
    if prob_map.dim() == 2:
        prob_map = prob_map.unsqueeze(0)
    if prob_map.dim() != 3:
        raise ValueError("Probability map must be 2D or 3D.")
    channels = prob_map.shape[0]
    if channels == 1:
        return _to_three_channel_grayscale(prob_map.squeeze(0))
    if channels == 2:
        return _to_three_channel_grayscale(prob_map[1])
    return onehot_to_rgb(prob_map)


def format_image_tensor_for_display(image_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the raw image is shown as grayscale or RGB based on its channels."""
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.shape[0] == 1:
        return _to_three_channel_grayscale(image_tensor.squeeze(0))
    return _minmax_normalize(image_tensor[:3])


def prepare_image_tensor(image_data) -> torch.Tensor:
    """Convert stored image arrays (H,W)/(H,W,3)/(3,H,W) into normalized tensors."""
    image = torch.as_tensor(image_data, dtype=torch.float32)
    if image.dim() == 2:
        image = image.unsqueeze(0)
    elif image.dim() == 3:
        if image.shape[0] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        else:
            raise ValueError("Unsupported image channel arrangement.")
    else:
        raise ValueError("Unsupported image rank.")
    return _minmax_normalize(image[:3])

def pred_grid_computation(ckpt_path = "/home/jloch/Desktop/diff/luzern/values/saves/lidc64/diffusion_swag/checkpoints/last.ckpt",
                          split = "id_test",
                          test_batch_size = 16):
    args.checkpoint_paths = [ckpt_path]
    args.test_split = split
    args.test_batch_size = test_batch_size
    tester = Tester(args)
    raw_batches = tester.collect_raw_predictions(max_batches=1)

    batch0 = raw_batches[0]
    data = Path(ckpt_path).parts[-4]
    p = f"/home/jloch/Desktop/diff/luzern/values_datasets/{data}/preprocessed/images/{{id}}.npy"
    images = [np.load(p.format(id=f"{batch0['image_id'][i]}")) for i in range(len(batch0["image_id"]))]
    return images, batch0

def plot_pred_grid(images, batch0, i=0, crop_unused=False, s=1.3, entropy_cmap="viridis"):
    image = prepare_image_tensor(images[i])

    pred = torch.stack(batch0["softmax_pred_groups"], dim=0).float().cpu()
    per_image_preds = pred[:, :, i]  # (n_AU, n_EU, C, H, W)
    if per_image_preds.shape[2] > 1:
        foreground_mass = per_image_preds[:, :, 1:, :, :]
    else:
        foreground_mass = per_image_preds
    sum_image = foreground_mass.sum(dim=(0, 1, 2))
    if crop_unused:
        bbox_crop = sum_image > sum_image.mean() * 0.01
        d1_indices = torch.where(bbox_crop.sum(1) > 0)[0]
        d2_indices = torch.where(bbox_crop.sum(0) > 0)[0]
        d1_min = d1_indices.min().item()
        d1_max = d1_indices.max().item() + 1
        d2_min = d2_indices.min().item()
        d2_max = d2_indices.max().item() + 1
        assert d1_max > d1_min and d2_max > d2_min, f"Invalid slice: {d1_min},{d1_max},{d2_min},{d2_max}"
        d1 = slice(d1_min, d1_max)
        d2 = slice(d2_min, d2_max)
    else:
        d1 = slice(None)
        d2 = slice(None)
    pred_cropped = per_image_preds[:, :, :, d1, d2].contiguous()
    num_classes = max(int(pred_cropped.shape[2]), 1)
    entropy_vmin = 0.0
    entropy_vmax = math.log(num_classes)
    E_y_p_full = batch0["softmax_pred"].float().cpu()[:, i, :, d1, d2].contiguous()
    image = image[:, d1, d2]
    image_display = format_image_tensor_for_display(image)
    n_AU, n_EU = pred_cropped.shape[:2]
    if n_EU < 4:
        raise ValueError("plot_pred_grid expects at least 4 EU samples to render the layout.")
    grid_dims = (n_AU + 2, n_EU + 4)
    height, width = pred_cropped.shape[-2:]
    blank_tile = torch.ones((3, height, width), dtype=pred_cropped.dtype)

    H_E_y_p = entropy(E_y_p_full, dim=1)
    AU_map = H_E_y_p.mean(0, keepdim=True)
    E_th_E_y_p = E_y_p_full.mean(0, keepdim=True)
    TU_map = entropy(E_th_E_y_p, dim=1)
    EU_map = TU_map - AU_map

    grid_rows = []
    for au_idx in range(n_AU):
        row_tiles = []
        for eu_idx in range(n_EU):
            row_tiles.append(prediction_to_display(pred_cropped[au_idx, eu_idx]))
        row_tiles.append(blank_tile.clone())
        row_tiles.append(prediction_to_display(E_y_p_full[au_idx]))
        row_tiles.append(blank_tile.clone())
        row_tiles.append(
            apply_colormap_tensor(
                H_E_y_p[au_idx],
                cmap_name=entropy_cmap,
                vmin=entropy_vmin,
                vmax=entropy_vmax,
            )
        )
        grid_rows.append([tile.detach().cpu() for tile in row_tiles])

    grid_rows.append([blank_tile.clone().detach().cpu() for _ in range(grid_dims[1])])

    bottom_row = [image_display]
    bottom_row.extend([blank_tile.clone() for _ in range(n_EU - 4)])
    bottom_row.extend([
        apply_colormap_tensor(
            EU_map.squeeze(0),
            cmap_name=entropy_cmap,
            vmin=entropy_vmin,
            vmax=entropy_vmax,
        ),
        blank_tile.clone(),
        apply_colormap_tensor(
            TU_map.squeeze(0),
            cmap_name=entropy_cmap,
            vmin=entropy_vmin,
            vmax=entropy_vmax,
        ),
        blank_tile.clone(),
        prediction_to_display(E_th_E_y_p.squeeze(0)),
        blank_tile.clone(),
        apply_colormap_tensor(
            AU_map.squeeze(0),
            cmap_name=entropy_cmap,
            vmin=entropy_vmin,
            vmax=entropy_vmax,
        ),
    ])
    grid_rows.append([tile.detach().cpu() for tile in bottom_row])

    assert len(grid_rows) == grid_dims[0]
    for row in grid_rows:
        assert len(row) == grid_dims[1]

    tiles = [tile.float() for row in grid_rows for tile in row]
    grid_stack = torch.stack(tiles, dim=0)

    grid2 = torchvision.utils.make_grid(
        grid_stack,
        nrow=grid_dims[1],
        padding=2,
        value_range=[0, 1],
        pad_value=1,
    )
    grid_img = grid2.permute(1, 2, 0).numpy()
    grid_h, grid_w = grid_img.shape[:2]
    grid_w_unit = grid_w / grid_dims[1]
    grid_h_unit = grid_h / grid_dims[0]

    # plot
    plt.figure(figsize=(8 * s, 8 * s))
    plt.imshow(grid_img)
    plt.axis("off")
    #plot a sequence of arrows to indicate direction from left (model samples) to right (ensemble)

    for j in range(10):
        arrow_x_positions = [
            grid_w_unit * (grid_dims[1] - 4 + 0.2),
            grid_w_unit * (grid_dims[1] - 2 + 0.2),
        ]
        for x, letter in zip(arrow_x_positions, ["E", "H"]):
            plt.arrow(
                x=x,
                y=grid_h_unit * (j + 0.5),
                dx=grid_w_unit * 0.6,
                dy=0,
                width=2,
                head_width=grid_h_unit * 0.2,
                head_length=grid_w_unit * 0.2,
                length_includes_head=True,
                color="black"
            )
            #plt expectation "E" ontop of arrows
            plt.text(
                x=x + grid_w_unit * 0.2,
                y=grid_h_unit * (j + 0.5),
                s=letter,
                color="white",
                fontsize=14*s,
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            )

    pos_to_title = {"top_row": {n_AU*1/6: "Varying AU \u2192", 
                                n_AU*1/2: "$p=p(y|x,\\theta)$", 
                                grid_dims[1]-2.5: "$E_y[p]$",
                                grid_dims[1]-0.5: "$H(E_y[p])$"},
                    "left_col": {n_EU*1/6: "\u2190 Varying EU ($\\theta$)"},
                    "btm_row": {grid_dims[1]-6.5: "$EU=TU-AU$\n(MI)",
                                grid_dims[1]-4.5: "$H(E_{\\theta}[E_y[p]])$\nTU",
                                grid_dims[1]-2.5: "$E_{\\theta}[E_y[p]]$",
                                grid_dims[1]-0.5: "$E_{\\theta}[H(E_y[p])]$\nAU"},
                                }

    fontsize = 10
    for pos, title in pos_to_title["top_row"].items():
        plt.text(
            x=pos * grid_w_unit,
            y=0,
            s=title,
            color="black",
            fontsize=fontsize*s,
            ha="center",
            va="bottom",
        )
    for pos, title in pos_to_title["left_col"].items():
        plt.text(
            x=0,
            y=pos * grid_h_unit,
            s=title,
            color="black",
            fontsize=fontsize*s,
            ha="right",
            va="center",
            rotation=90,
        )
    for pos, title in pos_to_title["btm_row"].items():
        plt.text(
            x=pos * grid_w_unit,
            y=grid_h,
            s=title,
            color="black",
            fontsize=fontsize*s,
            ha="center",
            va="top",
        )

    # add down arrows below each column
    for j in range(2):
        plt.arrow(
            x=grid_w_unit * (grid_dims[1] - 2.5 + j * 2),
            y=grid_h_unit * (n_EU + 0.2),
            dx=0,
            dy=grid_w_unit * 0.6,
            width=2,
            head_width=grid_h_unit * 0.2,
            head_length=grid_w_unit * 0.2,
            length_includes_head=True,
            color="black"
        )
        plt.text(
            x=grid_w_unit * (grid_dims[1] - 2.5 + j * 2),
            y=grid_h_unit * (n_EU + 0.2) + grid_w_unit * 0.2,
            s="E",
            color="white",
            fontsize=14*s,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
        )
    plt.arrow(
        x=grid_w_unit * (grid_dims[1] - 4 + 0.4) + grid_w_unit * 0.4,
        y=grid_h_unit * 11.5,
        dx=-grid_w_unit * 0.6,
        dy=0,
        width=2,
        head_width=grid_h_unit * 0.2,
        head_length=grid_w_unit * 0.2,
        length_includes_head=True,
        color="black"
    )
    plt.text(
        x=grid_w_unit * (grid_dims[1] - 4 + 0.2) + grid_w_unit * 0.4,
        y=grid_h_unit * 11.5,
        s="H",
        color="white",
        fontsize=14*s,
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")]
    )
    plt.tight_layout()

def get_bbox_col(matrix, value, is_ace):
    # get min and max of matrix
    mask = ~matrix.isna()
    vmin = matrix.values[mask].min()
    vmax = matrix.values[mask].max()
    # normalize value to [0,1]
    norm_value = (value - vmin) / (vmax - vmin + 1e-8)
    # get colormap
    #cmap = plt.get_cmap("RdYlGn")
    if is_ace:
        colors = ["#00FF00", "#FFFF00", "#FF0000"]
    else:
        colors = ["#FF0000", "#FFFF00", "#00FF00"]
    cmap = LinearSegmentedColormap.from_list(
        "pure_RdYlGn",
        colors,
        N=256
    )
    # get color from colormap
    color = cmap(norm_value)
    return color

GRID_ORDER_AU = ["softmax", "ssn", "prob_unet", "diffusion"]
GRID_ORDER_EU = ["none", "dropout", "swag_diag", "swag", "ensemble"]

def plot_metric_matrix(table,
            ax = None,
            metric = "val_ged",
            subkey = "id",
            index = "AU",
            columns = "EU",
            figsize_per_cell=(2,1.7),
            cmap="magma",
            reldiff = None,
            cbar_keys = None,
            cbar_vals = None,
            ):
    matrix = table.pivot(index=index, columns=columns, values=metric)
    #reorder rows and columns according to GRID_ORDER_AU and GRID_ORDER_EU
    matrix = matrix.reindex(index=GRID_ORDER_AU, columns=GRID_ORDER_EU)
    m,n = matrix.shape
    has_dicts = matrix.applymap(lambda x: isinstance(x, dict)).values.any()

    if ax is None:
        fig,ax = plt.subplots(figsize=(figsize_per_cell[0]*n, figsize_per_cell[1]*m))
    if has_dicts:
        matrix = matrix.applymap(lambda x: x[subkey] if isinstance(x, dict) and subkey in x else np.nan)
        ax.set_title(f"metric: {metric} ({subkey})", fontsize=20)
    else:
        ax.set_title(f"metric: {metric}", fontsize=20)
    if isinstance(figsize_per_cell, (int, float)):
        figsize_per_cell = (figsize_per_cell, figsize_per_cell)
    

    cmap = get_mm_colormap(metric)
    if cbar_vals is not None:
        vmin, vmax = cbar_vals
    elif cbar_keys is not None:
        vmin = float('inf')
        vmax = float('-inf')
        for key in cbar_keys:
            mat = table.pivot(index=index, columns=columns, values=key)
            mat = mat.reindex(index=GRID_ORDER_AU, columns=GRID_ORDER_EU)   
            has_dicts = mat.applymap(lambda x: isinstance(x, dict)).values.any()
            if has_dicts:
                vmin = mat.applymap(lambda x: min(x.values()) if isinstance(x, dict) else float('inf')).values.min()
                vmax = mat.applymap(lambda x: max(x.values()) if isinstance(x, dict) else float('-inf')).values.max()
            else:
                vmin = min(vmin, mat.values.min())
                vmax = max(vmax, mat.values.max())
    else:
        # non-nan:
        mask = ~matrix.isna()
        vmin = matrix.values[mask].min()
        vmax = matrix.values[mask].max()
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
              vmin=vmin,vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if reldiff:
        reldiff_matrix = table.pivot(index=index, columns=columns, values=reldiff)
        reldiff_matrix = reldiff_matrix.reindex(index=GRID_ORDER_AU, columns=GRID_ORDER_EU)
        has_dicts = reldiff_matrix.applymap(lambda x: isinstance(x, dict)).values.any()
        if has_dicts:
            reldiff_matrix = reldiff_matrix.applymap(lambda x: x[subkey] if isinstance(x, dict) and subkey in x else np.nan)
    for (i, row_label) in enumerate(matrix.index):
        for (j, col_label) in enumerate(matrix.columns):
            value = matrix.loc[row_label, col_label]
            if reldiff:
                bg_box_color = get_bbox_col(reldiff_matrix,reldiff_matrix.loc[row_label, col_label], is_ace="ace" in metric)
                ax.text(j,i, f"{-reldiff_matrix.loc[row_label, col_label]:+.1%}",
                ha="center", va="top",
                fontsize=plt.rcParams["font.size"] * 1.3,
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                bbox=dict(boxstyle="round,pad=0.3", fc=bg_box_color, ec="black", lw=1)
                )
            ax.text(
                j, i, f"{value:.3f}"+("\n" if reldiff else ""),
                ha="center", va="center",
                fontsize=plt.rcParams["font.size"] * 2,
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            )
    ax.set_xticks(ticks=np.arange(len(matrix.columns)))
    ax.set_xticklabels(labels=matrix.columns)
    ax.set_yticks(ticks=np.arange(len(matrix.index)))
    ax.set_yticklabels(labels=matrix.index)
    #add labels to axes
    ax.set_xlabel(columns, fontsize=16)
    ax.set_ylabel(index, fontsize=16)
    return {"fig": ax.figure, "ax": ax, "cbar": cbar, "matrix": matrix}

def get_mm_colormap(name):
    cmap = "viridis" if name.startswith("(") else "magma"
    if ("_ged" in name) or ("_ace" in name):
        cmap += "_r"
    return cmap

def plot_mm_grid(table,
                 kwarg_grid=[[{"metric": "val_dice"},{"metric": "id_dice"}],
                              [{"metric": "val_ged"},{"metric": "id_ged"}]],
                              same_cbar = True,
                              same_cbar_vals = None,):
    n_rows = len(kwarg_grid)
    n_cols = len(kwarg_grid[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*6))
    out_mm = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    #check if 1x1 plot
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes[:, None] if n_rows > 1 else axes[None, :]
    if same_cbar_vals:
        assert isinstance(same_cbar_vals, (tuple,list)) and len(same_cbar_vals) == 2, "same_cbar_vals should be a tuple/list of (vmin, vmax)"
        vmin, vmax = same_cbar_vals
    for i in range(n_rows):
        for j in range(n_cols):
            if not kwarg_grid[i][j]:
                axes[i,j].axis("off")
                continue
            kwargs = kwarg_grid[i][j]
            kwargs["cbar_vals"] = (vmin, vmax) if same_cbar_vals else None
            out = plot_metric_matrix(table, ax=axes[i,j], **kwargs)
            if same_cbar:
                #remove colorbar from individual plots
                saved_cbar = out["cbar"]
                saved_cbar.remove()
            out_mm[i][j] = out
    if same_cbar:
        #add a big colorbar to the right of the grid
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        #use the last saved_cbar mappable
        fig.colorbar(saved_cbar.mappable, cax=cbar_ax)
    return out_mm


AU_to_marker = {"softmax":"o","ssn":"s","diffusion":"*","prob_unet":"X"}
EU_to_color = {"dropout":"C3","swag":"C0","swag_diag":"C9","ensemble":"C1","none":"C7"}
def _idx(row, key_with_subkey):
    if not "[" in key_with_subkey or not key_with_subkey.endswith("]"):
        return row.get(key_with_subkey, np.nan)
    key, subkey = key_with_subkey.split("[")
    subkey = subkey[:-1]
    if key in row and isinstance(row[key], dict) and subkey in row[key]:
        return row[key][subkey]
    else:            
        raise ValueError(f"Key {key_with_subkey} not found in row or not a dict with subkey: {row}")
    
def plot_4_stat_arrows(table,x1="TU_ace[id]",y1="TU_ace[ood]",x2="min(AU,EU)_ace[id]",y2="min(AU,EU)_ace[ood]",
                        add_xy=True):
    # goes through all models, plots an arrow from (x1,y1) to (x2,y2) with marker and color determined by the model type

    xmin,xmax = float('inf'),float('-inf')
    ymin,ymax = float('inf'),float('-inf')
    plt.figure(figsize=(8,8))
    for idx, row in table.iterrows():
        AU_type = row["AU"]
        EU_type = row["EU"]
        marker = AU_to_marker.get(AU_type, "o")
        color = EU_to_color.get(EU_type, "C7")
        X1, Y1 = _idx(row, x1), _idx(row, y1)
        X2, Y2 = _idx(row, x2), _idx(row, y2)
        xmin, xmax = min(xmin, X1, X2), max(xmax, X1, X2)
        ymin, ymax = min(ymin, Y1, Y2), max(ymax, Y1, Y2)
        plt.scatter(X1, Y1, marker=marker, color=color, s=100)
        plt.scatter(X2, Y2, marker=marker, color=color, s=0)
        plt.annotate("",
                    xy=(X2, Y2),
                    xytext=(X1, Y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
    plt.xlabel(x1)
    plt.ylabel(y1)
    if add_xy:
        #add x=y line
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        lim_min = min(xmin, ymin)
        lim_max = max(xmax, ymax)
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k', linestyle='--')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    plt.title(f"Arrows from ({x1}, {y1}) to ({x2}, {y2})")
    plt.grid()
    #create legend for AU types
    for AU_type, marker in AU_to_marker.items():
        plt.scatter([], [], marker=marker, color="black", label=AU_type, s=100)
    for EU_type, color in EU_to_color.items():
        plt.scatter([], [], marker="o", color=color, label=EU_type, s=100)
    plt.legend()
    plt.tight_layout()


def model_scatter(table, x="AU_auc[ood]", y="EU_auc[ood]", add_xy=True, entangle_is_up=False,
                  xlabel=None, ylabel=None, title=None, plot_legend=True, equal_scaling=True, 
                  ignore_for_axis=None, only_legend=False, ax=None):
    # similar to the arrow plot without the arrows
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    
    # If only_legend is True, just create the legend and turn off axis
    if only_legend:
        ax.axis('off')
        for EU_type in GRID_ORDER_EU:
            color = EU_to_color.get(EU_type, "C7")
            ax.scatter([], [], marker="o", color=color, label=EU_type, s=100)
        for AU_type in GRID_ORDER_AU:
            marker = AU_to_marker.get(AU_type, "o")
            ax.scatter([], [], marker=marker, color="black", label=AU_type, s=100)
        legend = ax.legend(loc='center', frameon=True, fontsize=15, ncol=2)
        if ax is None:
            fig.tight_layout()
        return {"fig": fig, "ax": ax, "legend": legend}
    
    if ignore_for_axis is None:
        ignore_for_axis = {}

    # Separate points into those used for axis limits and those ignored
    scatter_points = []
    ignored_points = []
    
    for idx, row in table.iterrows():
        AU_type = row["AU"]
        EU_type = row["EU"]
        marker = AU_to_marker.get(AU_type, "o")
        color = EU_to_color.get(EU_type, "C7")
        X, Y = _idx(row, x), _idx(row, y)
        
        # Check if this point should be ignored for axis calculation
        should_ignore = False
        for key, ignore_values in ignore_for_axis.items():
            if key in row and row[key] in ignore_values:
                should_ignore = True
                break
        
        if should_ignore:
            ignored_points.append((X, Y, marker, color))
        else:
            scatter_points.append((X, Y))
            ax.scatter(X, Y, marker=marker, color=color, s=100)

    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Scatter of {y} vs {x}")
    
    # Create legend early to get its bounding box (if needed)
    legend = None
    if plot_legend:
        for EU_type, color in EU_to_color.items():
            ax.scatter([], [], marker="o", color=color, label=EU_type, s=100)
        for AU_type, marker in AU_to_marker.items():
            ax.scatter([], [], marker=marker, color="black", label=AU_type, s=100)
        legend = ax.legend()
    
    # Apply equal scaling if requested
    if equal_scaling:
        # Get current axis limits and figure size
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Get the physical size of the axes in inches
        bbox = ax.get_position()
        fig_width, fig_height = fig.get_size_inches()
        ax_width = bbox.width * fig_width
        ax_height = bbox.height * fig_height
        
        # Calculate aspect ratio of the physical axes
        aspect_ratio = ax_height / ax_width  # height / width
        
        # Apply equal scaling: maintain center, adjust ranges proportional to physical size
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        range_x = xmax - xmin
        range_y = ymax - ymin
        
        # For equal scaling, data_range_y / data_range_x = ax_height / ax_width
        # We want to expand the smaller range to match
        target_range_y = range_x * aspect_ratio
        target_range_x = range_y / aspect_ratio
        
        # Choose the larger of the two scenarios (expand to fit)
        if target_range_y >= range_y:
            # Expand y range
            new_range_x = range_x
            new_range_y = target_range_y
        else:
            # Expand x range
            new_range_x = target_range_x
            new_range_y = range_y
        
        # Set new limits centered around the original centers
        xmin, xmax = center_x - new_range_x / 2, center_x + new_range_x / 2
        ymin, ymax = center_y - new_range_y / 2, center_y + new_range_y / 2
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    # Store the axis limits before plotting additional elements
    xmin_final, xmax_final = ax.get_xlim()
    ymin_final, ymax_final = ax.get_ylim()
    
    # Add x=y line after axis limits are set
    if add_xy:
        lim_min = min(xmin_final, ymin_final)
        lim_max = max(xmax_final, ymax_final)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="k", linestyle="-", zorder=0)
        # Reset limits after plotting line
        ax.set_xlim(xmin_final, xmax_final)
        ax.set_ylim(ymin_final, ymax_final)
    
    # Now plot the ignored points (after axis limits are determined)
    for X, Y, marker, color in ignored_points:
        ax.scatter(X, Y, marker=marker, color=color, s=100)
        scatter_points.append((X, Y))  # Add to scatter_points for arrow placement
    
    # Reset limits again after plotting ignored points
    ax.set_xlim(xmin_final, xmax_final)
    ax.set_ylim(ymin_final, ymax_final)
    
    # Check for out-of-bounds points and display them at the border with faded alpha
    if ignore_for_axis:
        for X, Y in scatter_points:
            if X < xmin_final or X > xmax_final or Y < ymin_final or Y > ymax_final:
                # Point is outside bounds - find closest point on border
                X_clamped = np.clip(X, xmin_final, xmax_final)
                Y_clamped = np.clip(Y, ymin_final, ymax_final)
                
                # Find the corresponding marker and color for this point
                for idx, row in table.iterrows():
                    X_check, Y_check = _idx(row, x), _idx(row, y)
                    if abs(X_check - X) < 1e-10 and abs(Y_check - Y) < 1e-10:
                        AU_type = row["AU"]
                        EU_type = row["EU"]
                        marker = AU_to_marker.get(AU_type, "o")
                        color = EU_to_color.get(EU_type, "C7")
                        # Plot faded marker at border, on top of axis (clip_on=False)
                        ax.scatter(X_clamped, Y_clamped, marker=marker, color=color, s=100, alpha=0.5, 
                                zorder=10, clip_on=False, edgecolors='black', linewidths=1)
                        break
    
    if entangle_is_up is not None:
        # Get current axis limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Get legend bounding box in data coordinates (only if legend exists and is shown)
        legend_xmin, legend_ymin, legend_xmax, legend_ymax = None, None, None, None
        if plot_legend and legend is not None:
            fig.canvas.draw()
            legend_bbox = legend.get_window_extent().transformed(ax.transData.inverted())
            legend_xmin, legend_ymin = legend_bbox.x0, legend_bbox.y0
            legend_xmax, legend_ymax = legend_bbox.x1, legend_bbox.y1
        
        # Calculate center of x=y line within axis limits
        line_start = max(xmin, ymin)
        line_end = min(xmax, ymax)
        
        # Calculate arrow parameters using max axis length
        max_axis_len = max(xmax - xmin, ymax - ymin)
        arrow_len = 0.08 * max_axis_len
        
        # Direction orthogonal to x=y line (normalized)
        # Down-right: (1, -1)/sqrt(2), Up-left: (-1, 1)/sqrt(2)
        if entangle_is_up:
            main_dx, main_dy = arrow_len / np.sqrt(2), -arrow_len / np.sqrt(2)
            text_va = 'bottom'
        else:
            main_dx, main_dy = -arrow_len / np.sqrt(2), arrow_len / np.sqrt(2)
            text_va = 'top'
        
        # Sample 10 positions on x=y line within 20-80% central interval
        num_candidates = 10
        line_range = line_end - line_start
        candidates_on_line = []
        for _ in range(num_candidates):
            r = np.random.rand()
            # Sample from 20-80% of the line
            line_pos = line_start + (0.2 + 0.6 * r) * line_range
            candidates_on_line.append((line_pos, line_pos))
        
        # Find the candidate with the largest minimum distance to scatter points
        best_origin = None
        best_min_dist = -np.inf
        
        for cand_x, cand_y in candidates_on_line:
            # Distance to all scatter points
            distances_to_points = [np.sqrt((cand_x - sx)**2 + (cand_y - sy)**2) 
                                   for sx, sy in scatter_points]
            min_dist = min(distances_to_points) if distances_to_points else np.inf
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_origin = (cand_x, cand_y)
        
        # Use the best origin found
        center_point = best_origin if best_origin is not None else ((line_start + line_end) / 2, (line_start + line_end) / 2)
        
        # Main arrow starting from x=y line at the optimal position
        arrow_end = (center_point[0] - main_dx, center_point[1] - main_dy)
        ax.annotate(
            "",
            xy=arrow_end,
            xytext=center_point,
            arrowprops=dict(arrowstyle="->", color="black", lw=2)
        )
        # Add text near arrow head with central horizontal alignment
        ax.text(
            arrow_end[0], arrow_end[1],
            "Wrong Unc.\nBetter",
            fontsize=10,
            ha='center',
            va=text_va
        )
        
        # Two additional arrows for entanglement
        # Place them on a line orthogonal to x=y (not on x=y line itself)
        # Find optimal position by sampling and choosing point with max min-distance
        # to scatter points, x=y line, and legend
        num_candidates = 10
        candidates = []
        for _ in range(num_candidates):
            r1, r2 = np.random.rand(2)
            cand_x = xmin + (0.15 + 0.7 * r1) * (xmax - xmin)
            cand_y = ymin + (0.15 + 0.7 * r2) * (ymax - ymin)
            candidates.append((cand_x, cand_y))
        
        # Calculate minimum distance for each candidate
        best_candidate = None
        best_min_dist = -np.inf
        
        for cand_x, cand_y in candidates:
            # Distance to all scatter points
            distances_to_points = [np.sqrt((cand_x - sx)**2 + (cand_y - sy)**2) 
                                   for sx, sy in scatter_points]
            
            # Distance to x=y line: |x - y| / sqrt(2)
            dist_to_line = np.abs(cand_x - cand_y) / np.sqrt(2)
            
            # Distance to legend bounding box (only if legend is shown)
            if plot_legend and legend_xmin is not None:
                if legend_xmin <= cand_x <= legend_xmax:
                    # Point is horizontally within legend bounds
                    dist_to_legend = min(abs(cand_y - legend_ymin), abs(cand_y - legend_ymax))
                elif legend_ymin <= cand_y <= legend_ymax:
                    # Point is vertically within legend bounds
                    dist_to_legend = min(abs(cand_x - legend_xmin), abs(cand_x - legend_xmax))
                else:
                    # Point is outside both bounds - distance to nearest corner
                    corner_distances = [
                        np.sqrt((cand_x - legend_xmin)**2 + (cand_y - legend_ymin)**2),
                        np.sqrt((cand_x - legend_xmin)**2 + (cand_y - legend_ymax)**2),
                        np.sqrt((cand_x - legend_xmax)**2 + (cand_y - legend_ymin)**2),
                        np.sqrt((cand_x - legend_xmax)**2 + (cand_y - legend_ymax)**2),
                    ]
                    dist_to_legend = min(corner_distances)
            else:
                dist_to_legend = float('inf')
            
            # Minimum distance among all (only include legend distance if legend is shown)
            if plot_legend:
                min_dist = min(min(distances_to_points), dist_to_line, dist_to_legend)
            else:
                min_dist = min(min(distances_to_points), dist_to_line)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = (cand_x, cand_y)
        
        fallback_x, fallback_y = best_candidate
        
        # Direction along the line orthogonal to x=y: (1, -1)/sqrt(2)
        # This is the direction connecting the arrow origins/endpoints
        ortho_line_dx = 1 / np.sqrt(2)
        ortho_line_dy = -1 / np.sqrt(2)
        
        # Calculate spacing along this orthogonal line
        spacing = 0.015 * max_axis_len
        
        # Place two origins on this line, spaced apart
        origin1 = (fallback_x - spacing * ortho_line_dx, 
                   fallback_y - spacing * ortho_line_dy)
        origin2 = (fallback_x + spacing * ortho_line_dx, 
                   fallback_y + spacing * ortho_line_dy)
        
        # Arrow directions (both orthogonal to x=y, pointing in opposite directions)
        up_left_dx = -arrow_len / np.sqrt(2)
        up_left_dy = arrow_len / np.sqrt(2)
        down_right_dx = arrow_len / np.sqrt(2)
        down_right_dy = -arrow_len / np.sqrt(2)
        
        # Determine labels and positions based on entangle_is_up
        if entangle_is_up:
            # Up-left arrow gets "More Entangled"
            more_label, less_label = "More Entangled", "Less Entangled"
            more_origin, less_origin = origin1, origin2
            more_dx, more_dy = up_left_dx, up_left_dy
            less_dx, less_dy = down_right_dx, down_right_dy
            more_va = 'bottom'
            less_va = 'top'
        else:
            # Down-right arrow gets "More Entangled"
            more_label, less_label = "More Entangled", "Less Entangled"
            more_origin, less_origin = origin2, origin1
            more_dx, more_dy = down_right_dx, down_right_dy
            less_dx, less_dy = up_left_dx, up_left_dy
            more_va = 'top'
            less_va = 'bottom'
        
        more_end = (more_origin[0] + more_dx, more_origin[1] + more_dy)
        ax.annotate(
            "",
            xy=more_end,
            xytext=more_origin,
            arrowprops=dict(arrowstyle="->", color="black", lw=2)
        )
        ax.text(
            more_end[0], more_end[1],
            more_label,
            fontsize=10,
            ha='center',
            va=more_va
        )
        
        less_end = (less_origin[0] + less_dx, less_origin[1] + less_dy)
        ax.annotate(
            "",
            xy=less_end,
            xytext=less_origin,
            arrowprops=dict(arrowstyle="->", color="black", lw=2)
        )
        ax.text(
            less_end[0], less_end[1],
            less_label,
            fontsize=10,
            ha='center',
            va=less_va
        )
    
    #if equal_scaling:
    #    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    
    if ax is None:
        fig.tight_layout()
    
    return {"fig": fig, "ax": ax, "legend": legend}


def plot_scatter_grid(table, 
                      kwarg_list=None,
                      figsize_per_plot=(5, 4),
                      save_path=None):
    """
    Plot a grid of model scatter plots.
    
    Parameters:
    -----------
    table : pd.DataFrame
        The results table
    kwarg_list : list of dict or list of list of dict, optional
        Can be either:
        - 1D list of dicts for a single row
        - 2D list of lists of dicts for a grid
        Default creates a 2x3 grid with 5 plots and 1 legend-only subplot.
    figsize_per_plot : tuple
        Size of each subplot
    save_path : str or None
        Path to save the figure. If None, the figure is not saved.
    
    Returns:
    --------
    list of dict or list of list of dict
        Outputs from each model_scatter call, matching input structure
    """
    if kwarg_list is None:
        kwarg_list = [
            [{"only_legend": True},
            {"x": "EU_ncc[id]", "y": "AU_ncc[id]", "xlabel": "EU NCC", "ylabel": "AU NCC →",
             "title": "Ambiguity Modeling (ID)", "equal_scaling": True, "plot_legend": False},
            {"x": "AU_ace[id]", "y": "TU_ace[id]", "entangle_is_up": True, "xlabel": "AU ACE", "ylabel": "TU ACE ←",
             "title": "Calibration (ID)", "equal_scaling": True, "ignore_for_axis": {"EU": ["none"]},"plot_legend": False}],
            [{"x": "AU_auc[ood]", "y": "EU_auc[ood]", "xlabel": "AU AUC", "ylabel": "EU AUC →", 
             "title": "Out-of-Distribution Detection", "equal_scaling": True, "plot_legend": False},
            {"x": "EU_ncc[ood]", "y": "AU_ncc[ood]", "xlabel": "EU NCC", "ylabel": "AU NCC →",
             "title": "Ambiguity Modeling (OOD)", "equal_scaling": True, "plot_legend": False},
            {"x": "AU_ace[ood]", "y": "TU_ace[ood]", "entangle_is_up": True, "xlabel": "AU ACE", "ylabel": "TU ACE ←",
             "title": "Calibration (OOD)", "equal_scaling": True, "ignore_for_axis": {"EU": ["none"]},"plot_legend": False}]
        ]
    
    # Detect if kwarg_list is 1D or 2D
    is_2d = isinstance(kwarg_list[0], list)
    
    if is_2d:
        # 2D grid
        n_rows = len(kwarg_list)
        n_cols = len(kwarg_list[0])
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
        
        # Ensure axes is always 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]
        
        outputs = []
        for i in range(n_rows):
            row_outputs = []
            for j in range(n_cols):
                kwargs = kwarg_list[i][j].copy()
                out = model_scatter(table, ax=axes[i, j], **kwargs)
                row_outputs.append(out)
            outputs.append(row_outputs)
    else:
        # 1D row (backward compatibility)
        n_plots = len(kwarg_list)
        fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]))
        
        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        
        outputs = []
        for i, kwargs in enumerate(kwarg_list):
            kwargs = kwargs.copy()
            out = model_scatter(table, ax=axes[i], **kwargs)
            outputs.append(out)
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return outputs


"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p = "/home/jloch/Desktop/diff/luzern/values_datasets/origlidc128/splits/ood_aug/firstCycle/splits.pkl"
with open(p, "rb") as f:
    splits = pickle.load(f)
p = "/home/jloch/Desktop/diff/luzern/values_datasets/origlidc128/preprocessed/images"
pools_show = ["train","ood_contrast"]
n_images_per_pool = 10
m = len(pools_show)
#plot a m x n grid of images
concat_image = []
for pool in pools_show:
    pool_images = []
    pool_labels = []
    for i in range(n_images_per_pool):
        img_path = np.random.choice(splits[0][pool])
        img = np.load(Path(p)/img_path)
        pool_images.append(img)
        pool_labels.append([])
        for j in range(4):
            label_path = img_path.replace(".npy", f"_{j:02d}_mask.npy")
            label = np.load(Path(p.replace("/images", "/labels"))/label_path)
            pool_labels[-1].append(label)
    concat_image.append(np.concatenate(pool_images, axis=1))
final_image = np.concatenate(concat_image, axis=0)
plt.figure(figsize=(15,7))
plt.imshow(final_image, cmap="gray", vmin=-3,vmax=5)
plt.yticks([(i+0.5)*final_image.shape[0]/m for i in range(m)], pools_show)
plt.xticks([])
plt.title("Random samples from different pools")

"""

def plot_lidc(n_rows=3,train_cols=4, val_cols=1, test_cols=1,
              train_text="Train", val_text="Val", test_text="Test (id)", test_ood_text="Test (ood)",
              label_color=(1,0,0), label_lw=1.0, pad=1, pad_color=(0,0,0), layout_pad=32, layout_pad_color=(1,1,1),
              vmin=-3,vmax=3, arrow_width=1.0, arrow_color=(0,0,0), num_label_rows=2, text_fontsize=[16,10],
              crop_ratio=1.0, save_path=None, s=1.5,
              ):
    """
    Plots an array of images form the origlidc128 dataset. Layout:
    Train  | Val | Test (id) | Test (ood)
    with n_rows of images. Each of these sub-layouts should be a concatenated images 
    with pad pixels in between images in both vertical/horz.
    Afterwards, the sub-layouts should be concatenated horizontally with layout_pad pixels in between and layout_pad_color as color.
    An arrow should be pointed from test (id) to test (ood), but only if arrow_width > 0.
    Otherwise all images are randomly sampled from the respective pools. 
    The test (ood) columns are the same source images as test (id) but with different augmentations applied.
    The files should originate from the splits: "train", "val", "id_test", "ood_blur", "ood_noise", "ood_jpeg", "ood_contrast"
    When loading images, clip them to vmin and vmax and normalize to 0-1 for visualization. 
    
    Additionally, ground truth labels should be loaded and plotted on top of images for the first
    num_label_rows of the rows, with the specified label_color and transparency (if the tuple has 4 vals).
    
    The text labels should be plotted above each sub-layout center with fontsize specified by text_fontsize[0]. 
    Leave a vertical space between image and text for the following:
    Below the test_ood_text should be an additional text, e.g. "Blur" for the ood_blur pool, with fontsize specified by text_fontsize[1].    
    
    crop_ratio: float in (0, 1], controls what portion of the image to show (centered crop)
    """
    import pickle
    
    # Load splits
    p = "/home/jloch/Desktop/diff/luzern/values_datasets/origlidc128/splits/ood_aug/firstCycle/splits.pkl"
    with open(p, "rb") as f:
        splits = pickle.load(f)
    
    base_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/origlidc128/preprocessed"
    images_dir = Path(base_dir) / "images"
    labels_dir = Path(base_dir) / "labels"
    
    # Choose OOD augmentation types randomly
    ood_pools = ["ood_blur", "ood_noise", "ood_jpeg", "ood_contrast"]
    ood_pool_names = ["Blur", "Noise", "JPEG", "Contrast"]
    
    def crop_center(img, crop_ratio):
        """Crop the central portion of the image based on crop_ratio."""
        if crop_ratio >= 1.0:
            return img
        
        if img.ndim == 3:
            h, w, c = img.shape
            new_h = int(h * crop_ratio)
            new_w = int(w * crop_ratio)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            return img[start_h:start_h+new_h, start_w:start_w+new_w, :]
        else:
            h, w = img.shape
            new_h = int(h * crop_ratio)
            new_w = int(w * crop_ratio)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            return img[start_h:start_h+new_h, start_w:start_w+new_w]
    
    def load_and_normalize(img_path):
        img = np.load(images_dir / img_path)
        #Normalize with gauss
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        return img
    
    def load_label(img_path, annotator_idx=0):
        label_path = img_path.replace(".npy", f"_{annotator_idx:02d}_mask.npy")
        return np.load(labels_dir / label_path)
    
    def create_padded_grid(images, cols, pad, pad_color):
        # Create a grid with padding between images AND around the outer edges
        if len(images) == 0:
            return None
        
        # Calculate dimensions
        is_rgb = images[0].ndim == 3
        if is_rgb:
            img_h, img_w = images[0].shape[:2]
        else:
            img_h, img_w = images[0].shape
        n_imgs = len(images)
        rows = (n_imgs + cols - 1) // cols
        
        # Create grid with padding (including outer edges)
        grid_h = rows * img_h + (rows + 1) * pad  # +1 for top and bottom padding
        grid_w = cols * img_w + (cols + 1) * pad  # +1 for left and right padding
        
        if is_rgb:
            grid = np.ones((grid_h, grid_w, 3)) * np.array(pad_color)
        else:
            grid = np.ones((grid_h, grid_w)) * pad_color[0]
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            y = pad + row * (img_h + pad)  # Start with pad offset
            x = pad + col * (img_w + pad)  # Start with pad offset
            grid[y:y+img_h, x:x+img_w] = img
        
        return grid
    
    def overlay_labels(img, labels, label_color, label_lw):
        # Convert grayscale to RGB
        img_rgb = np.stack([img, img, img], axis=-1)
        
        # Create label outline
        from scipy import ndimage
        
        # Overlay all labels (there are 4 annotators)
        for label in labels:
            if label is None:
                continue
            label_binary = (label > 0).astype(float)
            if label_binary.sum() == 0:
                continue
            eroded = ndimage.binary_erosion(label_binary, iterations=int(label_lw))
            outline = label_binary - eroded
            
            # Apply label color with transparency
            alpha = label_color[3] if len(label_color) == 4 else 0.5
            for i in range(3):
                img_rgb[:, :, i] = np.where(
                    outline > 0,
                    img_rgb[:, :, i] * (1 - alpha) + label_color[i] * alpha,
                    img_rgb[:, :, i]
                )
        return img_rgb
    
    # Collect images for each section
    sections = []
    section_texts = []
    section_subtexts = []
    
    # Train section
    train_images = []
    for row_idx in range(n_rows):
        row_images = []
        for col_idx in range(train_cols):
            img_path = np.random.choice(splits[0]["train"])
            img = load_and_normalize(img_path)
            
            # Add labels for first num_label_rows
            if row_idx < num_label_rows:
                # Load all 4 annotator labels
                labels = [load_label(img_path, annotator_idx=i) for i in range(4)]
                img = overlay_labels(img, labels, label_color, label_lw)
            else:
                img = np.stack([img, img, img], axis=-1)
            
            # Apply crop
            img = crop_center(img, crop_ratio)
            row_images.append(img)
        train_images.extend(row_images)
    
    train_grid = create_padded_grid(train_images, train_cols, pad, pad_color)
    sections.append(train_grid)
    section_texts.append(train_text)
    section_subtexts.append("")
    
    # Val section
    val_images = []
    for row_idx in range(n_rows):
        for col_idx in range(val_cols):
            img_path = np.random.choice(splits[0]["val"])
            img = load_and_normalize(img_path)
            
            if row_idx < num_label_rows:
                labels = [load_label(img_path, annotator_idx=i) for i in range(4)]
                img = overlay_labels(img, labels, label_color, label_lw)
            else:
                img = np.stack([img, img, img], axis=-1)
            
            # Apply crop
            img = crop_center(img, crop_ratio)
            val_images.append(img)
    
    val_grid = create_padded_grid(val_images, val_cols, pad, pad_color)
    sections.append(val_grid)
    section_texts.append(val_text)
    section_subtexts.append("")
    
    # Test (id) section
    test_images = []
    test_img_paths = []
    for row_idx in range(n_rows):
        for col_idx in range(test_cols):
            img_path = np.random.choice(splits[0]["id_test"])
            test_img_paths.append(img_path)
            img = load_and_normalize(img_path)
            
            if row_idx < num_label_rows:
                labels = [load_label(img_path, annotator_idx=i) for i in range(4)]
                img = overlay_labels(img, labels, label_color, label_lw)
            else:
                img = np.stack([img, img, img], axis=-1)
            
            # Apply crop
            img = crop_center(img, crop_ratio)
            test_images.append(img)
    
    test_grid = create_padded_grid(test_images, test_cols, pad, pad_color)
    sections.append(test_grid)
    section_texts.append(test_text)
    section_subtexts.append("")
    
    # Test (ood) section - 4 columns, one for each OOD type
    # Use same source images as test (id) but from OOD pools
    test_ood_cols = 4  # One column per OOD type
    test_ood_images = []
    ood_col_labels = []
    
    for row_idx in range(n_rows):
        # Use the corresponding test (id) image path
        id_img_path = test_img_paths[row_idx]
        
        # For each OOD type, find the corresponding augmented image
        for ood_idx, ood_pool in enumerate(ood_pools):
            # Replace the parent folder to get the OOD version
            # id_test images are in id_test folder, OOD images are in ood_* folders
            ood_img_path = str(images_dir / id_img_path).replace("images", f"augmented/{ood_pool}/images")
            assert Path(ood_img_path).exists(), f"OOD image not found: {ood_img_path}"
            
            img = load_and_normalize(ood_img_path)
            
            if row_idx < num_label_rows:
                # Use original labels from the base image
                base_name = id_img_path
                labels = [load_label(base_name, annotator_idx=i) for i in range(4)]
                img = overlay_labels(img, labels, label_color, label_lw)
            else:
                img = np.stack([img, img, img], axis=-1)
            
            # Apply crop
            img = crop_center(img, crop_ratio)
            test_ood_images.append(img)
            
            # Track which OOD type this is for labeling
            if row_idx == 0:
                ood_col_labels.append(ood_pool_names[ood_idx])
    
    test_ood_grid = create_padded_grid(test_ood_images, test_ood_cols, pad, pad_color)
    sections.append(test_ood_grid)
    section_texts.append(test_ood_text)
    section_subtexts.append("")  # Remove the combined label
    
    # Concatenate sections horizontally with layout_pad
    max_height = max(s.shape[0] for s in sections)
    padded_sections = []
    for section in sections:
        if section.shape[0] < max_height:
            # Pad vertically with layout_pad_color
            pad_height = max_height - section.shape[0]
            if section.ndim == 3:
                padding = np.ones((pad_height, section.shape[1], section.shape[2])) * np.array(layout_pad_color)
            else:
                padding = np.ones((pad_height, section.shape[1])) * layout_pad_color[0]
            section = np.vstack([section, padding])
        padded_sections.append(section)
    
    # Add layout_pad between sections
    final_sections = []
    for i, section in enumerate(padded_sections):
        final_sections.append(section)
        if i < len(padded_sections) - 1:
            # Add spacer
            if section.ndim == 3:
                spacer = np.ones((section.shape[0], layout_pad, section.shape[2])) * np.array(layout_pad_color)
            else:
                spacer = np.ones((section.shape[0], layout_pad)) * layout_pad_color[0]
            final_sections.append(spacer)
    
    final_image = np.concatenate(final_sections, axis=1)
    
    # Plot
    # Calculate figure size based on image dimensions and s scaling factor
    fig_width = final_image.shape[1] / 100 * s
    fig_height = final_image.shape[0] / 100 * s
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(final_image, cmap='gray' if final_image.ndim == 2 else None, interpolation='nearest', resample=False)
    ax.axis('off')
    
    # Add text labels
    section_positions = []
    current_x = 0
    for i, section in enumerate(padded_sections):
        section_width = section.shape[1]
        section_positions.append((current_x, current_x + section_width))
        current_x += section_width + layout_pad
    
    for i, (x_start, x_end) in enumerate(section_positions):
        center_x = (x_start + x_end) / 2
        ax.text(center_x, -20, section_texts[i], fontsize=text_fontsize[0], 
                ha='center', va='bottom')
        if section_subtexts[i]:
            ax.text(center_x, -5, section_subtexts[i], fontsize=text_fontsize[1], 
                    ha='center', va='bottom')
    
    # Add individual column labels for OOD types above each column in test (ood) section
    if len(section_positions) >= 4:
        ood_section_start = section_positions[3][0]
        ood_section_width = section_positions[3][1] - section_positions[3][0]
        # Calculate width of each column (assuming 4 columns with padding)
        # Get first image dimensions from test_ood_images
        if len(test_ood_images) > 0:
            img_width = test_ood_images[0].shape[1]
            col_width = img_width + pad
            
            for col_idx, label in enumerate(ood_col_labels):
                # Account for the outer padding on the left
                col_center_x = ood_section_start + pad + col_idx * col_width + img_width / 2
                ax.text(col_center_x, -5, label, fontsize=text_fontsize[1], 
                        ha='center', va='bottom')
    
    # Add arrows for each row from test (id) to test (ood) - positioned in the gap
    if arrow_width > 0 and len(section_positions) >= 4:
        # Arrow should be in the gap between test (id) and test (ood)
        arrow_start_x = section_positions[2][1]
        arrow_end_x = section_positions[3][0]
        
        # Get image height to calculate row positions
        if len(test_images) > 0:
            img_height = test_images[0].shape[0]
            row_height = img_height + pad
            
            for row_idx in range(n_rows):
                # Account for the outer padding on the top
                arrow_y = pad + row_idx * row_height + img_height / 2
                ax.annotate('', xy=(arrow_end_x, arrow_y), 
                            xytext=(arrow_start_x, arrow_y),
                            arrowprops=dict(arrowstyle='->', lw=arrow_width, color=arrow_color))
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax

import warnings

def replace_images_with_corner_markers(fig,save_images="",save_folder="/home/jloch/Desktop/diff/writing/ECCV2026/ECCV_2026_AU_EU/images"):
    if save_images:
        assert len(save_images.split("/"))==1, f"save_images should be a single name, not a path"
        os.makedirs(os.path.join(save_folder,save_images),exist_ok=True)
        k = 0
    for ax in fig.axes:
        for im in ax.get_images():
            arr = im.get_array()
            new_img = np.ones((arr.shape), dtype=np.float32)
            new_img[[0,0,-1,-1],[0,-1,0,-1]] = 0
            im.set_data(new_img)
            if save_images:
                epss = 1e-10
                if arr.min()<-epss or arr.max()>1+epss:
                    warnings.warn(f"Image {im} has unexpected values outside of [0,1]")
                else:
                    arr = np.clip(arr,0,1)
                # Save the image to the specified path
                arr = (arr*255).astype(np.uint8)
                if not len(arr.shape)==3:
                    arr = arr[:,:,None]
                save_path = os.path.join(save_folder,save_images, f"im_{k}.png")
                Image.fromarray(arr).save(save_path)
                k += 1

def checkerboard_image(image):
    """Returns an image with identical dimensions as the input image, but with a checkerboard pattern. If multi channel, then each channel is identical"""
    if len(image.shape) == 2:  # Grayscale image
        cb_image = np.zeros_like(image)
        cb_image[::2, ::2] = 1
        cb_image[1::2, 1::2] = 1
    elif len(image.shape) == 3:  # Color image
        cb_image = np.zeros_like(image)
        cb_image[::2, ::2, :] = 1
        cb_image[1::2, 1::2, :] = 1
    else:
        raise ValueError("Input image must be either 2D or 3D (grayscale or color).")
    print(cb_image.shape)
    return cb_image

def plot_chaksu(n_rows=3, n_cols=10, scanners=["Remidio","Bosch","Forus"],scanner_titles=["Remidio\n(id)","Bosch\n(ood)","Forus\n(ood)"],
                text_fontsize=16, cup_color=(1,0,0), disc_color=(0,1,0), label_lw=1.0, pad=1, 
                pad_color=(0,0,0), label_cols=5, save_path=None, s=1.5):
    """
    plots a e.g. 3x10 matrix of images from the chaksu dataset. 
    Each row corresponds to a different scanner type, each image is a randomly sampled. 
    The first label_cols columns should have cup and disc segmentation masks overlaid on the images, 
    with cup_color and disc_color respectively, and label_lw as the line width for the labels. 
    (it should also accept 4 color RGBA tuples) 
    There should be pad pixels of pad_color in between images in both vertical and horizontal directions. If pad_pixels is a tuple width 2 elements,
    then it should specify the vertical and horizontal padding separately.
    To the left of each row, there should be a text label with the corresponding scanner title from scanner_titles, using text_fontsize for the font size.
    """
    import pickle
    import csv
    
    # Handle pad as tuple or scalar
    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h = pad_w = pad
    
    # Load splits
    base_dir = "/home/jloch/Desktop/diff/luzern/values_datasets/chaksu128"
    split_path = Path(base_dir) / "splits" / "scanner" / "firstCycle" / "splits.pkl"
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    
    # Load metadata to determine scanner for each image
    metadata_path = Path(base_dir) / "preprocessed" / "metadata.csv"
    metadata = {}
    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["image_file"]] = row["machine"]
    
    images_dir = Path(base_dir) / "preprocessed"
    labels_dir = Path(base_dir) / "preprocessed"
    
    def load_and_normalize(img_path):
        img = np.load(images_dir / img_path)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # Normalize to 0-1
        img = np.clip(img, 0, 1)
        return img
    
    def load_labels(img_path):
        # Chaksu has 5 annotations per image (_00 through _04)
        # Each mask has values 0 (background), 1 (disc), 2 (cup)
        base_name = Path(img_path).stem
        labels_path = Path(base_dir) / "preprocessed" / "labels"
        masks = []
        
        for i in range(5):
            label_path = labels_path / f"{base_name}_{i:02d}_mask.npy"
            if label_path.exists():
                masks.append(np.load(label_path))
        
        return masks if masks else None
    
    def overlay_labels(img, masks, cup_color, disc_color, label_lw):
        # Convert grayscale to RGB if needed
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img.copy()
        
        if masks is None:
            return img_rgb
        
        from scipy import ndimage
        
        # Each mask contains both disc (value 1) and cup (value 2)
        n_masks = len(masks)
        
        # Overlay all annotations with varying transparency
        for mask_idx, mask in enumerate(masks):
            # Vary alpha based on annotation index for visibility
            alpha_disc = disc_color[3] if len(disc_color) == 4 else 0.5
            alpha_cup = cup_color[3] if len(cup_color) == 4 else 0.5
            
            # Overlay disc (value >= 1)
            disc_binary = (mask >= 0.5).astype(float)
            if disc_binary.sum() > 0:
                eroded = ndimage.binary_erosion(disc_binary, iterations=max(1, int(label_lw)))
                outline = disc_binary - eroded
                
                for i in range(3):
                    img_rgb[:, :, i] = np.where(
                        outline > 0,
                        img_rgb[:, :, i] * (1 - alpha_disc) + disc_color[i] * alpha_disc,
                        img_rgb[:, :, i]
                    )
            
            # Overlay cup (value >= 2)
            cup_binary = (mask >= 1.5).astype(float)
            if cup_binary.sum() > 0:
                eroded = ndimage.binary_erosion(cup_binary, iterations=max(1, int(label_lw)))
                outline = cup_binary - eroded
                
                for i in range(3):
                    img_rgb[:, :, i] = np.where(
                        outline > 0,
                        img_rgb[:, :, i] * (1 - alpha_cup) + cup_color[i] * alpha_cup,
                        img_rgb[:, :, i]
                    )
        
        return img_rgb
    
    def get_images_for_scanner(scanner_name, n_images):
        # Collect images from the specified scanner
        all_pools = ["train", "val", "id_test", "ood_test"]
        available_images = []
        
        for pool in all_pools:
            if pool in splits[0]:
                for img_path in splits[0][pool]:
                    img_file = Path(img_path).name
                    if img_file in metadata and metadata[img_file] == scanner_name:
                        available_images.append(img_path)
        
        if len(available_images) == 0:
            # Fallback: just pick any images
            available_images = splits[0].get("train", [])
        
        # Sample randomly
        if len(available_images) >= n_images:
            selected = np.random.choice(available_images, size=n_images, replace=False)
        else:
            selected = np.random.choice(available_images, size=n_images, replace=True)
        
        return selected
    
    # Build grid rows for each scanner
    all_rows = []
    
    for scanner_idx, (scanner, title) in enumerate(zip(scanners, scanner_titles)):
        # Get images for this scanner
        img_paths = get_images_for_scanner(scanner, n_cols)
        
        # Create row images
        row_images = []
        for col_idx in range(n_cols):
            img_path = img_paths[col_idx]
            img = load_and_normalize(img_path)
            
            # First label_cols columns show labels, rest show raw images
            if col_idx < label_cols:
                masks = load_labels(img_path)
                img = overlay_labels(img, masks, cup_color, disc_color, label_lw)
            else:
                # Convert to RGB for consistency
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
            
            row_images.append(img)
        
        # Concatenate images in this row with horizontal padding
        row_parts = []
        for i, img in enumerate(row_images):
            row_parts.append(img)
            if i < len(row_images) - 1:
                # Add vertical padding strip between images
                pad_strip = np.ones((img.shape[0], pad_w, 3)) * np.array(pad_color)
                row_parts.append(pad_strip)
        
        row_concat = np.concatenate(row_parts, axis=1)
        all_rows.append((row_concat, title))
    
    # Stack all scanner rows vertically with horizontal padding
    grid_parts = []
    for i, (row, title) in enumerate(all_rows):
        grid_parts.append(row)
        if i < len(all_rows) - 1:
            # Add horizontal padding strip between rows
            pad_strip = np.ones((pad_h, row.shape[1], 3)) * np.array(pad_color)
            grid_parts.append(pad_strip)
    
    final_grid = np.concatenate(grid_parts, axis=0)
    
    # Add padding around the outer edges
    top_bottom_pad = np.ones((pad_h, final_grid.shape[1], 3)) * np.array(pad_color)
    final_grid = np.vstack([top_bottom_pad, final_grid, top_bottom_pad])
    left_right_pad = np.ones((final_grid.shape[0], pad_w, 3)) * np.array(pad_color)
    final_grid = np.hstack([left_right_pad, final_grid, left_right_pad])
    
    # Plot
    fig, ax = plt.subplots(figsize=(n_cols * s, n_rows * s))
    ax.imshow(final_grid, interpolation='nearest', resample=False)
    ax.axis('off')
    
    # Add text labels for each scanner row to the left
    img_height = all_rows[0][0].shape[0]
    current_y = pad_h + img_height / 2
    
    for i, (row, title) in enumerate(all_rows):
        # Place text to the left of each row
        ax.text(pad_w - 10, current_y, title, fontsize=text_fontsize, 
                ha='right', va='center', rotation=0)
        current_y += img_height + pad_h
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax
