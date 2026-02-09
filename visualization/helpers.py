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
        version = formatter.format(**add_dict).replace("ensemble","ens12[1,2,3,4,5]")
        if not (Path(f"{save_path}")/version).exists():
            print("Skipping missing version:", version)
            continue
        add_dict["version"] = version
        p = f"{save_path}{version}/e{epoch}{ema}/val/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["val_dice"] = loaded["mean"]["metrics"]["dice"]
        add_dict["val_ged"] = loaded["mean"]["metrics"]["ged"]
        add_dict["val_ged_bma"] = loaded["mean"]["metrics"]["ged_bma"]
        
        p = f"{save_path}{version}/e{epoch}{ema}/id/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["id_dice"] = loaded["mean"]["metrics"]["dice"]
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
        table[f"(min(AU,EU)-TU)_ace_{k2}"] = (table[[f"AU_ace_{k2}", f"EU_ace_{k2}"]].min(axis=1) - table[f"TU_ace_{k2}"])/table[f"TU_ace_{k2}"]
    #same as above but for non-ood keys
    table[f"(AU-EU)_ncc_id"] = (table["AU_ncc_id"] - table["EU_ncc_id"])/table["AU_ncc_id"]
    table[f"(min(AU,EU)-TU)_ace_id"] = (table[["AU_ace_id", "EU_ace_id"]].min(axis=1) - table["TU_ace_id"])/table["TU_ace_id"]
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
    plt.show()
from matplotlib.colors import LinearSegmentedColormap


def get_bbox_col(matrix, value, is_ace):
    # get min and max of matrix
    vmin = matrix.values.min()
    vmax = matrix.values.max()
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
    m,n = matrix.shape
    item00 = matrix.iloc[0,0]

    if ax is None:
        fig,ax = plt.subplots(figsize=(figsize_per_cell[0]*n, figsize_per_cell[1]*m))
    if isinstance(item00, dict):
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
            item00 = mat.iloc[0,0]
            if isinstance(item00, dict):
                vmin = mat.applymap(lambda x: min(x.values())).values.min()
                vmax = mat.applymap(lambda x: max(x.values())).values.max()
            else:
                vmin = min(vmin, mat.values.min())
                vmax = max(vmax, mat.values.max())
    else:
        vmin = matrix.values.min()
        vmax = matrix.values.max()
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
              vmin=vmin,vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if reldiff:
        reldiff_matrix = table.pivot(index=index, columns=columns, values=reldiff)
        reldiff_ij = reldiff_matrix.iloc[0,0]
        if isinstance(reldiff_ij, dict):
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
    if axes.ndim == 1:
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
            
            metric = kwargs["metric"]
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