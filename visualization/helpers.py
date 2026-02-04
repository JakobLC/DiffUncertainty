import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
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

AU, EU, TU = "aleatoric_uncertainty", "epistemic_uncertainty", "predictive_uncertainty"
# AU, EU, TU = "AU", "EU", "TU" # new
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
    assert aggregation_type in ["patch_level", "threshold"], "Invalid aggregation type"
    table = pd.DataFrame()

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

def pred_grid_computation(ckpt_path = "/home/jloch/Desktop/diff/luzern/values/saves/lidc64/diffusion_swag/checkpoints/last.ckpt",
                          split = "id_test"):
    args.checkpoint_paths = [ckpt_path]
    args.test_split = split
    tester = Tester(args)
    raw_batches = tester.collect_raw_predictions(max_batches=5)

    batch0 = raw_batches[2]
    probs = batch0["softmax_pred_groups"]  # shape: (n_models_or_samples, batch, C, H, W)
    batch0["image_id"]
    p = "/home/jloch/Desktop/diff/luzern/values_datasets/lidc64/images/{id}.npy"
    i = 6
    image = np.load(p.format(id=f"{batch0['image_id'][i]}"))
    return image, batch0

def plot_pred_grid(image, batch0, i=6):

    p = "/home/jloch/Desktop/diff/luzern/values_datasets/lidc64/images/{id}.npy"
    image = torch.tensor(np.load(p.format(id=f"{batch0['image_id'][i]}")))[None,None]
    image = (image - image.min()) / (image.max() - image.min())
    #means we should sample i=14 in dim 2, the batch dim of 16 images
    #plot 10x10 matrix of images. add padding between images
    pred = torch.stack(batch0["softmax_pred_groups"], dim=0)  # shape: (n_models_or_samples, batch, C, H, W)
    sum_image = pred[:,:,i,1].sum((0,1))
    bbox_crop = sum_image > sum_image.mean() * 0.01
    d1_indices = torch.where(bbox_crop.sum(1) > 0)[0]
    d2_indices = torch.where(bbox_crop.sum(0) > 0)[0]
    d1_min = d1_indices.min().item()
    d1_max = d1_indices.max().item()+1
    d2_min = d2_indices.min().item()
    d2_max = d2_indices.max().item()+1
    assert d1_max>d1_min and d2_max>d2_min, f"Invalid slice: {d1_min},{d1_max},{d2_min},{d2_max}"
    d1 = slice(d1_min,d1_max)
    d2 = slice(d2_min,d2_max)

    p = pred[:, :, i, 1, d1, d2]
    # flatten to [100, 1, 64, 64] for grid
    n_AU = p.shape[0]
    n_EU = p.shape[1]

    grid_dims = (n_AU+2, n_EU+4)

    E_y_p = batch0["softmax_pred"][:, i, 1, d1, d2].unsqueeze(1)
    H_E_y_p = entropy(E_y_p, dim=1).unsqueeze(1)

    E_th_H_E_y_p = H_E_y_p.mean(0).unsqueeze(1)
    E_th_E_y_p = E_y_p.mean(0).unsqueeze(1)
    H_E_th_E_y_p = entropy(E_th_E_y_p.unsqueeze(1), dim=1)
    EU = H_E_th_E_y_p - E_th_H_E_y_p
    w = torch.ones_like(E_th_E_y_p)

    btm_row_imgs = [image]+([w]*(n_EU-4))+[EU,w,E_th_H_E_y_p,w,E_th_E_y_p,w,H_E_th_E_y_p]


    grid = torch.cat([p, torch.ones_like(E_y_p), E_y_p, torch.ones_like(H_E_y_p), H_E_y_p], dim=1)
    #add btm row
    row = torch.cat(btm_row_imgs, dim=1)
    grid = torch.cat([grid, torch.ones_like(row), row], dim=0)
    grid = grid.reshape(-1, 1, grid.shape[-2], grid.shape[-1])  # (n_row, n_col, H, W)

    # make grid
    grid2 = torchvision.utils.make_grid(
        grid,
        nrow=grid_dims[1],
        padding=2,
        value_range=[0,1],
        pad_value=1,
    )[0]
    s = 2
    # plot
    plt.figure(figsize=(8*s, 8*s))
    plt.imshow(grid2, cmap="hot", vmin=0,vmax=1)
    plt.axis("off")
    #plot a sequence of arrows to indicate direction from left (model samples) to right (ensemble)

    for j in range(10):
        for x,letter in zip([grid2.shape[1]*(grid_dims[1]-4+0.2)/(grid_dims[1]),
                            grid2.shape[1]*(grid_dims[1]-2+0.2)/(grid_dims[1])],["E","H"]):
            plt.arrow(
                x=x,
                y=grid2.shape[0]*(j+0.5)/grid_dims[0],
                dx=grid2.shape[1]/grid_dims[1]*0.6,
                dy=0,
                width=2,
                head_width=grid2.shape[0]/grid_dims[0]*0.2,
                head_length=grid2.shape[1]/grid_dims[1]*0.2,
                length_includes_head=True,
                color="black"
            )
            #plt expectation "E" ontop of arrows
            plt.text(
                x=x+grid2.shape[1]/grid_dims[1]*0.2,
                y=grid2.shape[0]*(j+0.5)/grid_dims[0],
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
            x=pos * grid2.shape[1] / grid_dims[1],
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
            y=pos * grid2.shape[0] / grid_dims[0],
            s=title,
            color="black",
            fontsize=fontsize*s,
            ha="right",
            va="center",
            rotation=90,
        )
    for pos, title in pos_to_title["btm_row"].items():
        plt.text(
            x=pos * grid2.shape[1] / grid_dims[1],
            y=grid2.shape[0],
            s=title,
            color="black",
            fontsize=fontsize*s,
            ha="center",
            va="top",
        )

    # add down arrows below each column
    for j in range(2):
        plt.arrow(
            x=grid2.shape[1]*(grid_dims[1]-2.5+j*2)/(grid_dims[1]),
            y=grid2.shape[0]*(n_EU+0.2)/grid_dims[0],
            dx=0,
            dy=grid2.shape[1]/grid_dims[1]*0.6,
            width=2,
            head_width=grid2.shape[0]/grid_dims[0]*0.2,
            head_length=grid2.shape[1]/grid_dims[1]*0.2,
            length_includes_head=True,
            color="black"
        )
        plt.text(
            x=grid2.shape[1]*(grid_dims[1]-2.5+j*2)/(grid_dims[1]),
            y=grid2.shape[0]*(n_EU+0.2)/grid_dims[0]+grid2.shape[1]/grid_dims[1]*0.2,
            s="E",
            color="white",
            fontsize=14*s,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
        )
    plt.arrow(
        x=grid2.shape[1]*(grid_dims[1]-4+0.4)/(grid_dims[1])+grid2.shape[1]/grid_dims[1]*0.4,
        y=grid2.shape[0]*(11.5)/grid_dims[0],
        dx=-grid2.shape[1]/grid_dims[1]*0.6,
        dy=0,
        width=2,
        head_width=grid2.shape[0]/grid_dims[0]*0.2,
        head_length=grid2.shape[1]/grid_dims[1]*0.2,
        length_includes_head=True,
        color="black"
    )
    plt.text(
        x=grid2.shape[1]*(grid_dims[1]-4+0.2)/(grid_dims[1])+grid2.shape[1]/grid_dims[1]*0.4,
        y=grid2.shape[0]*(11.5)/grid_dims[0],
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
def get_bbox_col(matrix, value):
    # get min and max of matrix
    vmin = matrix.values.min()
    vmax = matrix.values.max()
    # normalize value to [0,1]
    norm_value = (value - vmin) / (vmax - vmin + 1e-8)
    # get colormap
    #cmap = plt.get_cmap("RdYlGn")
    cmap = LinearSegmentedColormap.from_list(
        "pure_RdYlGn",
        ["#FF0000", "#FFFF00", "#00FF00"],
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
            cbar_keys = None,):
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
    

    flip = False
    if "ged" in metric:
        flip = True
    elif "-" not in metric and "ace" in metric:
        flip = True
    if cbar_keys is not None:
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
    im = ax.imshow(matrix, cmap=f"{cmap}_r" if flip else f"{cmap}", aspect="auto",
              vmin=vmin,vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if reldiff:
        reldiff_matrix = table.pivot(index=index, columns=columns, values=reldiff)
        reldiff_ij = reldiff_matrix.iloc[0,0]
        if isinstance(reldiff_ij, dict):
            reldiff_matrix = reldiff_matrix.applymap(lambda x: x[subkey] if isinstance(x, dict) and subkey in x else np.nan)
    for (i, row_label) in enumerate(matrix.index):
        for (j, col_label) in enumerate(matrix.columns):
            value = matrix.loc[row_label, col_label]
            if reldiff:
                bg_box_color = get_bbox_col(reldiff_matrix,reldiff_matrix.loc[row_label, col_label])
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

def get_mm_colormap(name):
    cmap = "viridis" if name.startswith("(") else "magma"
    if ("_ged" in name) or ("_ace" in name):
        cmap += "_r"
    return cmap

def plot_mm_grid(table,
                 grid_layout=[[{"metric": "val_dice"},{"metric": "id_dice"}],
                              [{"metric": "val_ged"},{"metric": "id_ged"}]],
                shared_colorbar = True):
    # plots a grid of metric matrices, based on some grid layout
    # if the names in the grid layout is a tuple then split the 
    # first is the metric name and second is the subkey
    # if shared_colorbar is True then share colorbar across all plots
    # if shared_colorbar is a grid aswell then identical elements in the grid share colorbars
    n_rows = len(grid_layout)