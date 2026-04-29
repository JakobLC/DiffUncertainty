import json

import numpy as np

from evaluation.experiment_dataloader import ExperimentDataloader
from tqdm import tqdm


def compute_ncc(gt_unc_map: np.array, pred_unc_map: np.array):
    """
    Compute the normalized cross correlation between a ground truth uncertainty and a predicted uncertainty map,
    to determine how similar the maps are.
    :param gt_unc_map: the ground truth uncertainty map based on the rater variability
    :param pred_unc_map: the predicted uncertainty map
    :return: float: the normalized cross correlation between gt and predicted uncertainty map
    """
    mu_gt = np.mean(gt_unc_map)
    mu_pred = np.mean(pred_unc_map)
    sigma_gt = np.std(gt_unc_map, ddof=1)
    sigma_pred = np.std(pred_unc_map, ddof=1)
    gt_norm = gt_unc_map - mu_gt
    pred_norm = pred_unc_map - mu_pred
    prod = np.sum(np.multiply(gt_norm, pred_norm))
    if sigma_gt == 0 or sigma_pred == 0:
        return 0.0
    else:
        ncc = (1 / (np.size(gt_unc_map) * sigma_gt * sigma_pred)) * prod
        return ncc


def _prepare_image_for_display(image: np.ndarray):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.moveaxis(image, 0, -1)
    image = image.astype(np.float32)
    if image.size > 0 and (np.min(image) < 0.0 or np.max(image) > 1.0):
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def main(exp_dataloader: ExperimentDataloader, plot: bool = False):
    ncc_dict = {}
    ncc_dict["mean"] = {}
    rng = np.random.default_rng()
    sampled_image_ids = list(
        rng.choice(
            exp_dataloader.image_ids,
            size=min(5, len(exp_dataloader.image_ids)),
            replace=False,
        )
    )
    for unc_type in exp_dataloader.exp_version.unc_types:
        nccs_unc = []
        sampled_data = {}
        for image_id in tqdm(exp_dataloader.image_ids):
            if image_id not in ncc_dict.keys():
                ncc_dict[image_id] = {}
            gt_unc_map = exp_dataloader.get_gt_unc_map(image_id)
            pred_unc_map = exp_dataloader.get_unc_map(image_id, unc_type)

            ncc = compute_ncc(gt_unc_map, pred_unc_map)
            ncc_dict[image_id][unc_type] = {"metrics": {"ncc": ncc}}
            nccs_unc.append(ncc)
            if plot and image_id in sampled_image_ids:
                dataset = exp_dataloader.dataloader.dataset if exp_dataloader.dataloader is not None else None
                if dataset is not None and hasattr(dataset, "image_ids") and hasattr(dataset, "imgs"):
                    image_idx = dataset.image_ids.index(image_id)
                    raw_image = np.load(dataset.imgs[image_idx])
                else:
                    raw_image = None
                sampled_data[image_id] = {
                    "image": raw_image,
                    "gt": np.asarray(gt_unc_map),
                    "pred": np.asarray(pred_unc_map),
                    "ncc": float(ncc),
                }
        ncc_dict["mean"][unc_type] = {"metrics": {"ncc": np.mean(np.array(nccs_unc))}}

        if plot:
            import matplotlib.pyplot as plt

            sampled_image_ids_ordered = [image_id for image_id in sampled_image_ids if image_id in sampled_data]
            if not sampled_image_ids_ordered:
                continue
            fig, axes = plt.subplots(4, len(sampled_image_ids_ordered), figsize=(4.0 * len(sampled_image_ids_ordered), 13.5))
            if len(sampled_image_ids_ordered) == 1:
                axes = np.expand_dims(axes, axis=1)

            fig.suptitle(f"NCC diagnostics for {unc_type}", fontsize=14)

            for col, image_id in enumerate(sampled_image_ids_ordered):
                image = sampled_data[image_id]["image"]
                gt_unc_map = sampled_data[image_id]["gt"]
                pred_unc_map = sampled_data[image_id]["pred"]
                ncc_value = sampled_data[image_id]["ncc"]

                image_ax = axes[0, col]
                gt_ax = axes[1, col]
                pred_ax = axes[2, col]
                scatter_ax = axes[3, col]

                if image is not None:
                    image_ax.imshow(_prepare_image_for_display(image), cmap="gray" if np.asarray(image).ndim == 2 else None)
                image_ax.set_title(str(image_id))
                image_ax.set_ylabel("Image")
                image_ax.axis("off")

                gt_im = gt_ax.imshow(gt_unc_map, cmap="viridis")
                pred_im = pred_ax.imshow(pred_unc_map, cmap="viridis")
                fig.colorbar(gt_im, ax=gt_ax, fraction=0.046, pad=0.04)
                fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)

                gt_ax.set_title(str(image_id))
                pred_ax.set_title(str(image_id))
                gt_ax.set_ylabel("GT unc map")
                pred_ax.set_ylabel("Pred unc map")

                gt_flat = gt_unc_map.ravel().astype(np.float64)
                pred_flat = pred_unc_map.ravel().astype(np.float64)
                gt_jittered = gt_flat + rng.uniform(-0.01, 0.01, size=gt_flat.size)
                scatter_ax.scatter(pred_flat, gt_jittered, s=4, alpha=0.25, color="royalblue")
                scatter_ax.set_xlabel("Pred uncertainty")
                scatter_ax.set_ylabel("GT uncertainty")
                scatter_ax.set_title(str(image_id))

                x_min = float(np.min(pred_flat))
                x_max = float(np.max(pred_flat))
                if x_max <= x_min:
                    x_max = x_min + 1e-8

                if pred_flat.size >= 2 and np.std(pred_flat) > 0:
                    slope, intercept = np.polyfit(pred_flat, gt_flat, 1)
                    x_line = np.linspace(x_min, x_max, 200)
                    y_line = slope * x_line + intercept
                else:
                    x_line = np.array([x_min, x_max])
                    y_line = np.full_like(x_line, float(np.mean(gt_flat)))

                scatter_ax.plot(x_line, y_line, color="red", linewidth=1.5)
                scatter_ax.text(
                    0.02,
                    0.98,
                    f"NCC = {ncc_value:.4f}",
                    transform=scatter_ax.transAxes,
                    va="top",
                    ha="left",
                    color="red",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

            for row_ax in axes.reshape(-1):
                row_ax.tick_params(labelsize=8)

            fig.tight_layout()
            plt.show()
    save_path = exp_dataloader.dataset_path / "ambiguity_modeling.json"
    with open(save_path, "w") as f:
        json.dump(ncc_dict, f, indent=2)
