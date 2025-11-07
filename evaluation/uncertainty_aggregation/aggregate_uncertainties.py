import json

import hydra.utils
import jsbeautifier
import numpy as np
from medpy.io import load
from scipy.signal import convolve
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader


def patch_level_aggregation(image, patch_size, mean=False, **kwargs):
    if type(patch_size) == int:
        patch_size = len(image.shape) * [patch_size]
    kernel = np.ones(patch_size)
    patch_aggragated = convolve(image, kernel, mode="valid")
    if mean:
        patch_aggragated = patch_aggragated / (np.prod(patch_size))
    all_max_indices = np.where(np.isclose(patch_aggragated, np.max(patch_aggragated)))
    max_indices = []
    for indices in all_max_indices:
        max_indices.append(indices[0])

    max_indices_slice = []
    for idx, index in enumerate(max_indices):
        max_indices_slice.append((int(index), int(index + patch_size[idx])))
    return {
        "max_score": float(np.max(patch_aggragated)),
        "bounding_box": max_indices_slice,
    }


def image_level_aggregation(image, mean=False, **kwargs):
    if mean:
        return float(np.sum(image) / image.size)
    return {"max_score": float(np.sum(image))}


def threshold_aggregation(
    image,
    threshold=None,
    threshold_path=None,
    pred_model=None,
    unc_type=None,
    mean=True,
):
    if threshold is None:
        if threshold_path is None:
            raise Exception(
                "A threshold needs to be provided for threshold aggregation!"
            )
        with open(threshold_path) as f:
            threshold_json = json.load(f)
        if pred_model is None or unc_type is None:
            raise Exception(
                "If you want to load the threshold from a json file, you have to provide the prediction model and the uncertainty type"
            )
        unc_type_split = unc_type.split("_")[0]
        threshold = threshold_json[pred_model][f"Mean {unc_type_split} threshold"]
    uncertainty_sum = image[image >= threshold].sum()
    count = (image >= threshold).sum()
    if mean:
        if count > 0:
            uncertainty_mean = uncertainty_sum / count
            return {"max_score": uncertainty_mean, "threshold": threshold}
    return {"max_score": uncertainty_sum, "threshold": threshold}


def aggregate_uncertainties(exp_dataloader: ExperimentDataloader, aggregations):
    for unc, unc_path in exp_dataloader.unc_path_dict.items():
        all_uncs = {}
        print("Aggregate uncertainties")
        for image_id in tqdm(exp_dataloader.image_ids):
            all_uncs[f"{image_id}{exp_dataloader.exp_version.unc_ending}"] = {}
            for aggregation in aggregations:
                unc_image, _ = load(
                    unc_path / f"{image_id}{exp_dataloader.exp_version.unc_ending}"
                )
                # TODO: Probably pass pred model and unc more dynamically
                aggregation_config = aggregations[aggregation]
                instantiate_kwargs = {
                    "image": unc_image,
                    "pred_model": exp_dataloader.exp_version.pred_model,
                    "unc_type": unc,
                }
                target_path = aggregation_config.get("_target_", "")
                threshold_path_cfg = (
                    aggregation_config["threshold_path"]
                    if "threshold_path" in aggregation_config
                    else None
                )
                if (
                    "threshold_aggregation" in target_path
                    and (
                        threshold_path_cfg is None
                        or str(threshold_path_cfg).lower() == "none"
                    )
                ):
                    instantiate_kwargs["threshold_path"] = (
                        exp_dataloader.exp_version.exp_path
                        / "threshold_analysis.json"
                    )
                unc_dict = hydra.utils.instantiate(
                    aggregation_config,
                    **instantiate_kwargs,
                )
                all_uncs[f"{image_id}{exp_dataloader.exp_version.unc_ending}"][
                    aggregation
                ] = unc_dict
        save_path = exp_dataloader.dataset_path / f"aggregated_{unc}.json"
        print(save_path)
        opts = jsbeautifier.default_options()
        opts.indent_size = 4
        #convert to float from float32 numpy types for json serialization
        for image_id, unc_dict in all_uncs.items():
            for aggregation, values in unc_dict.items():
                for key, value in values.items():
                    if isinstance(value, np.floating):
                        all_uncs[image_id][aggregation][key] = float(value)
        with open(save_path, "w") as f:
            f.write(jsbeautifier.beautify(json.dumps(all_uncs), opts))
    print()
