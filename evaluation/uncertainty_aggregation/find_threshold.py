import json
from pathlib import Path

import numpy as np
from medpy.io import load

from evaluation.experiment_dataloader import ExperimentDataloader


def calculate_foreground_quantile_image(image):
    foreground = np.count_nonzero(image)
    return 1 - (foreground / image.size)


def get_foreground_quantile(exp_dataloader: ExperimentDataloader):
    print(exp_dataloader.dataset_path)
    quantile_dict = {exp_dataloader.exp_version.pred_model: {}}
    all_quantiles = []
    for image_id in exp_dataloader.image_ids:
        pred_segs = exp_dataloader.get_pred_segs(image_id)
        for pred_seg in pred_segs:
            perc = calculate_foreground_quantile_image(pred_seg)
            all_quantiles.append(perc)
    quantile_dict[exp_dataloader.exp_version.pred_model][
        exp_dataloader.exp_version.version_name
    ] = {
        "quantiles": all_quantiles,
        "exp_path": exp_dataloader.exp_version.exp_path.as_posix(),
    }
    return quantile_dict


def save_foreground_quantiles(results_dict, save_path=None):
    for method, versions in results_dict.items():
        for version_name, version_data in versions.items():
            exp_path = Path(version_data["exp_path"])
            exp_path.mkdir(parents=True, exist_ok=True)
            quantiles = version_data["quantiles"]
            if not quantiles:
                continue
            method_mean = float(np.mean(np.array(quantiles)))
            save_file = exp_path / "quantile_analysis.json"
            with open(save_file, "w") as f:
                json.dump({method: method_mean}, f, indent=2)
            print(save_file)


def threshold_images_paths(exp_dataloader: ExperimentDataloader):
    unc_image_path_dict = {
        exp_dataloader.exp_version.pred_model: {
            exp_dataloader.exp_version.version_name: {}
        }
    }
    version_dict = unc_image_path_dict[exp_dataloader.exp_version.pred_model][
        exp_dataloader.exp_version.version_name
    ]
    version_dict["exp_path"] = exp_dataloader.exp_version.exp_path.as_posix()
    version_dict["unc_paths"] = {}
    for unc_type in exp_dataloader.exp_version.unc_types:
        uncertainty_path = exp_dataloader.unc_path_dict[unc_type]
        version_dict["unc_paths"][unc_type] = []
        for image_id in exp_dataloader.image_ids:
            version_dict["unc_paths"][unc_type].append(
                (uncertainty_path / f"{image_id}{exp_dataloader.exp_version.unc_ending}").as_posix()
            )
    return unc_image_path_dict


def calculate_threshold_image(quantile_path: Path, image: np.array, method: str):
    quantile_path = Path(quantile_path)
    with open(quantile_path) as f:
        all_quantiles = json.load(f)
    method_quantile = all_quantiles[method]
    flattened_image = image.flatten() if isinstance(image, np.ndarray) else np.array(image).flatten()
    threshold = np.quantile(flattened_image, method_quantile)
    return float(threshold)


def find_threshold(results_dict, quantile_path=None, save_path=None):
    for pred_model, versions in results_dict.items():
        for version_name, version_data in versions.items():
            exp_path = Path(version_data["exp_path"])
            exp_path.mkdir(parents=True, exist_ok=True)
            quantile_file = exp_path / "quantile_analysis.json"
            if not quantile_file.is_file():
                raise FileNotFoundError(
                    f"Quantile file not found for {pred_model} {version_name}: {quantile_file}"
                )
            threshold_entries = {}
            for unc, paths in version_data["unc_paths"].items():
                if not paths:
                    continue
                unc_images = []
                for path in paths:
                    unc_image, _ = load(path)
                    unc_images.append(unc_image.ravel())
                if not unc_images:
                    continue
                concatenated_unc = np.concatenate(unc_images)
                threshold_value = calculate_threshold_image(
                    quantile_file, concatenated_unc, method=pred_model
                )
                key = f"Mean {unc.split('_')[0]} threshold"
                threshold_entries[key] = float(threshold_value)
                print(f"{pred_model} {version_name} {key}: {threshold_value}")
            if not threshold_entries:
                continue
            threshold_json = {pred_model: threshold_entries}
            save_file = exp_path / "threshold_analysis.json"
            with open(save_file, "w") as f:
                json.dump(threshold_json, f, indent=2)
            print(save_file)
