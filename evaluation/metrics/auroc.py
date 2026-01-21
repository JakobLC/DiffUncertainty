import json
from pathlib import Path

import jsbeautifier
import numpy as np
from sklearn.metrics import roc_curve, auc

from evaluation.experiment_dataloader import ExperimentDataloader
from evaluation.split_file_generation.split_files_second_cycle import (
    get_splits_first_cycle,
    get_aggregated_uncertainties,
    get_samples_to_query,
)
from evaluation.utils.sort_uncertainties import sort_uncertainties


def is_ood_toy(sample):
    # In used toy datasets, samples smaller than 21 are OoD
    # Caution: This is currently hardcoded
    if int(sample.split(".")[0]) > 20:
        return False
    else:
        return True


def is_ood_split(sample, splits, fold=0):
    id_unlabeled_pool = splits[fold]["id_unlabeled_pool"]
    if type(id_unlabeled_pool[0]) == tuple:
        id_unlabeled_pool = [image[0] for image in id_unlabeled_pool]
    ood_unlabeled_pool = splits[fold]["ood_unlabeled_pool"]
    if type(ood_unlabeled_pool[0]) == tuple:
        ood_unlabeled_pool = [image[0] for image in ood_unlabeled_pool]
    if sample in id_unlabeled_pool:
        sample_index = np.argwhere(id_unlabeled_pool == sample)
        if sample_index.size > 1:
            print("Sample found multiple times")
        else:
            return False
    elif sample in ood_unlabeled_pool:
        sample_index = np.argwhere(ood_unlabeled_pool == sample)
        if sample_index.size > 1:
            print("Sample found multiple times")
        else:
            return True
    else:
        print("Could not find sample {}!".format(sample))
    return None


def is_ood(sample, splits, fold=0):
    if splits is None:
        return is_ood_toy(sample)
    else:
        return is_ood_split(sample, splits, fold)


def get_ood_detection_rate(
    samples_to_query,
    splits=None,
    fold=0,
    sample_labels=None,
    num_ood_samples_override=None,
):
    if sample_labels is None:
        samples_to_query = [f"{sample.split('.')[0]}.npy" for sample in samples_to_query]
        id = 0
        ood = 0
        for sample in samples_to_query:
            if not is_ood(sample=sample, splits=splits, fold=fold):
                id += 1
            elif is_ood(sample=sample, splits=splits, fold=fold):
                ood += 1
            else:
                print(f"Error for sample {sample}!")
        if splits is None:
            # In toy dataset, there are 21 OoD samples.
            # Caution: This is currently hardcoded
            num_ood_samples = 21
        else:
            num_ood_samples = len(splits[fold]["ood_unlabeled_pool"])
    else:
        ood = sum(1 for sample in samples_to_query if sample_labels.get(sample) == 1)
        if num_ood_samples_override is not None:
            num_ood_samples = num_ood_samples_override
        else:
            num_ood_samples = sum(1 for label in sample_labels.values() if label == 1)
        if num_ood_samples == 0:
            print("Warning: No OOD samples available for detection rate computation.")
            return 0.0
    ood_detection_rate = ood / num_ood_samples
    print("OOD Detection rate: ", ood_detection_rate)
    return ood_detection_rate


def get_auroc_input(
    uncertainties,
    aggregation,
    splits=None,
    fold=0,
    sample_labels=None,
):
    y_labels = []
    unc_scores = []
    if sample_labels is None:
        for sample, unc in uncertainties.items():
            sample = f"{sample.split('.')[0]}.npy"
            if not is_ood(sample=sample, splits=splits, fold=fold):
                y_labels.append(0)
                unc_scores.append(unc[aggregation]["max_score"])
            elif is_ood(sample=sample, splits=splits, fold=fold):
                y_labels.append(1)
                unc_scores.append(unc[aggregation]["max_score"])
            else:
                print("Error for sample {}!".format(sample))
    else:
        for sample, unc in uncertainties.items():
            if sample not in sample_labels:
                raise KeyError(
                    f"Missing label for sample '{sample}' while building AUROC inputs."
                )
            y_labels.append(sample_labels[sample])
            unc_scores.append(unc[aggregation]["max_score"])
    return y_labels, unc_scores


def ood_detection(
    exp_dataloader: ExperimentDataloader,
    base_splits_path=None,
):
    base_splits_path = Path(base_splits_path) if base_splits_path is not None else None
    if "shift" in exp_dataloader.exp_version.version_params:
        shift = exp_dataloader.exp_version.version_params["shift"]
    else:
        shift = None
    dataset_key = exp_dataloader.dataset_split or "full_dataset"
    ood_det_dict = {dataset_key: {"mean": {}}}
    pair_splits = getattr(exp_dataloader, "dataset_pair", None)
    if pair_splits:
        paired_unc_files = exp_dataloader.get_paired_aggregated_unc_files_dict()
        id_split, ood_split = pair_splits
        missing_uncs = set(paired_unc_files[id_split].keys()) ^ set(
            paired_unc_files[ood_split].keys()
        )
        if missing_uncs:
            raise ValueError(
                f"Aggregated uncertainty files differ between splits {id_split} and {ood_split}: {missing_uncs}"
            )
        unc_iterable = (
            (unc, (paired_unc_files[id_split][unc], paired_unc_files[ood_split][unc]))
            for unc in paired_unc_files[id_split].keys()
        )
    else:
        unc_iterable = exp_dataloader.get_aggregated_unc_files_dict().items()
    fold = exp_dataloader.exp_version.version_params["fold"]
    for unc, path_info in unc_iterable:
        if pair_splits:
            id_path, ood_path = path_info
            id_uncertainties = get_aggregated_uncertainties(id_path)
            ood_uncertainties = get_aggregated_uncertainties(ood_path)
            uncertainties = {}
            sample_labels = {}
            for split_name, source, label in [
                (pair_splits[0], id_uncertainties, 0),
                (pair_splits[1], ood_uncertainties, 1),
            ]:
                for sample, values in source.items():
                    combined_key = f"{split_name}::{sample}"
                    uncertainties[combined_key] = values
                    sample_labels[combined_key] = label
            num_ood_samples_override = len(ood_uncertainties)
            splits = None
        else:
            aggregated_unc_path = path_info
            uncertainties = get_aggregated_uncertainties(aggregated_unc_path)
            sample_labels = None
            num_ood_samples_override = None
        for aggregation in exp_dataloader.exp_version.aggregations:
            if not pair_splits:
                if base_splits_path is not None:
                    splits = get_splits_first_cycle(base_splits_path, shift=shift)
                else:
                    splits = None
            sorted_uncertainties = sort_uncertainties(uncertainties, aggregation)
            samples_to_query = get_samples_to_query(sorted_uncertainties, 0.5)
            ood_detection_rate = get_ood_detection_rate(
                samples_to_query=samples_to_query,
                splits=splits,
                fold=fold,
                sample_labels=sample_labels,
                num_ood_samples_override=num_ood_samples_override,
            )
            y_true, y_score = get_auroc_input(
                uncertainties=uncertainties,
                aggregation=aggregation,
                splits=splits,
                fold=fold,
                sample_labels=sample_labels,
            )
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ood_det_dict[dataset_key]["mean"].setdefault(unc, {})
            ood_det_dict[dataset_key]["mean"][unc][aggregation] = {
                "metrics": {"ood_detection_rate": ood_detection_rate, "auroc": roc_auc}
            }
            print("AUROC: ", roc_auc)
    save_path = exp_dataloader.exp_version.exp_path / "ood_detection.json"
    existing_payload = {}
    if save_path.exists():
        with open(save_path) as f:
            existing_payload = json.load(f)
    existing_payload.update(ood_det_dict)
    opts = jsbeautifier.default_options()
    opts.indent_size = 4
    with open(save_path, "w") as f:
        f.write(jsbeautifier.beautify(json.dumps(existing_payload), opts))
