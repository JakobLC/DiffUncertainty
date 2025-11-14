import copy
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import hydra
import yaml
from collections import OrderedDict
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
#from torchmetrics.functional import dice
from evaluation.metrics.dice_wrapped import dice

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

from uncertainty_modeling.data_carrier_3D import DataCarrier3D
from uncertainty_modeling.models.ssn_unet3D_module import SsnUNet3D
from loss_modules import SoftDiceLoss
from main import set_seed
from tqdm import tqdm

from global_utils.checkpoint_format import format_checkpoint_subdir#, infer_epoch_from_path


def test_cli(config_file: str = None) -> Namespace:
    """
    Set the arguments for testing
    Args:
        config_file: optional, path to default arguments for testing.

    Returns:
        args [Namespace]: all arguments needed for testing
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        help="The path to the checkpoint that should be used to load the model. "
        "Multiple paths can be given for an ensemble prediction. "
        "In this case, configuration parameters like the patch size should be the same for all models "
        "and will be inferred from the checkpoint of the first model.",
    )
    parser.add_argument(
        "-i",
        "--data_input_dir",
        type=str,
        default=None,
        help="If given, dataset root directory to load from. "
        "Otherwise, the input dir will be inferred from the checkpoint. "
        "Specify this if you train and test on different machines (E.g. training on cluster and local testing).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If given, uses this string as root directory to save results in. "
        "Otherwise, the save dir will be inferred from the checkpoint. "
        "Specify this if you train and test on different machines (E.g. training on cluster and local testing).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="If given, uses this string as experiment name. "
        "Otherwise, the experiment name will be inferred from the checkpoint.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="If given, uses this string as test input."
        "Otherwise, the test samples will be inferred from the test split used in training",
    )
    parser.add_argument(
        "--subject_ids",
        type=list,
        default=None,
        help="If given, only use these subject of the test folder."
        "Otherwise, all images from the test input directory will be used.",
    )
    parser.add_argument(
        "--n_pred",
        type=int,
        default=10,
        help="Number of predictions to make by the model",
    )
    parser.add_argument(
        "--n_reference_samples",
        type=int,
        default=5,
        help="Number of generated reference samples if samples are simulated",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        nargs="?",
        help="Size of the test batches to pass. If specified without number, uses same batch size as used in training.",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="id",
        help="Comma separated list of test splits to evaluate. Each split is processed in a separate run.",
    )
    parser.add_argument(
        "--test_time_augmentations", "-tta", dest="tta", action="store_true"
    )
    parser.add_argument(
        "--ema_mode",
        type=str,
        default="normal",
        choices=["normal", "ema", "both"],
        help="Select which weights to evaluate: 'normal', 'ema', or 'both'.",
    )
    parser.add_argument(
        "--wildcard_replace",
        type=str,
        default=None,
        help=(
            "Comma-separated replacement strings for '*' in checkpoint paths. "
            "Example: --wildcard_replace=120,121,122 and --checkpoint_paths=/path/aug0_s*/ckpt.ckpt"
        ),
    )
    parser.add_argument(
        "--ensemble_mode",
        action="store_true",
        default=False,
        help=(
            "Group checkpoints as ensembles instead of forming a cross-product. "
            "Files: combine all files into one ensemble. Folders: create one ensemble per common filename across folders."
        ),
    )
    parser.add_argument(
        "--skip_ged",
        action="store_true",
        default=False,
        help=(
            " Skip evaluation of the Generalized Energy Distance (GED) metric. "
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help=(
            "Skip evaluation if all expected output files for a job already exist. "
            "Checks at the deepest level (per model group, split, EMA mode)."
        ),
    )

    if config_file is not None:
        with open(os.path.join(os.path.dirname(__file__), config_file), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**config)

    args = parser.parse_args()
    return args


def _build_checkpoint_groups(
    entries: List[str],
    wildcard_replacements: Optional[List[str]] = None,
    ensemble_mode: bool = False,
) -> List[List[str]]:
    """
    Build checkpoint groups from raw entries according to ensemble semantics.

    Behavior:
    - When ensemble_mode is False (default):
      * Files: each file is its own group
      * Folders: union of all .ckpt files across folders, each as its own group
      * Mixing files and folders is allowed; all are treated individually
    - When ensemble_mode is True:
      * Files only: all files are combined into a single ensemble group (requires >=2 files)
      * Folders only: create one ensemble per common filename across all folders (requires >=2 folders)
      * Mixing files and folders is not supported and will raise an error
    - Wildcards ('*') in entries must be provided with --wildcard_replace and are expanded prior to grouping.
    """
    if not entries:
        raise ValueError("No checkpoint paths provided.")

    # Helper to materialize one raw entry into concrete paths or folders
    def _resolve_entry(e: str) -> Tuple[List[str], List[str]]:
        files: List[str] = []
        folders: List[str] = []
        if "*" in e:
            tokens = [t.strip() for t in (wildcard_replacements or []) if t and t.strip()]
            if not tokens:
                raise ValueError(
                    "checkpoint_paths contains '*' but --wildcard_replace was not provided or empty."
                )
            for token in tokens:
                replaced = e.replace("*", token)
                p = Path(replaced)
                if p.is_dir():
                    folders.append(p.as_posix())
                else:
                    files.append(p.as_posix())
        else:
            p = Path(e)
            if p.is_dir():
                folders.append(p.as_posix())
            else:
                files.append(p.as_posix())
        return files, folders

    all_files: List[str] = []
    folder_to_files: OrderedDict[str, List[str]] = OrderedDict()

    for raw in entries:
        files, folders = _resolve_entry(raw)
        # Collect files
        all_files.extend(files)
        # Collect folder contents
        for folder in folders:
            ckpts = sorted(Path(folder).glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt files found in directory: {folder}")
            folder_to_files[folder] = [c.as_posix() for c in ckpts]

    if not ensemble_mode:
        # Individual evaluation of each file
        union: List[str] = list(sorted(set(all_files)))
        for files in folder_to_files.values():
            union.extend(files)
        # Ensure uniqueness and stable order
        union = list(OrderedDict((p, None) for p in union).keys())
        return [[p] for p in union]

    # Ensemble mode
    has_files = len(all_files) > 0
    has_folders = len(folder_to_files) > 0
    if has_files and has_folders:
        raise ValueError("--ensemble_mode does not support mixing files and folders in one run. Provide either files or folders.")

    if has_folders:
        if len(folder_to_files) < 2:
            raise ValueError("--ensemble_mode with folders requires at least two folders to ensemble across.")
        # Find common basenames across all folders
        basename_sets = [set(Path(p).name for p in files) for files in folder_to_files.values()]
        common_basenames = set.intersection(*basename_sets) if basename_sets else set()
        if not common_basenames:
            raise ValueError("No common checkpoint filenames across folders for ensembling.")
        groups: List[List[str]] = []
        for base in sorted(common_basenames):
            group: List[str] = []
            for folder in folder_to_files.keys():
                group.append((Path(folder) / base).as_posix())
            groups.append(group)
        return groups

    # Files only
    unique_files = list(sorted(set(all_files)))
    if len(unique_files) < 2:
        raise ValueError("--ensemble_mode with files requires at least two files.")
    return [unique_files]


def _parse_test_splits(split_arg: str) -> List[str]:
    if split_arg is None:
        return ["id"]
    splits = [s.strip() for s in str(split_arg).split(",") if s.strip()]
    return splits or ["id"]


def _ema_mode_to_flags(mode: str) -> List[Tuple[str, bool]]:
    normalized = (mode or "normal").lower()
    if normalized == "normal":
        return [("normal", False)]
    if normalized == "ema":
        return [("ema", True)]
    if normalized == "both":
        # Evaluate EMA first, then normal to mirror manual workflow
        return [("ema", True), ("normal", False)]
    raise ValueError(f"Unsupported ema_mode '{mode}'.")


def prepare_evaluation_jobs(args: Namespace) -> List[Namespace]:
    raw_paths = args.checkpoint_paths
    if raw_paths is None:
        raise ValueError("--checkpoint_paths must be provided.")
    if isinstance(raw_paths, str):
        raw_list = [raw_paths]
    else:
        raw_list = list(raw_paths)

    # Parse wildcard replacements if provided
    wildcard_replacements: Optional[List[str]] = None
    if getattr(args, "wildcard_replace", None):
        wildcard_replacements = [s.strip() for s in str(args.wildcard_replace).split(",") if s.strip()]

    checkpoint_combos = _build_checkpoint_groups(
        raw_list, wildcard_replacements, bool(getattr(args, "ensemble_mode", False))
    )
    test_splits = _parse_test_splits(args.test_split)
    ema_options = _ema_mode_to_flags(getattr(args, "ema_mode", "normal"))

    total_jobs = len(checkpoint_combos) * len(test_splits) * len(ema_options)
    print(f"About to launch {total_jobs} evaluation jobs.")
    print(f"len(checkpoint_combos)={len(checkpoint_combos)}, len(test_splits)={len(test_splits)}, len(ema_options)={len(ema_options)}")

    # Helper: derive an informative ensemble version name for saving
    def _extract_version_from_ckpt_path(p: str) -> str:
        path = Path(p)
        # Typical: .../<version>/(checkpoints|scheduled_ckpts)/file.ckpt
        if path.suffix == ".ckpt":
            parent = path.parent  # checkpoints folder
            if parent.name in ("checkpoints", "scheduled_ckpts") and parent.parent:
                return parent.parent.name
            # Fallback to parent folder if structure differs
            return parent.name
        # If a folder path slipped through (shouldn't in groups), use folder name
        return path.name

    def _build_group_version_name(
        raw_entries: List[str],
        replacements: Optional[List[str]],
        group_paths: List[str],
    ) -> Optional[str]:
        # Only build a special name for ensembles
        if len(group_paths) <= 1:
            return None

        versions = [_extract_version_from_ckpt_path(p) for p in group_paths]

        # If user provided replacements, try to reconstruct prefix[specified_tokens]suffix
        # even if the shell expanded '*' and raw_entries no longer contain it.
        if replacements:
            reps = [t for t in replacements if t]
            # Attempt to factor each version as prefix + rep + suffix with a common prefix/suffix
            common_prefix: Optional[str] = None
            common_suffix: Optional[str] = None
            matched_tokens: List[str] = []
            success = True
            for v in versions:
                found_match = False
                for r in reps:
                    idx = v.find(r)
                    if idx != -1:
                        pre = v[:idx]
                        suf = v[idx + len(r):]
                        if common_prefix is None and common_suffix is None:
                            common_prefix, common_suffix = pre, suf
                            found_match = True
                            matched_tokens.append(r)
                            break
                        else:
                            if pre == common_prefix and suf == common_suffix:
                                found_match = True
                                matched_tokens.append(r)
                                break
                if not found_match:
                    success = False
                    break
            if success and common_prefix is not None:
                # Preserve the order provided by replacements for readability
                ordered_unique = []
                seen = set()
                for r in reps:
                    if r in matched_tokens and r not in seen:
                        seen.add(r)
                        ordered_unique.append(r)
                candidate = f"{common_prefix}[{','.join(ordered_unique)}]{common_suffix}"
                if len(candidate) <= 50:
                    return candidate
                short = ordered_unique[:2]
                return f"{common_prefix}[{','.join(short)},etc]{common_suffix}"

        # Prefer star-based naming when any raw entry had a '*':
        # derive tokens by factoring common prefix/suffix across concrete version names.
        if any("*" in e for e in raw_entries):
            def longest_common_prefix(strs: List[str]) -> str:
                if not strs:
                    return ""
                s1, s2 = min(strs), max(strs)
                i = 0
                for i, (c1, c2) in enumerate(zip(s1, s2)):
                    if c1 != c2:
                        return s1[:i]
                return s1[: len(s1) if len(s1) <= len(s2) else len(s2)]

            def longest_common_suffix(strs: List[str]) -> str:
                rev = [s[::-1] for s in strs]
                pref = longest_common_prefix(rev)
                return pref[::-1]

            lcp = longest_common_prefix(versions)
            lcs = longest_common_suffix(versions)

            # Extract the varying tokens between lcp and lcs for each version
            tokens: List[str] = []
            for v in versions:
                start = len(lcp)
                end = len(v) - len(lcs) if lcs else len(v)
                token = v[start:end]
                tokens.append(token)

            # If tokens are non-empty and represent the variation, build prefix[tokens]suffix
            if any(t for t in tokens):
                # Preserve order and remove duplicates
                seen = set()
                uniq_tokens: List[str] = []
                for t in tokens:
                    if t not in seen:
                        seen.add(t)
                        uniq_tokens.append(t)
                candidate = f"{lcp}[{','.join(uniq_tokens)}]{lcs}"
                if len(candidate) <= 50:
                    return candidate
                # Compress to first two + etc if exceeding limit
                short_tokens = uniq_tokens[:2]
                return f"{lcp}[{','.join(short_tokens)},etc]{lcs}"

        # Otherwise, build from the concrete group paths' version names
        # Unique while preserving order
        seen = set()
        uniq = []
        for v in versions:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        name = f"[{','.join(uniq)}]"
        if len(name) <= 50:
            return name
        short = uniq[:2]
        return f"[{','.join(short)},etc]"

    jobs: List[Namespace] = []
    for checkpoints in checkpoint_combos:
        for split in test_splits:
            for ema_label, use_ema in ema_options:
                job_args = copy.deepcopy(args)
                job_args.checkpoint_paths = list(checkpoints)
                job_args.test_split = split
                job_args.use_ema = use_ema
                job_args.current_ema_label = ema_label
                # Derive version override for ensembles so results reflect group members
                job_args.version_override = _build_group_version_name(
                    raw_list, wildcard_replacements, job_args.checkpoint_paths
                )
                jobs.append(job_args)
    return jobs

def _metrics_complete_3d(args: Namespace, hparams: Dict) -> bool:
    """Return True if metrics.json exists and appears complete.

    Completeness heuristic:
    - File exists
    - Parses as JSON containing a 'mean' key (written at end of evaluation)
    """
    root_dir = args.save_dir if args.save_dir is not None else hparams["save_dir"]
    exp_name = hparams["exp_name"] if args.exp_name is None else args.exp_name
    version = getattr(args, "version_override", None) or hparams["version"]
    checkpoint_tag = getattr(args, "checkpoint_subdir", None)
    parts = [root_dir, exp_name, "test_results", str(version)]
    if checkpoint_tag:
        parts.append(checkpoint_tag)
    parts.append(args.test_split)
    metrics_path = os.path.join(*parts, "metrics.json")
    if not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "mean" in data:
            return True
    except Exception:
        return False
    return False


def dir_and_subjects_from_train(
    hparams: Dict, args: Namespace
) -> Tuple[str, List[str]]:
    """
    Get the test samples from the training configuration loaded through the checkpoint
    Args:
        hparams: The hyperparameters from the checkpoint. Needed to infer the path where the test data is as well
                 as the subject ids that are in the test data
        args: Arguments for testing, possibly specifying a data_input_dir

    Returns:
        test_data_dir [str]: The directory which contains the test images
        subject_ids [List[str]]: The list of the subject ids which should be inferred during testing
    """

    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    dataset_name = hparams["datamodule"]["dataset_name"]

    with open(os.path.join(data_input_dir, dataset_name, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    fold = hparams["datamodule"]["data_fold_id"]
    print(args.test_split)
    subject_ids = splits[fold][args.test_split]

    test_data_dir = os.path.join(data_input_dir, dataset_name, "preprocessed")
    return test_data_dir, subject_ids


def dir_and_subjects_from_train_lidc(
    hparams: Dict, args: Namespace, test_split: str = "id"
) -> Tuple[str, List[str]]:
    """
    Get the test samples from the training configuration loaded through the checkpoint
    Args:
        hparams: The hyperparameters from the checkpoint. Needed to infer the path where the test data is as well
                 as the subject ids that are in the test data
        args: Arguments for testing, possibly specifying a data_input_dir
        id: whether to predict id cases

    Returns:
        test_data_dir [str]: The directory which contains the test images
        subject_ids [List[str]]: The list of the subject ids which should be inferred during testing
    """

    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    # dataset_name = hparams["datamodule"]["dataset_name"]
    shift_feature = hparams["datamodule"]["shift_feature"]

    if "splits_path" in hparams["datamodule"].keys():
        if hparams["datamodule"]["splits_path"] is not None:
            if args.data_input_dir is not None:
                splits_path = hparams["datamodule"]["splits_path"].replace(
                    hparams["data_input_dir"], args.data_input_dir
                )
            else:
                splits_path = hparams["datamodule"]["splits_path"]
        else:
            splits_path = os.path.join(
                data_input_dir,
                "splits_{}.pkl".format(shift_feature)
                if shift_feature is not None
                else "all",
            )
    else:
        splits_path = os.path.join(
            data_input_dir,
            "splits_{}.pkl".format(shift_feature)
            if shift_feature is not None
            else "all",
        )

    with open(
        splits_path,
        "rb",
    ) as f:
        splits = pickle.load(f)
    fold = hparams["datamodule"]["data_fold_id"]
    if test_split == "unlabeled":
        subject_ids = splits[fold]["id_unlabeled_pool"]
        subject_ids = np.concatenate((subject_ids, splits[fold]["ood_unlabeled_pool"]))
    elif test_split == "val":
        subject_ids = splits[fold]["val"]
    elif test_split == "train":
        subject_ids = splits[fold]["train"]
    else:
        subject_ids = splits[fold]["{}_test".format(test_split)]

    test_data_dir = os.path.join(data_input_dir, "preprocessed")
    return test_data_dir, subject_ids


def load_models_from_checkpoint(
    checkpoints: List[Dict], device="cpu", use_ema: bool = False
) -> List[nn.Module]:
    """
    Load the model for the predictions from a checkpoint
    Args:
        checkpoints: The checkpoints to load the model from

    Returns:
        model: The model for the predictions
    """
    all_models = []
    for checkpoint in checkpoints:
        hparams = checkpoint["hyper_parameters"]
        state_dict = OrderedDict()
        if use_ema:
            if "ema_state_dict" in checkpoint:
                source_items = checkpoint["ema_state_dict"].items()
                cleaned = []
                for k, v in source_items:
                    if k == "n_averaged":
                        continue
                    key = k.split("module.", 1)[1] if k.startswith("module.") else k
                    cleaned.append((key, v))
            else:
                source_items = [
                    (k, v)
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("ema_model.")
                ]
                if not source_items:
                    raise ValueError(
                        "EMA weights requested but checkpoint does not contain an ema_model."  # noqa: E501
                    )
                cleaned = []
                for k, v in source_items:
                    key = k.split("ema_model.", 1)[1]
                    if key == "n_averaged":
                        continue
                    key = key.split("module.", 1)[1] if key.startswith("module.") else key
                    cleaned.append((key, v))
        else:
            source_items = [
                (k, v)
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            ]
            cleaned = [(k.split(".", 1)[1], v) for k, v in source_items]

        for key, value in cleaned:
            state_dict[key] = value
        if "aleatoric_loss" in hparams and hparams["aleatoric_loss"] is not None:
            model = hydra.utils.instantiate(
                hparams["model"], aleatoric_loss=hparams["aleatoric_loss"]
            )
        else:
            model = hydra.utils.instantiate(hparams["model"])
        model.load_state_dict(state_dict=state_dict)
        # Ensure deterministic evaluation behavior (e.g., disable dropout/batchnorm updates)
        model.eval()
        all_models.append(model.to(device))
    return all_models


def calculate_test_metrics(
    output_softmax: torch.Tensor, ground_truth: torch.Tensor
) -> Dict:
    """
    Calculate the metrics for evaluation
    Args:
        output_softmax [torch.Tensor]: The output of the network after applying softmax.
        ground_truth [torch.Tensor]: The ground truth segmentation.

    Returns:
        metrics_dict [Dict]: A dictionary with the calculated metrics
    """
    dice_loss = SoftDiceLoss()
    nll_loss = torch.nn.NLLLoss()

    all_test_loss = []
    all_test_dice = []
    for rater in range(ground_truth.size(dim=0)):
        gt_seg = ground_truth[rater]
        gt_seg = torch.unsqueeze(gt_seg, 0).type(torch.LongTensor)

        test_loss = dice_loss(output_softmax, gt_seg) + nll_loss(
            torch.log(output_softmax), gt_seg
        )
        test_dice = dice(output_softmax, gt_seg, 
                        is_softmax=1,
                        num_classes=output_softmax.size(dim=1),
                        binary_dice=output_softmax.size(dim=1) == 2)

        all_test_loss.append(test_loss.item())
        all_test_dice.append(test_dice.item())
    metrics_dict = {
        "loss": np.mean(np.array(all_test_loss)),
        "dice": np.mean(np.array(all_test_dice)),
    }
    return metrics_dict


def calculate_ged(
    output_softmax: torch.Tensor,
    ground_truth: torch.Tensor,
    ignore_index: int = 0,
    additional_metrics: Optional[List[str]] = None,
) -> Dict:
    if additional_metrics is None:
        additional_metrics = ["dice"]

    n_pred = output_softmax.shape[0]
    n_gt = ground_truth.shape[0]
    num_classes = output_softmax.shape[1]


    device = output_softmax.device
    dice_matrix = torch.zeros((n_pred, n_gt), dtype=torch.float32, device=device)
    for pred_idx in range(n_pred):
        pred_softmax = output_softmax[pred_idx : pred_idx + 1].detach()
        for gt_idx in range(n_gt):
            gt_seg = ground_truth[gt_idx : gt_idx + 1].detach()
            dice_score = dice(
                pred_softmax.cpu(),
                gt_seg.cpu(),
                binary_dice=num_classes == 2,
                num_classes=num_classes,
                ignore_index=ignore_index,
                is_softmax=True,
            )
            dice_matrix[pred_idx, gt_idx] = float(dice_score)
    one_minus_dice = 1.0 - dice_matrix
    dist_gt_pred_2 = one_minus_dice.mean().item()

    if n_pred > 1:
        pred_labels = output_softmax.argmax(dim=1)
        pred_distances: List[float] = []
        for i in range(n_pred):
            for j in range(n_pred):
                dice_score = dice(
                    pred_labels[i : i + 1].cpu(),
                    pred_labels[j : j + 1].cpu(),
                    num_classes=num_classes,
                    binary_dice=num_classes == 2,
                    ignore_index=None,
                )
                pred_distances.append(1.0 - float(dice_score))
        dist_pred_pred_2 = float(torch.tensor(pred_distances).mean().item()) if pred_distances else 0.0
    else:
        dist_pred_pred_2 = 0.0

    if n_gt > 1:
        gt_distances: List[float] = []
        for i in range(n_gt):
            for j in range(n_gt):
                dice_score = dice(
                    ground_truth[i : i + 1].cpu(),
                    ground_truth[j : j + 1].cpu(),
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    binary_dice=num_classes == 2,
                )
                gt_distances.append(1.0 - float(dice_score))
        dist_gt_gt_2 = float(torch.tensor(gt_distances).mean().item()) if gt_distances else 0.0
    else:
        dist_gt_gt_2 = 0.0

    ged = 2 * dist_gt_pred_2 - dist_pred_pred_2 - dist_gt_gt_2

    results: Dict[str, float | torch.Tensor] = {"ged": ged}

    dice_mean = dice_matrix.mean().item()
    if "dice" in additional_metrics:
        results["dice"] = dice_mean

    if "max_dice_pred" in additional_metrics:
        results["max_dice_pred"] = dice_matrix.max(dim=1).values.mean().item()

    if "max_dice_gt" in additional_metrics:
        results["max_dice_gt"] = dice_matrix.max(dim=0).values.mean().item()

    if "major_dice" in additional_metrics:
        majority_pred = output_softmax.mean(dim=0).argmax(dim=0)
        if num_classes == 2:
            majority_gt = (ground_truth.float().mean(dim=0) >= 0.5).to(torch.long)
        else:
            majority_gt = torch.mode(ground_truth, dim=0).values
        dice_score = dice(
                    majority_pred,
                    majority_gt,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    binary_dice=num_classes == 2,
                )

    if "dice_matrix" in additional_metrics:
        results["dice_matrix"] = dice_matrix.cpu()

    return results


def predict_cases_ssn(
    test_datacarrier: DataCarrier3D,
    data_samples: List[Dict],
    model: nn.Module,
    n_pred: int = 1,
) -> DataCarrier3D:
    model = model.double()
    for sample in tqdm(data_samples):
        input = test_datacarrier.load_image(sample)
        input["data"] = np.expand_dims(input["data"], axis=0)
        to_tensor = Compose([NumpyToTensor()])
        input_tensor = to_tensor(**input)
        distribution = model.forward(input_tensor["data"])
        output_samples = distribution.sample([n_pred])
        output_samples = output_samples.view(
            [
                n_pred,
                1,
                model.num_classes,
                *input_tensor["data"].size()[-3:],
            ]
        )

        pred_idx = 0
        for output_sample in output_samples:
            output_softmax = F.softmax(output_sample, dim=1)

            test_datacarrier.concat_data(
                batch=input_tensor,
                softmax_pred=output_softmax,
                n_pred=n_pred,
                pred_idx=pred_idx,
                sigma=None,
            )
            pred_idx += 1
    return test_datacarrier


def predict_cases(
    test_datacarrier: DataCarrier3D,
    data_samples: List[Dict],
    models: List[nn.Module],
    n_pred: int = 1,
    n_aleatoric_samples: int = 10,
    tta: bool = False,
) -> DataCarrier3D:
    """
    Predict all test cases.
    Args:
        test_datacarrier: The datacarrier to save the data
        data_samples: The samples to predict
        model: The model used for prediction

    Returns:
        test_datacarrier: The datacarrier with the concatenated data
    """
    for sample in tqdm(data_samples):
        input = test_datacarrier.load_image(sample)
        input["data"] = np.expand_dims(input["data"], axis=0)
        to_tensor = Compose([NumpyToTensor()])
        input_tensor = to_tensor(**input)

        pred_idx = 0
        for model in models:
            model = model.double().to("cuda")
            if tta:
                input["data"] = input["data"].copy()
                noise_to_tensor = Compose([GaussianNoiseTransform(), NumpyToTensor()])
                input_noise_tensor = noise_to_tensor(**input)
                flip_dims = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
                sigma = None
                n_pred = 2 * len(flip_dims) + 2
                for x in [input_tensor["data"], input_noise_tensor["data"]]:
                    output = model.forward(x.double().to("cuda"))
                    output_softmax = F.softmax(output, dim=1).to("cpu")
                    test_datacarrier.concat_data(
                        batch=input_tensor,
                        softmax_pred=output_softmax,
                        n_pred=n_pred * len(models),
                        pred_idx=pred_idx,
                        sigma=sigma,
                    )
                    pred_idx += 1
                    for flip_dim in flip_dims:
                        output = torch.flip(
                            model.forward(torch.flip(x.to("cuda"), flip_dim)), flip_dim
                        )
                        output_softmax = F.softmax(output, dim=1).to("cpu")
                        test_datacarrier.concat_data(
                            batch=input_tensor,
                            softmax_pred=output_softmax,
                            n_pred=n_pred * len(models),
                            pred_idx=pred_idx,
                            sigma=sigma,
                        )
                        pred_idx += 1
            else:
                if hasattr(model, "aleatoric_loss") and model.aleatoric_loss == True:
                    n_pred = n_aleatoric_samples
                    mu, s = model.forward(input_tensor["data"].double().to("cuda"))
                    sigma = torch.exp(s / 2)
                for pred in range(n_pred):
                    if (
                        hasattr(model, "aleatoric_loss")
                        and model.aleatoric_loss == True
                    ):
                        epsilon = torch.randn(s.size())
                        output = mu + sigma * epsilon
                        output_softmax = F.softmax(output, dim=1)
                    else:
                        output = model.forward(input_tensor["data"].double().to("cuda"))
                        output_softmax = F.softmax(output, dim=1)
                        sigma = None

                    test_datacarrier.concat_data(
                        batch=input_tensor,
                        softmax_pred=output_softmax,
                        n_pred=n_pred * len(models),
                        pred_idx=pred_idx,
                        sigma=sigma,
                    )
                    pred_idx += 1
    return test_datacarrier


def calculate_uncertainty(softmax_preds: torch.Tensor):
    uncertainty_dict = {}
    # softmax_preds = torch.from_numpy(softmax_preds)
    mean_softmax = torch.mean(softmax_preds, dim=0)
    pred_entropy = torch.zeros(*softmax_preds.shape[2:], device=mean_softmax.device)
    for y in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[y] * torch.log(mean_softmax[y])
        nan_pos = torch.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1
    expected_entropy = torch.zeros(
        softmax_preds.shape[0], *softmax_preds.shape[2:], device=softmax_preds.device
    )
    for pred in range(softmax_preds.shape[0]):
        entropy = torch.zeros(*softmax_preds.shape[2:], device=softmax_preds.device)
        for y in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, y] * torch.log(softmax_preds[pred, y])
            nan_pos = torch.isnan(entropy_class)
            entropy[~nan_pos] += entropy_class[~nan_pos]
        entropy *= -1
        expected_entropy[pred] = entropy
    expected_entropy = torch.mean(expected_entropy, dim=0)
    mutual_information = pred_entropy - expected_entropy
    uncertainty_dict["pred_entropy"] = pred_entropy
    uncertainty_dict["aleatoric_uncertainty"] = mutual_information
    uncertainty_dict["epistemic_uncertainty"] = expected_entropy
    return uncertainty_dict


def calculate_one_minus_msr(softmax_pred: torch.Tensor):
    uncertainty_dict = {}
    max_softmax = softmax_pred.max(dim=0)[0]
    uncertainty_dict["pred_entropy"] = 1 - max_softmax
    return uncertainty_dict


def caculcate_uncertainty_multiple_pred(
    test_datacarrier: DataCarrier3D, ssn: bool = False
) -> None:
    for key, value in test_datacarrier.data.items():
        softmax_preds = torch.from_numpy(value["softmax_pred"])
        value.update(calculate_uncertainty(softmax_preds, ssn))
    return


def calculate_metrics(test_datacarrier: DataCarrier3D) -> None:
    """
    Calculate metrics when all slices for all cases have been predicted
    Args:
        test_datacarrier: The datacarrier with the concatenated data
    """
    print("New metrics calculation")
    for key, value in test_datacarrier.data.items():
        mean_softmax_pred = torch.mean(
            torch.from_numpy(
                value["softmax_pred"] / np.clip(value["num_predictions"], 1, None)[0]
            ),
            dim=0,
        )
        mean_softmax_pred = torch.unsqueeze(mean_softmax_pred, 0)
        gt_seg = torch.from_numpy(np.asarray(value["seg"]))
        metrics_dict = calculate_test_metrics(mean_softmax_pred, gt_seg)
        if value["seg"].shape[0] > 1 or value["softmax_pred"].shape[0] > 1:
            gt = np.asarray(
                value["seg"]
                / np.stack(
                    [np.clip(value["num_predictions"], 1, None)[0]]
                    * value["seg"].shape[0]
                ),
                dtype=np.intc,
            )
            softmax_pred = np.asarray(
                value["softmax_pred"]
                / np.stack(
                    [np.clip(value["num_predictions"], 1, None)]
                    * value["softmax_pred"].shape[0]
                )
            )
            ged_dict = calculate_ged(
                torch.from_numpy(softmax_pred),
                torch.from_numpy(gt),
                ignore_index=0,
                additional_metrics=[
                    "dice",
                    "major_dice",
                    "max_dice_pred",
                    "max_dice_gt",
                ],
            )
            metrics_dict.update(ged_dict)
        test_datacarrier.data[key]["metrics"] = metrics_dict


def save_results(
    test_datacarrier: DataCarrier3D, hparams: Dict, args: Namespace
) -> None:
    """
    Save the results of the datacarrier to disc

    Args:
        test_datacarrier: The datacarrier which contains the data to save
        hparams: Dict with hyperparameters of training. Needed to infer the path where to store test results.
        args: Arguments for testing, possibly specifying a data_input_dir and a save_dir
    """
    save_dir = args.save_dir if args.save_dir is not None else hparams["save_dir"]
    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    exp_name = hparams["exp_name"] if args.exp_name is None else args.exp_name
    checkpoint_tag = getattr(args, "checkpoint_subdir", None)

    if "shift_feature" in hparams["datamodule"]:
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=exp_name,
            version=(getattr(args, "version_override", None) or hparams["version"]),
            org_data_path=os.path.join(data_input_dir, "images"),
            test_split=args.test_split,
            checkpoint_tag=checkpoint_tag,
        )
    else:
        if args.test_data_dir is not None:
            org_data_path = None
        else:
            if args.test_split == "val" or args.test_split == "train":
                imagesDir = "imagesTr"
            else:
                imagesDir = "imagesTs"
            org_data_path = os.path.join(
                data_input_dir, hparams["datamodule"]["dataset_name"], imagesDir
            )
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=exp_name,
            version=(getattr(args, "version_override", None) or hparams["version"]),
            org_data_path=org_data_path,
            test_split=args.test_split,
            checkpoint_tag=checkpoint_tag,
        )
    test_datacarrier.log_metrics()


def run_test(args: Namespace) -> None:
    """
    Run test and save the results in the end
    Args:
        args: Arguments for testing, including checkpoint_path, test_data_dir and subject_ids.
              test_data_dir and subject_ids might be None.
    """
    test_data_dir = args.test_data_dir
    subject_ids = args.subject_ids

    all_checkpoints = []
    for checkpoint_path in args.checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        all_checkpoints.append(checkpoint)
    hparams = all_checkpoints[0]["hyper_parameters"]

    set_seed(hparams["seed"])
    checkpoint_epoch = all_checkpoints[0].get("epoch")
    checkpoint_tag = format_checkpoint_subdir(checkpoint_epoch, args.use_ema)
    args.checkpoint_subdir = checkpoint_tag
    # No test data dir specified, so data should be in same input dir as training data and split should be specified
    if test_data_dir is None:
        if "shift_feature" in hparams["datamodule"]:
            test_data_dir, subject_ids = dir_and_subjects_from_train_lidc(
                hparams, args, args.test_split
            )
        else:
            test_data_dir, subject_ids = dir_and_subjects_from_train(hparams, args)

    test_datacarrier = DataCarrier3D()
    if "shift_feature" in hparams["datamodule"]:
        from uncertainty_modeling.lidc_idri_datamodule_3D import (
            get_val_test_data_samples,
        )
    else:
        from uncertainty_modeling.toy_datamodule_3D import get_val_test_data_samples
    if args.test_split == "val" or args.test_split == "train":
        test = False
    else:
        test = True
    data_samples = get_val_test_data_samples(
        base_dir=test_data_dir,
        subject_ids=subject_ids,
        test=test,
        num_raters=hparams["datamodule"]["num_raters"],
        patch_size=hparams["datamodule"]["patch_size"],
        patch_overlap=hparams["datamodule"]["patch_overlap"],
    )

    models = load_models_from_checkpoint(
        all_checkpoints, use_ema=bool(getattr(args, "use_ema", False))
    )
    # Early skip logic: only rely on metrics.json completeness
    if getattr(args, "skip_existing", False) and _metrics_complete_3d(args, hparams):
        print(f"[skip_existing] metrics.json complete for split='{args.test_split}' (version={getattr(args, 'version_override', None) or hparams['version']}, ckpt_tag={checkpoint_tag}). Skipping evaluation.")
        return
    # data_samples = [data_samples[0]]
    ssn = False
    if isinstance(models[0], SsnUNet3D) and len(models) == 1:
        test_datacarrier = predict_cases_ssn(
            test_datacarrier, data_samples, models[0], args.n_pred
        )
        ssn = True
        print(ssn)
    elif "n_aleatoric_samples" in hparams:
        test_datacarrier = predict_cases(
            test_datacarrier,
            data_samples,
            models,
            args.n_pred,
            hparams["n_aleatoric_samples"],
            tta=args.tta,
        )
    else:
        test_datacarrier = predict_cases(
            test_datacarrier, data_samples, models, args.n_pred, tta=args.tta
        )
    if args.n_pred > 1 or len(models) > 1 or args.tta:
        caculcate_uncertainty_multiple_pred(test_datacarrier, ssn)
    calculate_metrics(test_datacarrier)
    save_results(test_datacarrier, hparams, args)


if __name__ == "__main__":
    arguments = test_cli()
    jobs = prepare_evaluation_jobs(arguments)
    if not jobs:
        raise ValueError("No evaluation jobs were generated. Check your checkpoint paths.")
    total_jobs = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        ema_label = getattr(job, "current_ema_label", "normal")
        ckpt_summary = ", ".join(Path(p).name for p in job.checkpoint_paths)
        print(
            f"[{idx}/{total_jobs}] Evaluating split='{job.test_split}' "
            f"ema='{ema_label}' with checkpoints: {ckpt_summary}"
        )
        run_test(job)
