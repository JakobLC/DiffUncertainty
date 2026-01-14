import copy
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import yaml

from evaluation.metrics.dice_wrapped import dice


def test_cli(
    config_file: Optional[str] = None,
    extra_args_fn: Optional[Callable[[ArgumentParser], None]] = None,
) -> Namespace:
    """Common CLI used by 2D testing utilities.

    Args:
        config_file: Optional path to a YAML file with default argument values.
        extra_args_fn: Optional callback that can register additional CLI arguments
            on the parser before parsing occurs (used by specialized scripts).
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        help="One or more checkpoint paths to evaluate. Accepts directories or concrete files.",
    )
    parser.add_argument(
        "-i",
        "--data_input_dir",
        type=str,
        default=None,
        help="Dataset root directory override. Defaults to the dir encoded in the checkpoint.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional override for save directory.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional override for experiment name used when writing outputs.",
    )
    parser.add_argument(
        "--version_name",
        type=str,
        default=None,
        help="Optional override for the version directory used when writing outputs.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Optional override for the raw test data directory.",
    )
    parser.add_argument(
        "--subject_ids",
        type=list,
        default=None,
        help="Optional subset of subject ids to evaluate.",
    )
    parser.add_argument(
        "--n_pred",
        type=int,
        default=10,
        help="Number of predictions to sample per model (for generative AU models).",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=10,
        help="Number of models to sample (for EU models (SWAG, MC Dropout, etc.)).",
    )
    parser.add_argument(
        "--swag-blockwise",
        action="store_true",
        default=False,
        help="Use blockwise SWAG sampling when generating ensemble members.",
    )
    parser.add_argument(
        "--swag-low-rank-cov",
        dest="swag_low_rank_cov",
        action="store_true",
        default=True,
        help="Include the low-rank covariance component during SWAG sampling (ignored for diag-only stats).",
    )
    parser.add_argument(
        "--no-swag-low-rank-cov",
        dest="swag_low_rank_cov",
        action="store_false",
        help="Disable the low-rank covariance component even when checkpoints provide it.",
    )
    parser.add_argument(
        "--n_reference_samples",
        type=int,
        default=5,
        help="Number of reference samples for augmentation strategies.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        nargs="?",
        help="Override the evaluation batch size.",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="id",
        help="Comma-separated list of dataset splits to evaluate.",
    )
    parser.add_argument(
        "--test_time_augmentations",
        "-tta",
        dest="tta",
        action="store_true",
    )
    parser.add_argument(
        "--ema_mode",
        type=str,
        default="normal",
        choices=["normal", "ema", "both"],
        help="Select which checkpoint weights to load (normal, ema, or both).",
    )
    parser.add_argument(
        "--wildcard_replace",
        type=str,
        default=None,
        help=(
            "Comma-separated replacement strings for '*' in checkpoint paths. "
            "Example: --wildcard_replace=120,121 and --checkpoint_paths=/path/aug0_s*/ckpt.ckpt"
        ),
    )
    parser.add_argument(
        "--ensemble_mode",
        action="store_true",
        default=False,
        help="Treat the provided checkpoints as ensembles rather than independent runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Override the random seed used for SWAG sampling and dataloaders (set >=0 to enable).",
    )
    parser.add_argument(
        "--skip_ged",
        action="store_true",
        default=False,
        help="Skip GED metric computation (useful for deterministic single-rater datasets).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help=(
            "Skip evaluation when metrics.json already contains a mean entry for the job (best-effort check)."
        ),
    )

    if extra_args_fn is not None:
        extra_args_fn(parser)

    if config_file is not None:
        config_path = os.path.join(os.path.dirname(__file__), "..", config_file)
        config_path = os.path.abspath(config_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
        parser.set_defaults(**config)

    return parser.parse_args()


def _resolve_checkpoint_entry(
    entry: str,
    wildcard_replacements: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    files: List[str] = []
    folders: List[str] = []
    if "*" in entry:
        tokens = [tok.strip() for tok in (wildcard_replacements or []) if tok.strip()]
        if not tokens:
            raise ValueError(
                "checkpoint_paths contains '*' but --wildcard_replace was not provided or empty."
            )
        for token in tokens:
            candidate = entry.replace("*", token)
            path = Path(candidate)
            if path.is_dir():
                folders.append(path.as_posix())
            else:
                files.append(path.as_posix())
    else:
        path = Path(entry)
        if path.is_dir():
            folders.append(path.as_posix())
        else:
            files.append(path.as_posix())
    return files, folders


def _build_checkpoint_groups(
    entries: List[str],
    wildcard_replacements: Optional[List[str]] = None,
    ensemble_mode: bool = False,
) -> List[List[str]]:
    if not entries:
        raise ValueError("No checkpoint paths provided.")

    all_files: List[str] = []
    folder_to_files: OrderedDict[str, List[str]] = OrderedDict()

    for raw in entries:
        files, folders = _resolve_checkpoint_entry(raw, wildcard_replacements)
        all_files.extend(files)
        for folder in folders:
            ckpts = sorted(Path(folder).glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt files found in directory: {folder}")
            folder_to_files[folder] = [c.as_posix() for c in ckpts]

    if not ensemble_mode:
        union: List[str] = list(sorted(set(all_files)))
        for files in folder_to_files.values():
            union.extend(files)
        union = list(OrderedDict((p, None) for p in union).keys())
        return [[p] for p in union]

    has_files = len(all_files) > 0
    has_folders = len(folder_to_files) > 0
    if has_files and has_folders:
        raise ValueError("--ensemble_mode does not support mixing files and folders.")

    if has_folders:
        if len(folder_to_files) < 2:
            raise ValueError("--ensemble_mode with folders requires at least two folders.")
        basename_sets = [set(Path(p).name for p in files) for files in folder_to_files.values()]
        common_basenames = set.intersection(*basename_sets) if basename_sets else set()
        if not common_basenames:
            raise ValueError("No common checkpoint filenames across folders for ensembling.")
        groups: List[List[str]] = []
        for base in sorted(common_basenames):
            group = [(Path(folder) / base).as_posix() for folder in folder_to_files.keys()]
            groups.append(group)
        return groups

    unique_files = list(sorted(set(all_files)))
    if len(unique_files) < 2:
        raise ValueError("--ensemble_mode with files requires at least two files.")
    return [unique_files]


def _parse_test_splits(split_arg: Optional[str]) -> List[str]:
    if split_arg is None:
        return ["id"]
    splits = [chunk.strip() for chunk in str(split_arg).split(",") if chunk.strip()]
    return splits or ["id"]


def _ema_mode_to_flags(mode: str) -> List[Tuple[str, bool]]:
    normalized = (mode or "normal").lower()
    if normalized == "normal":
        return [("normal", False)]
    if normalized == "ema":
        return [("ema", True)]
    if normalized == "both":
        return [("ema", True), ("normal", False)]
    raise ValueError(f"Unsupported ema_mode '{mode}'.")


def _extract_version_from_ckpt_path(path_str: str) -> str:
    path = Path(path_str)
    if path.suffix == ".ckpt":
        parent = path.parent
        if parent.name in ("checkpoints", "scheduled_ckpts") and parent.parent:
            return parent.parent.name
        return parent.name
    return path.name


def _build_group_version_name(
    raw_entries: List[str],
    replacements: Optional[List[str]],
    group_paths: List[str],
) -> Optional[str]:
    if len(group_paths) <= 1:
        return None

    versions = [_extract_version_from_ckpt_path(p) for p in group_paths]

    if replacements:
        tokens = [t for t in replacements if t]
        common_prefix: Optional[str] = None
        common_suffix: Optional[str] = None
        matched_tokens: List[str] = []
        success = True
        for version in versions:
            found = False
            for token in tokens:
                idx = version.find(token)
                if idx != -1:
                    prefix = version[:idx]
                    suffix = version[idx + len(token) :]
                    if common_prefix is None and common_suffix is None:
                        common_prefix, common_suffix = prefix, suffix
                        matched_tokens.append(token)
                        found = True
                        break
                    if prefix == common_prefix and suffix == common_suffix:
                        matched_tokens.append(token)
                        found = True
                        break
            if not found:
                success = False
                break
        if success and common_prefix is not None:
            ordered_unique: List[str] = []
            seen = set()
            for token in tokens:
                if token in matched_tokens and token not in seen:
                    ordered_unique.append(token)
                    seen.add(token)
            candidate = f"{common_prefix}[{','.join(ordered_unique)}]{common_suffix}"
            if len(candidate) <= 50:
                return candidate
            short = ordered_unique[:2]
            return f"{common_prefix}[{','.join(short)},etc]{common_suffix}"

    if any("*" in entry for entry in raw_entries):
        def _lcp(strings: List[str]) -> str:
            if not strings:
                return ""
            s1, s2 = min(strings), max(strings)
            for idx, (c1, c2) in enumerate(zip(s1, s2)):
                if c1 != c2:
                    return s1[:idx]
            return s1[: min(len(s1), len(s2))]

        def _lcs(strings: List[str]) -> str:
            reversed_strings = [s[::-1] for s in strings]
            pref = _lcp(reversed_strings)
            return pref[::-1]

        lcp = _lcp(versions)
        lcs = _lcs(versions)
        tokens = []
        for version in versions:
            start = len(lcp)
            end = len(version) - len(lcs) if lcs else len(version)
            tokens.append(version[start:end])
        if any(tokens):
            uniq = []
            seen_tokens = set()
            for token in tokens:
                if token not in seen_tokens:
                    seen_tokens.add(token)
                    uniq.append(token)
            candidate = f"{lcp}[{','.join(uniq)}]{lcs}"
            if len(candidate) <= 50:
                return candidate
            short_uniq = uniq[:2]
            return f"{lcp}[{','.join(short_uniq)},etc]{lcs}"

    uniq_versions: List[str] = []
    seen_versions = set()
    for version in versions:
        if version not in seen_versions:
            seen_versions.add(version)
            uniq_versions.append(version)
    name = f"[{','.join(uniq_versions)}]"
    if len(name) <= 50:
        return name
    short = uniq_versions[:2]
    return f"[{','.join(short)},etc]"


def prepare_evaluation_jobs(args: Namespace) -> List[Namespace]:
    raw_paths = args.checkpoint_paths
    if raw_paths is None:
        raise ValueError("--checkpoint_paths must be provided.")
    raw_list = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)

    wildcard_replacements: Optional[List[str]] = None
    if getattr(args, "wildcard_replace", None):
        wildcard_replacements = [s.strip() for s in str(args.wildcard_replace).split(",") if s.strip()]

    checkpoint_combos = _build_checkpoint_groups(
        raw_list, wildcard_replacements, bool(getattr(args, "ensemble_mode", False))
    )
    test_splits = _parse_test_splits(args.test_split)
    ema_options = _ema_mode_to_flags(getattr(args, "ema_mode", "normal"))

    total_jobs = len(checkpoint_combos) * len(test_splits) * len(ema_options)
    print(
        f"About to launch {total_jobs} evaluation jobs. len(checkpoint_combos)={len(checkpoint_combos)}, "
        f"len(test_splits)={len(test_splits)}, len(ema_options)={len(ema_options)}"
    )

    jobs: List[Namespace] = []
    for checkpoints in checkpoint_combos:
        for split in test_splits:
            for ema_label, use_ema in ema_options:
                job_args = copy.deepcopy(args)
                job_args.checkpoint_paths = list(checkpoints)
                job_args.test_split = split
                job_args.use_ema = use_ema
                job_args.current_ema_label = ema_label
                auto_version = _build_group_version_name(
                    raw_list, wildcard_replacements, job_args.checkpoint_paths
                )
                job_args.version_override = getattr(args, "version_name", None) or auto_version
                jobs.append(job_args)
    return jobs


def load_models_from_checkpoint(
    checkpoints: List[Dict],
    device: str = "cpu",
    use_ema: bool = False,
) -> List[nn.Module]:
    all_models: List[nn.Module] = []
    for checkpoint in checkpoints:
        hparams = checkpoint["hyper_parameters"]
        state_dict = OrderedDict()
        if use_ema:
            if "ema_state_dict" in checkpoint:
                source_items = checkpoint["ema_state_dict"].items()
                cleaned = []
                for key, value in source_items:
                    if key == "n_averaged":
                        continue
                    stripped = key.split("module.", 1)[1] if key.startswith("module.") else key
                    cleaned.append((stripped, value))
            else:
                source_items = [
                    (k, v)
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("ema_model.")
                ]
                if not source_items:
                    raise ValueError("EMA weights requested but checkpoint does not contain an ema_model.")
                cleaned = []
                for key, value in source_items:
                    stripped = key.split("ema_model.", 1)[1]
                    stripped = stripped.split("module.", 1)[1] if stripped.startswith("module.") else stripped
                    if stripped == "n_averaged":
                        continue
                    cleaned.append((stripped, value))
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
            model = hydra.utils.instantiate(hparams["model"], aleatoric_loss=hparams["aleatoric_loss"])
        else:
            model = hydra.utils.instantiate(hparams["model"])
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        all_models.append(model.to(device))
    return all_models


def calculate_ged(
    output_softmax: torch.Tensor,
    ground_truth: torch.Tensor,
    ignore_index: int = 0,
    additional_metrics: Optional[List[str]] = None,
) -> Dict[str, float | torch.Tensor]:
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

    if "dice" in additional_metrics:
        results["dice"] = dice_matrix.mean().item()
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
        major_score = dice(
            majority_pred,
            majority_gt,
            num_classes=num_classes,
            ignore_index=ignore_index,
            binary_dice=num_classes == 2,
        )
        results["major_dice"] = float(major_score)
    if "dice_matrix" in additional_metrics:
        results["dice_matrix"] = dice_matrix.cpu()

    return results


def calculate_uncertainty(softmax_preds: torch.Tensor) -> Dict[str, torch.Tensor]:
    uncertainty_dict: Dict[str, torch.Tensor] = {}
    mean_softmax = torch.mean(softmax_preds, dim=0)
    pred_entropy = torch.zeros(*softmax_preds.shape[2:], device=mean_softmax.device)
    for cls_idx in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[cls_idx] * torch.log(mean_softmax[cls_idx])
        nan_pos = torch.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1

    expected_entropy = torch.zeros(
        softmax_preds.shape[0], *softmax_preds.shape[2:], device=softmax_preds.device
    )
    for pred in range(softmax_preds.shape[0]):
        entropy = torch.zeros(*softmax_preds.shape[2:], device=softmax_preds.device)
        for cls_idx in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, cls_idx] * torch.log(softmax_preds[pred, cls_idx])
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


def calculate_one_minus_msr(softmax_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    max_softmax = softmax_pred.max(dim=0)[0]
    return {"pred_entropy": 1 - max_softmax}


__all__ = [
    "calculate_ged",
    "calculate_one_minus_msr",
    "calculate_uncertainty",
    "load_models_from_checkpoint",
    "prepare_evaluation_jobs",
    "test_cli",
]
