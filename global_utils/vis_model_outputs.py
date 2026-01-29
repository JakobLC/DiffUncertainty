#!/usr/bin/env python3
"""Visualize segmentation outputs for multiple checkpoints in a configurable grid."""

from __future__ import annotations

import copy
import os
import random
import sys
import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.append(PROJECT_ROOT.as_posix())

UNCERTAINTY_ROOT = PROJECT_ROOT / "uncertainty_modeling"
if UNCERTAINTY_ROOT.as_posix() not in sys.path:
    sys.path.append(UNCERTAINTY_ROOT.as_posix())

import uncertainty_modeling.data.cityscapes_labels as cs_labels  # noqa: E402
from uncertainty_modeling.main import set_seed  # noqa: E402
from uncertainty_modeling.test_2D import Tester  # noqa: E402
from uncertainty_modeling.unc_mod_utils.swag import SWAG  # noqa: E402
from uncertainty_modeling.unc_mod_utils.test_utils import (  # noqa: E402
    calculate_one_minus_msr,
    calculate_uncertainty,
    load_models_from_checkpoint,
    prepare_evaluation_jobs,
    test_cli,
)

LIDC_PREFIX = "lidc"
GTA_PREFIX = "gta"
LIDC_PATIENT_AUG_SCHEMA = "lidc_patient_aug_v1"
AUGMENTED_TEST_SPLITS = {"ood_noise", "ood_blur", "ood_contrast", "ood_jpeg"}


@dataclass
class ModelGroup:
    label: str
    checkpoint_path: str
    models: List[torch.nn.Module]
    dataset_name: str


class ModelOutputVisualizer:
    """Create visualization grids for model predictions."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        checkpoint_paths = getattr(args, "checkpoint_paths", None)
        if checkpoint_paths is None:
            raise ValueError("--checkpoint_paths must be provided.")
        self.checkpoint_paths = list(checkpoint_paths)
        if not self.checkpoint_paths:
            raise ValueError("At least one checkpoint path is required.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregate = int(getattr(args, "aggregate", 1))
        if self.aggregate not in {0, 1, 2}:
            raise ValueError("--aggregate must be one of {0,1,2}.")
        self.show = not bool(getattr(args, "no_show", False))
        self.save_path = getattr(args, "save_path", "")
        self.max_images = max(int(getattr(args, "num_images", 1)), 1)
        self.figure_dpi = int(getattr(args, "figure_dpi", 120))
        self.label_size = max(1, int(getattr(args, "label_size", 12)))
        base_wrap = 50
        base_font = 12
        wrap_scale = base_font / self.label_size
        self.label_wrap_limit = max(1, int(round(base_wrap * wrap_scale)))
        requested_uq = getattr(args, "with_uq", None)
        if requested_uq is None:
            self.with_uq = self.aggregate == 1
        else:
            self.with_uq = bool(requested_uq)
        self.n_resample_largest_mask = max(1, int(getattr(args, "n_resample_largest_mask", 1)))
        self.tta = bool(getattr(args, "tta", False))
        self.n_pred = max(int(getattr(args, "n_pred", 1)), 1)
        self.n_models = max(int(getattr(args, "n_models", 0) or 0), 0)
        self.selection_seed = int(getattr(args, "selection_seed", -1))
        self._rng = random.Random(None if self.selection_seed < 0 else self.selection_seed)
        self.palette_cache: Dict[tuple[str, int], np.ndarray] = {}
        self.uq_cmap = plt.get_cmap("viridis")

        print(f"Loading checkpoints: {', '.join(Path(p).name for p in self.checkpoint_paths)}")
        self.all_checkpoints = Tester.get_checkpoints(self.checkpoint_paths)
        if not self.all_checkpoints:
            raise ValueError("Failed to load any checkpoints.")

        self.hparams = [copy.deepcopy(ckpt["hyper_parameters"]) for ckpt in self.all_checkpoints]
        self.dataset_name = self.hparams[0]["data"]["name"]
        self.ignore_index = self.hparams[0]["data"].get("ignore_index", -1)
        self._assert_same_dataset()
        set_seed(self.hparams[0]["seed"])

        dataset_cfg = self.hparams[0].get("data", {}).get("dataset", {})
        splits_path = dataset_cfg.get("splits_path")
        if splits_path is None:
            raise ValueError("Checkpoint hyper-parameters do not define data.dataset.splits_path.")
        (
            self.split_schema,
            self.available_splits,
            self.has_unlabeled_pool,
        ) = Tester._inspect_splits_file(splits_path)
        normalized_split = self._normalize_split_name(args.test_split)
        self._ensure_split_is_supported(args.test_split, normalized_split)
        self.args.test_split = normalized_split

        self.test_dataloader = self._build_dataloader()
        use_ema = bool(getattr(args, "use_ema", False))
        self.base_models = load_models_from_checkpoint(
            self.all_checkpoints,
            device=self.device,
            use_ema=use_ema,
        )
        self.model_groups = self._expand_model_groups()
        self.group_labels = [group.label for group in self.model_groups]
        self._prepare_label_template()
        print(f"Prepared {len(self.model_groups)} checkpoint group(s) with "
              f"{sum(len(group.models) for group in self.model_groups)} instantiated model(s).")

    def _assert_same_dataset(self) -> None:
        reference = self.dataset_name
        for idx, hp in enumerate(self.hparams[1:], start=1):
            candidate = hp["data"]["name"]
            if candidate != reference:
                raise ValueError(
                    f"Checkpoint at index {idx} targets dataset '{candidate}', but '{reference}' was expected."
                )

    def _normalize_split_name(self, split_name: str) -> str:
        available = self.available_splits or set()
        if split_name == "id" and "id_test" in available:
            return "id_test"
        if split_name == "ood" and "ood_test" in available:
            return "ood_test"
        return split_name

    def _ensure_split_is_supported(self, requested_split: str, normalized_split: str) -> None:
        if not requested_split:
            raise ValueError("Test split must be provided.")
        available = self.available_splits or set()

        if requested_split == "unlabeled":
            if self.has_unlabeled_pool:
                return
            raise ValueError(
                "Requested split 'unlabeled' but both id/ood unlabeled pools are empty in the splits file."
            )

        if (
            self.split_schema == LIDC_PATIENT_AUG_SCHEMA
            and requested_split in {"ood", "ood_test"}
            and any(name in available for name in AUGMENTED_TEST_SPLITS)
        ):
            options = ", ".join(sorted(name for name in AUGMENTED_TEST_SPLITS if name in available))
            raise ValueError(
                "This LIDC schema stores OOD variants separately. Use one of: "
                + (options or "<none>")
            )

        if available and normalized_split not in available:
            choices = ", ".join(sorted(available))
            raise ValueError(
                f"Requested split '{requested_split}' is not present in the configured splits file. Available: {choices}"
            )

    def _build_dataloader(self):
        args = self.args
        hparams = copy.deepcopy(self.hparams[0])
        data_input_dir = args.data_input_dir or hparams["data"]["data_input_dir"]
        if args.data_input_dir is not None:
            dataset_section = hparams["data"].get("dataset", {})
            splits_path = dataset_section.get("splits_path")
            if splits_path:
                dataset_section["splits_path"] = splits_path.replace(
                    hparams["data"]["data_input_dir"],
                    args.data_input_dir,
                )
        hparams = Tester.set_n_reference_samples(hparams, args.n_reference_samples)
        if getattr(args, "test_batch_size", None):
            hparams["data"]["val_batch_size"] = args.test_batch_size
        datamodule = hydra.utils.instantiate(
            hparams["data"],
            data_input_dir=data_input_dir,
            augmentations=hparams["data"]["augmentations"],
            seed=hparams["seed"],
            test_split=self.args.test_split,
            tta=self.tta,
            _recursive_=False,
        )
        datamodule.setup("test")
        return datamodule.test_dataloader()

    def _expand_model_groups(self) -> List[ModelGroup]:
        expanded: List[ModelGroup] = []
        for idx, (model, checkpoint, ckpt_path) in enumerate(
            zip(self.base_models, self.all_checkpoints, self.checkpoint_paths)
        ):
            models_for_group: List[torch.nn.Module] = []
            eu_type = getattr(model, "EU_type", None)
            if self.n_models > 0 and eu_type in {"swag", "swag_diag"}:
                if "swag_config" in checkpoint:
                    swag_config = checkpoint["swag_config"]
                else:
                    swag_config = checkpoint["hyper_parameters"].get("swag")
                swag_state = checkpoint.get("swag_state_dict")
                if swag_config is None or swag_state is None:
                    warnings.warn(
                        f"Checkpoint '{ckpt_path}' has EU_type='{eu_type}' but no SWAG statistics; using base model only."
                    )
                    models_for_group.append(model)
                else:
                    models_for_group.extend(
                        self._sample_swag_draws(model, swag_state, swag_config, ckpt_path)
                    )
            elif self.n_models > 0 and eu_type == "dropout":
                models_for_group.extend([model] * self.n_models)
            else:
                models_for_group.append(model)

            label = self._format_label(checkpoint["hyper_parameters"], idx)
            expanded.append(
                ModelGroup(
                    label=label,
                    checkpoint_path=ckpt_path,
                    models=models_for_group,
                    dataset_name=checkpoint["hyper_parameters"]["data"]["name"],
                )
            )
        return expanded

    def _sample_swag_draws(
        self,
        template_model: torch.nn.Module,
        swag_state: Dict[str, torch.Tensor],
        swag_config: Dict,
        checkpoint_label: str,
    ) -> List[torch.nn.Module]:
        if self.n_models <= 0:
            return []
        config = swag_config
        swag = SWAG(
            diag_only=config["diag_only"],
            max_num_models=config["max_snapshots"],
            var_clamp=config["min_variance"],
        )
        swag.prepare(template_model)
        swag.load_state_dict(swag_state)
        requested_low_rank = bool(getattr(self.args, "swag_low_rank_cov", True))
        use_low_rank = requested_low_rank and not config["diag_only"]
        if requested_low_rank and config["diag_only"]:
            warnings.warn(
                f"Checkpoint '{checkpoint_label}' only stores diagonal SWAG stats; falling back to diagonal sampling."
            )
        sampled: List[torch.nn.Module] = []
        for _ in range(self.n_models):
            sampled_model = copy.deepcopy(template_model)
            swag.sample(
                sampled_model,
                scale=1.0,
                use_low_rank=use_low_rank,
                blockwise=bool(getattr(self.args, "swag_blockwise", False)),
            )
            sampled_model.eval()
            sampled.append(sampled_model.to(self.device))
        print(f"[SWAG] Sampled {len(sampled)} draw(s) from '{checkpoint_label}'.")
        return sampled

    def run(self) -> None:
        cases = self._collect_cases()
        if not cases:
            print("No samples were collected from the dataloader; nothing to visualize.")
            return
        for case_idx, case in enumerate(cases):
            fig = self._render_case(case, case_idx)
            if self.save_path:
                target_path = self._resolve_save_path(case_idx, case["image_id"])
                target_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target_path, dpi=self.figure_dpi, bbox_inches="tight")
                print(f"Saved visualization to {target_path}.")
            if self.show:
                plt.show()
            plt.close(fig)

    def _collect_cases(self) -> List[Dict]:
        selected: List[Dict] = []
        candidate_pool: List[Dict] = []
        total_considered = 0
        n_resample = self.n_resample_largest_mask

        def flush_candidates(force: bool = False) -> bool:
            nonlocal total_considered
            if not candidate_pool:
                return False
            if len(candidate_pool) < n_resample and not force:
                return False
            best_case = max(candidate_pool, key=lambda entry: entry.get("label_mass", 0.0))
            candidate_pool.clear()
            total_considered += 1
            if len(selected) < self.max_images:
                selected.append(best_case)
            else:
                swap_idx = self._rng.randint(0, total_considered - 1)
                if swap_idx < self.max_images:
                    selected[swap_idx] = best_case
            return True

        with torch.no_grad():
            stop_early = False
            for batch in self.test_dataloader:
                batch_cases = self._process_batch(batch)
                for case in batch_cases:
                    candidate_pool.append(case)
                    flushed = flush_candidates(force=False)
                    if (
                        flushed
                        and self.selection_seed >= 0
                        and len(selected) >= self.max_images
                    ):
                        stop_early = True
                        break
                if stop_early:
                    break
        flush_candidates(force=True)
        return selected

    def _process_batch(self, batch: Dict) -> List[Dict]:
        gt = batch["seg"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        batch_size = gt.shape[0]
        group_preds = self._run_models(batch)
        cases: List[Dict] = []
        for sample_idx in range(batch_size):
            gt_sample = gt[sample_idx]
            case = {
                "image_id": str(batch["image_id"][sample_idx]),
                "dataset": str(batch["dataset"][sample_idx]),
                "image_rgb": self._tensor_to_rgb(batch["data"][sample_idx]),
                "gt_rgb": self._gt_to_display(gt_sample),
                "group_preds": [],
            }
            case["label_mass"] = self._calculate_label_mass(gt_sample)
            for group_tensor in group_preds:
                case["group_preds"].append(group_tensor[:, :, sample_idx].clone())
            cases.append(case)
            if len(cases) >= self.max_images:
                break
        return cases

    def _run_models(self, batch: Dict) -> List[torch.Tensor]:
        group_tensors: List[torch.Tensor] = []
        for group in self.model_groups:
            model_outputs: List[torch.Tensor] = []
            for model in group.models:
                preds = self._forward_model(model, batch)
                model_outputs.append(preds)
            stacked = torch.stack(model_outputs, dim=0)
            group_tensors.append(stacked.cpu())
        return group_tensors

    def _forward_model(self, model: torch.nn.Module, batch: Dict) -> torch.Tensor:
        au_type = getattr(model, "AU_type", "softmax")
        if au_type == "ssn":
            inputs = batch["data"]
            if isinstance(inputs, list):
                inputs = inputs[0]
            tensor_inputs = inputs.to(self.device)
            distribution, cov_failed_flag = model.forward(tensor_inputs)
            if cov_failed_flag:
                raise RuntimeError("SSN covariance matrix was not positive definite.")
            samples = distribution.sample([self.n_pred])
            samples = samples.view(
                self.n_pred,
                tensor_inputs.size()[0],
                model.num_classes,
                *tensor_inputs.size()[2:],
            )
            softmax_samples = F.softmax(samples, dim=2)
            return softmax_samples
        elif au_type == "diffusion":
            inputs = batch["data"]
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.to(self.device).float()
            sample_list = []
            num_steps = int(getattr(model, "diffusion_num_steps", self.n_pred))
            sampler_type = getattr(model, "diffusion_sampler_type", "ddpm") or "ddpm"
            for _ in range(self.n_pred):
                x_init = torch.randn(
                    (inputs.shape[0], model.num_classes, *inputs.shape[2:]),
                    device=self.device,
                    dtype=inputs.dtype,
                )
                sample_output = model.diffusion_sample_loop(
                    x_init=x_init,
                    im=inputs,
                    num_steps=num_steps,
                    sampler_type=sampler_type,
                    clip_x=False,
                    guidance_weight=0.0,
                    progress_bar=False,
                    self_cond=False,
                )
                sample_list.append(sample_output)
            return torch.stack(sample_list, dim=0)
        elif au_type == "prob_unet":
            inputs = batch["data"]
            if isinstance(inputs, list):
                inputs = inputs[0]
            tensor_inputs = inputs.to(self.device).float()
            model.forward(tensor_inputs, segm=None, training=False)
            logits_stack = model.sample_multiple(self.n_pred, from_prior=True, testing=True)
            return torch.softmax(logits_stack, dim=2)
        elif self.tta:
            preds: List[torch.Tensor] = []
            transforms = batch.get("transforms")
            data_iterable = batch["data"] if isinstance(batch["data"], list) else batch["data"]
            for index, image in enumerate(data_iterable):
                output = model.forward(image.to(self.device))
                output_softmax = F.softmax(output, dim=1)
                flip_applied = False
                if transforms is not None and index < len(transforms):
                    flip_applied = any("HorizontalFlip" in sl for sl in transforms[index])
                if flip_applied:
                    preds.append(torch.flip(output_softmax, dims=[-1]))
                else:
                    preds.append(output_softmax)
            stacked = torch.stack(preds, dim=0)
            return stacked.unsqueeze(0)
        else:
            inputs = batch["data"]
            if isinstance(inputs, list):
                inputs = inputs[0]
            output = model.forward(inputs.to(self.device))
            output_softmax = F.softmax(output, dim=1)
            return output_softmax.unsqueeze(0)

    def _render_case(self, case: Dict, case_index: int):
        if self.aggregate == 0:
            return self._render_full_grid(case, case_index)
        return self._render_aggregated_grid(case, case_index)

    def _render_aggregated_grid(self, case: Dict, case_index: int):
        group_infos = [
            self._prepare_group_display(tensor, self._get_group_display_label(idx))
            for idx, tensor in enumerate(case["group_preds"])
        ]
        blank_cell = self._blank_like(case["image_rgb"])

        max_models = max(info["n_models"] for info in group_infos) if group_infos else 0
        if self.aggregate == 2:
            pred_rows = 1
            row_labels = ["Pred (mean)"]
        else:
            pred_rows = max_models
            row_labels = [f"Model #{idx + 1}" for idx in range(pred_rows)]

        grid_rows: List[List[np.ndarray]] = []
        if self.aggregate == 2:
            row_cells = [info["rows"][0][0] for info in group_infos]
            grid_rows.append(row_cells)
        elif self.aggregate == 1:
            for model_idx in range(pred_rows):
                row_cells: List[np.ndarray] = []
                for info in group_infos:
                    if model_idx < info["n_models"]:
                        row_cells.append(info["rows"][model_idx][0])
                    else:
                        row_cells.append(blank_cell.copy())
                grid_rows.append(row_cells)
        else:  # fallback
            for model_idx in range(pred_rows):
                row_cells: List[np.ndarray] = []
                for info in group_infos:
                    if model_idx < info["n_models"]:
                        row_cells.extend(info["rows"][model_idx])
                    else:
                        row_cells.extend([blank_cell.copy() for _ in range(info["width"])])
                grid_rows.append(row_cells)

        col_labels: List[str] = []
        for info in group_infos:
            if self.aggregate in (1, 2):
                col_labels.append(info["label"])
            else:
                for col_idx in range(info["width"]):
                    if info["width"] == 1:
                        col_labels.append(info["label"])
                    else:
                        col_labels.append(f"{info['label']}·p{col_idx + 1}")

        uq_rows, uq_labels = self._build_uncertainty_rows(group_infos, blank_cell)
        if uq_rows:
            grid_rows.extend(uq_rows)
            row_labels.extend(uq_labels)

        # Image row
        image_row: List[np.ndarray] = []
        gt_row: List[np.ndarray] = []
        for info in group_infos:
            width = 1 if self.aggregate in (1, 2) else info["width"]
            for col_idx in range(width):
                use_blank = self.aggregate == 0 and width > 1 and col_idx > 0
                if use_blank:
                    image_row.append(blank_cell.copy())
                    gt_row.append(blank_cell.copy())
                else:
                    image_row.append(case["image_rgb"].copy())
                    gt_row.append(case["gt_rgb"].copy())
        grid_rows.append(image_row)
        grid_rows.append(gt_row)
        row_labels.extend(["Image", "GT"])

        title = self._format_fig_title(case_index, case["image_id"], aggregate=self.aggregate)
        return self._build_grid_figure(grid_rows, col_labels, row_labels, title)

    def _render_full_grid(self, case: Dict, case_index: int):
        group_infos = [
            self._prepare_group_display(tensor, self._get_group_display_label(idx))
            for idx, tensor in enumerate(case["group_preds"])
        ]
        max_models = max(info["n_models"] for info in group_infos) if group_infos else 0
        row_labels = [f"Model #{idx + 1}" for idx in range(max_models)]

        col_labels: List[str] = []
        for info in group_infos:
            if info["width"] == 1:
                col_labels.append(info["label"])
            else:
                for pred_idx in range(info["width"]):
                    col_labels.append(f"{info['label']}·p{pred_idx + 1}")

        blank_cell = self._blank_like(case["image_rgb"])
        grid_rows: List[List[np.ndarray]] = []
        for model_idx in range(max_models):
            row_cells: List[np.ndarray] = []
            for info in group_infos:
                if model_idx < info["n_models"]:
                    row_cells.extend(info["rows"][model_idx])
                else:
                    row_cells.extend([blank_cell.copy() for _ in range(info["width"])])
            grid_rows.append(row_cells)

        uq_rows, uq_labels = self._build_uncertainty_rows(group_infos, blank_cell)
        if uq_rows:
            grid_rows.extend(uq_rows)
            row_labels.extend(uq_labels)

        image_row: List[np.ndarray] = []
        gt_row: List[np.ndarray] = []
        for info in group_infos:
            for col_idx in range(info["width"]):
                if col_idx == 0:
                    image_row.append(case["image_rgb"].copy())
                    gt_row.append(case["gt_rgb"].copy())
                else:
                    image_row.append(blank_cell.copy())
                    gt_row.append(blank_cell.copy())
        grid_rows.append(image_row)
        grid_rows.append(gt_row)
        row_labels.extend(["Image", "GT"])

        title = self._format_fig_title(case_index, case["image_id"], aggregate=0)
        return self._build_grid_figure(grid_rows, col_labels, row_labels, title)

    def _tensor_to_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().float()
        if arr.ndim == 4:
            arr = arr.squeeze(0)
        if arr.shape[0] > 3:
            arr = arr[:3]
        if arr.shape[0] == 1:
            arr = arr.repeat(3, 1, 1)
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        denom = arr_max - arr_min if arr_max > arr_min else 1.0
        arr = (arr - arr_min) / denom
        return arr.permute(1, 2, 0).numpy()

    def _gt_to_display(self, gt_tensor: torch.Tensor) -> np.ndarray:
        tensor = gt_tensor.clone()
        if tensor.ndim == 3:
            majority = torch.mode(tensor, dim=0).values
        else:
            majority = tensor
        if self.ignore_index is not None:
            majority = majority.masked_fill(majority == self.ignore_index, 0)
        if self.dataset_name.lower().startswith(LIDC_PREFIX):
            if tensor.ndim == 3:
                mask = (tensor == 1).float().mean(dim=0)
            else:
                mask = (tensor == 1).float()
            mask = mask.clamp(0, 1).cpu().numpy()
            return np.repeat(mask[..., None], 3, axis=2)
        palette = self._get_palette(int(majority.max().item() + 1 or 1))
        labels = majority.cpu().numpy().astype(int)
        labels = np.clip(labels, 0, palette.shape[0] - 1)
        rgb = palette[labels]
        return rgb

    def _calculate_label_mass(self, gt_tensor: torch.Tensor) -> float:
        tensor = gt_tensor.clone()
        if self.ignore_index is not None:
            tensor = tensor.masked_fill(tensor == self.ignore_index, 0)
        return float(tensor.float().sum().item())

    def _prob_to_rgb(self, prob_tensor: torch.Tensor) -> np.ndarray:
        prob = prob_tensor.detach().cpu()
        clipped = prob.clamp(0.0, 1.0)
        if torch.any(clipped != prob):
            warnings.warn("Clipping probabilities outside [0,1] for visualization.")
        prob = clipped
        num_classes = prob.shape[0]
        if self.dataset_name.lower().startswith(LIDC_PREFIX):
            if num_classes == 1:
                gray = prob.squeeze(0).numpy()
            else:
                gray = prob[1].numpy()
            gray = np.clip(gray, 0.0, 1.0)
            return np.repeat(gray[..., None], 3, axis=2)
        palette = self._get_palette(num_classes)
        prob_np = prob.numpy().transpose(1, 2, 0)
        rgb = np.tensordot(prob_np, palette, axes=(2, 0))
        return np.clip(rgb, 0.0, 1.0)

    def _heatmap_to_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr_min = float(arr.min()) if arr.size else 0.0
        arr_max = float(arr.max()) if arr.size else 0.0
        denom = arr_max - arr_min
        if denom > 0:
            norm = (arr - arr_min) / denom
        else:
            norm = np.zeros_like(arr)
        rgb = self.uq_cmap(norm)[..., :3]
        return rgb

    def _compute_uncertainty_maps(self, per_model_probs: torch.Tensor) -> Dict[str, np.ndarray]:
        if per_model_probs.numel() == 0:
            return {}
        n_models = per_model_probs.shape[0]
        if n_models == 0:
            return {}
        if n_models == 1:
            uq_tensors = calculate_one_minus_msr(per_model_probs[0])
        else:
            uq_tensors = calculate_uncertainty(per_model_probs)
        label_map = [
            ("aleatoric_uncertainty", "AU"),
            ("epistemic_uncertainty", "EU"),
            ("pred_entropy", "TU"),
        ]
        outputs: Dict[str, np.ndarray] = {}
        for key, label in label_map:
            tensor = uq_tensors.get(key)
            if tensor is None:
                continue
            outputs[label] = self._heatmap_to_rgb(tensor)
        return outputs

    def _get_palette(self, num_classes: int) -> np.ndarray:
        key = (self.dataset_name.lower(), num_classes)
        if key in self.palette_cache:
            return self.palette_cache[key]
        if self.dataset_name.lower().startswith(GTA_PREFIX):
            palette = np.zeros((num_classes, 3), dtype=np.float32)
            for cls_idx in range(num_classes):
                color = cs_labels.trainId2color.get(cls_idx, (255, 255, 255))
                palette[cls_idx] = np.array(color, dtype=np.float32) / 255.0
        else:
            cmap = plt.get_cmap("viridis", num_classes)
            palette = cmap(np.arange(num_classes))[:, :3]
        self.palette_cache[key] = palette
        return palette

    def _blank_like(self, reference: np.ndarray) -> np.ndarray:
        return np.zeros_like(reference)

    def _build_uncertainty_rows(
        self,
        group_infos: List[Dict],
        blank_cell: np.ndarray,
    ) -> tuple[List[List[np.ndarray]], List[str]]:
        if not self.with_uq or not group_infos:
            return [], []
        order = ["AU", "EU", "TU"]
        rows: List[List[np.ndarray]] = []
        labels: List[str] = []
        for label in order:
            row_cells: List[np.ndarray] = []
            has_value = False
            for info in group_infos:
                width = 1 if self.aggregate in (1, 2) else info["width"]
                cell_img = (info.get("uncertainty") or {}).get(label)
                if cell_img is not None:
                    has_value = True
                    base_cell = cell_img
                else:
                    base_cell = blank_cell.copy()
                if self.aggregate == 0 and width > 1:
                    row_cells.append(base_cell)
                    row_cells.extend([blank_cell.copy() for _ in range(width - 1)])
                else:
                    row_cells.append(base_cell)
            if has_value:
                rows.append(row_cells)
                labels.append(label)
        return rows, labels

    def _strip_axes(self, axes: np.ndarray) -> None:
        for axis_row in axes:
            for ax in axis_row:
                ax.set_xticks([])
                ax.set_yticks([])

    def _ensure_axes(self, axes, rows: int, cols: int) -> np.ndarray:
        if rows == 1 and cols == 1:
            return np.array([[axes]])
        if rows == 1:
            return np.array([axes])
        if cols == 1:
            return np.array([[ax] for ax in axes])
        return axes

    def _figure_size(self, rows: int, cols: int) -> tuple[float, float]:
        width = max(4.0, cols * 2.0)
        height = max(4.0, rows * 2.0)
        return (width, height)

    def _build_grid_figure(
        self,
        grid_rows: List[List[np.ndarray]],
        col_labels: List[str],
        row_labels: List[str],
        title: str,
    ):
        if not grid_rows or not grid_rows[0]:
            raise ValueError("Cannot render an empty grid.")
        n_rows = len(grid_rows)
        n_cols = len(grid_rows[0])
        cell_h, cell_w = grid_rows[0][0].shape[:2]
        row_images = [np.concatenate(row, axis=1) for row in grid_rows]
        composite = np.concatenate(row_images, axis=0)

        fig, ax = plt.subplots(figsize=self._figure_size(n_rows, n_cols), dpi=self.figure_dpi)
        ax.imshow(composite)
        width_total = n_cols * cell_w
        height_total = n_rows * cell_h
        x_positions = (np.arange(n_cols) + 0.5) * cell_w
        y_positions = (np.arange(n_rows) + 0.5) * cell_h
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self._wrap_label_text(lbl) for lbl in col_labels], fontsize=self.label_size)
        ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([self._wrap_label_text(lbl) for lbl in row_labels], fontsize=self.label_size)
        ax.tick_params(axis="y", left=False, right=False)
        ax.set_xlim(0, width_total)
        ax.set_ylim(height_total, 0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig

    def _prepare_label_template(self) -> None:
        labels = self.group_labels or []
        if not labels:
            self._shared_prefix = ""
            self._shared_suffix = ""
            self._group_unique_labels = []
            self._label_template = None
            return

        def _common_prefix(strings: List[str]) -> str:
            prefix = strings[0]
            for s in strings[1:]:
                while not s.startswith(prefix) and prefix:
                    prefix = prefix[:-1]
                if not prefix:
                    break
            return prefix

        def _common_suffix(strings: List[str]) -> str:
            reversed_strings = [s[::-1] for s in strings]
            suffix_rev = _common_prefix(reversed_strings)
            return suffix_rev[::-1]

        prefix = _common_prefix(labels)
        suffix = _common_suffix(labels) if prefix != "" or len(set(labels)) > 1 else ""

        # Avoid suffix covering entire string
        max_suffix = min(len(min(labels, key=len)), len(suffix))
        suffix = suffix[:max_suffix]
        # Ensure prefix and suffix do not overlap fully
        while prefix and suffix and any(len(label) < len(prefix) + len(suffix) for label in labels):
            suffix = suffix[1:]

        unique_parts: List[str] = []
        for label in labels:
            core_start = len(prefix)
            core_end = len(label) - len(suffix) if suffix else len(label)
            if core_end < core_start:
                core_end = core_start
            core = label[core_start:core_end]
            unique_parts.append(core)

        has_template = bool(prefix or suffix)
        self._shared_prefix = prefix if has_template else ""
        self._shared_suffix = suffix if has_template else ""
        self._group_unique_labels = unique_parts if has_template else labels
        self._label_template = (
            f"{self._shared_prefix}[model]{self._shared_suffix}" if has_template else None
        )

    def _get_group_display_label(self, idx: int) -> str:
        if self._label_template:
            core = self._group_unique_labels[idx]
            return core if core else "-"
        return self.group_labels[idx]

    def _format_fig_title(self, case_index: int, image_id: str, aggregate: int) -> str:
        base = f"Case {case_index + 1}: {image_id} (aggregate={aggregate})"
        if self._label_template:
            return f"{base}\nLabel template: {self._label_template}"
        return base

    def _prepare_group_display(self, tensor: torch.Tensor, label: str) -> Dict:
        n_models = tensor.shape[0]
        n_pred = tensor.shape[1]
        per_model = tensor.mean(dim=1)
        info: Dict = {
            "n_models": n_models,
            "n_pred": n_pred,
            "width": 1 if self.aggregate in (1, 2) else n_pred,
            "label": label,
        }
        # label will be injected by caller to maintain sequencing
        if self.aggregate == 2:
            mean_tensor = per_model.mean(dim=0)
            info["rows"] = [[self._prob_to_rgb(mean_tensor)]]
        elif self.aggregate == 1:
            rows: List[List[np.ndarray]] = []
            for model_idx in range(n_models):
                rows.append([self._prob_to_rgb(per_model[model_idx])])
            info["rows"] = rows
        else:
            rows = []
            for model_idx in range(n_models):
                cols = [self._prob_to_rgb(tensor[model_idx, pred_idx]) for pred_idx in range(n_pred)]
                rows.append(cols)
            info["rows"] = rows
        info["uncertainty"] = self._compute_uncertainty_maps(per_model) if self.with_uq else {}
        return info

    def _wrap_label_text(self, text: str) -> str:
        if not text:
            return text
        limit = self.label_wrap_limit
        if len(text) <= limit:
            return text
        wrapped: List[str] = []
        for segment in str(text).split("\n"):
            if len(segment) <= limit:
                wrapped.append(segment)
                continue
            for idx in range(0, len(segment), limit):
                wrapped.append(segment[idx : idx + limit])
        return "\n".join(wrapped)

    def _format_label(self, hparams: Dict, idx: int) -> str:
        exp_override = getattr(self.args, "exp_name", None)
        exp_name = exp_override if exp_override is not None else hparams.get("exp_name", f"exp_{idx}")
        version = hparams.get("version", "version")
        return f"{exp_name}/{version}"

    def _resolve_save_path(self, case_index: int, image_id: str) -> Path:
        sanitized_id = image_id.replace(os.sep, "_")
        raw_path = Path(self.save_path)
        treat_as_dir = raw_path.suffix == "" or self.save_path.endswith(os.sep)
        if treat_as_dir:
            return raw_path / f"{case_index:03d}_{sanitized_id}.png"
        if self.max_images > 1:
            return raw_path.with_name(f"{raw_path.stem}_{case_index:03d}_{sanitized_id}{raw_path.suffix}")
        return raw_path


def _freeze_value(value):
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, Namespace):
        return tuple(sorted((k, _freeze_value(v)) for k, v in vars(value).items()))
    return value


def _combine_jobs_for_visualization(jobs: List[Namespace]) -> List[Namespace]:
    grouped: Dict[tuple, List[Namespace]] = {}
    for job in jobs:
        key_items = []
        for name, value in sorted(vars(job).items()):
            if name in {"checkpoint_paths", "version_override"}:
                continue
            key_items.append((name, _freeze_value(value)))
        key = tuple(key_items)
        grouped.setdefault(key, []).append(job)

    combined_jobs: List[Namespace] = []
    for job_list in grouped.values():
        if not job_list:
            continue
        base = copy.deepcopy(job_list[0])
        merged_paths: List[str] = []
        for job in job_list:
            for path in job.checkpoint_paths:
                if path not in merged_paths:
                    merged_paths.append(path)
        base.checkpoint_paths = merged_paths
        combined_jobs.append(base)
    return combined_jobs


def add_visualization_args(parser) -> None:
    parser.add_argument(
        "--aggregate",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="0: raw model x sample grid, 1: mean across EU members, 2: mean across EU members and draws.",
    )
    uq_group = parser.add_mutually_exclusive_group()
    uq_group.set_defaults(with_uq=None)
    uq_group.add_argument(
        "--with_uq",
        dest="with_uq",
        action="store_true",
        help="Include AU/EU/TU uncertainty rows (default when --aggregate=1).",
    )
    uq_group.add_argument(
        "--without_uq",
        dest="with_uq",
        action="store_false",
        help="Hide AU/EU/TU uncertainty rows (default otherwise).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of test images to visualize.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional output path (file or directory). Empty string disables saving.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable plt.show(). Useful when only saving figures.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=120,
        help="DPI for matplotlib figures.",
    )
    parser.add_argument(
        "--label_size",
        type=int,
        default=15,
        help="Font size for axis labels (wrap limit scales inversely).",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=-1,
        help="Seed controlling which images are visualized (-1 picks different images each run).",
    )
    parser.add_argument(
        "--n_resample_largest_mask",
        type=int,
        default=1,
        help="Draw this many random candidates per slot and keep the case with the largest summed GT mask.",
    )


def main() -> None:
    args = test_cli(extra_args_fn=add_visualization_args)
    jobs = prepare_evaluation_jobs(args)
    jobs = _combine_jobs_for_visualization(jobs)
    if not jobs:
        raise ValueError("No visualization jobs were created. Check the provided checkpoint paths.")
    torch.set_grad_enabled(False)
    total_jobs = len(jobs)
    for job_idx, job in enumerate(jobs, start=1):
        ckpt_summary = ", ".join(Path(p).name for p in job.checkpoint_paths)
        print(f"[{job_idx}/{total_jobs}] Visualizing checkpoints: {ckpt_summary}")
        visualizer = ModelOutputVisualizer(job)
        visualizer.run()


if __name__ == "__main__":
    main()
# 
# 6 softmax models
# python global_utils/vis_model_outputs.py --checkpoint_paths /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_softmax_sgd_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_softmax_sgd_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_softmax_sgd_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_softmax_standard_lidc_2d_small/checkpoints/last.ckpt
# 9 models: swag, swag_diag, dropout for softmax, ssn, diffusion
# python global_utils/vis_model_outputs.py --checkpoint_paths /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_ssn_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diffusion_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_ssn_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_diffusion_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_ssn_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_diffusion_standard_lidc_2d_small/checkpoints/last.ckpt  --test_split ood --n_models 4 --n_pred 5 --aggregate 1 --n_resample_largest_mask 10
# 3 softmax models (only standard training)
# python global_utils/vis_model_outputs.py --checkpoint_paths /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_softmax_standard_lidc_2d_small/checkpoints/last.ckpt   
# --test_split ood --label_size 30 --aggregate 1

# python global_utils/vis_model_outputs.py --checkpoint_paths /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/swag_diag_softmax_standard_lidc_2d_small/checkpoints/last.ckpt /home/jloch/Desktop/diff/luzern/values/saves/eu_tests/dropout_softmax_standard_lidc_2d_small/checkpoints/last.ckpt --test_split ood --n_models 4 --n_pred 5 --aggregate 1 --n_resample_largest_mask 10