#!/usr/bin/env python3
"""Visualization utility that assembles fixed prediction grids for qualitative review."""

from __future__ import annotations

import copy
import os
import random
import sys
import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.colors as mcolors
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
from evaluation.metrics.dice_wrapped import dice  # noqa: E402
from evaluation.metrics.ged_fast import ged_binary_fast  # noqa: E402
from uncertainty_modeling.main import set_seed  # noqa: E402
from uncertainty_modeling.test_2D import Tester  # noqa: E402
from uncertainty_modeling.unc_mod_utils.swag import SWAG  # noqa: E402
from uncertainty_modeling.unc_mod_utils.test_utils import (  # noqa: E402
    calculate_ged,
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


@dataclass
class CaseRow:
    image_id: str
    image_cell: np.ndarray
    gt_sum_cell: np.ndarray
    pred_sum_cell: np.ndarray
    prediction_cells: List[np.ndarray]
    metrics: Dict[str, Optional[float]]
    predictions_tensor: torch.Tensor
    ground_truth_tensor: torch.Tensor
    prediction_sources: List[str]
    label_mass: float


class ModelOutputVisualizer:
    """Render grids where each row is an image and columns cover derived outputs."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        checkpoint_paths = getattr(args, "checkpoint_paths", None)
        if not checkpoint_paths:
            raise ValueError("--checkpoint_paths must be provided.")
        self.checkpoint_paths = list(checkpoint_paths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.show = not bool(getattr(args, "no_show", False))
        self.save_path = getattr(args, "save_path", "")
        self.max_images = max(int(getattr(args, "num_images", 4)), 1)
        raw_num_preds = getattr(args, "num_preds", None)
        fallback_preds = getattr(args, "n_pred", 4)
        self.num_predictions = max(int(raw_num_preds if raw_num_preds is not None else fallback_preds), 1)
        self.figure_dpi = int(getattr(args, "figure_dpi", 120))
        self.label_size = max(1, int(getattr(args, "label_size", 12)))
        base_wrap = 50
        self.label_wrap_limit = max(1, int(round(base_wrap * (12 / self.label_size))))
        self.pred_threshold = float(getattr(args, "pred_threshold", 0.5))
        self.row_mode = str(getattr(args, "row_mode", "images")).lower()
        if self.row_mode != "images":
            raise ValueError(
                "Only row_mode='images' is supported right now. Future modes will reuse the same core pipeline."
            )
        self.n_resample_largest_mask = max(1, int(getattr(args, "n_resample_largest_mask", 1)))
        self.tta = bool(getattr(args, "tta", False))
        if self.tta:
            raise NotImplementedError("vis_model_outputs_new.py currently does not support --tta.")
        self.n_models = max(int(getattr(args, "n_models", 0) or 0), 0)
        self.selection_seed = int(getattr(args, "selection_seed", -1))
        self._rng = random.Random(None if self.selection_seed < 0 else self.selection_seed)
        self.palette_cache: Dict[tuple[str, int], np.ndarray] = {}
        self.sum_cmap = plt.get_cmap("viridis")
        self.overlay_alpha = 0.3

        print(f"Loading checkpoints: {', '.join(Path(p).name for p in self.checkpoint_paths)}")
        self.all_checkpoints = Tester.get_checkpoints(self.checkpoint_paths)
        if not self.all_checkpoints:
            raise ValueError("Failed to load any checkpoints.")

        self.hparams = [copy.deepcopy(ckpt["hyper_parameters"]) for ckpt in self.all_checkpoints]

        self.dataset_name = self.hparams[0]["data"]["name"]
        self._dataset_lower = str(self.dataset_name or "").lower()
        self._is_chaksu = "chaksu" in self._dataset_lower

        model_section = self.hparams[0].get("model", {}) if self.hparams else {}
        self._num_model_classes = int(model_section.get("num_classes", 0) or 0)
        self._chaksu_palette: Optional[np.ndarray] = None
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
        for model in self.base_models:
            self._apply_diffusion_overrides(model)
        self.model_groups = self._expand_model_groups()
        self.group_labels = [group.label for group in self.model_groups]
        self._prepare_label_template()
        self.model_sampling_pool = self._build_model_sampling_pool()
        if not self.model_sampling_pool:
            raise ValueError("No models available after sampling configuration was applied.")
        total_models = len(self.model_sampling_pool)
        print(f"Prepared {len(self.model_groups)} model group(s) and {total_models} sampling candidates.")

    def _apply_diffusion_overrides(self, model: torch.nn.Module) -> None:
        override_steps = getattr(self.args, "diffusion_num_steps", None)
        if override_steps is not None and hasattr(model, "diffusion_num_steps"):
            model.diffusion_num_steps = int(override_steps)
        override_sampler = getattr(self.args, "diffusion_sampler_type", None)
        if override_sampler is not None:
            setattr(model, "diffusion_sampler_type", override_sampler)

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

    def _build_model_sampling_pool(self) -> List[Tuple[int, int, torch.nn.Module]]:
        pool: List[Tuple[int, int, torch.nn.Module]] = []
        for group_idx, group in enumerate(self.model_groups):
            for model_idx, model in enumerate(group.models):
                pool.append((group_idx, model_idx, model))
        return pool

    def run(self) -> None:
        rows = self._collect_rows()
        if not rows:
            print("No samples were collected from the dataloader; nothing to visualize.")
            return
        fig = self._render_rows(rows)
        if self.save_path:
            first_id = rows[0].image_id if rows else "grid"
            target_path = self._resolve_save_path(0, first_id)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, dpi=self.figure_dpi, bbox_inches="tight")
            print(f"Saved visualization to {target_path}.")
        if self.show:
            plt.show()
        plt.close(fig)

    def _collect_rows(self) -> List[CaseRow]:
        selected: List[CaseRow] = []
        candidate_pool: List[CaseRow] = []
        total_considered = 0

        def flush_candidates(force: bool = False) -> bool:
            nonlocal total_considered
            if not candidate_pool:
                return False
            if len(candidate_pool) < self.n_resample_largest_mask and not force:
                return False
            best_case = max(candidate_pool, key=lambda entry: entry.label_mass)
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
                batch_rows = self._process_batch(batch)
                for row in batch_rows:
                    candidate_pool.append(row)
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

    def _process_batch(self, batch: Dict) -> List[CaseRow]:
        seg = batch["seg"]
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        batch_rows: List[CaseRow] = []
        batch_size = seg.shape[0]
        for sample_idx in range(batch_size):
            case_row = self._build_case_row(batch, seg, sample_idx)
            if case_row is None:
                continue
            batch_rows.append(case_row)
            if len(batch_rows) >= self.max_images:
                break
        return batch_rows

    def _build_case_row(self, batch: Dict, seg_tensor: torch.Tensor, sample_idx: int) -> Optional[CaseRow]:
        image_tensor = self._extract_image_tensor(batch["data"], sample_idx)
        ground_truth = seg_tensor[sample_idx].clone()
        predictions_tensor, prediction_sources = self._sample_predictions_for_sample(batch, sample_idx)
        if predictions_tensor.numel() == 0:
            return None
        num_classes = predictions_tensor.shape[1] if predictions_tensor.ndim >= 2 else 0
        use_chaksu_colors = self._use_chaksu_rendering(num_classes)
        if use_chaksu_colors:
            prediction_cells = [self._render_chaksu_probability_map(pred) for pred in predictions_tensor]
            pred_sum_tensor = self._compute_chaksu_prediction_sum(predictions_tensor)
        else:
            foreground_probs = torch.stack(
                [self._extract_foreground_probability(pred) for pred in predictions_tensor],
                dim=0,
            )
            prediction_cells = self._prepare_prediction_cells(foreground_probs)
            pred_sum_tensor = (foreground_probs >= self.pred_threshold).float().sum(dim=0)
        pred_sum_cell = self._render_sum_map(pred_sum_tensor, max_value=self.num_predictions)
        gt_masks = self._extract_gt_masks(ground_truth)
        if use_chaksu_colors:
            gt_sum_tensor = self._compute_chaksu_gt_sum(ground_truth)
            gt_max = max(1, self._count_gt_annotators(ground_truth))
        else:
            gt_sum_tensor = gt_masks.float().sum(dim=0)
            gt_max = max(1, gt_masks.shape[0])
        gt_sum_cell = self._render_sum_map(gt_sum_tensor, max_value=gt_max)
        metrics = self._calculate_metrics(predictions_tensor, ground_truth)
        label_mass = self._calculate_label_mass(gt_masks)
        return CaseRow(
            image_id=str(batch["image_id"][sample_idx]),
            image_cell=self._tensor_to_rgb(image_tensor),
            gt_sum_cell=gt_sum_cell,
            pred_sum_cell=pred_sum_cell,
            prediction_cells=prediction_cells,
            metrics=metrics,
            predictions_tensor=predictions_tensor.cpu(),
            ground_truth_tensor=ground_truth.cpu(),
            prediction_sources=prediction_sources,
            label_mass=label_mass,
        )

    def _extract_image_tensor(self, batch_data, sample_idx: int) -> torch.Tensor:
        if isinstance(batch_data, list):
            if not batch_data:
                raise ValueError("Encountered an empty data list when extracting an image tensor.")
            tensor = batch_data[0]
        else:
            tensor = batch_data
        return tensor[sample_idx].detach().cpu()

    def _sample_predictions_for_sample(self, batch: Dict, sample_idx: int) -> Tuple[torch.Tensor, List[str]]:
        predictions: List[torch.Tensor] = []
        sources: List[str] = []
        for _ in range(self.num_predictions):
            group_idx, model_idx, model = self._select_model_entry()
            pred = self._draw_single_prediction(model, batch, sample_idx)
            predictions.append(pred.cpu())
            sources.append(self._format_prediction_source(group_idx, model_idx))
        stacked = torch.stack(predictions, dim=0)
        return stacked, sources

    def _select_model_entry(self) -> Tuple[int, int, torch.nn.Module]:
        return self._rng.choice(self.model_sampling_pool)

    def _format_prediction_source(self, group_idx: int, model_idx: int) -> str:
        label = self._get_group_display_label(group_idx)
        return f"{label}::m{model_idx + 1}"

    def _draw_single_prediction(self, model: torch.nn.Module, batch: Dict, sample_idx: int) -> torch.Tensor:
        inputs = batch["data"]
        if isinstance(inputs, list):
            inputs = inputs[0]
        tensor_inputs = inputs[sample_idx : sample_idx + 1].to(self.device)
        au_type = getattr(model, "AU_type", "softmax")
        if au_type == "ssn":
            distribution, cov_failed_flag = model.forward(tensor_inputs)
            if cov_failed_flag:
                raise RuntimeError("SSN covariance matrix was not positive definite.")
            samples = distribution.sample([1])
            samples = samples.view(
                1,
                tensor_inputs.size()[0],
                model.num_classes,
                *tensor_inputs.size()[2:],
            )
            softmax_samples = F.softmax(samples, dim=2)
            return softmax_samples[0, 0].detach()
        if au_type == "diffusion":
            tensor_inputs = tensor_inputs.float()
            num_steps = int(getattr(model, "diffusion_num_steps", 50))
            sampler_type = getattr(model, "diffusion_sampler_type", "ddpm") or "ddpm"
            x_init = torch.randn(
                (tensor_inputs.shape[0], model.num_classes, *tensor_inputs.shape[2:]),
                device=self.device,
                dtype=tensor_inputs.dtype,
            )
            sample_output = model.diffusion_sample_loop(
                x_init=x_init,
                im=tensor_inputs,
                num_steps=num_steps,
                sampler_type=sampler_type,
                clip_x=False,
                guidance_weight=0.0,
                progress_bar=False,
                self_cond=False,
            )
            return sample_output.squeeze(0)
        if au_type == "prob_unet":
            tensor_inputs = tensor_inputs.float()
            model.forward(tensor_inputs, segm=None, training=False)
            logits_stack = model.sample_multiple(1, from_prior=True, testing=True)
            softmax_stack = torch.softmax(logits_stack, dim=2)
            return softmax_stack[0, 0].detach()
        output = model.forward(tensor_inputs)
        return F.softmax(output, dim=1)[0].detach()

    def _extract_foreground_probability(self, prob_tensor: torch.Tensor) -> torch.Tensor:
        if prob_tensor.shape[0] == 1:
            return prob_tensor[0]
        dataset_lower = str(self.dataset_name or "").lower()
        if dataset_lower.startswith(LIDC_PREFIX) and prob_tensor.shape[0] >= 2:
            return prob_tensor[1]
        return 1.0 - prob_tensor[0]

    def _prepare_prediction_cells(self, foreground_probs: torch.Tensor) -> List[np.ndarray]:
        return [self._render_probability_map(prob) for prob in foreground_probs]

    def _render_probability_map(self, prob_tensor: torch.Tensor) -> np.ndarray:
        prob_np = prob_tensor.detach().cpu().numpy()
        clipped = np.clip(prob_np, 0.0, 1.0)
        rgb = np.stack([clipped, clipped, clipped], axis=-1)
        mask = (prob_np < 0) | (prob_np > 1)
        if np.any(mask):
            overlay = np.zeros_like(rgb)
            overlay[..., 0] = 1.0
            mask_exp = mask[..., None]
            rgb = np.where(mask_exp, (1 - self.overlay_alpha) * rgb + self.overlay_alpha * overlay, rgb)
        return rgb.astype(np.float32)

    def _use_chaksu_rendering(self, num_classes: Optional[int] = None) -> bool:
        if not self._is_chaksu:
            return False
        if num_classes is None or num_classes <= 0:
            return True
        return num_classes >= 3

    def _get_chaksu_palette(self) -> np.ndarray:
        if self._chaksu_palette is not None:
            return self._chaksu_palette
        palette = np.zeros((3, 3), dtype=np.float32)
        user_defined = ["#d600a4", "#006dd4"]
        for idx, hex_color in enumerate(user_defined[:2]):
            palette[idx + 1] = np.array(mcolors.to_rgb(hex_color), dtype=np.float32)
        self._chaksu_palette = palette
        return palette

    def _render_chaksu_probability_map(self, prob_tensor: torch.Tensor) -> np.ndarray:
        probs = prob_tensor.detach().cpu().float()
        if probs.ndim != 3 or probs.shape[0] < 3:
            fallback = self._extract_foreground_probability(prob_tensor)
            return self._render_probability_map(fallback)
        palette = self._get_chaksu_palette()
        color = np.zeros((probs.shape[1], probs.shape[2], 3), dtype=np.float32)
        positive_probs = probs[1:3].clamp(0.0, 1.0).cpu().numpy()
        classes = min(positive_probs.shape[0], palette.shape[0] - 1)
        for idx in range(classes):
            color += positive_probs[idx][..., None] * palette[idx + 1]
        return np.clip(color, 0.0, 1.0).astype(np.float32)

    def _compute_chaksu_prediction_sum(self, predictions_tensor: torch.Tensor) -> torch.Tensor:
        if predictions_tensor.ndim != 4 or predictions_tensor.shape[1] < 3:
            h, w = predictions_tensor.shape[-2:]
            return torch.zeros((1, h, w), dtype=torch.float32)
        probs = predictions_tensor[:, 1:3]
        votes = (probs >= self.pred_threshold).float()
        if votes.shape[1] > 1:
            multi_mask = votes.sum(dim=1, keepdim=True) > 1.0
            if multi_mask.any():
                winners = probs.argmax(dim=1, keepdim=True)
                one_hot = torch.zeros_like(votes)
                one_hot.scatter_(1, winners, 1.0)
                votes = torch.where(multi_mask, one_hot, votes)
        return votes.sum(dim=0)

    def _count_gt_annotators(self, gt_tensor: torch.Tensor) -> int:
        if gt_tensor.ndim == 2:
            return 1
        return max(1, gt_tensor.shape[0])

    def _compute_chaksu_gt_sum(self, gt_tensor: torch.Tensor) -> torch.Tensor:
        tensor = gt_tensor.clone()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if self.ignore_index is not None and self.ignore_index >= 0:
            tensor = tensor.masked_fill(tensor == self.ignore_index, 0)
        tensor = tensor.long()
        max_label = int(tensor.max().item()) if tensor.numel() > 0 else 0
        num_classes = max(self._num_model_classes, max_label + 1, 3)
        one_hot = F.one_hot(tensor, num_classes=num_classes)
        positive = one_hot[..., 1:3].permute(0, 3, 1, 2).float()
        return positive.sum(dim=0)

    def _render_sum_map(self, tensor: torch.Tensor, max_value: int) -> np.ndarray:
        if tensor.ndim == 3 and self._use_chaksu_rendering(tensor.shape[0] + 1):
            return self._render_chaksu_sum_map(tensor, max_value)
        arr = tensor.detach().cpu().numpy()
        denom = max(1.0, float(max_value))
        norm = np.clip(arr / denom, 0.0, 1.0)
        rgb = self.sum_cmap(norm)[..., :3]
        return rgb.astype(np.float32)

    def _render_chaksu_sum_map(self, tensor: torch.Tensor, max_value: int) -> np.ndarray:
        counts = tensor.detach().cpu().float().numpy()
        if counts.ndim != 3:
            squeeze = tensor
            if tensor.ndim > 3:
                squeeze = tensor.mean(dim=0)
            return self._render_probability_map(squeeze)
        counts = np.clip(counts, 0.0, None)
        totals = counts.sum(axis=0, keepdims=True)
        palette = self._get_chaksu_palette()
        height, width = counts.shape[1], counts.shape[2]
        color = np.zeros((height, width, 3), dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        classes = min(counts.shape[0], palette.shape[0] - 1)
        for idx in range(classes):
            color += weights[idx][..., None] * palette[idx + 1]
        return np.clip(color, 0.0, 1.0).astype(np.float32)

    def _extract_gt_masks(self, gt_tensor: torch.Tensor) -> torch.Tensor:
        tensor = gt_tensor.clone()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if self.ignore_index is not None and self.ignore_index >= 0:
            tensor = tensor.masked_fill(tensor == self.ignore_index, 0)
        dataset_lower = str(self.dataset_name or "").lower()
        if dataset_lower.startswith(LIDC_PREFIX):
            masks = (tensor == 1).float()
        else:
            masks = (tensor > 0).float()
        return masks

    def _calculate_label_mass(self, gt_masks: torch.Tensor) -> float:
        if gt_masks.numel() == 0:
            return 0.0
        return float(gt_masks.sum().item())

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {"dice": None, "ged": None}
        if predictions.numel() == 0:
            return metrics
        mean_softmax = predictions.mean(dim=0)
        pred_idx = mean_softmax.argmax(dim=0)
        gt = ground_truth.squeeze(0)
        dataset_lower = str(self.dataset_name or "").lower()
        is_lidc = "lidc" in dataset_lower
        if is_lidc:
            if gt.ndim == 4:
                gt = gt.squeeze(1)
            ignore = self.ignore_index
            pred_rep = pred_idx.unsqueeze(0).expand_as(gt)
            valid = gt != ignore
            pred_pos = (pred_rep == 1) & valid
            gt_pos = (gt == 1) & valid
            tp = (pred_pos & gt_pos).sum(dim=(1, 2)).to(torch.float32)
            pred_sum = pred_pos.sum(dim=(1, 2)).to(torch.float32)
            gt_sum = gt_pos.sum(dim=(1, 2)).to(torch.float32)
            denom = 2 * tp + (pred_sum - tp) + (gt_sum - tp)
            both_empty = (pred_sum == 0) & (gt_sum == 0)
            one_empty = (pred_sum == 0) ^ (gt_sum == 0)
            dice_vals = torch.zeros_like(denom)
            dice_vals[both_empty] = 1.0
            dice_vals[one_empty] = 0.0
            regular = ~(both_empty | one_empty)
            safe = denom > 0
            idx = regular & safe
            dice_vals[idx] = (2 * tp[idx]) / denom[idx]
            metrics["dice"] = float(dice_vals.mean().item())
        else:
            output_idx = pred_idx.unsqueeze(0)
            per_rater = []
            if gt.ndim == 3:
                raters = gt
            else:
                raters = gt.squeeze(1)
            for rater in raters:
                rater = rater.unsqueeze(0)
                score = dice(
                    output_idx,
                    rater,
                    ignore_index=self.ignore_index,
                    include_background=False,
                    num_classes=int(mean_softmax.shape[0]),
                    binary_dice=False,
                    average="macro",
                    is_softmax=False,
                )
                per_rater.append(float(score if isinstance(score, float) else score.item()))
            metrics["dice"] = float(sum(per_rater) / len(per_rater)) if per_rater else None
        if gt.shape[0] > 1 and not getattr(self.args, "skip_ged", False):
            ignore_index = self.ignore_index if self.ignore_index >= 0 else None
            try:
                if is_lidc:
                    ged_result = ged_binary_fast(
                        predictions.to(self.device),
                        gt.to(self.device),
                        ignore_index=ignore_index,
                        additional_metrics=["dice"],
                    )
                else:
                    ged_result = calculate_ged(
                        predictions.to(self.device),
                        gt.to(self.device),
                        ignore_index=ignore_index,
                        additional_metrics=["dice"],
                    )
                if ged_result and "ged" in ged_result:
                    metrics["ged"] = float(ged_result["ged"])
            except Exception:
                metrics["ged"] = None
        return metrics

    def _render_rows(self, rows: List[CaseRow]):
        col_labels = ["Image", "GT sum", "Pred sum"] + [f"Pred {idx + 1}" for idx in range(self.num_predictions)]
        grid_rows: List[List[np.ndarray]] = []
        row_labels: List[str] = []
        blank_cell = self._blank_like(rows[0].image_cell)
        for idx, row in enumerate(rows):
            row_cells = [row.image_cell, row.gt_sum_cell, row.pred_sum_cell]
            row_cells.extend(row.prediction_cells)
            if len(row_cells) < len(col_labels):
                row_cells.extend([blank_cell.copy() for _ in range(len(col_labels) - len(row_cells))])
            grid_rows.append(row_cells)
            row_labels.append(self._format_row_label(idx, row))
        title = self._build_title(rows)
        return self._build_grid_figure(grid_rows, col_labels, row_labels, title)

    def _build_title(self, rows: List[CaseRow]) -> str:
        sample_ids = ", ".join(row.image_id for row in rows[:3])
        if len(rows) > 3:
            sample_ids += ", …"
        return f"split={self.args.test_split} | rows={len(rows)} | samples={sample_ids}"

    def _format_row_label(self, row_idx: int, row: CaseRow) -> str:
        dice_val = row.metrics.get("dice")
        ged_val = row.metrics.get("ged")
        dice_text = "n/a" if dice_val is None else f"{dice_val:.3f}"
        ged_text = "n/a" if ged_val is None else f"{ged_val:.3f}"
        return f"#{row_idx}\ndice={dice_text}, ged={ged_text}"

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
        return arr.permute(1, 2, 0).numpy().astype(np.float32)

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
        return np.zeros_like(reference, dtype=np.float32)

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

    def _figure_size(self, rows: int, cols: int) -> Tuple[float, float]:
        width = max(4.0, cols * 2.0)
        height = max(4.0, rows * 2.0)
        return (width, height)

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
        max_suffix = min(len(min(labels, key=len)), len(suffix))
        suffix = suffix[:max_suffix]
        while prefix and suffix and any(len(label) < len(prefix) + len(suffix) for label in labels):
            suffix = suffix[1:]

        unique_parts: List[str] = []
        for label in labels:
            core_start = len(prefix)
            core_end = len(label) - len(suffix) if suffix else len(label)
            if core_end < core_start:
                core_end = core_start
            unique_parts.append(label[core_start:core_end])

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
        "--num-images",
        type=int,
        default=4,
        help="Number of test images (rows) to visualize.",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=None,
        help="Override the number of sampled predictions per row (defaults to --n_pred or 4).",
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
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used before summing predictions for the 'pred_sum' column.",
    )
    parser.add_argument(
        "--row-mode",
        type=str,
        default="images",
        choices=["images"],
        help="Controls how rows are organized. Only 'images' is implemented in this version.",
    )
    parser.set_defaults(n_pred=4)


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
