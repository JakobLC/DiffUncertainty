import copy
import json
import os
import pickle
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
#from torchmetrics.functional import dice
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())

from evaluation.metrics.dice_wrapped import dice
from evaluation.metrics.ged_fast import ged_binary_fast
from uncertainty_modeling.main import set_seed
from uncertainty_modeling.unc_mod_utils.test_utils import (
    test_cli,
    load_models_from_checkpoint,
    calculate_ged,
    calculate_uncertainty,
    calculate_one_minus_msr,
    prepare_evaluation_jobs,
)
import uncertainty_modeling.data.cityscapes_labels as cs_labels
from global_utils.checkpoint_format import format_checkpoint_subdir
from uncertainty_modeling.unc_mod_utils.swag import SWAG

LIDC_PATIENT_AUG_SCHEMA = "lidc_patient_aug_v1"
AUGMENTED_TEST_SPLITS = {"ood_noise", "ood_blur", "ood_contrast", "ood_jpeg"}


class Tester:
    def __init__(self, args: Namespace):
        checkpoint_paths = getattr(args, "checkpoint_paths", None)
        self.checkpoint_paths = list(checkpoint_paths) if checkpoint_paths is not None else []
        if not self.checkpoint_paths:
            raise ValueError("Tester requires at least one checkpoint path.")
        self.all_checkpoints = self.get_checkpoints(self.checkpoint_paths)
        hparams = self.all_checkpoints[0]["hyper_parameters"]
        requested_seed = int(getattr(args, "seed", -1) or -1)
        if requested_seed >= 0:
            hparams["seed"] = requested_seed
        set_seed(hparams["seed"])
        self.ignore_index = hparams["data"]["ignore_index"]
        self.skip_ged = args.skip_ged
        dataset_cfg = hparams.get("data", {}).get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            dataset_cfg = {}
        splits_path = dataset_cfg.get("splits_path")
        (
            self.split_schema,
            self.available_splits,
            self.has_unlabeled_pool,
        ) = self._inspect_splits_file(splits_path)
        normalized_split = self._normalize_split_name(args.test_split)
        self._ensure_split_is_supported(args.test_split, normalized_split)
        args.test_split = normalized_split
        self.test_batch_size = args.test_batch_size
        self.tta = args.tta
        self.discretize = args.discretize
        self.direct_au = bool(getattr(args, "direct_au", False))
        self.test_dataloader = self.get_test_dataloader(args, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.swag_blockwise = bool(getattr(args, "swag_blockwise", False))
        self.swag_low_rank_cov = bool(getattr(args, "swag_low_rank_cov", True))
        self.ensemble_mode = bool(getattr(args, "ensemble_mode", False))
        if self.direct_au and self.ensemble_mode:
            raise ValueError("direct_au cannot be combined with --ensemble_mode.")
        self.diffusion_steps_override = getattr(args, "diffusion_num_steps", None)
        self.diffusion_sampler_override = getattr(args, "diffusion_sampler_type", None)
        base_models = load_models_from_checkpoint(
            self.all_checkpoints,
            device="cpu",
            use_ema=bool(getattr(args, "use_ema", False)),
        )
        requested_n_models = max(int(getattr(args, "n_models", 0) or 0), 0)
        if self.ensemble_mode and requested_n_models > 0:
            print(
                f"[ensemble_mode] Ignoring --n_models={requested_n_models} because ensemble size is determined by provided checkpoints."
            )
        if self.direct_au:
            requested_n_models = self._apply_direct_au_overrides(
                base_models, requested_n_models
            )
        self.n_models = 0 if self.ensemble_mode else requested_n_models
        self.models = self.expand_eu_models(
            base_models,
            self.all_checkpoints,
            self.checkpoint_paths,
        )
        self._maybe_override_diffusion_steps()
        if not self.models:
            raise RuntimeError(
                "No models were instantiated; ensure checkpoints are valid and --n_models is not zero with ensemble_mode disabled."
            )
        self.n_pred = args.n_pred
        self.results_dict = {}
        self.save_root_dir = (
            args.save_dir if args.save_dir is not None else hparams["save_dir"]
        )
        self.exp_name = hparams["exp_name"] if args.exp_name is None else args.exp_name
        # Prefer an override when running ensembles so saved folder reflects all members
        self.version = str(getattr(args, "version_override", None) or hparams["version"])
        self.test_split = args.test_split
        self.save_split_name = self._format_split_for_output(self.test_split)
        self.use_ema = bool(getattr(args, "use_ema", False))
        self.dataset_name = hparams["data"]["name"]
        self.cityscapes_palette = self._build_cityscapes_palette()
        self.checkpoint_epoch = self.all_checkpoints[0]["epoch"] + 1
        self.checkpoint_subdir = format_checkpoint_subdir(
            self.checkpoint_epoch, self.use_ema
        )
        self.metrics_only = bool(getattr(args, "metrics_only", False))
        self.skip_saving = bool(getattr(args, "skip_saving", False))
        self.create_save_dirs()
        self.skip_existing = bool(getattr(args, "skip_existing", False))

    @staticmethod
    def get_checkpoints(checkpoint_paths):
        all_checkpoints = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            # Minimal fix: if top-level hyper_parameters is a Lightning AttributeDict, cast to plain dict
            if checkpoint["hyper_parameters"].__class__.__name__ == "AttributeDict":
                checkpoint["hyper_parameters"] = dict(checkpoint["hyper_parameters"])
            conf = OmegaConf.create(checkpoint["hyper_parameters"])
            resolved = OmegaConf.to_container(conf, resolve=True)
            Tester._disable_hrnet_pretrained_flag(resolved)
            checkpoint["hyper_parameters"] = resolved
            all_checkpoints.append(checkpoint)
        return all_checkpoints

    @staticmethod
    def set_n_reference_samples(hparams, n_reference_samples):
        transforms_cfg = hparams["data"]["augmentations"].get("TEST", [])
        updated = False
        for compose in transforms_cfg:
            compose_cfg = compose.get("Compose")
            if not compose_cfg:
                continue
            for transform in compose_cfg.get("transforms", []):
                if "StochasticLabelSwitches" in transform:
                    transform["StochasticLabelSwitches"][
                        "n_reference_samples"
                    ] = n_reference_samples
                    updated = True
        return hparams

    @staticmethod
    def _disable_hrnet_pretrained_flag(hparams):
        if not isinstance(hparams, dict):
            return

        def _set_flag(section):
            if isinstance(section, dict) and "PRETRAINED" in section:
                section["PRETRAINED"] = False
                return True
            return False

        legacy_model = hparams.get("MODEL") if isinstance(hparams, dict) else None
        if _set_flag(legacy_model):
            return

        network_section = hparams.get("network") if isinstance(hparams, dict) else None
        if not isinstance(network_section, dict):
            return

        candidates = []
        cfg_section = network_section.get("cfg")
        if isinstance(cfg_section, dict):
            candidates.append(cfg_section.get("MODEL"))

        nested_model = network_section.get("model")
        if isinstance(nested_model, dict):
            if isinstance(nested_model.get("cfg"), dict):
                candidates.append(nested_model["cfg"].get("MODEL"))
            candidates.append(nested_model.get("MODEL"))

        for candidate in candidates:
            if _set_flag(candidate):
                return

    def _apply_direct_au_overrides(self, models, requested_n_models):
        if len(models) != 1:
            raise ValueError(
                f"[direct_au] Expected exactly one checkpoint/model, received {len(models)}."
            )
        if requested_n_models != 1:
            print(
                f"[direct_au] Ignoring --n_models={requested_n_models}; using 1 ensemble member."
            )
        allowed_eu = {"none", "swag", "swag_diag"}
        for idx, model in enumerate(models):
            eu_type = getattr(model, "EU_type", "none") or "none"
            au_type = getattr(model, "AU_type", "softmax") or "softmax"
            if eu_type not in allowed_eu:
                raise ValueError(
                    f"[direct_au] Model #{idx} has unsupported EU_type='{eu_type}'. Only 'none' or 'swag' are allowed."
                )
            if au_type == "softmax":
                raise ValueError(
                    "[direct_au] Provided model has AU_type='softmax'. Supply a generative AU model (diffusion, ssn, prob_unet, ...)."
                )
            if eu_type in {"swag", "swag_diag"}:
                setattr(model, "EU_type", "none")
                print(
                    f"[direct_au] Treating SWAG checkpoint '{self.checkpoint_paths[idx]}' as a standard model (no SWAG sampling)."
                )
        return 1

    @staticmethod
    def _inspect_splits_file(splits_path):
        with open(splits_path, "rb") as handle:
            splits_obj = pickle.load(handle)
        entry = splits_obj[0]
        meta = entry.get("_meta", {})
        schema = meta.get("schema")
        available = {k for k in entry.keys() if not k.startswith("_")}

        id_pool = entry.get("id_unlabeled_pool")
        if id_pool is None:
            id_count = 0
        elif isinstance(id_pool, np.ndarray):
            id_count = int(id_pool.size)
        else:
            id_count = len(id_pool)

        ood_pool = entry.get("ood_unlabeled_pool")
        if ood_pool is None:
            ood_count = 0
        elif isinstance(ood_pool, np.ndarray):
            ood_count = int(ood_pool.size)
        else:
            ood_count = len(ood_pool)

        unlabeled_pool_nonempty = (id_count + ood_count) > 0
        return schema, available, unlabeled_pool_nonempty

    def _normalize_split_name(self, split_name):
        available = self.available_splits or set()
        if split_name == "id" and "id_test" in available:
            return "id_test"
        if split_name == "ood" and "ood_test" in available:
            return "ood_test"
        return split_name

    @staticmethod
    def _format_split_for_output(split_name):
        if split_name in ["ood_test", "id_test"]:
            return split_name.replace("_test", "")
        return split_name

    def _ensure_split_is_supported(self, requested_split, normalized_split):
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

    def expand_eu_models(self, base_models, checkpoints, checkpoint_paths):
        if self.direct_au:
            return list(base_models)
        expanded_models = []
        total = len(base_models)
        for idx in range(total):
            model = base_models[idx]
            checkpoint = checkpoints[idx]
            if self.ensemble_mode:
                expanded_models.append(model)
                continue
            expanded = False
            if self.n_models > 0:
                if model.EU_type in ["swag", "swag_diag"]:
                    if "swag_config" in checkpoint:
                        swag_config = checkpoint["swag_config"]
                    else:
                        hyper_params = checkpoint["hyper_parameters"]
                        swag_config = hyper_params["swag"]
                    swag_models = self._sample_swag_draws(
                        model,
                        checkpoint["swag_state_dict"],
                        swag_config,
                        checkpoint_paths[idx],
                    )
                    expanded_models.extend(swag_models)
                    expanded = bool(swag_models)
                elif model.EU_type == "dropout":
                    expanded_models.extend([model] * self.n_models)
                    expanded = self.n_models > 0
                else:
                    print(
                        f"[n_models={self.n_models}] EU sampling is not implemented for EU_type='{model.EU_type}'. Using base model instead."
                    )
            if not expanded:
                expanded_models.append(model)
        return expanded_models

    def _maybe_override_diffusion_steps(self):
        if self.diffusion_steps_override is None and self.diffusion_sampler_override is None:
            return
        steps_override = (
            None
            if self.diffusion_steps_override is None
            else int(self.diffusion_steps_override)
        )
        sampler_override = (
            None
            if self.diffusion_sampler_override is None
            else str(self.diffusion_sampler_override)
        )
        for model in self.models:
            if getattr(model, "diffusion", False):
                if steps_override is not None:
                    setattr(model, "diffusion_num_steps", steps_override)
                if sampler_override:
                    setattr(model, "diffusion_sampler_type", sampler_override)

    def _sample_swag_draws(self, template_model, swag_state, swag_config, checkpoint_label):
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
        requested_low_rank = self.swag_low_rank_cov
        use_low_rank = requested_low_rank and not config["diag_only"]
        if requested_low_rank and config["diag_only"]:
            print(
                f"[SWAG] Checkpoint '{checkpoint_label}' only stores diagonal statistics; falling back to diagonal sampling."
            )
        sampled_models = []
        for sample_idx in range(self.n_models):
            sampled_model = copy.deepcopy(template_model)
            swag.sample(
                sampled_model,
                scale=1.0,
                use_low_rank=use_low_rank,
                blockwise=self.swag_blockwise,
            )
            sampled_model.eval()
            sampled_models.append(sampled_model.to("cpu"))
        print(
            f"[SWAG] Sampled {len(sampled_models)} model(s) from '{checkpoint_label}' (n_snapshots={config['max_snapshots']})"
        )
        return sampled_models

    def _activate_model_for_inference(self, model):
        if self.device == "cpu":
            return
        model.to(self.device)

    def _deactivate_model_after_inference(self, model):
        if self.device == "cpu":
            return
        model.to("cpu")

    @contextmanager
    def _model_device_scope(self, model):
        self._activate_model_for_inference(model)
        try:
            yield
        finally:
            self._deactivate_model_after_inference(model)

    def create_save_dirs(self):
        path_parts = [
            self.save_root_dir,
            self.exp_name,
            "test_results",
            self.version,
        ]
        if self.checkpoint_subdir:
            path_parts.append(self.checkpoint_subdir)
        path_parts.append(self.save_split_name)
        self.save_dir = os.path.join(*path_parts)
        self.save_pred_dir = os.path.join(self.save_dir, "pred_seg")
        self.save_pred_prob_dir = os.path.join(self.save_dir, "pred_prob")
        if self.skip_saving:
            return
        print(f"saving to results dir {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        if not self.metrics_only:
            os.makedirs(self.save_pred_dir, exist_ok=True)
            # os.makedirs(self.save_pred_prob_dir, exist_ok=True)
        return

    def _build_cityscapes_palette(self):
        palette = [0] * (256 * 3)
        for train_id, color in cs_labels.trainId2color.items():
            if train_id < 0 or train_id > 255:
                continue
            base = int(train_id) * 3
            palette[base : base + 3] = [int(channel) for channel in color]
        return palette

    def _save_palettized_prediction(self, mask, img_name):
        palette_img = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="P")
        palette_img.putpalette(self.cityscapes_palette)
        palette_img.save(os.path.join(self.save_pred_dir, f"{img_name}.png"))

    def should_skip(self):
        if not self.skip_existing:
            return False
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return False
        with open(metrics_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected metrics.json at '{metrics_path}' to contain a dict, got {type(data)}."
            )
        if "mean" in data:
            return True
        return False

    def get_test_dataloader(self, args: Namespace, hparams):
        data_input_dir = (
            args.data_input_dir
            if args.data_input_dir is not None
            else hparams["data"]["data_input_dir"]
        )
        if args.data_input_dir is not None:
            hparams["data"]["dataset"]["splits_path"] = hparams["data"][
                "dataset"
            ]["splits_path"].replace(hparams["data"]["data_input_dir"], args.data_input_dir)
        hparams = self.set_n_reference_samples(hparams, args.n_reference_samples)
        if self.test_batch_size:
            hparams["data"]["val_batch_size"] = self.test_batch_size
        dm = hydra.utils.instantiate(
            hparams["data"],
            data_input_dir=data_input_dir,
            augmentations=hparams["data"]["augmentations"],
            seed=hparams["seed"],
            test_split=args.test_split,
            tta=self.tta,
            _recursive_=False,
        )
        dm.setup("test")
        return dm.test_dataloader()

    def save_prediction(self, image_id, image_preds, mean_pred, ignore_index_map):
        multiple_preds = False
        if image_preds.shape[0] > 1:
            image_preds_mean = torch.cat([mean_pred.unsqueeze(0), image_preds], dim=0)
            multiple_preds = True
        else:
            image_preds_mean = image_preds
        for output_idx, output in enumerate(image_preds_mean):
            output = torch.moveaxis(output, 0, -1)
            # output_softmax_np = output.detach().cpu().numpy()
            output = torch.argmax(output, dim=-1, keepdim=True)
            output_np = output.detach().long().cpu().numpy().astype(np.uint8)
            if not multiple_preds:
                output_idx += 1
            img_name = (
                f"{image_id}_mean"
                if output_idx == 0 and multiple_preds
                else f"{image_id}_{str(output_idx).zfill(2)}"
            )

            if "lidc" in self.dataset_name.lower():
                mask = output_np.squeeze(-1)
                ignore_mask = np.asarray(ignore_index_map, dtype=bool)
                if ignore_mask.shape != mask.shape:
                    ignore_mask = np.squeeze(ignore_mask)
                mask = mask.copy()
                mask[ignore_mask] = 0
                mask = (mask > 0).astype(np.uint8) * 255
                if mask.ndim > 2:
                    mask = mask.squeeze()
                mask = np.asarray(mask, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.save_pred_dir, f"{img_name}.png"), mask)
            else:
                mask = output_np.squeeze(-1)
                ignore_mask = np.asarray(ignore_index_map, dtype=bool)
                if ignore_mask.shape != mask.shape:
                    ignore_mask = np.squeeze(ignore_mask)
                mask = mask.copy()
                mask[ignore_mask] = cs_labels.name2trainId["unlabeled"]
                self._save_palettized_prediction(mask, img_name)
        return


    def save_uncertainty(self, image_id, uncertainty_dict):
        for unc_type, unc_map in uncertainty_dict.items():
            unc_dir = os.path.join(self.save_dir, unc_type)
            os.makedirs(unc_dir, exist_ok=True)
            # TODO: Choose good file format that can handle floating point numbers
            # unc_map_np = (unc_map.detach().cpu().numpy() * 255).astype(np.uint8)
            unc_map_np = unc_map.detach().cpu().numpy()
            cv2.imwrite(os.path.join(unc_dir, f"{image_id}.tif"), unc_map_np)
            # save(unc_map_np, os.path.join(unc_dir, f"{image_id}.nii.gz"))

    def calculate_test_metrics(self, output_softmax, ground_truth):
        """Compute per-image mean Dice across raters.

        Optimizations:
        - Compute argmax of mean softmax once and reuse for all raters.
        - Use a vectorized binary Dice path for LIDC (two-class) datasets.
        """
        metrics_dict = {}

        # output_softmax: (C,H,W) for this image
        # Convert to predicted indices once
        pred_idx = output_softmax.argmax(dim=0)  # (H,W)

        if "lidc" in self.dataset_name.lower():
            # Vectorized binary dice versus all raters on device
            gt = ground_truth.to(pred_idx.device)  # (R,H,W)
            ignore = self.ignore_index
            # Expand pred to (R,H,W)
            pred_rep = pred_idx.unsqueeze(0).expand_as(gt)
            valid = (gt != ignore)

            pred_pos = (pred_rep == 1) & valid
            gt_pos = (gt == 1) & valid

            tp = (pred_pos & gt_pos).sum(dim=(1, 2)).to(torch.float32)
            pred_sum = pred_pos.sum(dim=(1, 2)).to(torch.float32)
            gt_sum = gt_pos.sum(dim=(1, 2)).to(torch.float32)
            denom = 2 * tp + (pred_sum - tp) + (gt_sum - tp)  # 2TP+FP+FN

            both_empty = (pred_sum == 0) & (gt_sum == 0)
            one_empty = (pred_sum == 0) ^ (gt_sum == 0)
            dice_vals = torch.zeros_like(denom, dtype=torch.float32)
            dice_vals[both_empty] = 1.0
            dice_vals[one_empty] = 0.0
            regular = ~(both_empty | one_empty)
            safe = denom > 0
            idx = regular & safe
            dice_vals[idx] = (2 * tp[idx]) / denom[idx]
            metrics_dict["dice"] = float(dice_vals.mean().item())
            return metrics_dict

        # Fallback: multi-class/general path via wrapper (single argmax per call avoided)
        output_idx = pred_idx.unsqueeze(0)  # (1,H,W)
        all_test_dice = []
        for rater in ground_truth:
            rater = rater.unsqueeze(0).to(output_idx.device)
            test_dice = dice(
                output_idx,
                rater,
                ignore_index=self.ignore_index,
                include_background=False,
                num_classes=int(output_softmax.shape[0]),
                binary_dice=False,
                average="macro",
                is_softmax=False,
            )
            all_test_dice.append(test_dice.item() if not isinstance(test_dice, float) else test_dice)
        metrics_dict["dice"] = float(np.mean(np.array(all_test_dice)))
        return metrics_dict

    def _compute_ged_backend(self, preds, ground_truth, ignore_index, additional_metrics=None):
        addl = additional_metrics if additional_metrics is not None else ["dice"]
        if "lidc" in self.dataset_name.lower():
            return ged_binary_fast(
                preds,
                ground_truth,
                ignore_index=ignore_index,
                additional_metrics=addl,
            )
        return calculate_ged(
            preds,
            ground_truth,
            ignore_index=ignore_index,
            additional_metrics=addl,
        )

    def _compute_grouped_ged(self, grouped_preds, ground_truth, ignore_index):
        if not grouped_preds:
            return None
        ged_scores = []
        for preds in grouped_preds:
            if preds is None or preds.numel() == 0:
                continue
            metrics = self._compute_ged_backend(
                preds,
                ground_truth,
                ignore_index,
                additional_metrics=[],
            )
            if "ged" in metrics:
                ged_scores.append(float(metrics["ged"]))
        if not ged_scores:
            return None
        return float(np.mean(ged_scores))

    def process_output(self, all_preds):
        ignore_index_map = all_preds["gt"] == self.ignore_index
        compute_ged = not self.skip_ged and all_preds["gt"].shape[1] > 1
        ged_ignore_index = (
            self.ignore_index if compute_ged and self.ignore_index >= 0 else None
        )
        n_batch = all_preds["softmax_pred"].shape[1]
        raw_pred_groups = all_preds.get("softmax_pred_groups", [])
        is_lidc_dataset = "lidc" in self.dataset_name.lower()
        for image_idx in range(n_batch):
            image_preds = all_preds["softmax_pred"][:, image_idx]  # (P,C,H,W)
            image_id = all_preds["image_id"][image_idx]
            mean_softmax_pred = torch.mean(image_preds, dim=0)
            self.results_dict[image_id] = {"dataset": all_preds["dataset"][image_idx]}
            self.results_dict[image_id]["metrics"] = {}
            gt_tensor = all_preds["gt"][image_idx]
            gt_for_backend = gt_tensor if is_lidc_dataset else gt_tensor.to(self.device)
            self.results_dict[image_id]["metrics"].update(
                self.calculate_test_metrics(mean_softmax_pred, gt_tensor)
            )
            if compute_ged:
                bma_metrics = self._compute_ged_backend(
                    image_preds,
                    gt_for_backend,
                    ged_ignore_index,
                    additional_metrics=["dice"],
                )
                bma_ged = bma_metrics.pop("ged", None)
                if bma_ged is not None:
                    self.results_dict[image_id]["metrics"]["ged_bma"] = float(bma_ged)
                self.results_dict[image_id]["metrics"].update(bma_metrics)
                per_image_raw_groups = (
                    [group[:, image_idx] for group in raw_pred_groups]
                    if raw_pred_groups
                    else []
                )
                grouped_ged = self._compute_grouped_ged(
                    per_image_raw_groups,
                    gt_for_backend,
                    ged_ignore_index,
                )
                if grouped_ged is not None:
                    self.results_dict[image_id]["metrics"]["ged"] = grouped_ged
            if image_preds.shape[0] > 1:
                uncertainty_dict = calculate_uncertainty(image_preds)
            else:
                uncertainty_dict = calculate_one_minus_msr(image_preds.squeeze(0))
            if not self.metrics_only:
                ignore_index_map_image = ignore_index_map[image_idx][0]
                self.save_prediction(
                    image_id,
                    image_preds,
                    mean_softmax_pred,
                    ignore_index_map_image.detach().long().cpu().numpy().astype(np.uint8),
                )
                self.save_uncertainty(image_id, uncertainty_dict)
    def _build_batch_predictions(self, batch):
        gt = batch["seg"]
        if isinstance(gt, torch.Tensor) and gt.ndim == 3:
            gt = gt.unsqueeze(1)
        all_preds = {
            "softmax_pred": [],
            "softmax_pred_groups": [],
            "image_id": batch["image_id"],
            "gt": gt,
            "dataset": batch["dataset"],
        }
        generative_count = sum(1 for m in self.models if getattr(m, "is_generative", False))
        multiple_generative = generative_count > 1 and not self.direct_au
        for model in self.models:
            #print("allpreds groups shape:", all_preds["softmax_pred_groups"][-1].shape if all_preds["softmax_pred_groups"] else "N/A")
            with self._model_device_scope(model):
                au_type = getattr(model, "AU_type", "softmax")
                if au_type == "ssn":
                    distribution, cov_failed_flag = model.forward(batch["data"].to(self.device))
                    assert not cov_failed_flag, "Covariance matrix was not positive definite"
                    output_samples = distribution.sample([self.n_pred])
                    output_samples = output_samples.view(
                        [
                            self.n_pred,
                            batch["data"].size()[0],
                            model.num_classes,
                            *batch["data"].size()[2:],
                        ]
                    )
                    if multiple_generative:
                        softmax_samples = F.softmax(output_samples, dim=2)
                        all_preds["softmax_pred_groups"].append(softmax_samples)
                        #all_preds["softmax_pred"].append(softmax_samples.mean(dim=0))
                    else:
                        for output_sample in output_samples:
                            output_softmax = F.softmax(output_sample, dim=1)
                            #all_preds["softmax_pred"].append(output_softmax)
                            all_preds["softmax_pred_groups"].append(output_softmax.unsqueeze(0))
                elif au_type == "diffusion":
                    inputs = batch["data"]
                    if isinstance(inputs, list):
                        inputs = inputs[0]
                    inputs = inputs.to(self.device).float()
                    sample_list = []
                    num_steps = int(getattr(model, "diffusion_num_steps", 10))
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
                    sample_stack = torch.stack(sample_list)
                    if multiple_generative:
                        all_preds["softmax_pred_groups"].append(sample_stack)
                        #all_preds["softmax_pred"].append(sample_stack.mean(dim=0))
                    else:
                        for sample in sample_stack:
                            #all_preds["softmax_pred"].append(sample)
                            all_preds["softmax_pred_groups"].append(sample.unsqueeze(0))
                elif au_type == "prob_unet" or getattr(model, "prob_unet_enabled", False):
                    inputs = batch["data"]
                    if isinstance(inputs, list):
                        inputs = inputs[0]
                    tensor_inputs = inputs.to(self.device).float()
                    model.forward(tensor_inputs, segm=None, training=False)
                    logits_stack = model.sample_multiple(self.n_pred, from_prior=True, testing=True)
                    softmax_stack = torch.softmax(logits_stack, dim=2)
                    if multiple_generative:
                        all_preds["softmax_pred_groups"].append(softmax_stack)
                        #all_preds["softmax_pred"].append(softmax_stack.mean(dim=0))
                    else:
                        for sample in softmax_stack:
                            #all_preds["softmax_pred"].append(sample)
                            all_preds["softmax_pred_groups"].append(sample.unsqueeze(0))
                elif self.tta:
                    for index, image in enumerate(batch["data"]):
                        output = model.forward(image.to(self.device))
                        output_softmax = F.softmax(output, dim=1)
                        if any(
                            "HorizontalFlip" in sl for sl in batch["transforms"][index]
                        ):
                            pred_tensor = torch.flip(output_softmax, [-1])
                        else:
                            pred_tensor = output_softmax
                        #all_preds["softmax_pred"].append(pred_tensor)
                        all_preds["softmax_pred_groups"].append(pred_tensor.unsqueeze(0))
                else:
                    output = model.forward(batch["data"].to(self.device))
                    output_softmax = F.softmax(output, dim=1)
                    #all_preds["softmax_pred"].append(output_softmax)
                    all_preds["softmax_pred_groups"].append(output_softmax.unsqueeze(0))
        if self.discretize:
            def _discretize(group):
                    return F.one_hot(torch.argmax(group, dim=2), num_classes=group.shape[2]).permute(0, 1, 4, 2, 3).float()
            all_preds["softmax_pred_groups"] = [_discretize(group) for group in all_preds["softmax_pred_groups"]]

        all_preds["softmax_pred"] = torch.stack(all_preds["softmax_pred_groups"]).mean(dim=1)
        #p1 = torch.stack(all_preds["softmax_pred_groups"])
        #p2 = all_preds["softmax_pred"]
        return all_preds

    def _prepare_raw_batch(self, batch_preds, detach=True, move_to_cpu=True):
        def _process_tensor(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            result = tensor.detach() if detach else tensor
            return result.cpu() if move_to_cpu else result

        return {
            "image_id": list(batch_preds["image_id"]),
            "dataset": list(batch_preds["dataset"]),
            "gt": _process_tensor(batch_preds["gt"]),
            "softmax_pred": _process_tensor(batch_preds["softmax_pred"]),
            "softmax_pred_groups": [
                _process_tensor(tensor) for tensor in batch_preds["softmax_pred_groups"]
            ],
        }

    def collect_raw_predictions(
        self,
        max_batches=None,
        detach=True,
        move_to_cpu=True,
        show_progress=False,
        random_seed=None,
        n_resamples_for_largest=1,
        skip_saving=False,
    ):
        """Return unreduced prediction tensors for downstream analysis.
        
        Args:
            max_batches: Maximum number of batches to process
            detach: Whether to detach tensors from computation graph
            move_to_cpu: Whether to move tensors to CPU
            show_progress: Whether to show progress bar
            random_seed: Random seed for selecting images. If None, uses sequential order.
                        If non-negative integer, uses it as seed. If negative, samples a random seed.
            n_resamples_for_largest: Number of times to resample batches to select ones with largest label areas.
            skip_saving: If True, suppresses any folder creation or file saving (overrides the
                         instance-level skip_saving set at construction time).
        """
        import random
        import numpy as np

        # Honour skip_saving even if it wasn't set at construction time
        if skip_saving:
            self.skip_saving = True

        # Validate parameters
        if random_seed is None and n_resamples_for_largest > 1:
            raise ValueError(
                f"n_resamples_for_largest={n_resamples_for_largest} requires random_seed to be set. "
                "Please provide a random_seed value (non-negative integer or negative for auto-sampling)."
            )
        
        # Handle random_seed
        actual_seed = None
        if random_seed is not None:
            if random_seed < 0:
                # Sample a random seed
                actual_seed = random.randint(0, 2**31 - 1)
                print(f"[collect_raw_predictions] Sampled random seed: {actual_seed}")
            else:
                actual_seed = random_seed
        
        # Get the underlying dataset from the dataloader
        dataset = self.test_dataloader.dataset
        dataset_size = len(dataset)
        
        # Determine batch size
        batch_size = self.test_dataloader.batch_size
        
        # Determine how many images we need (for the final output)
        if max_batches is not None:
            n_batches_needed = max_batches
            n_images_needed = n_batches_needed * batch_size
        else:
            n_batches_needed = len(self.test_dataloader)
            n_images_needed = n_batches_needed * batch_size
        
        # Sample n_images_needed * n_resamples_for_largest candidates (like qualitative_plot_models)
        n_candidate_images = n_images_needed * n_resamples_for_largest
        
        # Sample indices
        if actual_seed is not None:
            # Random sampling with seed
            np.random.seed(actual_seed)
            random.seed(actual_seed)
            candidate_indices = np.random.choice(dataset_size, size=n_candidate_images, replace=True)
        else:
            # Sequential sampling (original behavior)
            candidate_indices = np.arange(min(n_candidate_images, dataset_size))
        
        # Load candidate samples
        candidate_samples = [dataset[int(idx)] for idx in candidate_indices]
        
        # Helper function to compute label area for a single sample
        def compute_label_area(sample):
            return sample['seg'].sum()
        
        # Select images based on n_resamples_for_largest
        if n_resamples_for_largest > 1:
            # Compute areas for all candidate samples
            areas = np.array([compute_label_area(sample) for sample in candidate_samples])
            # Reshape to (n_images_needed, n_resamples_for_largest)
            areas_matrix = areas.reshape(n_images_needed, n_resamples_for_largest)
            
            # Get indices of largest areas across resampling dimension
            largest_indices = np.argmax(areas_matrix, axis=1)
            
            # Reshape candidate_indices similarly and select based on largest_indices
            candidate_indices_matrix = candidate_indices.reshape(n_images_needed, n_resamples_for_largest)
            selected_indices = candidate_indices_matrix[np.arange(n_images_needed), largest_indices]
            
            # Load the selected samples
            selected_samples = [dataset[int(idx)] for idx in selected_indices]
        else:
            # Just use the first n_images_needed
            selected_samples = candidate_samples[:n_images_needed]
        
        # Collate individual images into batches
        from torch.utils.data import default_collate
        selected_batches = []
        for i in range(0, len(selected_samples), batch_size):
            batch_samples = selected_samples[i:i+batch_size]
            if batch_samples:
                batched = default_collate(batch_samples)
                selected_batches.append(batched)
        
        # Now process the selected batches
        iterator = selected_batches
        if show_progress:
            iterator = tqdm(iterator)
        
        results = []
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            for batch_idx, batch in enumerate(iterator):
                batch_preds = self._build_batch_predictions(batch)
                results.append(
                    self._prepare_raw_batch(
                        batch_preds,
                        detach=detach,
                        move_to_cpu=move_to_cpu,
                    )
                )
        finally:
            torch.set_grad_enabled(prev_grad_state)
        return results

    def save_results_dict(self):
        filename = os.path.join(self.save_dir, "metrics.json")
        mean_metrics_dict = {}
        for image_id, value in self.results_dict.items():
            for metric, score in value["metrics"].items():
                if metric not in mean_metrics_dict:
                    mean_metrics_dict[metric] = []
                mean_metrics_dict[metric].append(score)
        self.results_dict["mean"] = {}
        self.results_dict["mean"]["metrics"] = {}
        for metric, scores in mean_metrics_dict.items():
            self.results_dict["mean"]["metrics"][metric] = np.asarray(scores).mean()
        with open(filename, "w") as f:
            json.dump(self.results_dict, f, indent=2)

    def predict_cases(self):
        for batch in tqdm(self.test_dataloader):
            all_preds = self._build_batch_predictions(batch)
            self.process_output(all_preds)
        self.save_results_dict()


def run_test(args: Namespace) -> None:
    """
    Run test and save the results in the end
    Args:
        args: Arguments for testing, including checkpoint_path, test_data_dir and subject_ids.
              test_data_dir and subject_ids might be None.
    """
    torch.set_grad_enabled(False)
    tester = Tester(args)
    if tester.should_skip():
        print(f"[skip_existing] All expected outputs already exist for split='{tester.test_split}' (version={tester.version}, ckpt_tag={tester.checkpoint_subdir}). Skipping evaluation.")
        return
    tester.predict_cases()


def collect_raw_predictions_from_args(
    args: Namespace,
    max_batches=None,
    detach=True,
    move_to_cpu=True,
    show_progress=False,
):
    """Instantiate a Tester and return raw probability tensors without running the full eval loop."""
    torch.set_grad_enabled(False)
    tester = Tester(args)
    return tester.collect_raw_predictions(
        max_batches=max_batches,
        detach=detach,
        move_to_cpu=move_to_cpu,
        show_progress=show_progress,
    )


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
