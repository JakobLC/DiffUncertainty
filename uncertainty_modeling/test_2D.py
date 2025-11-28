import copy
import json
import os
from argparse import Namespace
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
#from torchmetrics.functional import dice
import torch.nn.functional as F
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


class Tester:
    def __init__(self, args: Namespace):
        checkpoint_paths = getattr(args, "checkpoint_paths", None)
        self.checkpoint_paths = list(checkpoint_paths) if checkpoint_paths is not None else []
        if not self.checkpoint_paths:
            raise ValueError("Tester requires at least one checkpoint path.")
        self.all_checkpoints = self.get_checkpoints(self.checkpoint_paths)
        hparams = self.all_checkpoints[0]["hyper_parameters"]
        set_seed(hparams["seed"])
        self.ignore_index = hparams["data"]["ignore_index"]
        #self.evaluate_all_raters = True#bool(hparams.get("evaluate_all_raters", True))
        self.skip_ged = args.skip_ged
        self.test_batch_size = args.test_batch_size
        self.tta = args.tta
        self.test_dataloader = self.get_test_dataloader(args, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.swag_blockwise = bool(getattr(args, "swag_blockwise", False))
        self.swag_low_rank_cov = bool(getattr(args, "swag_low_rank_cov", False))
        base_models = load_models_from_checkpoint(
            self.all_checkpoints,
            device=self.device,
            use_ema=bool(getattr(args, "use_ema", False)),
        )
        self.n_models = max(int(getattr(args, "n_models", 0) or 0), 0)
        self.models = self._maybe_expand_swag_models(
            base_models,
            self.all_checkpoints,
            self.checkpoint_paths,
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
        self.use_ema = bool(getattr(args, "use_ema", False))
        self.dataset_name = hparams.get("dataset") if isinstance(hparams, dict) else None
        self.checkpoint_epoch = self.all_checkpoints[0].get("epoch")+1 if self.all_checkpoints else None
        self.checkpoint_subdir = format_checkpoint_subdir(
            self.checkpoint_epoch, self.use_ema
        )
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

    @staticmethod
    def _normalize_swag_config(cfg):
        base = {
            "enabled": False,
            "snapshot_frequency": 1,
            "max_snapshots": 20,
            "min_variance": 1e-30,
            "diag_only": True,
        }
        if cfg is None:
            return base
        container = None
        if isinstance(cfg, dict):
            container = cfg
        else:
            try:
                container = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                try:
                    container = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
                except Exception:
                    container = None
        if isinstance(container, dict):
            for key in base:
                if key in container:
                    base[key] = container[key]
        base["snapshot_frequency"] = max(1, int(base["snapshot_frequency"]))
        base["max_snapshots"] = max(1, int(base["max_snapshots"]))
        base["min_variance"] = float(base["min_variance"])
        base["diag_only"] = bool(base["diag_only"])
        base["enabled"] = bool(base["enabled"])
        return base

    def _maybe_expand_swag_models(self, base_models, checkpoints, checkpoint_paths):
        if not base_models or not checkpoints:
            return base_models
        if self.n_models <= 0:
            return base_models
        expanded_models = []
        total = len(base_models)
        for idx in range(total):
            model = base_models[idx]
            checkpoint = checkpoints[idx] if idx < len(checkpoints) else None
            if checkpoint is None:
                expanded_models.append(model)
                continue
            swag_state = checkpoint.get("swag_state_dict")
            swag_config = checkpoint.get("swag_config") or checkpoint.get("hyper_parameters", {}).get("swag")
            ckpt_label = checkpoint_paths[idx] if idx < len(checkpoint_paths) else f"checkpoint_{idx}"
            swag_models = self._sample_swag_draws(model, swag_state, swag_config, ckpt_label)
            if swag_models:
                expanded_models.extend(swag_models)
            else:
                expanded_models.append(model)
        return expanded_models

    def _sample_swag_draws(self, template_model, swag_state, swag_config, checkpoint_label):
        if not swag_state or self.n_models <= 0:
            return []
        if isinstance(swag_state, dict) and "mean" in swag_state and "sq_mean" in swag_state:
            print(
                f"[SWAG] Checkpoint '{checkpoint_label}' uses an unsupported legacy SWAG format. "
                "Please regenerate checkpoints with the updated SWAG implementation."
            )
            return []
        config = self._normalize_swag_config(swag_config)
        swag = SWAG(
            diag_only=config["diag_only"],
            max_num_models=config["max_snapshots"],
            var_clamp=config["min_variance"],
        )
        swag.prepare(template_model)
        try:
            swag.load_state_dict(swag_state)
        except RuntimeError as exc:
            print(f"[SWAG] Failed to load statistics for '{checkpoint_label}': {exc}")
            return []
        snapshots = int(swag.n_models.item())
        if snapshots < 2:
            print(
                f"[SWAG] Checkpoint '{checkpoint_label}' has only {snapshots} snapshot(s); sampling skipped."
            )
            return []
        use_low_rank = self.swag_low_rank_cov and not config["diag_only"]
        if self.swag_low_rank_cov and config["diag_only"]:
            print(
                f"[SWAG] Checkpoint '{checkpoint_label}' stores diagonal-only stats; ignoring --swag-low-rank-cov."
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
            sampled_models.append(sampled_model.to(self.device))
        print(
            f"[SWAG] Sampled {len(sampled_models)} model(s) from '{checkpoint_label}' (n_snapshots={snapshots})."
        )
        return sampled_models

    def create_save_dirs(self):
        path_parts = [
            self.save_root_dir,
            self.exp_name,
            "test_results",
            self.version,
        ]
        if self.checkpoint_subdir:
            path_parts.append(self.checkpoint_subdir)
        path_parts.append(self.test_split)
        self.save_dir = os.path.join(*path_parts)
        print(f"saving to results dir {self.save_dir}")
        self.save_pred_dir = os.path.join(self.save_dir, "pred_seg")
        self.save_pred_prob_dir = os.path.join(self.save_dir, "pred_prob")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_pred_dir, exist_ok=True)
        # os.makedirs(self.save_pred_prob_dir, exist_ok=True)
        return

    def should_skip(self):
        if not self.skip_existing:
            return False
        metrics_path = os.path.join(self.save_dir, "metrics.json")
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

            if self.dataset_name and "lidc" in self.dataset_name.lower():
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
                output_np_color = np.zeros((*output_np.shape[:-1], 3), dtype=np.uint8)
                output_np[ignore_index_map.astype(bool)] = cs_labels.name2trainId[
                    "unlabeled"
                ]
                for k, v in cs_labels.trainId2color.items():
                    output_np_color[(output_np == k).squeeze(-1), :] = v
                output_np_color = cv2.cvtColor(output_np_color, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(self.save_pred_dir, f"{img_name}.png"), output_np_color
                )
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
        is_lidc = self.dataset_name and "lidc" in self.dataset_name.lower()

        # output_softmax: (C,H,W) for this image
        # Convert to predicted indices once
        pred_idx = output_softmax.argmax(dim=0)  # (H,W)

        if is_lidc:
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

    def process_output(self, all_preds):
        ignore_index_map = all_preds["gt"] == self.ignore_index
        compute_ged = not self.skip_ged and all_preds["gt"].shape[1] > 1
        ged_ignore_index = (
            self.ignore_index if compute_ged and self.ignore_index >= 0 else None
        )
        n_batch = all_preds["softmax_pred"].shape[1]
        for image_idx in range(n_batch):
            image_preds = all_preds["softmax_pred"][:, image_idx]  # (P,C,H,W)
            image_id = all_preds["image_id"][image_idx]
            mean_softmax_pred = torch.mean(image_preds, dim=0)
            self.results_dict[image_id] = {"dataset": all_preds["dataset"][image_idx]}
            self.results_dict[image_id]["metrics"] = {}
            self.results_dict[image_id]["metrics"].update(
                self.calculate_test_metrics(
                    mean_softmax_pred, all_preds["gt"][image_idx]
                )
            )
            if compute_ged:
                # Fast GED path for binary LIDC: compute with argmax on-device
                is_lidc = self.dataset_name and "lidc" in self.dataset_name.lower()
                if is_lidc:
                    fast_ged = ged_binary_fast(
                        image_preds,
                        all_preds["gt"][image_idx],
                        ignore_index=ged_ignore_index,
                        additional_metrics=["dice"],
                    )
                    self.results_dict[image_id]["metrics"].update(fast_ged)
                else:
                    self.results_dict[image_id]["metrics"].update(
                        calculate_ged(
                            image_preds,
                            all_preds["gt"][image_idx].to(self.device),
                            ignore_index=ged_ignore_index,
                            additional_metrics=["dice"],
                        )
                    )
            if image_preds.shape[0] > 1:
                uncertainty_dict = calculate_uncertainty(image_preds)
            else:
                uncertainty_dict = calculate_one_minus_msr(image_preds.squeeze(0))
            ignore_index_map_image = ignore_index_map[image_idx][0]
            self.save_prediction(
                image_id,
                image_preds,
                mean_softmax_pred,
                ignore_index_map_image.detach().long().cpu().numpy().astype(np.uint8),
            )
            self.save_uncertainty(image_id, uncertainty_dict)

    

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
            # dataloader_iterator = iter(self.test_dataloader)
            # for i in tqdm(range(2)):
            #     batch = next(dataloader_iterator)
            gt = batch["seg"]
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            all_preds = {
                "softmax_pred": [],
                "image_id": batch["image_id"],
                "gt": gt,
                "dataset": batch["dataset"],
            }
            # If we ensemble multiple generative models (e.g., SSNs, diffusion),
            # average the inner samples per model and only aggregate across models.
            multiple_generative = sum(1 for m in self.models if getattr(m, "is_generative", False)) > 1
            for model in self.models:
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
                        # Vectorized softmax over class dim (2), mean over inner samples
                        softmax_samples = F.softmax(output_samples, dim=2)
                        output_softmax_mean = softmax_samples.mean(dim=0)
                        all_preds["softmax_pred"].append(output_softmax_mean)
                    else:
                        for output_sample in output_samples:
                            output_softmax = F.softmax(output_sample, dim=1)
                            all_preds["softmax_pred"].append(output_softmax)
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
                        sample_list.append(F.softmax(sample_output, dim=1))
                    sample_stack = torch.stack(sample_list)
                    if multiple_generative:
                        all_preds["softmax_pred"].append(sample_stack.mean(dim=0))
                    else:
                        for sample in sample_stack:
                            all_preds["softmax_pred"].append(sample)
                elif self.tta:
                    for index, image in enumerate(batch["data"]):
                        output = model.forward(image.to(self.device))
                        output_softmax = F.softmax(output, dim=1)  # .to("cpu")
                        if any(
                            "HorizontalFlip" in sl for sl in batch["transforms"][index]
                        ):
                            # all_preds["softmax_pred"].append(output_softmax)
                            all_preds["softmax_pred"].append(
                                torch.flip(output_softmax, [-1])
                            )
                        else:
                            all_preds["softmax_pred"].append(output_softmax)
                else:
                    output = model.forward(batch["data"].to(self.device))
                    output_softmax = F.softmax(output, dim=1)  # .to("cpu")
                    all_preds["softmax_pred"].append(output_softmax)
            all_preds["softmax_pred"] = torch.stack(all_preds["softmax_pred"])
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
