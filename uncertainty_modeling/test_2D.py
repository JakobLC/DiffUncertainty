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
from uncertainty_modeling.test_3D import (
    test_cli,
    load_models_from_checkpoint,
    calculate_ged,
    calculate_uncertainty,
    calculate_one_minus_msr,
    prepare_evaluation_jobs,
)
import uncertainty_modeling.data.cityscapes_labels as cs_labels
from global_utils.checkpoint_format import format_checkpoint_subdir


class Tester:
    def __init__(self, args: Namespace):
        self.all_checkpoints = self.get_checkpoints(args.checkpoint_paths)
        hparams = self.all_checkpoints[0]["hyper_parameters"]
        set_seed(hparams["seed"])
        self.ignore_index = hparams["datamodule"]["ignore_index"]
        self.evaluate_all_raters = True#bool(hparams.get("evaluate_all_raters", True))
        self.test_batch_size = args.test_batch_size
        self.tta = args.tta
        self.test_dataloader = self.get_test_dataloader(args, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = load_models_from_checkpoint(
            self.all_checkpoints,
            device=self.device,
            use_ema=bool(getattr(args, "use_ema", False)),
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

    @staticmethod
    def get_checkpoints(checkpoint_paths):
        all_checkpoints = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint["hyper_parameters"]["MODEL"]["PRETRAINED"] = False
            conf = OmegaConf.create(checkpoint["hyper_parameters"])
            resolved = OmegaConf.to_container(conf, resolve=True)
            checkpoint["hyper_parameters"] = resolved
            all_checkpoints.append(checkpoint)
        return all_checkpoints

    @staticmethod
    def set_n_reference_samples(hparams, n_reference_samples):
        transforms_cfg = hparams["AUGMENTATIONS"].get("TEST", [])
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

    def get_test_dataloader(self, args: Namespace, hparams):
        data_input_dir = (
            args.data_input_dir
            if args.data_input_dir is not None
            else hparams["data_input_dir"]
        )
        if args.data_input_dir is not None:
            hparams["datamodule"]["dataset"]["splits_path"] = hparams["datamodule"][
                "dataset"
            ]["splits_path"].replace(hparams["data_input_dir"], args.data_input_dir)
        hparams = self.set_n_reference_samples(hparams, args.n_reference_samples)
        if self.test_batch_size:
            hparams["datamodule"]["val_batch_size"] = self.test_batch_size
        dm = hydra.utils.instantiate(
            hparams["datamodule"],
            data_input_dir=data_input_dir,
            augmentations=hparams["AUGMENTATIONS"],
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
            rater = rater.unsqueeze(0)
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
        compute_ged = self.evaluate_all_raters and all_preds["gt"].shape[1] > 1
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
            # If we ensemble multiple generative models (e.g., SSNs),
            # average the inner samples per model and only aggregate across models.
            multiple_generative = sum(getattr(m, "ssn", False) for m in self.models) > 1
            for model in self.models:
                if model.ssn:
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
                    for pred in range(self.n_pred):
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


"""
Code commented out because it was unused
    def process_image_prediction(
        self, all_preds, image_idx, image_preds, ignore_index_map
    ):
        image_id = all_preds["image_id"][image_idx]
        mean_softmax_pred = torch.mean(image_preds, dim=0)
        self.results_dict[image_id] = {"dataset": all_preds["dataset"][image_idx]}
        self.results_dict[image_id]["metrics"] = {}
        self.results_dict[image_id]["metrics"].update(
            self.calculate_test_metrics(mean_softmax_pred, all_preds["gt"][image_idx])
        )
        if self.evaluate_all_raters and all_preds["gt"][image_idx].shape[0] > 1:
            ged_ignore_index = self.ignore_index if self.ignore_index >= 0 else None
            self.results_dict[image_id]["metrics"].update(
                calculate_ged(
                    image_preds,
                    all_preds["gt"][image_idx],
                    ignore_index=ged_ignore_index,
                    additional_metrics=["dice", "major_dice", "max_dice_pred", "max_dice_gt"],
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
"""