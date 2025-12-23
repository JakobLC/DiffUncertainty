import json
import math
import os
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np

from evaluation.metrics.dice_wrapped import dice
from evaluation.metrics.ged_fast import ged_binary_fast
from global_utils.checkpoint_format import format_checkpoint_subdir
import uncertainty_modeling.data.cityscapes_labels as cs_labels
from uncertainty_modeling.main import set_seed
from uncertainty_modeling.unc_mod_utils.test_utils import (
    calculate_ged,
    calculate_one_minus_msr,
    calculate_uncertainty,
    load_models_from_checkpoint,
    prepare_evaluation_jobs,
    test_cli,
)


def _register_qualitative_args(parser) -> None:
    parser.add_argument(
        "--quantile_interval",
        type=str,
        default="0,1",
        help=(
            "Comma separated lower,upper quantiles to sample from. "
            "The default '0,1' selects a uniformly random image."
        ),
    )
    parser.add_argument(
        "--quantile_metric",
        type=str,
        default="dice",
        help=(
            "Metric key in metrics.json used to rank images when selecting from quantile intervals."
        ),
    )
    parser.add_argument(
        "--save_artifact",
        action="store_true",
        default=False,
        help="Persist the sampled predictions/metadata bundle to disk.",
    )
    parser.add_argument(
        "--save_plot_png",
        action="store_true",
        default=False,
        help="Save the rendered qualitative figure as a PNG file.",
    )
    parser.add_argument(
        "--show_plot",
        dest="show_plot",
        action="store_true",
        default=True,
        help="Display the qualitative figure using an interactive backend.",
    )
    parser.add_argument(
        "--no_show_plot",
        dest="show_plot",
        action="store_false",
        help="Disable figure display and force a non-interactive backend.",
    )


class QualitativeInspector:
    """Loads checkpoints and quickly evaluates a single image for qualitative inspection."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.all_checkpoints = self._load_checkpoints(args.checkpoint_paths)
        if not self.all_checkpoints:
            raise ValueError("No checkpoints found for qualitative inspection.")
        self.hparams = self.all_checkpoints[0]["hyper_parameters"]
        set_seed(self.hparams["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ignore_index = self.hparams["datamodule"].get("ignore_index", 0)
        self.dataset_name = self.hparams.get("dataset")
        self.test_batch_size = args.test_batch_size
        self.tta = bool(args.tta)
        self.n_pred = args.n_pred
        self.n_reference_samples = args.n_reference_samples
        self.test_split = args.test_split
        self.save_root_dir = args.save_dir if args.save_dir is not None else self.hparams["save_dir"]
        self.exp_name = args.exp_name if args.exp_name is not None else self.hparams["exp_name"]
        self.version = str(getattr(args, "version_override", None) or self.hparams["version"])
        self.use_ema = bool(getattr(args, "use_ema", False))
        self.checkpoint_epoch = self.all_checkpoints[0].get("epoch", None)
        if self.checkpoint_epoch is not None:
            self.checkpoint_epoch += 1
        self.checkpoint_subdir = format_checkpoint_subdir(self.checkpoint_epoch, self.use_ema)
        self.random_state = random.Random(self.hparams.get("seed", 123))
        self.quantile_bounds = self._parse_quantile_interval(args.quantile_interval)
        self.quantile_metric = args.quantile_metric
        self.save_artifact = bool(getattr(args, "save_artifact", False))
        self.save_plot_png = bool(getattr(args, "save_plot_png", False))
        self.show_plot = bool(getattr(args, "show_plot", True))
        self._output_dir: Optional[str] = None

        self.test_dataloader = self._build_test_dataloader(args)
        self.models = load_models_from_checkpoint(
            self.all_checkpoints,
            device=self.device,
            use_ema=self.use_ema,
        )
        self.num_classes = getattr(self.models[0], "num_classes", None)
        if self.num_classes is None:
            raise ValueError("Unable to determine number of classes from loaded model.")
        if self.save_artifact or self.save_plot_png:
            self._output_dir = self._create_output_dir()
        self._color_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _load_checkpoints(checkpoint_paths: List[str]) -> List[Dict[str, Any]]:
        checkpoints: List[Dict[str, Any]] = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            checkpoint["hyper_parameters"]["MODEL"]["PRETRAINED"] = False
            if checkpoint["hyper_parameters"].__class__.__name__ == "AttributeDict":
                checkpoint["hyper_parameters"] = dict(checkpoint["hyper_parameters"])
            conf = OmegaConf.create(checkpoint["hyper_parameters"])
            resolved = OmegaConf.to_container(conf, resolve=True)
            checkpoint["hyper_parameters"] = resolved
            checkpoints.append(checkpoint)
        return checkpoints

    def _build_test_dataloader(self, args: Namespace):
        hparams = self.hparams
        data_input_dir = args.data_input_dir if args.data_input_dir is not None else hparams["data_input_dir"]
        if args.data_input_dir is not None:
            dm_conf = hparams["datamodule"]["dataset"]
            dm_conf["splits_path"] = dm_conf["splits_path"].replace(hparams["data_input_dir"], args.data_input_dir)
        hparams = self._set_n_reference_samples(hparams, self.n_reference_samples)
        if self.test_batch_size:
            hparams["datamodule"]["val_batch_size"] = self.test_batch_size
        datamodule = hydra.utils.instantiate(
            hparams["datamodule"],
            data_input_dir=data_input_dir,
            augmentations=hparams["AUGMENTATIONS"],
            seed=hparams["seed"],
            test_split=self.test_split,
            tta=self.tta,
            _recursive_=False,
        )
        datamodule.setup("test")
        return datamodule.test_dataloader()

    @staticmethod
    def _set_n_reference_samples(hparams: Mapping[str, Any], n_reference_samples: int):
        transforms_cfg = hparams["AUGMENTATIONS"].get("TEST", [])
        for compose in transforms_cfg:
            compose_cfg = compose.get("Compose")
            if not compose_cfg:
                continue
            for transform in compose_cfg.get("transforms", []):
                if "StochasticLabelSwitches" in transform:
                    transform["StochasticLabelSwitches"]["n_reference_samples"] = n_reference_samples
        return hparams

    def _create_output_dir(self) -> str:
        parts = [self.save_root_dir, self.exp_name, "qualitative_inspection", self.version]
        if self.checkpoint_subdir:
            parts.append(self.checkpoint_subdir)
        parts.append(self.test_split)
        target_dir = os.path.join(*parts)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    @staticmethod
    def _primary_tensor(batch_data: Any) -> torch.Tensor:
        """Return the primary tensor from either a tensor or a TTA list."""
        if isinstance(batch_data, list):
            if not batch_data:
                raise ValueError("Encountered an empty data list for the selected sample.")
            return batch_data[0]
        if not isinstance(batch_data, torch.Tensor):
            raise TypeError("Expected batch data to be a Tensor or list of Tensors.")
        return batch_data

    def _get_class_colors(self, num_classes: int) -> torch.Tensor:
        cached = self._color_cache.get(num_classes)
        if cached is not None:
            return cached
        colors: List[List[float]] = []
        dataset_lower = str(self.dataset_name or "").lower()
        if num_classes == 2 and ("lidc" in dataset_lower or "binary" in dataset_lower):
            colors = [[0, 0, 0], [255, 255, 255]]
        else:
            sorted_ids = sorted(cs_labels.trainId2color.keys())
            for idx in range(min(num_classes, len(sorted_ids))):
                colors.append(list(cs_labels.trainId2color[sorted_ids[idx]]))
        if len(colors) < num_classes:
            import matplotlib.cm as cm

            cmap = cm.get_cmap("tab20", num_classes)
            for idx in range(len(colors), num_classes):
                rgba = cmap(idx)
                colors.append([rgba[0] * 255, rgba[1] * 255, rgba[2] * 255])
        palette = torch.tensor(colors, dtype=torch.float32) / 255.0
        self._color_cache[num_classes] = palette
        return palette

    def _probabilities_to_rgb(self, prob_map: torch.Tensor) -> np.ndarray:
        prob = prob_map.clamp(0.0, 1.0)
        colors = self._get_class_colors(prob.shape[0]).to(prob.device)
        rgb = torch.zeros(3, prob.shape[1], prob.shape[2], device=prob.device)
        for cls_idx in range(prob.shape[0]):
            rgb += colors[cls_idx].view(3, 1, 1) * prob[cls_idx]
        rgb = rgb.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        return rgb

    def _labels_to_rgb(self, label_map: torch.Tensor) -> np.ndarray:
        labels = label_map.clone().long()
        if self.ignore_index is not None and self.ignore_index >= 0:
            labels[labels == self.ignore_index] = 0
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        if labels.ndim == 4:
            labels = labels.squeeze(1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(2, 0, 1).float()
        return self._probabilities_to_rgb(one_hot)

    @staticmethod
    def _normalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
        tensor = image_tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] >= 3:
            tensor = tensor[:3]
        else:
            tensor = torch.cat([tensor, tensor[0:1].repeat(3 - tensor.shape[0], 1, 1)], dim=0)
        tensor = tensor.float()
        tensor -= tensor.min()
        denom = tensor.max().clamp_min(1e-6)
        tensor /= denom
        return tensor.permute(1, 2, 0).clamp(0.0, 1.0).numpy()

    @staticmethod
    def _entropy_from_prob(prob: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        safe_prob = prob.clamp(min=eps)
        entropy = -(safe_prob * safe_prob.log()).sum(dim=0)
        return entropy

    def _prepare_plot_payload(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        member_samples: List[torch.Tensor] = [tensor.clone() for tensor in blob["member_samples"]]
        row_means = [samples.mean(dim=0) for samples in member_samples]
        row_entropies = [self._entropy_from_prob(mean) for mean in row_means]
        full_mean = torch.stack(row_means).mean(dim=0)
        aleatoric_map = torch.stack(row_entropies).mean(dim=0)
        total_uncertainty = self._entropy_from_prob(full_mean)
        epistemic_map = (total_uncertainty - aleatoric_map).clamp(min=0.0)
        entropy_fields = row_entropies + [aleatoric_map, total_uncertainty, epistemic_map]
        entropy_vmax = max((field.max().item() for field in entropy_fields), default=1.0)
        entropy_vmax = max(entropy_vmax, 1e-6)
        payload = {
            "member_samples": member_samples,
            "row_means": row_means,
            "row_entropies": row_entropies,
            "full_mean": full_mean,
            "aleatoric": aleatoric_map,
            "total_uncertainty": total_uncertainty,
            "epistemic": epistemic_map,
            "entropy_vmax": entropy_vmax,
        }
        return payload

    @staticmethod
    def _split_ground_truth_slices(gt_tensor: torch.Tensor) -> List[torch.Tensor]:
        tensor = gt_tensor.clone()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            return [tensor[idx] for idx in range(tensor.shape[0])]
        return [tensor.squeeze(0)]

    def _render_plot(self, blob: Dict[str, Any], save_png: bool) -> Optional[str]:
        payload = self._prepare_plot_payload(blob)
        member_samples: List[torch.Tensor] = payload["member_samples"]
        max_samples = max(1, max(samples.shape[0] for samples in member_samples))
        num_rows = len(member_samples) + 2  # top row + aggregated row
        total_cols = max_samples + 8

        import matplotlib
        import sys

        if not self.show_plot:
            if "matplotlib.pyplot" in sys.modules:
                import matplotlib.pyplot as plt

                plt.switch_backend("Agg")
            else:
                matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            num_rows,
            total_cols,
            figsize=(1.6 * total_cols, 1.6 * num_rows),
            squeeze=False,
        )

        input_image = self._normalize_image_tensor(blob["input_tensor"])
        gt_slices = self._split_ground_truth_slices(blob["ground_truth"])
        max_gt_slots = max(1, total_cols - 1)
        gt_slices = gt_slices[:max_gt_slots]

        entropy_kwargs = {
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": payload["entropy_vmax"],
        }

        col_arrow_E = max_samples
        col_mean = max_samples + 1
        col_arrow_H = max_samples + 2
        col_entropy = max_samples + 3
        col_arrow_total = max_samples + 4
        col_total = max_samples + 5
        col_arrow_epistemic = max_samples + 6
        col_epistemic = max_samples + 7

        # Top row: image + ground truths
        axes[0, 0].imshow(input_image)
        axes[0, 0].set_title("Image")
        axes[0, 0].axis("off")
        for idx, gt_slice in enumerate(gt_slices):
            col = 1 + idx
            axes[0, col].imshow(self._labels_to_rgb(gt_slice))
            axes[0, col].set_title(f"GT{idx + 1}")
            axes[0, col].axis("off")
        for col in range(1 + len(gt_slices), total_cols):
            axes[0, col].axis("off")

        # Rows per epistemic member
        for row_idx, samples in enumerate(member_samples):
            row = 1 + row_idx
            axes[row, 0].set_ylabel(f"EU{row_idx + 1}", rotation=0, labelpad=40, va="center")
            for sample_idx in range(max_samples):
                ax = axes[row, sample_idx]
                ax.axis("off")
                if sample_idx >= samples.shape[0]:
                    continue
                ax.imshow(self._probabilities_to_rgb(samples[sample_idx]))
            axes[row, col_arrow_E].axis("off")
            axes[row, col_arrow_E].text(0.5, 0.5, r"$\mathbb{E}\rightarrow$", ha="center", va="center")
            axes[row, col_mean].imshow(self._probabilities_to_rgb(payload["row_means"][row_idx]))
            axes[row, col_mean].set_title("Mean" if row_idx == 0 else "")
            axes[row, col_mean].axis("off")
            axes[row, col_arrow_H].axis("off")
            axes[row, col_arrow_H].text(0.5, 0.5, "H→", ha="center", va="center")
            axes[row, col_entropy].imshow(payload["row_entropies"][row_idx].detach().cpu().numpy(), **entropy_kwargs)
            axes[row, col_entropy].set_title("Entropy" if row_idx == 0 else "")
            axes[row, col_entropy].axis("off")
            axes[row, col_arrow_total].axis("off")
            axes[row, col_arrow_total].text(0.5, 0.5, r"$\mathbb{E}\downarrow$", ha="center", va="center")
            for col in (col_total, col_arrow_epistemic, col_epistemic):
                axes[row, col].axis("off")

        agg_row = num_rows - 1
        for col in range(max_samples):
            axes[agg_row, col].axis("off")
        axes[agg_row, col_mean].imshow(self._probabilities_to_rgb(payload["full_mean"]))
        axes[agg_row, col_mean].set_title("Full mean")
        axes[agg_row, col_mean].axis("off")

        axes[agg_row, col_entropy].imshow(payload["aleatoric"].detach().cpu().numpy(), **entropy_kwargs)
        axes[agg_row, col_entropy].set_title("Aleatoric unc.")
        axes[agg_row, col_entropy].axis("off")

        axes[agg_row, col_arrow_total].axis("off")
        axes[agg_row, col_arrow_total].text(0.5, 0.5, "H→", ha="center", va="center")

        axes[agg_row, col_total].imshow(payload["total_uncertainty"].detach().cpu().numpy(), **entropy_kwargs)
        axes[agg_row, col_total].set_title("Total unc.")
        axes[agg_row, col_total].axis("off")

        axes[agg_row, col_arrow_epistemic].axis("off")
        axes[agg_row, col_arrow_epistemic].text(0.5, 0.5, "→", ha="center", va="center")

        axes[agg_row, col_epistemic].imshow(payload["epistemic"].detach().cpu().numpy(), **entropy_kwargs)
        axes[agg_row, col_epistemic].set_title("Epistemic unc.")
        axes[agg_row, col_epistemic].axis("off")

        metric_value = blob["metrics"].get(blob["quantile_metric"])
        metric_text = "n/a" if metric_value is None else f"{metric_value:.3f}"
        selection_mode = blob["selection"].get("mode", "random")
        fig.suptitle(
            f"Sample {blob['image_id']} | mode={selection_mode} | {blob['quantile_metric']}={metric_text}",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        png_path: Optional[str] = None
        if save_png:
            if self._output_dir is None:
                self._output_dir = self._create_output_dir()
            png_path = os.path.join(self._output_dir, f"{blob['image_id']}.png")
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
        if self.show_plot:
            plt.show()
        plt.close(fig)
        return png_path

    @staticmethod
    def _parse_quantile_interval(raw: str) -> Tuple[float, float]:
        tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
        if len(tokens) != 2:
            raise ValueError("quantile_interval must be formatted as 'low,high'.")
        low, high = float(tokens[0]), float(tokens[1])
        if not (0.0 <= low < high <= 1.0):
            raise ValueError("quantile_interval bounds must satisfy 0 <= low < high <= 1.")
        return low, high

    def _metrics_path(self) -> str:
        parts = [self.save_root_dir, self.exp_name, "test_results", self.version]
        if self.checkpoint_subdir:
            parts.append(self.checkpoint_subdir)
        parts.append(self.test_split)
        parts.append("metrics.json")
        return os.path.join(*parts)

    def _load_metrics(self) -> Dict[str, Any]:
        path = self._metrics_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Requested quantile-based sampling but metrics file '{path}' does not exist."
            )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _sample_image_from_metrics(self) -> Tuple[str, Optional[float]]:
        metrics = self._load_metrics()
        entries: List[Tuple[str, float]] = []
        for image_id, payload in metrics.items():
            if image_id == "mean":
                continue
            metric_value = payload.get("metrics", {}).get(self.quantile_metric)
            if metric_value is None:
                continue
            entries.append((image_id, float(metric_value)))
        if not entries:
            raise ValueError(
                f"Metric '{self.quantile_metric}' was not found in metrics.json for any image."
            )
        entries.sort(key=lambda x: x[1])
        low, high = self.quantile_bounds
        total = len(entries)
        low_idx = min(math.floor(low * total), total - 1)
        high_idx = min(max(math.ceil(high * total) - 1, low_idx), total - 1)
        candidate_slice = entries[low_idx : high_idx + 1]
        if not candidate_slice:
            raise RuntimeError(
                f"No samples available in quantile range {low}-{high} for metric '{self.quantile_metric}'."
            )
        chosen_id, chosen_metric = self.random_state.choice(candidate_slice)
        return chosen_id, chosen_metric

    def _extract_single_item(self, batch: Dict[str, Any], index: int) -> Dict[str, Any]:
        single: Dict[str, Any] = {}
        data = batch["data"]
        if isinstance(data, list):
            single["data"] = [tensor[index : index + 1].clone() for tensor in data]
        else:
            single["data"] = data[index : index + 1].clone()
        single["seg"] = batch["seg"][index : index + 1].clone()
        single["image_id"] = [batch["image_id"][index]]
        dataset_entry = None
        if "dataset" in batch:
            dataset_entry = batch["dataset"][index]
        single["dataset"] = [dataset_entry]
        if "transforms" in batch:
            transforms = batch["transforms"]
            if isinstance(data, list) and isinstance(transforms, list) and len(transforms) == len(data):
                single["transforms"] = list(transforms)
            elif isinstance(transforms, list):
                single["transforms"] = [transforms[index]]
            else:
                single["transforms"] = [transforms]
        return single

    def _reservoir_sample(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        selected_sample: Optional[Dict[str, Any]] = None
        selected_id: Optional[str] = None
        total_seen = 0
        for batch in self.test_dataloader:
            batch_size = len(batch["image_id"])
            for local_idx in range(batch_size):
                total_seen += 1
                if self.random_state.random() <= 1.0 / total_seen:
                    selected_sample = self._extract_single_item(batch, local_idx)
                    selected_id = selected_sample["image_id"][0]
        if selected_sample is None or selected_id is None:
            raise RuntimeError("Could not sample an image from the test split.")
        metadata = {"mode": "random", "seen": total_seen}
        return selected_id, selected_sample, metadata

    def _locate_image(self, target_image_id: str) -> Dict[str, Any]:
        for batch in self.test_dataloader:
            for local_idx, image_id in enumerate(batch["image_id"]):
                if image_id == target_image_id:
                    return self._extract_single_item(batch, local_idx)
        raise ValueError(f"Image id '{target_image_id}' was not found in the dataloader.")

    def _select_sample(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        if self.quantile_bounds == (0.0, 1.0):
            return self._reservoir_sample()
        image_id, metric_value = self._sample_image_from_metrics()
        sample = self._locate_image(image_id)
        metadata = {
            "mode": "quantile",
            "metric": self.quantile_metric,
            "metric_value": metric_value,
            "interval": self.quantile_bounds,
        }
        return image_id, sample, metadata

    def _generate_softmax_samples(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        member_samples: List[torch.Tensor] = []
        tta_payload = sample["data"] if isinstance(sample["data"], list) else None
        for model in self.models:
            au_type = getattr(model, "AU_type", "softmax")
            outputs: List[torch.Tensor] = []
            if au_type == "ssn":
                data_tensor = self._primary_tensor(sample["data"]).to(self.device)
                distribution, cov_failed_flag = model.forward(data_tensor)
                if cov_failed_flag:
                    raise RuntimeError("Covariance matrix was not positive definite for SSN model.")
                output_samples = distribution.sample([self.n_pred])
                output_samples = output_samples.view(
                    [
                        self.n_pred,
                        data_tensor.shape[0],
                        model.num_classes,
                        *data_tensor.shape[2:],
                    ]
                )
                output_samples = F.softmax(output_samples, dim=2)
                for sample_tensor in output_samples:
                    outputs.append(sample_tensor[0].detach().cpu())
            elif au_type == "diffusion":
                inputs = self._primary_tensor(sample["data"]).to(self.device).float()
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
                    outputs.append(F.softmax(sample_output, dim=1)[0].detach().cpu())
            elif self.tta and tta_payload is not None:
                transforms = sample.get("transforms") or []
                for index, image in enumerate(tta_payload):
                    output = model.forward(image.to(self.device))
                    output_softmax = F.softmax(output, dim=1)
                    flipped = False
                    if index < len(transforms):
                        transform_desc = transforms[index]
                        if isinstance(transform_desc, list) and any(
                            "HorizontalFlip" in spec for spec in transform_desc
                        ):
                            flipped = True
                    tensor = torch.flip(output_softmax, dims=[-1]) if flipped else output_softmax
                    outputs.append(tensor[0].detach().cpu())
            else:
                data_tensor = self._primary_tensor(sample["data"]).to(self.device)
                output = model.forward(data_tensor)
                outputs.append(F.softmax(output, dim=1)[0].detach().cpu())
            if not outputs:
                raise RuntimeError("Model did not produce any predictions for the selected image.")
            member_samples.append(torch.stack(outputs))
        concatenated = torch.cat(member_samples, dim=0)
        return concatenated, member_samples

    def _calculate_metrics(
        self,
        mean_softmax: torch.Tensor,
        ground_truth: torch.Tensor,
        all_samples: torch.Tensor,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        pred_idx = mean_softmax.argmax(dim=0)
        gt = ground_truth.squeeze(0)
        is_lidc = bool(self.dataset_name and "lidc" in str(self.dataset_name).lower())
        if is_lidc:
            if gt.ndim == 4:
                gt = gt.squeeze(1)
            ignore = self.ignore_index
            pred_rep = pred_idx.unsqueeze(0).expand_as(gt)
            valid = (gt != ignore)
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
            metrics["dice"] = float(sum(per_rater) / len(per_rater))
        if gt.shape[0] > 1 and not self.args.skip_ged:
            ignore_index = self.ignore_index if self.ignore_index >= 0 else None
            if is_lidc:
                ged_result = ged_binary_fast(
                    all_samples,
                    gt,
                    ignore_index=ignore_index,
                    additional_metrics=["dice"],
                )
            else:
                ged_result = calculate_ged(
                    all_samples.to(self.device),
                    gt.to(self.device),
                    ignore_index=ignore_index,
                    additional_metrics=["dice"],
                )
            metrics.update({k: float(v) for k, v in ged_result.items()})
        return metrics

    def _build_result_blob(
        self,
        image_id: str,
        sample: Dict[str, Any],
        predictions: torch.Tensor,
        member_samples: List[torch.Tensor],
        selection_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        mean_softmax = predictions.mean(dim=0)
        if predictions.shape[0] > 1:
            uncertainty = calculate_uncertainty(predictions)
        else:
            uncertainty = calculate_one_minus_msr(predictions.squeeze(0))
        metrics = self._calculate_metrics(mean_softmax, sample["seg"], predictions)
        result = {
            "image_id": image_id,
            "dataset": sample["dataset"][0],
            "softmax_samples": predictions,
            "mean_softmax": mean_softmax,
            "ground_truth": sample["seg"],
            "member_samples": member_samples,
            "input_tensor": self._primary_tensor(sample["data"]).detach().cpu(),
            "metrics": metrics,
            "selection": selection_meta,
            "quantile_bounds": self.quantile_bounds,
            "quantile_metric": self.quantile_metric,
            "checkpoint_paths": [Path(p).as_posix() for p in self.args.checkpoint_paths],
        }
        if isinstance(uncertainty, dict):
            result["uncertainty_maps"] = {k: v.detach().cpu() for k, v in uncertainty.items()}
        else:
            result["uncertainty_maps"] = {"one_minus_msr": uncertainty.detach().cpu()}
        return result

    def _persist_result(self, blob: Dict[str, Any]) -> str:
        if self._output_dir is None:
            self._output_dir = self._create_output_dir()
        target_path = os.path.join(self._output_dir, f"{blob['image_id']}.pt")
        torch.save(blob, target_path)
        meta = {
            "image_id": blob["image_id"],
            "metrics": blob["metrics"],
            "selection": blob["selection"],
            "quantile_bounds": blob["quantile_bounds"],
            "quantile_metric": blob["quantile_metric"],
        }
        json_path = os.path.join(self._output_dir, f"{blob['image_id']}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return target_path

    def run(self) -> Dict[str, Any]:
        torch.set_grad_enabled(False)
        image_id, sample, selection_meta = self._select_sample()
        predictions, member_samples = self._generate_softmax_samples(sample)
        result_blob = self._build_result_blob(
            image_id,
            sample,
            predictions,
            member_samples,
            selection_meta,
        )
        artifact_path: Optional[str] = None
        if self.save_artifact:
            artifact_path = self._persist_result(result_blob)
            print(f"Saved qualitative sample '{image_id}' to {artifact_path}")
        png_path: Optional[str] = None
        if self.save_plot_png:
            png_path = self._render_plot(result_blob, save_png=True)
        else:
            self._render_plot(result_blob, save_png=False)
        if png_path:
            print(f"Saved qualitative figure to {png_path}")
        return result_blob


def main() -> None:
    args = test_cli(extra_args_fn=_register_qualitative_args)
    jobs = prepare_evaluation_jobs(args)
    if len(jobs) != 1:
        raise ValueError(
            "qual_plot.py currently supports exactly one job. "
            "Provide a single checkpoint combination / split / ema selection."
        )
    inspector = QualitativeInspector(jobs[0])
    inspector.run()


if __name__ == "__main__":
    main()
