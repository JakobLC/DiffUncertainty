from __future__ import annotations

import argparse
import copy
import json
import math
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from evaluation.metrics.dice_wrapped import dice
from uncertainty_modeling.lightning_experiment import LightningExperiment
from uncertainty_modeling.models.masked_subensemble import (
    MaskedConv2d,
    MaskedLinear,
    combined_mask_logits,
    freeze_unmasked_parameters,
    initialize_mask_logits_for_target_fraction,
    iter_masked_layers,
    mean_pairwise_iou,
    replace_with_masked_layers,
    set_normalize_mode,
    set_rows_only_mode,
    set_active_submodel,
    submodel_size_penalty,
)


def _checkpoint_to_dict_hparams(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None:
        raise KeyError("checkpoint is missing 'hyper_parameters'")
    if hparams.__class__.__name__ == "AttributeDict":
        hparams = dict(hparams)
    cfg = OmegaConf.create(hparams)
    return OmegaConf.to_container(cfg, resolve=True)


def _infer_experiment_dir(ckpt_path: Path) -> Path:
    parent = ckpt_path.parent
    if parent.name in {"checkpoints", "scheduled_ckpts"}:
        return parent.parent
    return parent


def _build_dest_root(source_ckpt: Path, dest_version: str) -> Path:
    src_root = _infer_experiment_dir(source_ckpt)
    return src_root.parent / dest_version


def _apply_destination_hparams_identity(
    hparams: dict[str, Any],
    dest_root: Path,
    explicit_dest_root: bool,
) -> dict[str, Any]:
    """Align save_dir/exp_name/version in extracted checkpoints with destination layout.

    Expected destination layout is: save_dir/exp_name/version.
    """
    updated = copy.deepcopy(hparams)
    updated["version"] = dest_root.name
    if explicit_dest_root:
        # If --dest_root is passed, treat it as canonical experiment/version path.
        if dest_root.parent != dest_root:
            updated["exp_name"] = dest_root.parent.name
        if dest_root.parent.parent != dest_root.parent:
            updated["save_dir"] = dest_root.parent.parent.as_posix()
    return updated


def _instantiate_datamodule(hparams: dict[str, Any]):
    if "data" not in hparams:
        raise KeyError("checkpoint hparams do not contain 'data' config")
    seed = hparams.get("seed", None)
    dm = hydra.utils.instantiate(hparams["data"], seed=seed, _recursive_=False)
    dm.prepare_data()
    dm.setup("fit")
    return dm


def _instantiate_id_test_datamodule(hparams: dict[str, Any]):
    if "data" not in hparams:
        raise KeyError("checkpoint hparams do not contain 'data' config")
    seed = hparams.get("seed", None)
    data_cfg = copy.deepcopy(hparams["data"])
    if not isinstance(data_cfg, dict):
        raise TypeError("hparams['data'] must resolve to a dict")
    data_cfg["test_split"] = "id"
    dm = hydra.utils.instantiate(data_cfg, seed=seed, _recursive_=False)
    dm.prepare_data()
    dm.setup("test")
    return dm


def _maybe_override_train_loader(train_loader, batch_size_override: int):
    if int(batch_size_override) <= 0:
        return train_loader
    if not hasattr(train_loader, "dataset"):
        raise TypeError("train_loader has no dataset attribute; cannot apply batch size override")

    dataset = train_loader.dataset
    shuffle = isinstance(getattr(train_loader, "sampler", None), RandomSampler)
    return DataLoader(
        dataset,
        batch_size=int(batch_size_override),
        shuffle=shuffle,
        num_workers=int(getattr(train_loader, "num_workers", 0)),
        pin_memory=bool(getattr(train_loader, "pin_memory", False)),
        drop_last=bool(getattr(train_loader, "drop_last", False)),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )


def _dtype_to_precision_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


def _instantiate_lightning_experiment(hparams: dict[str, Any]) -> LightningExperiment:
    cfg = OmegaConf.create(hparams)
    init_kwargs = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(init_kwargs, dict):
        raise TypeError("Resolved hparams must be a dict")
    return LightningExperiment(cfg, **init_kwargs)


def _load_source_checkpoint(source_ckpt_path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    hparams = _checkpoint_to_dict_hparams(checkpoint)
    return checkpoint, hparams


def _remove_dropout_from_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of hparams with model dropout disabled for extraction."""
    updated = copy.deepcopy(hparams)
    model_cfg = updated.get("model")
    if not isinstance(model_cfg, dict):
        return updated

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                key_l = str(key).lower()
                if key_l in {"dropout", "dropout_rate"}:
                    node[key] = 0.0
                    continue
                if key_l == "dropout_cfg":
                    node[key] = {
                        "enabled": False,
                        "probability": [0.0],
                        "encoder": False,
                        "mid": False,
                        "decoder": False,
                        "skip_connections": False,
                        "residual_connections": False,
                        "per_block": True,
                    }
                    continue
                _visit(value)
            return
        if isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(model_cfg)
    return updated


def _compute_softmax_predictions(model: torch.nn.Module, batch: dict[str, Any], au_type: str) -> torch.Tensor | None:
    x = batch["data"]
    if isinstance(x, list):
        x = x[0]
    x = x.float()

    if au_type == "softmax":
        logits = model(x)
        return torch.softmax(logits, dim=1)

    if au_type == "ssn":
        output = model(x)
        if isinstance(output, tuple):
            distribution = output[0]
        else:
            distribution = output
        if hasattr(distribution, "mean"):
            mean = distribution.mean.view(x.shape[0], model.num_classes, *x.shape[2:])
            return torch.softmax(mean, dim=1)
        return None

    if au_type == "diffusion" and hasattr(model, "diffusion_sample_loop"):
        x_init = torch.randn(
            (x.shape[0], model.num_classes, *x.shape[2:]),
            device=x.device,
            dtype=x.dtype,
        )
        with torch.no_grad():
            sample = model.diffusion_sample_loop(
                x_init=x_init,
                im=x,
                num_steps=int(getattr(model, "diffusion_num_steps", 50)),
                sampler_type=getattr(model, "diffusion_sampler_type", "ddpm") or "ddpm",
                clip_x=False,
                guidance_weight=0.0,
                progress_bar=False,
                self_cond=False,
            )
        if sample.ndim == 4:
            return torch.softmax(sample, dim=1)
        return None

    if au_type == "prob_unet" and hasattr(model, "forward"):
        with torch.no_grad():
            logits = model(x, segm=None, training=False)
        if logits.ndim == 4:
            return torch.softmax(logits, dim=1)
        return None

    return None


def _compute_mean_dice_for_submodels(
    model: torch.nn.Module,
    batch: dict[str, Any],
    num_submodels: int,
    device: torch.device,
    au_type: str,
    ignore_index: int,
) -> tuple[float | None, torch.Tensor | None]:
    target = batch["seg"].to(device)
    if target.ndim == 4 and target.shape[1] > 1:
        target = target[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.long()

    values = []
    preview_probs = None
    for mask_idx in range(num_submodels):
        set_active_submodel(model, mask_idx=mask_idx, temp=1.0)
        probs = _compute_softmax_predictions(model, batch, au_type=au_type)
        if probs is None:
            return None, None
        if preview_probs is None:
            preview_probs = probs.detach()
        d = dice(
            probs,
            target,
            num_classes=int(probs.shape[1]),
            ignore_index=ignore_index,
            binary_dice=int(probs.shape[1]) == 2,
            is_softmax=True,
        )
        values.append(float(d))
    return float(sum(values) / max(len(values), 1)), preview_probs


def _prediction_probs_to_display_images(probs: torch.Tensor) -> torch.Tensor:
    """Convert class probabilities to displayable RGB images per sample."""
    num_classes = int(probs.shape[1])
    if num_classes == 3:
        pred = torch.argmax(probs, dim=1)
        images = torch.nn.functional.one_hot(pred, num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
        return images

    # For binary and num_classes >= 4, display channel index 1 as grayscale.
    channel = probs[:, 1:2].to(torch.float32)
    return channel.repeat(1, 3, 1, 1)


def _grid_rows_cols(num_images: int) -> tuple[int, int]:
    rows = max(1, int(math.ceil(math.sqrt(float(num_images) / 2.0))))
    cols = max(1, int(math.ceil(float(num_images) / float(rows))))
    return rows, cols


def _make_prediction_grid(probs: torch.Tensor) -> torch.Tensor:
    images = _prediction_probs_to_display_images(probs)
    num_images, _, height, width = images.shape
    rows, cols = _grid_rows_cols(num_images)
    total_slots = rows * cols

    if total_slots > num_images:
        pad = torch.zeros(
            (total_slots - num_images, images.shape[1], height, width),
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat([images, pad], dim=0)

    grid = torch.zeros((images.shape[1], rows * height, cols * width), dtype=images.dtype, device=images.device)
    for idx in range(total_slots):
        r = idx // cols
        c = idx % cols
        h0 = r * height
        w0 = c * width
        grid[:, h0 : h0 + height, w0 : w0 + width] = images[idx]
    return grid


def _get_logits_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    nats = log_probs * probs
    nats = torch.nan_to_num(nats, nan=0.0, posinf=0.0, neginf=0.0)
    return -torch.sum(nats, dim=-1)


def _get_average_pred_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.logsumexp(log_probs, dim=dim, keepdim=True) - torch.log(
        torch.tensor(logits.shape[dim], device=logits.device, dtype=logits.dtype)
    )


def _compute_mask_mutual_information_loss(model: torch.nn.Module) -> torch.Tensor:
    # Match reference: stack [logit, -logit] and compute MI over submodel axis.
    mask_logits = combined_mask_logits(model)
    binary_logits = torch.stack([mask_logits, -mask_logits], dim=-1)
    conditional_entropy = _get_logits_entropy(binary_logits).mean(dim=0, keepdim=True)
    mean_entropy = _get_logits_entropy(_get_average_pred_logits(binary_logits, dim=0))
    mutual_information = torch.clamp(mean_entropy - conditional_entropy, min=0.0)
    return mutual_information.mean()


def _collect_mask_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = []
    for module in iter_masked_layers(model):
        params.append(module.mask_logits_inputs)
        params.append(module.mask_logits_outputs)
    return params


def _compute_actual_fraction(model: torch.nn.Module, hard: bool = False) -> torch.Tensor:
    ratios = []
    for module in iter_masked_layers(model):
        if hard:
            fill = module.hard_binary_weights().to(torch.float32).mean()
        else:
            fill = module.soft_binary_weights().mean()
        ratios.append(fill)
    if not ratios:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(ratios).mean()


def _clone_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.to(device)
        elif isinstance(value, list):
            cloned[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            cloned[key] = value
    return cloned


def _build_model_only_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict()
    for key, value in model.state_dict().items():
        state[f"model.{key}"] = value.detach().cpu()
    return state


def _materialize_dense_base_model_from_masked(model: torch.nn.Module) -> torch.nn.Module:
    """Convert masked wrappers back to plain Linear/Conv2d while preserving dense weights."""
    dense_model = copy.deepcopy(model)

    def _replace(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, MaskedLinear):
                layer = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                layer = layer.to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    layer.weight.copy_(child.weight)
                    if child.bias is not None and layer.bias is not None:
                        layer.bias.copy_(child.bias)
                setattr(module, name, layer)
                continue
            if isinstance(child, MaskedConv2d):
                layer = torch.nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                layer = layer.to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    layer.weight.copy_(child.weight)
                    if child.bias is not None and layer.bias is not None:
                        layer.bias.copy_(child.bias)
                setattr(module, name, layer)
                continue
            _replace(child)

    _replace(dense_model)
    return dense_model


def _build_subensemble_masks_payload(model: torch.nn.Module) -> dict[str, Any]:
    """Serialize hard channel masks from masked layers in a compact checkpoint payload."""
    layers: list[dict[str, Any]] = []
    num_submodels: int | None = None

    for module_name, module in model.named_modules():
        if not isinstance(module, (MaskedLinear, MaskedConv2d)):
            continue
        if num_submodels is None:
            num_submodels = int(module.num_masks)
        elif int(module.num_masks) != num_submodels:
            raise RuntimeError(
                f"Inconsistent num_masks across layers ({num_submodels} vs {module.num_masks})."
            )

        input_masks = (module.mask_logits_inputs >= 0.0).to(torch.uint8).cpu()
        output_masks = (module.mask_logits_outputs >= 0.0).to(torch.uint8).cpu()
        layer_type = "conv2d" if isinstance(module, MaskedConv2d) else "linear"
        layers.append(
            {
                "name": module_name,
                "type": layer_type,
                "rows_only": bool(module.rows_only),
                "normalize": bool(module.normalize),
                "input_masks": input_masks,
                "output_masks": output_masks,
            }
        )

    if num_submodels is None:
        raise RuntimeError("No masked layers found when exporting sub-ensemble masks.")

    return {
        "format": "binary_channel_masks_v1",
        "num_submodels": int(num_submodels),
        "layers": layers,
    }


def _save_checkpoint(
    dest: Path,
    base_checkpoint: dict[str, Any],
    hparams: dict[str, Any],
    model_state: OrderedDict[str, torch.Tensor],
    subensemble_masks: dict[str, Any] | None = None,
) -> None:
    payload = copy.deepcopy(base_checkpoint)
    payload["hyper_parameters"] = hparams
    payload["state_dict"] = model_state
    if subensemble_masks is not None:
        payload["subensemble_masks"] = subensemble_masks
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)


def _save_metadata(dest_root: Path, payload: dict[str, Any]) -> None:
    metadata_path = dest_root / "subensemble_extraction.json"
    metadata_path.write_text(json.dumps(payload, indent=2))


def train_subensemble_masks(args: argparse.Namespace) -> None:
    source_ckpt = Path(args.source_ckpt).expanduser().resolve()
    if not source_ckpt.exists():
        raise FileNotFoundError(f"source checkpoint not found: {source_ckpt}")
    explicit_dest_root = bool(args.dest_root)
    if args.dest_root:
        dest_root = Path(args.dest_root).expanduser().resolve()
        # Keep checkpoint version identity aligned with destination folder name.
        args.dest_version = dest_root.name
    else:
        assert args.dest_version, f"Either --dest_root or --dest_version must be provided to determine where to save extracted sub-ensemble masks."
        dest_root = _build_dest_root(source_ckpt, args.dest_version)
    #dest_root = Path(args.dest_root).expanduser().resolve() if args.dest_root else _build_dest_root(source_ckpt, args.dest_version)
    dest_ckpt_dir = dest_root / "checkpoints"
    tb_dir = dest_root
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir.as_posix())

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint, hparams = _load_source_checkpoint(source_ckpt, device=device)
    if bool(args.remove_dropout):
        hparams = _remove_dropout_from_hparams(hparams)
        print("[extract] remove_dropout enabled: forcing model dropout settings to zero")
    experiment = _instantiate_lightning_experiment(hparams)
    # We call training_step manually, outside a PL Trainer context.
    experiment.log = lambda *a, **k: None  # type: ignore[assignment]
    experiment.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    experiment.to(device)
    experiment.train()

    replace_summary = replace_with_masked_layers(experiment.model, num_masks=args.num_submodels)
    if replace_summary.total_replaced == 0:
        raise RuntimeError("No nn.Linear/nn.Conv2d layers were replaced. Nothing to optimize.")
    rows_only_mode = not bool(args.both_col_row)
    set_rows_only_mode(experiment.model, rows_only_mode)
    set_normalize_mode(experiment.model, bool(args.normalize))
    initialize_mask_logits_for_target_fraction(
        experiment.model,
        target_fraction=float(args.target_fraction),
        use_zero_init=not bool(args.use_fraction_init),
    )
    if bool(args.use_fraction_init):
                print(
            "[extract] using target-fraction mask-logit initialization "
            f"(target_fraction={float(args.target_fraction):.4f}, rows_only={rows_only_mode}, both_col_row={bool(args.both_col_row)})"
        )
    else:
        print("[extract] using zero mask-logit initialization")

    freeze_unmasked_parameters(experiment.model)

    # Build optimizer only over mask logits.
    mask_params = _collect_mask_parameters(experiment.model)
    optimizer = torch.optim.Adam(mask_params, lr=args.lr, weight_decay=0.0)

    dm = _instantiate_datamodule(hparams)
    train_loader = dm.train_dataloader()
    id_test_dm = _instantiate_id_test_datamodule(hparams)
    id_test_loader = id_test_dm.test_dataloader()
    original_batch_size = getattr(train_loader, "batch_size", None)
    if int(args.batch_size_override) > 0:
        train_loader = _maybe_override_train_loader(train_loader, int(args.batch_size_override))
        new_batch_size = getattr(train_loader, "batch_size", None)
        print(f"[extract] batch_size override: {original_batch_size} -> {new_batch_size}")

    base_precision = _dtype_to_precision_label(next(experiment.model.parameters()).dtype)
    precision_override = str(args.precision or "").strip().lower()
    if precision_override and precision_override not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision override must be one of: fp32, fp16, bf16")
    precision = precision_override or base_precision
    if precision_override:
        print(f"[extract] precision override: {base_precision} -> {precision}")

    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and precision == "fp16"))
    loss_base_weight = float(args.loss_base_weight)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else int(args.num_epochs) * len(train_loader)
    if total_steps <= 0:
        raise ValueError("Computed total_steps <= 0. Check num_epochs/max_steps and dataloader size.")

    iterator = iter(train_loader)
    id_test_iterator = iter(id_test_loader)
    pbar = tqdm(range(total_steps), desc="Extracting sub-ensemble masks", dynamic_ncols=True)

    try:
        for step_idx in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            batch = _clone_batch_to_device(batch, device=device)
            optimizer.zero_grad(set_to_none=True)

            epoch_idx = step_idx // steps_per_epoch
            temp = float(args.temp_start) * (float(args.temp_decay) ** float(epoch_idx))

            masks_this_step = int(args.num_submodels) if int(args.masks_per_step) <= 0 else int(args.masks_per_step)

            def _autocast_ctx():
                return (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp
                    else nullcontext()
                )

            effective_masks = max(1, masks_this_step)
            base_loss_acc = 0.0

            # Memory-friendly accumulation: backprop each mask branch immediately.
            for local_idx in range(effective_masks):
                with _autocast_ctx():
                    mask_idx = (step_idx + local_idx) % int(args.num_submodels)
                    set_active_submodel(experiment.model, mask_idx=mask_idx, temp=temp)
                    mask_loss = experiment.training_step(batch, batch_idx=step_idx)
                    scaled_mask_loss = (loss_base_weight * mask_loss) / float(effective_masks)
                base_loss_acc += float(mask_loss.detach().item())
                if scaler.is_enabled():
                    scaler.scale(scaled_mask_loss).backward()
                else:
                    scaled_mask_loss.backward()

            with _autocast_ctx():
                overlap_soft = mean_pairwise_iou(experiment.model, hard=False)
                mask_mi = _compute_mask_mutual_information_loss(experiment.model)
                size_pen = submodel_size_penalty(experiment.model, target_fraction=float(args.target_fraction))
                actual_fraction = _compute_actual_fraction(experiment.model, hard=False)
                if bool(args.use_overlap):
                    diversity_term = float(args.overlap_weight) * overlap_soft
                else:
                    diversity_term = -float(args.mi_weight) * mask_mi
                reg_loss = diversity_term + float(args.size_weight) * size_pen

            if scaler.is_enabled():
                scaler.scale(reg_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reg_loss.backward()
                optimizer.step()

            base_loss_value = base_loss_acc / float(effective_masks)
            weighted_base_loss_value = loss_base_weight * base_loss_value
            reg_loss_value = float(reg_loss.detach().item())
            overlap_soft_value = float(overlap_soft.detach().item())
            mask_mi_value = float(mask_mi.detach().item())
            size_pen_value = float(size_pen.detach().item())
            actual_fraction_value = float(actual_fraction.detach().item())
            loss_value = weighted_base_loss_value + reg_loss_value

            # TensorBoard logging: per-step loss components and schedule values.
            global_step = step_idx + 1
            writer.add_scalar("extract/loss_total", loss_value, global_step)
            writer.add_scalar("extract/loss_base", base_loss_value, global_step)
            writer.add_scalar("extract/loss_reg", reg_loss_value, global_step)
            writer.add_scalar("extract/loss_overlap", overlap_soft_value, global_step)
            writer.add_scalar("extract/loss_mask_mi", mask_mi_value, global_step)
            writer.add_scalar("extract/loss_size", size_pen_value, global_step)
            writer.add_scalar("extract/actual_fraction", actual_fraction_value, global_step)

            base_loss = torch.tensor(weighted_base_loss_value, device=device)
            loss = torch.tensor(loss_value, device=device)

            should_log_eval = ((step_idx + 1) % int(args.log_every) == 0 or step_idx == 0)
            if should_log_eval:
                with torch.no_grad():
                    overlap_hard = float(mean_pairwise_iou(experiment.model, hard=True).item())
                    active_ratio_train = float(_compute_actual_fraction(experiment.model, hard=False).item())
                    active_ratio_test = float(_compute_actual_fraction(experiment.model, hard=True).item())
                    au_type = str(getattr(experiment.model, "AU_type", "softmax") or "softmax")
                    mean_dice, preview_probs = _compute_mean_dice_for_submodels(
                        experiment.model,
                        batch=batch,
                        num_submodels=int(args.num_submodels),
                        device=device,
                        au_type=au_type,
                        ignore_index=int(getattr(experiment, "ignore_index", 0) or 0),
                    )

                    try:
                        id_test_batch = next(id_test_iterator)
                    except StopIteration:
                        id_test_iterator = iter(id_test_loader)
                        id_test_batch = next(id_test_iterator)
                    id_test_batch = _clone_batch_to_device(id_test_batch, device=device)

                    was_training = experiment.model.training
                    experiment.model.eval()
                    mean_dice_eval, preview_probs_eval = _compute_mean_dice_for_submodels(
                        experiment.model,
                        batch=id_test_batch,
                        num_submodels=int(args.num_submodels),
                        device=device,
                        au_type=au_type,
                        ignore_index=int(getattr(experiment, "ignore_index", 0) or 0),
                    )
                    if was_training:
                        experiment.model.train()

                #print(f"[eval step {global_step}] active_param_ratio train={active_ratio_train:.4f} test={active_ratio_test:.4f}")

                writer.add_scalar("extract/overlap_hard", overlap_hard, global_step)
                if mean_dice is not None:
                    writer.add_scalar("extract/dice_mean", float(mean_dice), global_step)
                if mean_dice_eval is not None:
                    writer.add_scalar("extract/dice_mean_eval", float(mean_dice_eval), global_step)
                if preview_probs is not None:
                    pred_grid = _make_prediction_grid(preview_probs)
                    writer.add_image("extract_images/pred_train", pred_grid.detach().cpu(), global_step, dataformats="CHW")
                if preview_probs_eval is not None:
                    pred_grid_eval = _make_prediction_grid(preview_probs_eval)
                    writer.add_image("extract_images/pred_test", pred_grid_eval.detach().cpu(), global_step, dataformats="CHW")

                if args.verbose:
                    postfix = {
                        "loss": f"{float(loss.item()):.4f}",
                        "base": f"{float(base_loss.item()):.4f}",
                        "iou": f"{overlap_hard:.3f}",
                        "temp": f"{temp:.3f}",
                    }
                    postfix["dice"] = "n/a" if mean_dice is None else f"{mean_dice:.3f}"
                    postfix["dice_eval"] = "n/a" if mean_dice_eval is None else f"{mean_dice_eval:.3f}"
                    pbar.set_postfix(postfix)
    finally:
        writer.flush()
        writer.close()

    extracted_cfg = {
        "enabled": True,
        "num_submodels": int(args.num_submodels),
        "rows_only": rows_only_mode,
        "both_col_row": bool(args.both_col_row),
        "normalize": bool(args.normalize),
        "remove_dropout": bool(args.remove_dropout),
        "save_checkpoint": bool(args.save_checkpoint),
        "loss_base_weight": float(args.loss_base_weight),
        "use_fraction_init": bool(args.use_fraction_init),
        "temp_start": float(args.temp_start),
        "temp_decay": float(args.temp_decay),
        "target_fraction": float(args.target_fraction),
        "source_checkpoint": str(source_ckpt),
        "dest_version": str(args.dest_version),
        "mask_storage": "binary_channel_masks_v1",
    }

    new_hparams = _apply_destination_hparams_identity(
        hparams,
        dest_root=dest_root,
        explicit_dest_root=explicit_dest_root,
    )
    new_hparams["subensemble_extraction"] = extracted_cfg
    new_hparams["is_subensemble_masked_model"] = True
    new_hparams["version"] = str(args.dest_version)
    if bool(args.save_checkpoint):
        masked_ckpt_path = dest_ckpt_dir / "last.ckpt"
        dense_model = _materialize_dense_base_model_from_masked(experiment.model)
        subensemble_masks = _build_subensemble_masks_payload(experiment.model)
        _save_checkpoint(
            dest=masked_ckpt_path,
            base_checkpoint=checkpoint,
            hparams=new_hparams,
            model_state=_build_model_only_state_dict(dense_model),
            subensemble_masks=subensemble_masks,
        )

        hparams_yaml = dest_root / "hparams.yaml"
        hparams_yaml.write_text(OmegaConf.to_yaml(OmegaConf.create(new_hparams), resolve=False))
        _save_metadata(
            dest_root,
            {
                "source": str(source_ckpt),
                "dest": str(masked_ckpt_path),
                "tensorboard_dir": str(tb_dir),
                "replaced_layers": {
                    "linear": int(replace_summary.linear_replaced),
                    "conv2d": int(replace_summary.conv_replaced),
                },
                "num_submodels": int(args.num_submodels),
                "rows_only": rows_only_mode,
                "both_col_row": bool(args.both_col_row),
                "normalize": bool(args.normalize),
                "remove_dropout": bool(args.remove_dropout),
                "save_checkpoint": bool(args.save_checkpoint),
                "loss_base_weight": float(args.loss_base_weight),
                "use_fraction_init": bool(args.use_fraction_init),
                "use_overlap": bool(args.use_overlap),
                "mi_weight": float(args.mi_weight),
                "overlap_weight": float(args.overlap_weight),
                "temp_start": float(args.temp_start),
                "temp_decay": float(args.temp_decay),
                "mask_storage": "binary_channel_masks_v1",
            },
        )

        print(f"Saved masked sub-ensemble checkpoint to {masked_ckpt_path}")
        print(
            "Saved one checkpoint with dense weights + serialized mask list "
            f"({int(args.num_submodels)} submodels)"
        )
    else:
        print("Skipped checkpoint/hparams/metadata save (--save_checkpoint not set).")
    print(f"Saved TensorBoard logs to {tb_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract implicit sub-ensembles from a trained segmentation checkpoint using trainable masks.",
    )
    parser.add_argument(
        "--source_ckpt",
        type=str,
        default="./saves/chaksu128/softmax_ens_00/checkpoints/last.ckpt",
        help="Path to source .ckpt file",
    )
    parser.add_argument(
        "--dest_version",
        type=str,
        default="",
        help="Destination version folder name (sibling of source version)",
    )
    parser.add_argument(
        "--dest_root",
        type=str,
        default=None,
        help="Optional full output directory override. Defaults to sibling of source version.",
    )
    parser.add_argument(
        "--num_submodels",
        type=int,
        default=5,
        help="Number of implicit ensemble members (distinct masks) to extract.",
    )
    parser.add_argument(
        "--both_col_row",
        action="store_true",
        help="Use both column/input and row/output masks. By default only rows/outputs are masked.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Rescale masked layers by inverse keep ratios (dropout-style) during extraction and exported inference.",
    )
    parser.add_argument(
        "--remove_dropout",
        action="store_true",
        help="Disable model dropout (including dropout_cfg) during sub-ensemble extraction.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of full passes over the training dataloader when --max_steps is 0.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional fixed number of optimization steps. If > 0, overrides --num_epochs.",
    )
    parser.add_argument(
        "--masks_per_step",
        type=int,
        default=0,
        help="How many submodel indices to average per optimization step. Use 0 to use all submodels each step.",
    )
    parser.add_argument(
        "--lr", 
        type=float,
        default=0.1,
        help="Learning rate for mask-logit optimization (base model weights remain frozen).",
    )
    parser.add_argument(
        "--temp_start",  # from the code ref
        type=float,
        default=1.0,
        help="Initial mask sampling temp (higher = softer mask probabilities).",
    )
    parser.add_argument(
        "--temp_decay",  # from the code ref
        type=float,
        default=0.9,
        help="Multiplicative decay applied to temp once per epoch.",
    )
    parser.add_argument(
        "--target_fraction",  # guessed value
        type=float,
        default=0.2,
        help="Target fraction of active units/weights per extracted submodel for size regularization.",
    )
    parser.add_argument(
        "--use_fraction_init",
        action="store_true",
        help="Use fraction initialization instead of zeros.",
    )
    parser.add_argument(
        "--use_overlap",
        action="store_true",
        help="Use overlap-based diversity surrogate instead of mutual-information regularization.",
    )
    parser.add_argument(
        "--mi_weight",  # from the code ref
        type=float,
        default=0.5,
        help="Weight for mutual-information mask diversity regularization (default behavior).",
    )
    parser.add_argument(
        "--overlap_weight",  # from the code ref
        type=float,
        default=0.5,
        help="Weight for overlap surrogate diversity regularization when --use_overlap is enabled.",
    )
    parser.add_argument(
        "--size_weight",  # from the code ref
        type=float,
        default=3.0,
        help="Weight for penalty pushing each submodel toward --target_fraction.",
    )
    parser.add_argument(
        "--loss_base_weight",  # new param
        type=float,
        default=1.0,
        help="Weight for base task loss before adding regularization terms.",
    )
    parser.add_argument(
        "--batch_size_override",  # new param
        type=int,
        default=0,
        help="Override train dataloader batch size for extraction (0 keeps checkpoint/datamodule default).",
    )
    parser.add_argument(
        "--precision",  # new param
        type=str,
        default="",
        help="Optional precision override for extraction passes (fp32/fp16/bf16). Empty keeps model/checkpoint precision.",
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Save masked/materialized checkpoints plus extraction metadata. Disabled by default for training-only runs.",
    )
    parser.add_argument(
        "--verbose",  # from the code ref
        action="store_true",
        help="Enable periodic tqdm postfix logging (loss, overlap, temp, and Dice when available).",
    )
    parser.add_argument(
        "--log_every",  # from the code ref
        type=int,
        default=10,
        help="Logging interval in optimization steps when --verbose is enabled.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")  # from the code ref
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_subensemble_masks(args)


if __name__ == "__main__":
    main()