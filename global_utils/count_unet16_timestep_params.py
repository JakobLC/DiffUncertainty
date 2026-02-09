from __future__ import annotations
import sys
#add to path
sys.path.append("/home/jloch/Desktop/diff/luzern/values/")
import argparse
from typing import Dict, Tuple

from uncertainty_modeling.models.diff_unet2D_module import DiffUnet, MLPBlock, ResBlock

DEFAULT_DIFFUSION_KWARGS = {
    "schedule_name": "cosine",
    "input_scale": 0.1,
    "model_pred_type": "X",
    "weights_type": "uniform",
    "sampler_type": "uniform_low_d",
    "var_type": "large",
    "loss_type": "MSE",
    "logsnr_min": -10.0,
    "logsnr_max": 10.0,
    "decouple_loss_weights": False,
}


def _build_unet16_diffusion(
    num_classes: int,
    image_channels: int,
    model_channels: int,
    dropout: float,
    diffusion_steps: int,
    sampler: str,
) -> DiffUnet:
    in_channels = image_channels + num_classes
    return DiffUnet(
        name="unet16",
        in_channels=in_channels,
        out_channels=num_classes,
        model_channels=model_channels,
        num_res_blocks=[2, 2, 2, 2],
        num_middle_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=[-2, -1],
        dropout=dropout,
        final_act="softmax",
        diffusion=True,
        diffusion_kwargs=dict(DEFAULT_DIFFUSION_KWARGS),
        diffusion_num_steps=diffusion_steps,
        diffusion_sampler_type=sampler,
    )


def _collect_timestep_assignments(model: DiffUnet) -> Dict[int, str]:
    assignments: Dict[int, str] = {}
    time_embed = getattr(model, "time_embed", None)
    if time_embed is not None:
        for param in time_embed.parameters():
            assignments[id(param)] = "time_mlp"
    for module in model.modules():
        if isinstance(module, ResBlock):
            for param in module.emb_layers.parameters():
                assignments[id(param)] = "resblock_projections"
        elif isinstance(module, MLPBlock):
            for param in module.emb_layers.parameters():
                assignments[id(param)] = "mlpblock_projections"
    return assignments


def _partition_parameter_counts(model: DiffUnet, assignments: Dict[int, str]) -> Tuple[Dict[str, int], int, int, int]:
    category_totals: Dict[str, int] = {
        "time_mlp": 0,
        "resblock_projections": 0,
        "mlpblock_projections": 0,
    }
    timestep_total = 0
    grand_total = 0
    for param in model.parameters():
        count = param.numel()
        grand_total += count
        category = assignments.get(id(param))
        if category is None:
            continue
        category_totals[category] += count
        timestep_total += count
    non_timestep_total = grand_total - timestep_total
    return category_totals, timestep_total, non_timestep_total, grand_total


def _format_count(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Instantiate the standard diffusion UNet-16 configuration and split its parameters "
            "into timestep-related and non-timestep groups."
        )
    )
    parser.add_argument("--num-classes", type=int, default=4, help="Segmentation classes / output channels")
    parser.add_argument("--image-channels", type=int, default=3, help="Input image channels before diffusion concat")
    parser.add_argument(
        "--model-channels",
        type=int,
        default=32,
        help="Base channel width (32 replicates the unet16 sweep configuration)",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate inside residual blocks")
    parser.add_argument("--diffusion-steps", type=int, default=10, help="Number of sampling steps to store in the model")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        help="Sampler type recorded on the DiffUnet instance",
    )
    args = parser.parse_args()

    if args.num_classes <= 0:
        parser.error("--num-classes must be positive")
    if args.image_channels <= 0:
        parser.error("--image-channels must be positive")
    if args.model_channels <= 0:
        parser.error("--model-channels must be positive")
    if args.diffusion_steps <= 0:
        parser.error("--diffusion-steps must be positive")

    model = _build_unet16_diffusion(
        num_classes=args.num_classes,
        image_channels=args.image_channels,
        model_channels=args.model_channels,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        sampler=args.sampler,
    )

    assignments = _collect_timestep_assignments(model)
    category_totals, timestep_total, non_timestep_total, grand_total = _partition_parameter_counts(
        model, assignments
    )

    print("UNet16 diffusion parameter counts:")
    print(f"  total params         : {_format_count(grand_total)}")
    print(f"  timestep-conditioned  : {_format_count(timestep_total)} ({timestep_total / grand_total:.2%})")
    print(f"    - time MLP           : {_format_count(category_totals['time_mlp'])}")
    print(f"    - resblock proj      : {_format_count(category_totals['resblock_projections'])}")
    print(f"    - mlp block proj     : {_format_count(category_totals['mlpblock_projections'])}")
    print(f"  non-timestep          : {_format_count(non_timestep_total)} ({non_timestep_total / grand_total:.2%})")


if __name__ == "__main__":
    main()
