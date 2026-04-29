import argparse
from pathlib import Path

import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt


def get_gt_unc_map(image_id, ref_seg_dir, n_reference_segs, image_ending):
    reference_segs_paths = [
        Path(ref_seg_dir) / f"{image_id}_{i:02d}{image_ending}"
        for i in range(n_reference_segs)
    ]
    reference_segs = []
    for reference_seg_path in reference_segs_paths:
        if str(reference_seg_path).endswith(".npy"):
            reference_seg = np.load(reference_seg_path)
        else:
            reference_seg, _ = load(reference_seg_path)
        reference_segs.append(reference_seg)
    reference_segs = np.array(reference_segs)
    per_pixel_variance = np.var(reference_segs, axis=0)
    return per_pixel_variance


def get_unc_map(image_id, unc_dir, unc_ending):
    unc_map_path = Path(unc_dir) / f"{image_id}{unc_ending}"
    unc_map, _ = load(unc_map_path)
    #import tifffile
    #print(f"Loading uncertainty map from {unc_map_path}")
    #unc_map = tifffile.imread(unc_map_path)
    return unc_map


def main():
    parser = argparse.ArgumentParser(description="Debug NCC transposition by plotting GT and pred uncertainty maps.")
    parser.add_argument(
        "--exp_path",
        default="/home/jloch/Desktop/diff/luzern/values/saves/chaksu128/test_results/ssn_dropout_0/e500_ema",
        help="Experiment path containing split directories.",
    )
    parser.add_argument("--split", default="id", help="Dataset split folder name (default: id).")
    parser.add_argument("--image_id", default="t_001081", help="Image ID to visualize.")
    parser.add_argument("--unc_type", default="AU", help="Uncertainty type folder name (e.g. AU, EU, TU).")
    parser.add_argument("--n_reference_segs", type=int, default=5, help="Number of GT rater masks.")
    parser.add_argument("--image_ending", default=".png", help="GT segmentation file extension.")
    parser.add_argument("--unc_ending", default=".tif", help="Uncertainty map file extension.")
    parser.add_argument("--gt_ending", default="_mask.npy", help="GT segmentation file extension.")
    parser.add_argument(
        "--data_root",
        default="/home/jloch/Desktop/diff/luzern/values_datasets/chaksu128",
        help="Dataset root containing preprocessed labels.",
    )
    args = parser.parse_args()

    exp_path = Path(args.exp_path)
    split_path = exp_path / args.split
    ref_seg_dir = Path(args.data_root) / "preprocessed" / "labels"
    unc_dir = split_path / args.unc_type

    gt_unc_map = get_gt_unc_map(
        args.image_id,
        ref_seg_dir=ref_seg_dir,
        n_reference_segs=args.n_reference_segs,
        image_ending=args.gt_ending,
    )
    pred_unc_map = get_unc_map(
        args.image_id,
        unc_dir=unc_dir,
        unc_ending=args.unc_ending,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pred_unc_map, cmap="viridis")
    axes[0].set_title("Pred unc")
    axes[1].imshow(gt_unc_map, cmap="viridis")
    axes[1].set_title("GT unc")
    axes[2].imshow(np.swapaxes(pred_unc_map, 0, 1), cmap="viridis")
    axes[2].set_title("Pred unc (swapaxes)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
