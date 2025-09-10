import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import pylidc as pl
import pylidc.utils
from tqdm import tqdm


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to the folder where the cropped nodules will be stored",
        required=True,
    )
    args = parser.parse_args()
    return args


def has_large_mask(nod):
    """Checks if the consensus mask is larger than 64 voxels in any dimension."""
    consensus_mask, _, _ = pylidc.utils.consensus(nod, clevel=0.1)
    max_size_mask = max(consensus_mask.shape)
    if max_size_mask > 64:
        return True


def append_metadata(metadata_nod, nod, first=False):
    features = [
        "subtlety",
        "internal Structure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "texture",
        "malignancy",
    ]
    if first:
        for feature in features:
            metadata_nod[feature] = []
    if nod is not None:
        for feature in features:
            metadata_nod[feature].append(getattr(nod, feature.replace(" ", "")))
    else:
        for feature in features:
            metadata_nod[feature].append(None)


def save_nodules(args: Namespace):
    # Set up the paths to store the data
    save_path = Path(args.save_path)
    images_save_dir = save_path / "images"
    labels_save_dir = save_path / "labels"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    scans = pl.query(pl.Scan)
    all_metadata = []
    global_idx = 0
    for scan in tqdm(scans):
        nods = scan.cluster_annotations()
        for nod_idx, nod in enumerate(nods):
            if has_large_mask(nod):
                continue
            metadata_nod = {}
            masks = []
            for ann_idx in range(4):
                if ann_idx == 0:
                    image_size = 64
                    vol, mask, irp_pts = nod[ann_idx].uniform_cubic_resample(
                        image_size - 1, return_irp_pts=True
                    )
                    metadata_nod["Patient ID"] = str(nod[0].scan.patient_id)
                    metadata_nod["Scan ID"] = str(nod[0].scan.id).zfill(4)
                    append_metadata(metadata_nod, nod[ann_idx], first=True)
                if ann_idx < len(nod):
                    mask = nod[ann_idx].uniform_cubic_resample(
                        image_size - 1, resample_vol=False, irp_pts=irp_pts
                    )
                    annotation = nod[ann_idx]
                else:
                    mask = np.zeros([64, 64, 64])
                    annotation = None
                masks.append(mask)
                if ann_idx > 0:
                    append_metadata(metadata_nod, annotation)
            # determine which slices to save (at least one rater positive)
            positive_slices = [
                s for s in range(vol.shape[2]) if any(m[:, :, s].any() for m in masks)
            ]
            for slice_idx in positive_slices:
                image_slice = vol[:, :, slice_idx]
                image_save_path = (
                    images_save_dir
                    / f"{str(nod[0].scan.id).zfill(4)}_{str(global_idx).zfill(3)}.npy"
                )
                np.save(image_save_path, image_slice)
                seg_paths = []
                for rater in range(4):
                    label_slice = masks[rater][:, :, slice_idx]
                    segmentation_save_path = (
                        labels_save_dir
                        / f"{str(nod[0].scan.id).zfill(4)}_{str(global_idx).zfill(3)}_{str(rater).zfill(2)}_mask.npy"
                    )
                    np.save(segmentation_save_path, label_slice.astype(np.intc))
                    seg_paths.append(segmentation_save_path)
                metadata_slice = metadata_nod.copy()
                metadata_slice["Nodule Index"] = str(global_idx).zfill(3)
                metadata_slice["Image Save Path"] = image_save_path
                metadata_slice["Segmentation Save Paths"] = seg_paths
                all_metadata.append(pd.Series(metadata_slice))
                global_idx += 1
    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(save_path / "metadata.csv", index=False)


if __name__ == "__main__":
    cli_args = main_cli()
    save_nodules(cli_args)
