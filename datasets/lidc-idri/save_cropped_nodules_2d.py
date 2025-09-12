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
    parser.add_argument(
        "-large",
        action="store_true",
        help="If set, saves 128x128 images instead of 64x64.",
        default=False,
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

    if args.large:
        vs = 63.0/127.0
        assert not "small" in args.save_path, "You are trying to save large images in a folder with 'small' in the name."
    else:
        vs = 1.0
        assert "small" in args.save_path, "You are trying to save small images in a folder without 'small' in the name."

    scans = pl.query(pl.Scan)
    all_metadata = []
    
    for scan in tqdm(scans, total=scans.count(), desc="Processing scans"):
        nods = scan.cluster_annotations()
        local_nod_idx = 0
        for nod_idx, nod in enumerate(nods):
            if has_large_mask(nod):
                continue
            metadata_nod = {}
            masks = []
            for ann_idx in range(4):
                if ann_idx == 0:
                    vol, mask, irp_pts = nod[ann_idx].uniform_cubic_resample(
                        side_length = 63,
                        voxel_size=vs,
                        raw_z_sampling=True,
                        return_irp_pts=True,
                        verbose=False
                    )
                    metadata_nod["Patient ID"] = str(nod[0].scan.patient_id)
                    metadata_nod["Scan ID"] = str(nod[0].scan.id).zfill(4)
                    append_metadata(metadata_nod, nod[ann_idx], first=True)
                if ann_idx < len(nod):
                    mask = nod[ann_idx].uniform_cubic_resample(
                        side_length = 63,
                        voxel_size=vs,
                        raw_z_sampling=True,
                        resample_vol=False, 
                        irp_pts=irp_pts,
                        verbose=False
                    )
                    annotation = nod[ann_idx]
                else:
                    mask = np.zeros(vol.shape)
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
                    / f"{str(nod[0].scan.id).zfill(4)}_{str(local_nod_idx).zfill(3)}.npy"
                )
                np.save(image_save_path, image_slice)
                seg_paths = []
                for rater in range(4):
                    label_slice = masks[rater][:, :, slice_idx]
                    segmentation_save_path = (
                        labels_save_dir
                        / f"{str(nod[0].scan.id).zfill(4)}_{str(local_nod_idx).zfill(3)}_{str(rater).zfill(2)}_mask.npy"
                    )
                    np.save(segmentation_save_path, label_slice.astype(np.intc))
                    seg_paths.append(segmentation_save_path)
                metadata_slice = metadata_nod.copy()
                metadata_slice["Nodule Index"] = str(local_nod_idx).zfill(3)
                metadata_slice["Image Save Path"] = image_save_path
                metadata_slice["Segmentation Save Paths"] = seg_paths
                all_metadata.append(pd.Series(metadata_slice))
                local_nod_idx += 1
    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(save_path / "metadata.csv", index=False)


if __name__ == "__main__":
    cli_args = main_cli()
    save_nodules(cli_args)
