"""
Debug script: compute ACE (adaptive calibration error) for
  saves/origlidc128/test_results/prob_unet_swag_diag_0/e1000_ema
on the "id" dataset split using the pre-computed Platt scaling parameters from
  .../e1000_ema/platt_scale_params.json

Prints per-image and mean ACE for TU, AU, EU.

Run from values/:
    python global_utils/debug_ace.py
"""
import json
import sys
from pathlib import Path

# Ensure values/ and values/evaluation/ are importable
_values_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_values_dir / "evaluation"))
sys.path.insert(0, str(_values_dir))

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sklearn import preprocessing as sk_preprocess
from sklearn import utils as sk_utils
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader
from experiment_version import ExperimentVersion

# ── Platt scaling parameters file ─────────────────────────────────────────────
PLATT_SCALE_FILE = Path(
    "/home/jloch/Desktop/diff/luzern/values/saves/origlidc128"
    "/test_results/prob_unet_swag_diag_0/e1000_ema/platt_scale_params.json"
)


# ── Copied from ace.py (independent for debugging) ────────────────────────────

def platt_scale_confid(uncalib_confid, params_dict, uncertainty):
    params = params_dict[uncertainty]
    return 1.0 / (1.0 + np.exp(uncalib_confid * params["a"] + params["b"]))


def calib_stats(correct, calib_confids):
    calib_confids = np.clip(calib_confids, 0, 1)
    n_bins = 20
    y_true = sk_utils.column_or_1d(correct)
    y_prob = sk_utils.column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Labels: {labels}.")
    y_true = sk_preprocess.label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums  = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true  = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = int(nonzero.sum())
    prob_true  = bin_true[nonzero]  / bin_total[nonzero]
    prob_pred  = bin_sums[nonzero]  / bin_total[nonzero]
    prob_total = bin_total[nonzero] / bin_total.sum()

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, prob_total, num_nonzero


def calc_ace(correct, calib_confids):
    bin_discrepancies, _, num_nonzero = calib_stats(correct, calib_confids)
    return (1.0 / num_nonzero) * np.sum(bin_discrepancies)


# ── Global (dataset-level) ACE accumulator ────────────────────────────────────

class GlobalCalibAccumulator:
    """Accumulates calibration bin statistics incrementally across many images,
    then computes ACE once over the full dataset — equivalent to treating all
    pixels as a single super-image but without ever concatenating them.

    Bins are the same 20 uniform bins over [0, 1+1e-8] used in calib_stats.
    """

    N_BINS = 20

    def __init__(self):
        n = self.N_BINS + 1          # calib_stats passes minlength=len(bins)
        self.bin_sums  = np.zeros(n, dtype=np.float64)   # sum of confidences
        self.bin_true  = np.zeros(n, dtype=np.float64)   # sum of correct labels
        self.bin_total = np.zeros(n, dtype=np.int64)     # count of samples

    def accumulate(self, correct, calib_confids):
        """Add one batch of (correct, calib_confids) — both 1-D arrays."""
        calib_confids = np.clip(calib_confids, 0.0, 1.0)
        y_true = correct.astype(np.float64).ravel()
        y_prob = calib_confids.ravel()

        bins   = np.linspace(0.0, 1.0 + 1e-8, self.N_BINS + 1)
        binids = np.digitize(y_prob, bins) - 1

        n = self.N_BINS + 1
        self.bin_sums  += np.bincount(binids, weights=y_prob, minlength=n)
        self.bin_true  += np.bincount(binids, weights=y_true, minlength=n)
        self.bin_total += np.bincount(binids,                 minlength=n)

    def compute_ace(self):
        """Return the global ACE computed from all accumulated data."""
        nonzero     = self.bin_total > 0
        num_nonzero = int(nonzero.sum())
        if num_nonzero == 0:
            return float("nan")
        prob_true = self.bin_true[nonzero]  / self.bin_total[nonzero]
        prob_pred = self.bin_sums[nonzero]  / self.bin_total[nonzero]
        discrepancies = np.abs(prob_true - prob_pred)
        return float((1.0 / num_nonzero) * np.sum(discrepancies))


def calibration_error_debug(
    exp_dataloader: ExperimentDataloader,
    params_dict: dict,
    ignore_value=None,
):
    """Compute ACE two ways and print both:
      1. Original: mean of per-image ACEs.
      2. Global:   single ACE over all pixels at once (incremental binning).
    """
    for unc_type in exp_dataloader.exp_version.unc_types:
        aces_per_image = []
        global_accum   = GlobalCalibAccumulator()

        for image_id in tqdm(exp_dataloader.image_ids, desc=f"[{unc_type}] ACE"):
            reference_segs = exp_dataloader.get_reference_segs(image_id)
            pred_seg       = exp_dataloader.get_mean_pred_seg(image_id)
            unc_map        = exp_dataloader.get_unc_map(image_id, unc_type)

            if pred_seg.shape != unc_map.shape:
                import warnings
                warnings.warn(
                    f"Uncertainty map shape {unc_map.shape} does not match "
                    f"pred_seg shape {pred_seg.shape} for image {image_id}. "
                    "Swapping axes."
                )
                unc_map = np.swapaxes(unc_map, 0, 1)

            n_gt     = reference_segs.shape[0]
            pred_rep = np.repeat(pred_seg[np.newaxis, :], n_gt, axis=0)
            unc_rep  = np.repeat(unc_map[np.newaxis, :],  n_gt, axis=0)
            rater_correct = (reference_segs == pred_rep).astype(int)

            if ignore_value is not None:
                mask          = reference_segs != ignore_value
                flat_correct  = rater_correct[mask]
                flat_confid   = platt_scale_confid(-unc_rep[mask], params_dict, unc_type)
            else:
                flat_correct  = rater_correct.flatten()
                flat_confid   = platt_scale_confid(-unc_rep.flatten(), params_dict, unc_type)

            aces_per_image.append(calc_ace(flat_correct, flat_confid))
            global_accum.accumulate(flat_correct, flat_confid)

        mean_ace   = float(np.mean(aces_per_image))
        global_ace = global_accum.compute_ace()
        print(
            f"\n[{unc_type}]"
            f"  mean-of-images ACE = {mean_ace:.6f}"
            f"  |  global ACE = {global_ace:.6f}"
            f"  (over {len(aces_per_image)} images)"
        )

        # Print pixel counts per bin
        bins = np.linspace(0.0, 1.0 + 1e-8, GlobalCalibAccumulator.N_BINS + 1)
        print(f"  {'Bin':>5}  {'[lo, hi)':>20}  {'pixels':>12}")
        for i, count in enumerate(global_accum.bin_total[: GlobalCalibAccumulator.N_BINS]):
            lo, hi = bins[i], bins[i + 1]
            print(f"  {i:>5}  [{lo:>8.4f}, {hi:>8.4f})  {count:>12,}")


def main():
    GlobalHydra.instance().clear()
    config_dir = str(_values_dir / "evaluation" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="eval_config_lidc")

    exp_version = ExperimentVersion(
        base_path=Path("/home/jloch/Desktop/diff/luzern/values/saves"),
        naming_scheme_pred_model="origlidc128",
        naming_scheme_version="{pred_model}_{eu}_{seed1}/e1000_ema",
        pred_model="prob_unet",
        image_ending=".png",
        unc_ending=".tif",
        unc_types=["TU", "AU", "EU"],
        aggregations=["all"],
        n_reference_segs=4,
        datamodule_config=cfg.datamodule_config,
        seed="120",
        seed1="0",
        eu="swag_diag",
    )

    print(f"exp_path     : {exp_version.exp_path}")
    assert exp_version.exp_path.exists(), f"Experiment path not found: {exp_version.exp_path}"

    with open(PLATT_SCALE_FILE) as f:
        params_dict = json.load(f)
    print(f"Platt params : {PLATT_SCALE_FILE}")
    for unc, p in params_dict.items():
        print(f"  {unc:4s}  a = {p['a']:.6f},  b = {p['b']:.6f}")
    print()

    id_dataloader = ExperimentDataloader(exp_version, "id")
    print(f"dataset_path : {id_dataloader.dataset_path}")
    print(f"n_images     : {len(id_dataloader.image_ids)}")
    print()

    calibration_error_debug(id_dataloader, params_dict, ignore_value=None)


if __name__ == "__main__":
    main()
