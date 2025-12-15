import json
import os
import warnings

import numpy as np
from sklearn.calibration import _sigmoid_calibration as calib
from sklearn import utils as sk_utils
from sklearn import preprocessing as sk_preprocess
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader


def platt_scale_params(val_exp_dataloader: ExperimentDataloader, ignore_value=None, n_bins: int = 256):
    """Compute Platt scaling parameters using compressed binned data.

    - Bin uncertainties on a logspace grid between 1e-12 and 1e2 (inclusive)
    - Accumulate counts for correct==1 and correct==0 per bin
    - Accumulate sum of uncertainties per bin to compute a mean per bin
    - Build a small weighted dataset (two samples per non-empty bin: one for positives, one for negatives)
    - Run calib once per uncertainty type using sample_weight
    """
    ps_params_dict = {}
    # Precompute bin edges
    bin_edges = np.logspace(-12, 2, num=n_bins + 1, dtype=np.float64)

    for unc_type in val_exp_dataloader.exp_version.unc_types:
        # Aggregation buffers
        pos_counts = np.zeros(n_bins, dtype=np.int64)
        neg_counts = np.zeros(n_bins, dtype=np.int64)
        sum_unc = np.zeros(n_bins, dtype=np.float64)
        total_counts = np.zeros(n_bins, dtype=np.int64)
        saw_oob = False
        oob_low_count = 0
        oob_high_count = 0

        for image_id in tqdm(val_exp_dataloader.image_ids):
            reference_segs = val_exp_dataloader.get_reference_segs(image_id)
            pred_seg = val_exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = val_exp_dataloader.get_unc_map(image_id, unc_type)
            reference_segs = np.asarray(reference_segs)
            pred_seg = np.asarray(pred_seg)
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            assert reference_segs.ndim == pred_seg.ndim + 1, f"Reference segs should have shape (n_raters, W, H), pred_seg (W, H). found {reference_segs.shape} vs {pred_seg.shape}"
            assert reference_segs.shape[1:] == pred_seg.shape, f"Reference segs and pred_seg spatial dimensions should match. found {reference_segs.shape} vs {pred_seg.shape}"
            # Broadcast instead of repeat to avoid copies
            rater_correct = (reference_segs == pred_seg[None, ...])

            if ignore_value is not None:
                valid_mask = (reference_segs != ignore_value)
            else:
                valid_mask = np.ones(reference_segs.shape, dtype=bool)

            # Uncertainty values per rater/pixel (broadcast as a view)
            u = np.broadcast_to(unc_map[None, ...], reference_segs.shape)[valid_mask].ravel()
            c = rater_correct[valid_mask].ravel().astype(np.int8)

            if u.size == 0:
                continue

            # Bin by uncertainty magnitude (not negated); clamp out-of-range
            bin_idx = np.digitize(u, bin_edges) - 1  # 0..n_bins-1 ideally
            oob_low = bin_idx < 0
            oob_high = bin_idx >= n_bins
            if oob_low.any() or oob_high.any():
                saw_oob = True
                oob_low_count += int(oob_low.sum())
                oob_high_count += int(oob_high.sum())
                # clamp to endpoints
                bin_idx[oob_low] = 0
                bin_idx[oob_high] = n_bins - 1

            # Accumulate sums and counts
            sum_unc += np.bincount(bin_idx, weights=u, minlength=n_bins)
            total_counts += np.bincount(bin_idx, minlength=n_bins)

            # Positive/negative counts per bin
            if (c == 1).any():
                pos_counts += np.bincount(bin_idx[c == 1], minlength=n_bins)
            if (c == 0).any():
                neg_counts += np.bincount(bin_idx[c == 0], minlength=n_bins)

        if saw_oob:
            warnings.warn(
                (
                    f"Uncertainty values outside [1e-12, 1e2] for '{unc_type}': "
                    f"{oob_low_count} below 1e-12, {oob_high_count} above 1e2. "
                    "They were clamped to the nearest end bin."
                ),
                RuntimeWarning,
            )

        # Mean uncertainty per bin (use total_counts to average across both classes)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_unc = np.divide(sum_unc, total_counts, out=np.zeros_like(sum_unc), where=total_counts > 0)

        # Build compressed dataset: at most 2 samples per bin (pos and neg)
        samples_F = []
        samples_y = []
        samples_w = []
        for b in range(n_bins):
            if total_counts[b] == 0:
                continue
            F_b = -mean_unc[b]  # match previous sign convention
            if pos_counts[b] > 0:
                samples_F.append(F_b)
                samples_y.append(1)
                samples_w.append(int(pos_counts[b]))
            if neg_counts[b] > 0:
                samples_F.append(F_b)
                samples_y.append(0)
                samples_w.append(int(neg_counts[b]))

        if len(samples_F) == 0:
            # Fallback to neutral parameters if no data collected
            a, b_param = 0.0, 0.0
        else:
            F_arr = np.asarray(samples_F, dtype=np.float64)
            y_arr = np.asarray(samples_y, dtype=np.float64)
            w_arr = np.asarray(samples_w, dtype=np.float64)
            a, b_param = calib(F_arr, y_arr, sample_weight=w_arr)

        ps_params_dict[unc_type] = {"a": float(a), "b": float(b_param)}

    with open(
        val_exp_dataloader.exp_version.exp_path / "platt_scale_params.json", "w"
    ) as f:
        json.dump(ps_params_dict, f, indent=2)



def platt_scale_params_old(val_exp_dataloader: ExperimentDataloader, ignore_value=None):
    raise ValueError("platt_scale_params_old has been deprecated. Use platt_scale_params instead.")
    ps_params_dict = {}
    for unc_type in val_exp_dataloader.exp_version.unc_types:
        ps_params_dict[unc_type] = {"a": [], "b": []}
        for image_id in tqdm(val_exp_dataloader.image_ids):
            reference_segs = val_exp_dataloader.get_reference_segs(image_id)
            pred_seg = val_exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = val_exp_dataloader.get_unc_map(image_id, unc_type)
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            pred_seg = np.repeat(pred_seg[np.newaxis, :], reference_segs.shape[0], 0)
            unc_map = np.repeat(unc_map[np.newaxis, :], reference_segs.shape[0], 0)
            rater_correct = (reference_segs == pred_seg).astype(int)
            if ignore_value is not None:
                ignore_mask = reference_segs != ignore_value
                a, b = calib(-unc_map[ignore_mask], rater_correct[ignore_mask])
            else:
                a, b = calib(-unc_map.flatten(), np.array(rater_correct).flatten())
            ps_params_dict[unc_type]["a"].append(a)
            ps_params_dict[unc_type]["b"].append(b)
        ps_params_dict[unc_type]["a"] = np.mean(np.array(ps_params_dict[unc_type]["a"]))
        ps_params_dict[unc_type]["b"] = np.mean(np.array(ps_params_dict[unc_type]["b"]))
    with open(
        val_exp_dataloader.exp_version.exp_path / "platt_scale_params.json", "w"
    ) as f:
        json.dump(ps_params_dict, f, indent=2)

def platt_scale_confid(uncalib_confid, platt_scale_file, uncertainty):
    with open(platt_scale_file) as f:
        params_dict = json.load(f)
    params = params_dict[uncertainty]
    return 1 / (1 + np.exp(uncalib_confid * params["a"] + params["b"]))


def calib_stats(correct, calib_confids):
    # calib_confids = np.clip(self.confids, 0, 1)
    n_bins = 20
    y_true = sk_utils.column_or_1d(correct)
    y_prob = sk_utils.column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob has values outside [0, 1] and normalize is " "set to False."
        )

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. " f"Provided labels {labels}."
        )
    y_true = sk_preprocess.label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = len(nonzero[nonzero == True])
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_total = bin_total[nonzero] / bin_total.sum()

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, prob_total, num_nonzero


def calc_ace(correct, calib_confids):
    bin_discrepancies, _, num_nonzero = calib_stats(correct, calib_confids)
    return (1 / num_nonzero) * np.sum(bin_discrepancies)


def calibration_error(exp_dataloader: ExperimentDataloader, ignore_value=None):
    calib_dict = {}
    calib_dict["mean"] = {}
    for unc_type in exp_dataloader.exp_version.unc_types:
        aces_unc = []
        for image_id in tqdm(exp_dataloader.image_ids):
            if image_id not in calib_dict.keys():
                calib_dict[image_id] = {}
            reference_segs = exp_dataloader.get_reference_segs(image_id)
            pred_seg = exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = exp_dataloader.get_unc_map(image_id, unc_type)
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            pred_seg = np.repeat(pred_seg[np.newaxis, :], reference_segs.shape[0], 0)
            unc_map = np.repeat(unc_map[np.newaxis, :], reference_segs.shape[0], 0)
            rater_correct = (reference_segs == pred_seg).astype(int)
            platt_scale_file = (
                exp_dataloader.exp_version.exp_path / "platt_scale_params.json"
            )
            if ignore_value is not None:
                ignore_mask = reference_segs != ignore_value
                unc_map = platt_scale_confid(
                    -unc_map[ignore_mask],
                    platt_scale_file=platt_scale_file,
                    uncertainty=unc_type,
                )
                ace = calc_ace(rater_correct[ignore_mask], unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace}}
                aces_unc.append(ace)
            else:
                unc_map = platt_scale_confid(
                    -unc_map.flatten(),
                    platt_scale_file=platt_scale_file,
                    uncertainty=unc_type,
                )
                ace = calc_ace(rater_correct.flatten(), unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace}}
                aces_unc.append(ace)
        calib_dict["mean"][unc_type] = {"metrics": {"ace": np.mean(np.array(aces_unc))}}
    save_path = exp_dataloader.dataset_path / "calibration.json"
    with open(save_path, "w") as f:
        json.dump(calib_dict, f, indent=2)


def main(exp_dataloader: ExperimentDataloader, ignore_value=None):
    platt_scale_params_file = (
        exp_dataloader.exp_version.exp_path / "platt_scale_params.json"
    )
    # replace by checking whether platt scale params file exists
    if not os.path.isfile(platt_scale_params_file):
        val_exp_dataloader = ExperimentDataloader(exp_dataloader.exp_version, "val")
        platt_scale_params(val_exp_dataloader, ignore_value=ignore_value)
    calibration_error(exp_dataloader, ignore_value=ignore_value)
