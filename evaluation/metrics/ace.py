import json
import os
import warnings

import numpy as np
from sklearn.calibration import _sigmoid_calibration as calib
from sklearn import utils as sk_utils
from sklearn import preprocessing as sk_preprocess
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader


def platt_scale_params(
    val_exp_dataloader: ExperimentDataloader,
    ignore_value=None,
    n_bins: int = 256,
    plot: bool = False,
):
    """Compute Platt scaling parameters using compressed binned data.

    - Bin uncertainties on a logspace grid between 1e-12 and 1e2 (inclusive)
    - Accumulate counts for correct==1 and correct==0 per bin
    - Accumulate sum of uncertainties per bin to compute a mean per bin
    - Build a small weighted dataset (two samples per non-empty bin: one for positives, one for negatives)
    - Run calib once per uncertainty type using sample_weight
    """
    ps_params_dict = {}
    rng = np.random.default_rng(0)
    # Precompute bin edges
    bin_edges = np.logspace(-12, 2, num=n_bins + 1, dtype=np.float64)

    row_labels = ["AU", "EU", "TU"]
    row_index = {label: idx for idx, label in enumerate(row_labels)}
    fig = None
    axes = None
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)
        for r, label in enumerate(row_labels):
            axes[r, 0].set_ylabel(f"{label} correctness")
            axes[r, 0].set_xlim(bin_edges[0], bin_edges[-1])
            axes[r, 1].set_xscale("log")
            axes[r, 1].set_xlim(bin_edges[0], bin_edges[-1])
            axes[r, 0].set_ylim(-0.05, 1.05)
            axes[r, 1].set_ylim(-0.05, 1.05)
            axes[r, 0].grid(alpha=0.2)
            axes[r, 1].grid(alpha=0.2)

        axes[0, 0].set_title("Linear x-axis")
        axes[0, 1].set_title("Log x-axis")
        axes[2, 0].set_xlabel("Uncertainty")
        axes[2, 1].set_xlabel("Uncertainty")

    for unc_type in val_exp_dataloader.exp_version.unc_types:
        # Aggregation buffers
        pos_counts = np.zeros(n_bins, dtype=np.int64)
        neg_counts = np.zeros(n_bins, dtype=np.int64)
        sum_unc = np.zeros(n_bins, dtype=np.float64)
        total_counts = np.zeros(n_bins, dtype=np.int64)
        saw_oob = False
        oob_low_count = 0
        oob_high_count = 0
        sampled_x_parts = []
        sampled_y_parts = []
        sampled_n_gt_parts = []
        per_image_quota = max(1, int(np.ceil(10000 / max(1, len(val_exp_dataloader.image_ids)))))

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

            if plot:
                valid_counts = valid_mask.sum(axis=0)
                pixel_valid = valid_counts > 0
                if pixel_valid.any():
                    frac_correct = np.divide(
                        (rater_correct & valid_mask).sum(axis=0),
                        valid_counts,
                        out=np.zeros_like(valid_counts, dtype=np.float64),
                        where=valid_counts > 0,
                    )
                    pixel_unc = np.asarray(unc_map)[pixel_valid].ravel()
                    pixel_corr = np.asarray(frac_correct)[pixel_valid].ravel()
                    pixel_n_gt = np.asarray(valid_counts)[pixel_valid].ravel().astype(np.float64)
                    sample_count = min(per_image_quota, pixel_unc.size)
                    if sample_count > 0:
                        sample_idx = rng.choice(pixel_unc.size, size=sample_count, replace=False)
                        sampled_x_parts.append(pixel_unc[sample_idx])
                        sampled_y_parts.append(pixel_corr[sample_idx])
                        sampled_n_gt_parts.append(pixel_n_gt[sample_idx])

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

        if plot and axes is not None:
            row = row_index.get(str(unc_type).upper())
            if row is None:
                warnings.warn(
                    f"Skipping plotting for uncertainty type '{unc_type}' because it is not one of {row_labels}.",
                    RuntimeWarning,
                )
                continue

            left_ax = axes[row, 0]
            right_ax = axes[row, 1]

            if sampled_x_parts:
                sampled_x = np.concatenate(sampled_x_parts)
                sampled_y = np.concatenate(sampled_y_parts)
                sampled_n_gt = np.concatenate(sampled_n_gt_parts)
                if sampled_x.size > 10000:
                    keep_idx = rng.choice(sampled_x.size, size=10000, replace=False)
                    sampled_x = sampled_x[keep_idx]
                    sampled_y = sampled_y[keep_idx]
                    sampled_n_gt = sampled_n_gt[keep_idx]
            else:
                sampled_x = np.array([], dtype=np.float64)
                sampled_y = np.array([], dtype=np.float64)
                sampled_n_gt = np.array([], dtype=np.float64)

            if sampled_x.size > 0:
                in_range = (sampled_x >= bin_edges[0]) & (sampled_x <= bin_edges[-1])
                sampled_x = sampled_x[in_range]
                sampled_y = sampled_y[in_range]
                sampled_n_gt = sampled_n_gt[in_range]

            denom = pos_counts + neg_counts
            valid_bins = denom > 0
            frac_pos = np.divide(
                pos_counts,
                denom,
                out=np.zeros_like(pos_counts, dtype=np.float64),
                where=valid_bins,
            )

            non_empty_bins = np.where(valid_bins)[0]
            if non_empty_bins.size > 0:
                x_min = float(bin_edges[non_empty_bins[0]])
                x_max = float(bin_edges[non_empty_bins[-1] + 1])
            else:
                x_min = float(bin_edges[0])
                x_max = float(bin_edges[-1])

            if x_max <= x_min:
                x_max = x_min * (1.0 + 1e-6)

            left_ax.set_xlim(x_min, x_max)
            right_ax.set_xlim(x_min, x_max)

            for b in np.where(valid_bins)[0]:
                left_ax.hlines(
                    y=frac_pos[b],
                    xmin=bin_edges[b],
                    xmax=bin_edges[b + 1],
                    colors="black",
                    linewidth=1.0,
                    alpha=0.7,
                )
                right_ax.hlines(
                    y=frac_pos[b],
                    xmin=bin_edges[b],
                    xmax=bin_edges[b + 1],
                    colors="black",
                    linewidth=1.0,
                    alpha=0.7,
                )

            if sampled_x.size > 0:
                # Uniform y-jitter with total interval width 0.2 / n_GTs.
                jitter_width = np.divide(
                    0.2,
                    sampled_n_gt,
                    out=np.zeros_like(sampled_n_gt, dtype=np.float64),
                    where=sampled_n_gt > 0,
                )
                sampled_y_jittered = sampled_y + rng.uniform(-0.5, 0.5, size=sampled_y.size) * jitter_width
                sampled_y_jittered = np.clip(sampled_y_jittered, 0.0, 1.0)
                left_ax.scatter(sampled_x, sampled_y_jittered, s=5, color="royalblue", alpha=0.2)
                right_ax.scatter(sampled_x, sampled_y_jittered, s=5, color="royalblue", alpha=0.2)

            x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
            y_curve = 1.0 / (1.0 + np.exp((-x_curve) * a + b_param))
            left_ax.plot(x_curve, y_curve, color="red", linewidth=1.5)
            right_ax.plot(x_curve, y_curve, color="red", linewidth=1.5)

            left_ax.text(
                0.02,
                0.98,
                str(unc_type),
                transform=left_ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

    with open(
        val_exp_dataloader.exp_version.exp_path / "platt_scale_params.json", "w"
    ) as f:
        json.dump(ps_params_dict, f, indent=2)

    if plot and fig is not None:
        fig.suptitle("Platt Scaling Diagnostics", fontsize=14)
        fig.tight_layout()
        import matplotlib.pyplot as plt

        plt.show()



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
    calib_confids = np.clip(calib_confids, 0, 1)
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


def calc_ece(correct, calib_confids):
    bin_discrepancies, prob_total, _ = calib_stats(correct, calib_confids)
    return float(np.sum(bin_discrepancies * prob_total))


def calc_eqace(correct, calib_confids, n_bins: int = 20):
    calib_confids = np.clip(calib_confids, 0.0, 1.0)
    y_true = sk_utils.column_or_1d(correct).astype(np.float64)
    y_prob = sk_utils.column_or_1d(calib_confids).astype(np.float64)

    if y_prob.size == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0 + 1e-8
    bin_edges = np.maximum.accumulate(bin_edges)

    binids = np.digitize(y_prob, bin_edges) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    num_nonzero = int(nonzero.sum())
    if num_nonzero == 0:
        return float("nan")

    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    bin_discrepancies = np.abs(prob_true - prob_pred)
    return float((1.0 / num_nonzero) * np.sum(bin_discrepancies))


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
        prob_true     = self.bin_true[nonzero]  / self.bin_total[nonzero]
        prob_pred     = self.bin_sums[nonzero]  / self.bin_total[nonzero]
        discrepancies = np.abs(prob_true - prob_pred)
        return float((1.0 / num_nonzero) * np.sum(discrepancies))

    def compute_ece(self):
        """Return the global ECE computed from all accumulated data."""
        nonzero = self.bin_total > 0
        total = float(self.bin_total.sum())
        if total == 0.0:
            return float("nan")
        prob_true = self.bin_true[nonzero] / self.bin_total[nonzero]
        prob_pred = self.bin_sums[nonzero] / self.bin_total[nonzero]
        discrepancies = np.abs(prob_true - prob_pred)
        prob_total = self.bin_total[nonzero] / total
        return float(np.sum(discrepancies * prob_total))


def calibration_error(exp_dataloader: ExperimentDataloader, ignore_value=None):
    calib_dict = {}
    calib_dict["mean"] = {}
    for unc_type in exp_dataloader.exp_version.unc_types:
        aces_unc = []
        eces_unc = []
        eqaces_unc = []
        global_accum = GlobalCalibAccumulator()
        for image_id in tqdm(exp_dataloader.image_ids):
            if image_id not in calib_dict.keys():
                calib_dict[image_id] = {}
            reference_segs = exp_dataloader.get_reference_segs(image_id)
            pred_seg = exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = exp_dataloader.get_unc_map(image_id, unc_type)
            
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                import warnings
                warnings.warn(
                    f"Uncertainty map shape {unc_map.shape} does not match pred_seg shape {pred_seg.shape} for image {image_id}. Attempting to swap axes."
                )
                unc_map = np.swapaxes(unc_map, 0, 1)
            n_gt = reference_segs.shape[0]
            pred_seg = np.repeat(pred_seg[np.newaxis, :], n_gt, 0)
            unc_map = np.repeat(unc_map[np.newaxis, :], n_gt, 0)
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
                correct_vals = rater_correct[ignore_mask]
                ace = calc_ace(correct_vals, unc_map)
                ece = calc_ece(correct_vals, unc_map)
                eqace = calc_eqace(correct_vals, unc_map)
                global_accum.accumulate(correct_vals, unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace, "ece": ece, "eqace": eqace}}
                aces_unc.append(ace)
                eces_unc.append(ece)
                eqaces_unc.append(eqace)
            else:
                unc_map = platt_scale_confid(
                    -unc_map.flatten(),
                    platt_scale_file=platt_scale_file,
                    uncertainty=unc_type,
                )
                correct_vals = rater_correct.flatten()
                ace = calc_ace(correct_vals, unc_map)
                ece = calc_ece(correct_vals, unc_map)
                eqace = calc_eqace(correct_vals, unc_map)
                global_accum.accumulate(correct_vals, unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace, "ece": ece, "eqace": eqace}}
                aces_unc.append(ace)
                eces_unc.append(ece)
                eqaces_unc.append(eqace)
        calib_dict["mean"][unc_type] = {
            "metrics": {
                "ace": np.mean(np.array(aces_unc)),
                "ece": np.mean(np.array(eces_unc)),
                "eqace": np.mean(np.array(eqaces_unc)),
                "gace": global_accum.compute_ace(),
                "gece": global_accum.compute_ece(),
            }
        }
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
