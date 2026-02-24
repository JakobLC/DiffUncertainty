"""
Debug script: run Platt scale parameter estimation for
  saves/origlidc128/test_results/prob_unet_swag_diag_0/e1000_ema
on the "id" dataset split and print estimated a and b values.

Run from values/:
    python global_utils/debug_platt_scale.py
"""
import sys
import warnings
from pathlib import Path

# Ensure values/ and values/evaluation/ are importable
_values_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_values_dir / "evaluation"))  # for bare imports (experiment_version, etc.)
sys.path.insert(0, str(_values_dir))                 # for package imports (evaluation.*, uncertainty_modeling.*)

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sklearn.calibration import _sigmoid_calibration as calib
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader
from experiment_version import ExperimentVersion


def platt_scale_params_debug(val_exp_dataloader: ExperimentDataloader, ignore_value=None, n_bins: int = 256):
    """Compute Platt scaling parameters and print estimated a and b — no file I/O."""
    bin_edges = np.logspace(-12, 2, num=n_bins + 1, dtype=np.float64)

    for unc_type in val_exp_dataloader.exp_version.unc_types:
        pos_counts = np.zeros(n_bins, dtype=np.int64)
        neg_counts = np.zeros(n_bins, dtype=np.int64)
        sum_unc = np.zeros(n_bins, dtype=np.float64)
        total_counts = np.zeros(n_bins, dtype=np.int64)
        saw_oob = False
        oob_low_count = 0
        oob_high_count = 0

        for image_id in tqdm(val_exp_dataloader.image_ids, desc=f"[{unc_type}] accumulating bins"):
            reference_segs = val_exp_dataloader.get_reference_segs(image_id)
            pred_seg = val_exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = val_exp_dataloader.get_unc_map(image_id, unc_type)
            reference_segs = np.asarray(reference_segs)
            pred_seg = np.asarray(pred_seg)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            assert reference_segs.ndim == pred_seg.ndim + 1, (
                f"Reference segs should have shape (n_raters, W, H), pred_seg (W, H). "
                f"Found {reference_segs.shape} vs {pred_seg.shape}"
            )
            assert reference_segs.shape[1:] == pred_seg.shape, (
                f"Reference segs and pred_seg spatial dimensions must match. "
                f"Found {reference_segs.shape} vs {pred_seg.shape}"
            )
            rater_correct = (reference_segs == pred_seg[None, ...])

            if ignore_value is not None:
                valid_mask = reference_segs != ignore_value
            else:
                valid_mask = np.ones(reference_segs.shape, dtype=bool)

            u = np.broadcast_to(unc_map[None, ...], reference_segs.shape)[valid_mask].ravel()
            c = rater_correct[valid_mask].ravel().astype(np.int8)

            if u.size == 0:
                continue

            bin_idx = np.digitize(u, bin_edges) - 1
            oob_low = bin_idx < 0
            oob_high = bin_idx >= n_bins
            if oob_low.any() or oob_high.any():
                saw_oob = True
                oob_low_count += int(oob_low.sum())
                oob_high_count += int(oob_high.sum())
                bin_idx[oob_low] = 0
                bin_idx[oob_high] = n_bins - 1

            sum_unc += np.bincount(bin_idx, weights=u, minlength=n_bins)
            total_counts += np.bincount(bin_idx, minlength=n_bins)
            if (c == 1).any():
                pos_counts += np.bincount(bin_idx[c == 1], minlength=n_bins)
            if (c == 0).any():
                neg_counts += np.bincount(bin_idx[c == 0], minlength=n_bins)

        if saw_oob:
            warnings.warn(
                f"Uncertainty values outside [1e-12, 1e2] for '{unc_type}': "
                f"{oob_low_count} below 1e-12, {oob_high_count} above 1e2. "
                "They were clamped to the nearest end bin.",
                RuntimeWarning,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_unc = np.divide(
                sum_unc, total_counts,
                out=np.zeros_like(sum_unc),
                where=total_counts > 0,
            )

        samples_F = []
        samples_y = []
        samples_w = []
        for b in range(n_bins):
            if total_counts[b] == 0:
                continue
            F_b = -mean_unc[b]
            if pos_counts[b] > 0:
                samples_F.append(F_b)
                samples_y.append(1)
                samples_w.append(int(pos_counts[b]))
            if neg_counts[b] > 0:
                samples_F.append(F_b)
                samples_y.append(0)
                samples_w.append(int(neg_counts[b]))

        if len(samples_F) == 0:
            a, b_param = 0.0, 0.0
        else:
            F_arr = np.asarray(samples_F, dtype=np.float64)
            y_arr = np.asarray(samples_y, dtype=np.float64)
            w_arr = np.asarray(samples_w, dtype=np.float64)
            a, b_param = calib(F_arr, y_arr, sample_weight=w_arr)

        print(f"\n[{unc_type}]  a = {a:.8f},  b = {b_param:.8f}")

        # ── scatter plot ──────────────────────────────────────────────────────
        # Collect per-bin (mean_unc, label, weight) for non-empty bins only
        plot_x_pos, plot_x_neg = [], []
        plot_w_pos, plot_w_neg = [], []
        for b in range(n_bins):
            if total_counts[b] == 0:
                continue
            mu = mean_unc[b]
            if pos_counts[b] > 0:
                plot_x_pos.append(mu)
                plot_w_pos.append(int(pos_counts[b]))
            if neg_counts[b] > 0:
                plot_x_neg.append(mu)
                plot_w_neg.append(int(neg_counts[b]))

        plot_x_pos = np.asarray(plot_x_pos)
        plot_x_neg = np.asarray(plot_x_neg)
        plot_w_pos = np.asarray(plot_w_pos, dtype=float)
        plot_w_neg = np.asarray(plot_w_neg, dtype=float)

        # Normalise weights to [0.05, 1.0] so even small bins are visible
        def _norm_alpha(w):
            if w.size == 0:
                return w
            return 0.05 + 0.95 * (w - w.min()) / (w.max() - w.min() + 1e-12)

        alpha_pos = _norm_alpha(plot_w_pos)
        alpha_neg = _norm_alpha(plot_w_neg)

        fig, ax = plt.subplots(figsize=(9, 5))

        # Positive bins (label = 1) — one scatter point per bin, alpha ∝ weight
        for x, al in zip(plot_x_pos, alpha_pos):
            ax.scatter(x, 1, color="steelblue", alpha=float(al), s=40, linewidths=0)

        # Negative bins (label = 0) — one scatter point per bin, alpha ∝ weight
        for x, al in zip(plot_x_neg, alpha_neg):
            ax.scatter(x, 0, color="tomato", alpha=float(al), s=40, linewidths=0)

        # ── correct / incorrect rate curves (normalised to max=1) ──────────
        # Use all non-empty bins regardless of whether pos/neg is zero
        nonempty = total_counts > 0
        curve_x = mean_unc[nonempty]
        sort_idx = np.argsort(curve_x)
        curve_x = curve_x[sort_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            correct_rate   = np.where(nonempty, pos_counts / total_counts.clip(1), 0.0)[nonempty][sort_idx]
            incorrect_rate = np.where(nonempty, neg_counts / total_counts.clip(1), 0.0)[nonempty][sort_idx]

        def _norm_to_max(arr):
            m = arr.max()
            return arr / m if m > 0 else arr

        bin_weight = total_counts[nonempty][sort_idx].astype(float)

        ax.plot(curve_x, _norm_to_max(correct_rate),   color="steelblue", linewidth=1.5,
                linestyle="--", label="correct rate (norm)")
        ax.plot(curve_x, _norm_to_max(incorrect_rate), color="tomato",    linewidth=1.5,
                linestyle="--", label="incorrect rate (norm)")
        ax.plot(curve_x, _norm_to_max(bin_weight),     color="grey",      linewidth=1.5,
                linestyle="--", label="bin weight (norm)")

        # Sigmoid fit: P = 1 / (1 + exp(F*a + b))  where F = -unc
        all_x = np.concatenate([plot_x_pos, plot_x_neg])
        if all_x.size > 0:
            x_line = np.logspace(
                np.log10(max(all_x.min(), 1e-12)),
                np.log10(all_x.max()),
                500,
            )
            F_line = -x_line
            y_line = 1.0 / (1.0 + np.exp(F_line * a + b_param))
            ax.plot(x_line, y_line, color="black", linewidth=2,
                    label=f"sigmoid fit  (a={a:.4f}, b={b_param:.4f})")

        # Dummy handles for the scatter legend
        ax.scatter([], [], color="steelblue", s=40, label="correct (pos bins)")
        ax.scatter([], [], color="tomato",    s=40, label="incorrect (neg bins)")

        #ax.set_xscale("log")
        ax.set_xlabel("mean uncertainty (bin average)")
        ax.set_ylabel("label  (1 = correct, 0 = incorrect)")
        ax.set_title(f"Platt scaling bins — {unc_type}  |  alpha ∝ bin weight")
        ax.legend(loc="center right")
        ax.set_ylim(-0.15, 1.15)
        fig.tight_layout()
        plt.show()


def main():
    # Load the hydra config to obtain the origlidc128 datamodule_config
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
        unc_types=["EU"],
        aggregations=["all"],
        n_reference_segs=4,
        datamodule_config=cfg.datamodule_config,
        # kwargs stored in version_params (seed is required by ExperimentDataloader)
        seed="120",
        seed1="0",
        eu="swag_diag",
    )

    print(f"exp_path : {exp_version.exp_path}")
    assert exp_version.exp_path.exists(), f"Experiment path not found: {exp_version.exp_path}"

    id_dataloader = ExperimentDataloader(exp_version, "id")
    print(f"dataset_path : {id_dataloader.dataset_path}")
    print(f"n_images     : {len(id_dataloader.image_ids)}")
    print(f"unc_types    : {exp_version.unc_types}")
    print()

    platt_scale_params_debug(id_dataloader, ignore_value=None)


if __name__ == "__main__":
    main()
