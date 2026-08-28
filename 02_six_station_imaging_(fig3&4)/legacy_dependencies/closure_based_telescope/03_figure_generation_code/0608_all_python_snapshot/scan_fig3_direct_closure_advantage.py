from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hawaii3_compact_case
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


FALSE_POSITIVES = [0.02, 0.035, 0.05, 0.075, 0.10]
TARGET_AB_MAGS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
BANDS_NM = [(550.0, 650.0), (500.0, 700.0), (450.0, 750.0), (400.0, 800.0)]

SCAN_N_PIX = 64
SCAN_COARSE_BINS = 36
RECON_MODE = "coarse_interp"


def scaled_source_for_mag(source: ngc.SourceModel, target_mag_550: float) -> ngc.SourceModel:
    current_mag = ngc.sed_effective_ab_mag(source, 550.0)
    scale = 10.0 ** (-0.4 * (target_mag_550 - current_mag))
    return replace(
        source,
        sed_fnu_mjy=tuple(float(scale * value) for value in source.sed_fnu_mjy),
        note=source.note + f" SED scaled to m_AB(550 nm)={target_mag_550:.2f} for Fig.3 parameter scan.",
    )


def configure_for_scan(false_positive: float, band_nm: tuple[float, float]) -> dict:
    old = {
        "aug_observing_days": aug.OBSERVING_DAYS,
        "aug_n_time": aug.N_TIME_WINDOWS,
        "aug_exp": aug.EXPOSURE_S,
        "aug_gap": aug.EXPOSURE_GAP_S,
        "aug_lmin": aug.LAMBDA_MIN_NM,
        "aug_lmax": aug.LAMBDA_MAX_NM,
        "aug_lstep": aug.LAMBDA_STEP_NM,
        "aug_loss": aug.FIBER_LOSS_DB_PER_KM,
        "aug_len_scale": aug.FIBER_LENGTH_SCALE,
        "aug_mode_fp": aug.MODE_FALSE_POSITIVE,
        "aug_pair_fp": aug.PAIR_FALSE_POSITIVE,
        "aug_base_fp": aug.BASELINE_FALSE_POSITIVE,
        "aug_drift": aug.POST_AVERAGE_DRIFT_STD,
        "wt_npix": wt.N_PIX,
        "wt_half_width": wt.HALF_WIDTH_UAS,
        "wt_snr_boost": wt.SNR_BOOST,
        "wt_observing_days": wt.OBSERVING_DAYS,
        "wt_base_fp": wt.BASELINE_FALSE_POSITIVE,
        "wt_amp_fp": wt.AMPLITUDE_MODE_FALSE_POSITIVE,
        "wt_bins_u": wt.COARSE_BINS_U,
        "wt_bins_v": wt.COARSE_BINS_V,
    }
    aug.OBSERVING_DAYS = 30
    aug.N_TIME_WINDOWS = 36
    aug.EXPOSURE_S = 600.0
    aug.EXPOSURE_GAP_S = 150.0
    aug.LAMBDA_MIN_NM = float(band_nm[0])
    aug.LAMBDA_MAX_NM = float(band_nm[1])
    aug.LAMBDA_STEP_NM = 10.0
    aug.FIBER_LOSS_DB_PER_KM = 0.20
    aug.FIBER_LENGTH_SCALE = 0.75
    aug.MODE_FALSE_POSITIVE = float(false_positive)
    aug.PAIR_FALSE_POSITIVE = 0.0
    aug.BASELINE_FALSE_POSITIVE = 0.0
    aug.POST_AVERAGE_DRIFT_STD = math.pi / 5.0

    wt.N_PIX = SCAN_N_PIX
    wt.HALF_WIDTH_UAS = aug.HALF_WIDTH_UAS
    wt.SNR_BOOST = 1.0
    wt.OBSERVING_DAYS = aug.OBSERVING_DAYS
    wt.BASELINE_FALSE_POSITIVE = 0.0
    wt.AMPLITUDE_MODE_FALSE_POSITIVE = float(false_positive)
    wt.COARSE_BINS_U = SCAN_COARSE_BINS
    wt.COARSE_BINS_V = SCAN_COARSE_BINS
    return old


def restore_settings(old: dict) -> None:
    aug.OBSERVING_DAYS = old["aug_observing_days"]
    aug.N_TIME_WINDOWS = old["aug_n_time"]
    aug.EXPOSURE_S = old["aug_exp"]
    aug.EXPOSURE_GAP_S = old["aug_gap"]
    aug.LAMBDA_MIN_NM = old["aug_lmin"]
    aug.LAMBDA_MAX_NM = old["aug_lmax"]
    aug.LAMBDA_STEP_NM = old["aug_lstep"]
    aug.FIBER_LOSS_DB_PER_KM = old["aug_loss"]
    aug.FIBER_LENGTH_SCALE = old["aug_len_scale"]
    aug.MODE_FALSE_POSITIVE = old["aug_mode_fp"]
    aug.PAIR_FALSE_POSITIVE = old["aug_pair_fp"]
    aug.BASELINE_FALSE_POSITIVE = old["aug_base_fp"]
    aug.POST_AVERAGE_DRIFT_STD = old["aug_drift"]
    wt.N_PIX = old["wt_npix"]
    wt.HALF_WIDTH_UAS = old["wt_half_width"]
    wt.SNR_BOOST = old["wt_snr_boost"]
    wt.OBSERVING_DAYS = old["wt_observing_days"]
    wt.BASELINE_FALSE_POSITIVE = old["wt_base_fp"]
    wt.AMPLITUDE_MODE_FALSE_POSITIVE = old["wt_amp_fp"]
    wt.COARSE_BINS_U = old["wt_bins_u"]
    wt.COARSE_BINS_V = old["wt_bins_v"]


def reconstruct_coarse_only(bands: list[dict[str, np.ndarray]], truth: np.ndarray) -> dict[str, np.ndarray]:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    images = {key: np.zeros_like(truth) for key in ("all", "split", "direct")}
    weights = {key: 0.0 for key in images}
    for band in bands:
        for key in images:
            image, weight = wt.reconstruct_band_coarse_interp(band, key, fov_rad)
            images[key] += weight * image
            weights[key] += weight
    return {key: wt.normalize_stack(images[key] / max(weights[key], 1e-30)) for key in images}


def simulate_one(
    case: aug.NetworkCase,
    false_positive: float,
    target_mag: float,
    band_nm: tuple[float, float],
) -> dict:
    source = scaled_source_for_mag(ngc.NGC4151, target_mag)
    old = configure_for_scan(false_positive, band_nm)
    try:
        with ngc.patched_source(source):
            bands, stats, truth, axis_uas = wt.simulate_bands(case)
        images = reconstruct_coarse_only(bands, truth)
        metrics = {key: ngc.image_metrics(truth, image, axis_uas, source) for key, image in images.items()}
    finally:
        restore_settings(old)

    best_competitor_blr = max(metrics["all"]["blr_corr"], metrics["split"]["blr_corr"])
    best_competitor_global = max(metrics["all"]["global_corr"], metrics["split"]["global_corr"])
    direct_margin_blr = metrics["direct"]["blr_corr"] - best_competitor_blr
    direct_margin_global = metrics["direct"]["global_corr"] - best_competitor_global
    direct_margin_ring = metrics["direct"]["ring_contrast"] - max(
        metrics["all"]["ring_contrast"], metrics["split"]["ring_contrast"]
    )
    advantage_score = direct_margin_blr + 0.35 * direct_margin_global + 0.15 * direct_margin_ring
    row = {
        "false_positive": float(false_positive),
        "target_ab_mag_550": float(target_mag),
        "lambda_min_nm": float(band_nm[0]),
        "lambda_max_nm": float(band_nm[1]),
        "bandwidth_nm": float(band_nm[1] - band_nm[0]),
        "effective_ab_mag_550": ngc.sed_effective_ab_mag(source, 550.0),
        "n_bands": int((band_nm[1] - band_nm[0]) / 10.0),
        "amp_snr_median": stats["amplitude_snr"]["median"],
        "amp_sigma_median": stats["amplitude_sigma_abs"]["median"],
        "direct_margin_blr": float(direct_margin_blr),
        "direct_margin_global": float(direct_margin_global),
        "direct_margin_ring_contrast": float(direct_margin_ring),
        "advantage_score": float(advantage_score),
    }
    for strategy in ("all", "split", "direct"):
        for name, value in metrics[strategy].items():
            row[f"{strategy}_{name}"] = float(value)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scan(rows: list[dict], path_pdf: Path, path_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.8,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )
    mags = np.array(TARGET_AB_MAGS)
    fps = np.array(FALSE_POSITIVES)
    fig, axes = plt.subplots(2, len(BANDS_NM), figsize=(11.0, 5.0), constrained_layout=True, sharex=True, sharey=True)
    for col, band in enumerate(BANDS_NM):
        subset = [r for r in rows if r["lambda_min_nm"] == band[0] and r["lambda_max_nm"] == band[1]]
        grid_adv = np.full((len(fps), len(mags)), np.nan)
        grid_direct = np.full_like(grid_adv, np.nan)
        for r in subset:
            i = int(np.where(np.isclose(fps, r["false_positive"]))[0][0])
            j = int(np.where(np.isclose(mags, r["target_ab_mag_550"]))[0][0])
            grid_adv[i, j] = r["direct_margin_blr"]
            grid_direct[i, j] = r["direct_blr_corr"]

        ax = axes[0, col]
        im = ax.imshow(
            grid_adv,
            origin="lower",
            aspect="auto",
            extent=[mags[0] - 0.5, mags[-1] + 0.5, fps[0], fps[-1]],
            cmap="coolwarm",
            vmin=-0.12,
            vmax=0.12,
        )
        ax.set_title(f"{band[0]:.0f}-{band[1]:.0f} nm\nDirect BLR margin")
        if col == 0:
            ax.set_ylabel(r"$p_{\rm fp}$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, col]
        im = ax.imshow(
            grid_direct,
            origin="lower",
            aspect="auto",
            extent=[mags[0] - 0.5, mags[-1] + 0.5, fps[0], fps[-1]],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title("Direct BLR corr")
        ax.set_xlabel(r"$m_{\rm AB}(550\,{\rm nm})$")
        if col == 0:
            ax.set_ylabel(r"$p_{\rm fp}$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Fig.3 quick scan: Hawaii top4 + remote3, coarse nonparametric reconstruction, 30 nights",
        fontsize=10.5,
        weight="bold",
    )
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    rows = []
    total = len(FALSE_POSITIVES) * len(TARGET_AB_MAGS) * len(BANDS_NM)
    counter = 0
    for band in BANDS_NM:
        for false_positive in FALSE_POSITIVES:
            for mag in TARGET_AB_MAGS:
                counter += 1
                print(
                    f"[{counter:03d}/{total}] band={band[0]:.0f}-{band[1]:.0f} nm "
                    f"p_fp={false_positive:g} m={mag:g}",
                    flush=True,
                )
                rows.append(simulate_one(case, false_positive, mag, band))

    rows_sorted = sorted(rows, key=lambda r: r["advantage_score"], reverse=True)
    csv_path = OUTFIG / "fig3_direct_closure_advantage_scan_quick.csv"
    json_path = OUTFIG / "fig3_direct_closure_advantage_scan_quick.json"
    pdf_path = OUTFIG / "fig3_direct_closure_advantage_scan_quick_heatmaps.pdf"
    png_path = OUTFIG / "fig3_direct_closure_advantage_scan_quick_heatmaps.png"
    write_csv(rows_sorted, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "scan_model": "quick coarse nonparametric screen; no manuscript files changed",
                "case": case.key,
                "reconstruction": {
                    "mode": RECON_MODE,
                    "n_pix": SCAN_N_PIX,
                    "coarse_bins": SCAN_COARSE_BINS,
                },
                "fixed_parameters": {
                    "observing_days": 30,
                    "n_time_windows": 36,
                    "exposure_s": 600.0,
                    "fiber_loss_db_per_km": 0.20,
                    "fiber_length_scale": 0.75,
                    "post_average_drift_std_rad": math.pi / 5.0,
                    "snr_boost": 1.0,
                },
                "top_12": rows_sorted[:12],
                "rows": rows_sorted,
                "outputs": {
                    "csv": str(csv_path),
                    "heatmap_pdf": str(pdf_path),
                    "heatmap_png": str(png_path),
                },
            },
            indent=2,
        )
        + "\n"
    )
    plot_scan(rows_sorted, pdf_path, png_path)
    print(csv_path)
    print(json_path)
    print(pdf_path)
    print(png_path)
    print(json.dumps({"top_8": rows_sorted[:8]}, indent=2))


if __name__ == "__main__":
    main()
