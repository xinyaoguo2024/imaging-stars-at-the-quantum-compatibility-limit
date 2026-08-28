from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_hawaii3_rml_strategy_comparison as ngc_fig3
import run_hawaii3_rml_strategy_comparison_3c273 as threec
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_presentation_diagnostics_20260524"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#8d99ae"),
    ("split", "Edge-first closure", "split_dirty", "#0077b6"),
    ("direct", "Direct closure", "direct_dirty", "#d00000"),
]


def configure_common(source: ngc.SourceModel) -> None:
    amp_rml.SOURCE = source
    amp_rml.N_RML = 64
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = 600.0
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.FIBER_LOSS_DB_PER_KM = 0.2
    amp_rml.MODE_FALSE_POSITIVE = 0.05
    amp_rml.PAIR_FALSE_POSITIVE = 0.0
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.PRIOR_WEIGHT = 0.10
    amp_rml.TV_WEIGHT = 0.045
    amp_rml.ENTROPY_WEIGHT = 0.010
    amp_rml.AMP_GRAD_WEIGHT = 4.0
    amp_rml.PHASE_GRAD_WEIGHT = 1.5

    val.FIT_N_PIX = 32
    val.ADAM_ITER = 2000
    val.ADAM_LR = 0.012
    # Match the stable manuscript-style runs: no early stopping at chi^2 ~ 1.
    # The additional Adam steps mostly sharpen morphology/regularizer balance.
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0


def simulate_dataset(source_key: str):
    if source_key == "ngc4151":
        configure_common(ngc.NGC4151)
        case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
        bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
        source = ngc.NGC4151
    elif source_key == "3c273":
        threec.configure_fig3_runtime()
        case = threec.make_3c273_case()
        bands, stats, truth, axis_uas = threec.simulate_3c273_case(case)
        source = threec.SOURCE_3C273
    else:
        raise ValueError(source_key)
    return source, case, bands, stats, truth, axis_uas


def run_full_rml(source_key: str) -> dict:
    source, case, bands, stats, truth, axis_uas = simulate_dataset(source_key)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    results = []
    for strategy, label, start_name, _color in STRATEGIES:
        print(f"[full-rml] {source_key} strategy={strategy}", flush=True)
        results.append(ngc_fig3.run_strategy(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts))
    return {
        "source_key": source_key,
        "source": source,
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "results": results,
    }


def result_map(results: list[dict]) -> dict[str, dict]:
    return {result["strategy"]: result for result in results}


def safe_display(image: np.ndarray) -> np.ndarray:
    return opt.normalize_blr_display(image)


def blr_masks(axis_uas: np.ndarray, source: ngc.SourceModel) -> tuple[np.ndarray, np.ndarray]:
    return ngc.blr_masks_for_source(axis_uas, source)


def angular_profile(
    image: np.ndarray,
    axis_uas: np.ndarray,
    source: ngc.SourceModel,
    *,
    n_bin: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    half_width = max(2.2 * source.blr_width_uas, 10.0)
    mask = (rr > source.blr_radius_uas - half_width) & (rr < source.blr_radius_uas + half_width)
    bins = np.linspace(-np.pi, np.pi, n_bin + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros(n_bin, dtype=float)
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = mask & (theta >= lo) & (theta < hi)
        prof[k] = float(np.mean(image[m])) if np.any(m) else 0.0
    mean = float(np.mean(prof[prof > 0])) if np.any(prof > 0) else 1.0
    return centers, prof / max(mean, 1e-30)


def circular_angle_error_deg(a: float, b: float) -> float:
    return float(np.rad2deg(np.angle(np.exp(1j * (a - b)))))


def radial_peak_radius(image: np.ndarray, axis_uas: np.ndarray, source: ngc.SourceModel) -> float:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.45 * source.blr_radius_uas, 1.65 * source.blr_radius_uas, 46)
    profile = []
    centers = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            profile.append(float(np.mean(image[mask])))
    if not profile:
        return float("nan")
    return float(centers[int(np.argmax(profile))])


def core_centroid(image: np.ndarray, axis_uas: np.ndarray, source: ngc.SourceModel) -> tuple[float, float]:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr < max(0.55 * source.blr_radius_uas, 22.0)
    weights = np.clip(image * mask, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0, 0.0
    return float(np.sum(weights * xx) / total), float(np.sum(weights * yy) / total)


def feature_summary(dataset: dict) -> list[dict[str, float | str]]:
    source = dataset["source"]
    axis = dataset["axis_uas"]
    truth = dataset["truth"]
    theta, truth_prof = angular_profile(truth, axis, source)
    truth_pa = theta[int(np.argmax(truth_prof))]
    truth_asym = abs(np.sum(truth_prof * np.exp(1j * theta))) / max(float(np.sum(truth_prof)), 1e-30)
    truth_radius = radial_peak_radius(truth, axis, source)
    truth_cx, truth_cy = core_centroid(truth, axis, source)
    ring_mask, core_mask = blr_masks(axis, source)
    rows: list[dict[str, float | str]] = []
    for strategy, label, _start_name, _color in STRATEGIES:
        image = result_map(dataset["results"])[strategy]["best"]["image"]
        _, prof = angular_profile(image, axis, source)
        pa = theta[int(np.argmax(prof))]
        asym = abs(np.sum(prof * np.exp(1j * theta))) / max(float(np.sum(prof)), 1e-30)
        radius = radial_peak_radius(image, axis, source)
        cx, cy = core_centroid(image, axis, source)
        profile_rmse = float(np.sqrt(np.mean((prof - truth_prof) ** 2)))
        rows.append(
            {
                "source": dataset["source_key"],
                "strategy": strategy,
                "label": label,
                "global_corr": float(result_map(dataset["results"])[strategy]["best"]["metrics"]["global_corr"]),
                "blr_corr": float(result_map(dataset["results"])[strategy]["best"]["metrics"]["blr_corr"]),
                "ring_contrast": float(opt.ring_contrast(image, ring_mask, core_mask)),
                "radial_corr": float(result_map(dataset["results"])[strategy]["best"]["metrics"].get("radial_corr", np.nan)),
                "profile_rmse": profile_rmse,
                "bright_pa_error_deg": circular_angle_error_deg(pa, truth_pa),
                "ring_radius_error_uas": float(radius - truth_radius),
                "asymmetry_error": float(asym - truth_asym),
                "core_centroid_error_uas": float(math.hypot(cx - truth_cx, cy - truth_cy)),
                "amp_chi2": float(result_map(dataset["results"])[strategy]["best"]["residuals"]["amp_reduced_chi2"]),
                "phase_chi2": float(result_map(dataset["results"])[strategy]["best"]["residuals"]["phase_reduced_chi2"]),
            }
        )
    return rows


def plot_residual_maps(dataset: dict) -> tuple[Path, Path]:
    axis = dataset["axis_uas"]
    truth = dataset["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    truth_disp = safe_display(truth)
    residuals = []
    for strategy, *_ in STRATEGIES:
        residuals.append(safe_display(result_map(dataset["results"])[strategy]["best"]["image"]) - truth_disp)
    vmax = max(0.05, float(np.percentile(np.abs(np.concatenate([r.ravel() for r in residuals])), 99.0)))
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), constrained_layout=True)
    panels = [("truth", "Input", truth)] + [
        (strategy, label, result_map(dataset["results"])[strategy]["best"]["image"])
        for strategy, label, *_ in STRATEGIES
    ]
    for col, (strategy, label, image) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(safe_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(label)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.96,
        (
            "Displayed-image residuals\n"
            r"$\Delta I_{\rm disp}=I_{\rm rec,disp}-I_{\rm true,disp}$"
            "\nCommon diverging color scale."
        ),
        va="top",
        fontsize=10,
    )
    for col, (resid, (strategy, label, *_rest)) in enumerate(zip(residuals, STRATEGIES), start=1):
        ax = axes[1, col]
        im = ax.imshow(resid, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        m = result_map(dataset["results"])[strategy]["best"]["metrics"]
        ax.set_title(f"{label}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"{dataset['source'].name}: RML images and residual maps", weight="bold")
    stem = f"{dataset['source_key']}_residual_maps"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_azimuth_profiles(dataset: dict) -> tuple[Path, Path]:
    source = dataset["source"]
    axis = dataset["axis_uas"]
    theta, truth_prof = angular_profile(dataset["truth"], axis, source)
    deg = np.rad2deg(theta)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(deg, truth_prof, color="black", lw=2.2, label="Input")
    for strategy, label, _start_name, color in STRATEGIES:
        _, prof = angular_profile(result_map(dataset["results"])[strategy]["best"]["image"], axis, source)
        axes[0].plot(deg, prof, color=color, lw=1.6, label=label)
        axes[1].plot(deg, prof - truth_prof, color=color, lw=1.4, label=label)
    axes[0].set_ylabel("annular brightness / mean")
    axes[0].set_title(f"{source.name}: BLR annular azimuthal profile")
    axes[0].legend(ncol=2, frameon=False)
    axes[1].axhline(0.0, color="0.3", lw=0.8)
    axes[1].set_xlabel("position angle on BLR annulus (deg)")
    axes[1].set_ylabel("profile residual")
    axes[1].legend(ncol=3, frameon=False)
    stem = f"{dataset['source_key']}_blr_azimuth_profile"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_feature_errors(dataset: dict, rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), constrained_layout=True)
    metrics = [
        ("global_corr", "Global corr.", False),
        ("blr_corr", "BLR corr.", False),
        ("profile_rmse", "BLR profile RMSE", True),
        ("bright_pa_error_deg", "bright-side PA error (deg)", True),
        ("ring_radius_error_uas", "ring-radius error ($\\mu$as)", True),
        ("phase_chi2", r"phase $\chi^2_\nu$", True),
    ]
    labels = [str(r["strategy"]) for r in rows]
    colors_by_strategy = {strategy: color for strategy, _label, _start, color in STRATEGIES}
    bar_colors = [colors_by_strategy[label] for label in labels]
    for ax, (key, title, absval) in zip(axes.flat, metrics):
        values = [float(r[key]) for r in rows]
        if absval and "error" in key:
            values = [abs(v) for v in values]
        ax.bar(labels, values, color=bar_colors, alpha=0.86)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{dataset['source'].name}: feature-level performance diagnostics", weight="bold")
    stem = f"{dataset['source_key']}_feature_errors"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.abs(values[np.isfinite(values)]))
    if v.size == 0:
        return np.array([0.0]), np.array([0.0])
    y = np.linspace(0.0, 1.0, v.size, endpoint=True)
    return v, y


def plot_residual_cdf(dataset: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for strategy, label, _start, color in STRATEGIES:
        arrays = result_map(dataset["results"])[strategy]["best"]["residual_arrays"]
        x, y = cdf(np.asarray(arrays["amp_z"]))
        axes[0].plot(x, y, color=color, lw=1.8, label=label)
        x, y = cdf(np.asarray(arrays["phase_z"]))
        axes[1].plot(x, y, color=color, lw=1.8, label=label)
    for ax, title in zip(axes, ("Amplitude residual CDF", "phase/closure residual CDF")):
        ax.axvline(1.0, color="0.3", lw=0.8, ls="--")
        ax.axvline(2.0, color="0.5", lw=0.8, ls=":")
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"absolute normalized residual $|z|$")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{dataset['source'].name}: data-domain residual diagnostics", weight="bold")
    stem = f"{dataset['source_key']}_residual_cdf"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_quick_ensemble_ngc(n_seed: int = 3) -> list[dict[str, float | str]]:
    print(f"[ensemble] NGC4151 quick ensemble with {n_seed} seeds", flush=True)
    configure_common(ngc.NGC4151)
    old = {
        "iter": val.ADAM_ITER,
        "fit_n": val.FIT_N_PIX,
        "seed": wt.RNG_SEED,
        "n_rml": amp_rml.N_RML,
        "n_time": amp_rml.N_TIME_WINDOWS,
        "lambda_step": getattr(amp_rml.aug, "LAMBDA_STEP_NM", 10.0),
        "aug_lambda_step": getattr(amp_rml.aug, "LAMBDA_STEP_NM", 10.0),
        "wt_npix": wt.N_PIX,
    }
    # This is only a stability-display stress test.  Use a reduced data grid so
    # the ensemble finishes quickly; production numbers must use the full run.
    val.ADAM_ITER = 220
    val.FIT_N_PIX = 24
    amp_rml.N_RML = 48
    amp_rml.N_TIME_WINDOWS = 10
    amp_rml.aug.LAMBDA_STEP_NM = 40.0
    wt.N_PIX = 48
    rows: list[dict[str, float | str]] = []
    try:
        for seed_index in range(n_seed):
            print(f"[ensemble] seed={seed_index}", flush=True)
            wt.RNG_SEED = 20260515 + 771 + 1009 * seed_index
            source, case, bands, _stats, truth, axis = simulate_dataset("ngc4151")
            prior_full = amp_rml.broad_gaussian_prior(axis)
            prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
            starts = val.build_starts(bands, truth, prior_full)
            for strategy, label, start_name, _color in STRATEGIES:
                print(f"[ensemble] seed={seed_index} strategy={strategy}", flush=True)
                old_strategy = val.STRATEGY
                old_optimizer = val.OPTIMIZER
                val.STRATEGY = strategy
                val.OPTIMIZER = "adam"
                config = {
                    "label": "quick_ensemble",
                    "prior": amp_rml.PRIOR_WEIGHT,
                    "tv": amp_rml.TV_WEIGHT,
                    "entropy": amp_rml.ENTROPY_WEIGHT,
                    "step": amp_rml.STEP,
                }
                result = val.run_single_reconstruction(
                    case=case,
                    bands=bands,
                    truth=truth,
                    axis_uas=axis,
                    prior=prior,
                    start_name=start_name,
                    start=starts[start_name],
                    config=config,
                    split_label="quick_seed_ensemble",
                )
                val.STRATEGY = old_strategy
                val.OPTIMIZER = old_optimizer
                metrics = result["metrics"]
                residuals = result["residuals"]
                rows.append(
                    {
                        "seed_index": seed_index,
                        "strategy": strategy,
                        "label": label,
                        "global_corr": float(metrics["global_corr"]),
                        "blr_corr": float(metrics["blr_corr"]),
                        "radial_corr": float(metrics.get("radial_corr", np.nan)),
                        "amp_chi2": float(residuals["amp_reduced_chi2"]),
                        "phase_chi2": float(residuals["phase_reduced_chi2"]),
                        "adam_iter": float(val.ADAM_ITER),
                        "quick_n_time_windows": float(amp_rml.N_TIME_WINDOWS),
                        "quick_lambda_step_nm": float(amp_rml.aug.LAMBDA_STEP_NM),
                        "quick_n_rml": float(amp_rml.N_RML),
                    }
                )
    finally:
        val.ADAM_ITER = old["iter"]
        val.FIT_N_PIX = old["fit_n"]
        wt.RNG_SEED = old["seed"]
        amp_rml.N_RML = old["n_rml"]
        amp_rml.N_TIME_WINDOWS = old["n_time"]
        amp_rml.aug.LAMBDA_STEP_NM = old["aug_lambda_step"]
        wt.N_PIX = old["wt_npix"]
    return rows


def plot_ensemble(rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), constrained_layout=True)
    keys = [("blr_corr", "BLR correlation"), ("global_corr", "global correlation"), ("phase_chi2", r"phase $\chi^2_\nu$")]
    strategies = [s[0] for s in STRATEGIES]
    colors_by_strategy = {strategy: color for strategy, _label, _start, color in STRATEGIES}
    for ax, (key, title) in zip(axes, keys):
        data = [[float(r[key]) for r in rows if r["strategy"] == strategy] for strategy in strategies]
        bp = ax.boxplot(data, tick_labels=strategies, patch_artist=True, showmeans=True)
        for patch, strategy in zip(bp["boxes"], strategies):
            patch.set_facecolor(colors_by_strategy[strategy])
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("NGC4151 quick stability ensemble; 3 seeds, reduced data grid, single start", weight="bold")
    png = OUT / "ngc4151_quick_noise_seed_ensemble.png"
    pdf = OUT / "ngc4151_quick_noise_seed_ensemble.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary_pdf(figure_paths: list[Path], payload: dict) -> Path:
    pdf_path = OUT / "rml_presentation_diagnostics_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 6.0))
        fig.text(0.06, 0.92, "RML presentation diagnostics", fontsize=18, weight="bold")
        fig.text(
            0.06,
            0.78,
            (
                "Goal: test display choices that make strategy differences visible when RML images are smooth.\n"
                "No manuscript files were modified.  Main diagnostics: residual maps, BLR azimuthal profiles,\n"
                "feature-level error bars, data-domain residual CDFs, and a quick NGC4151 noise-seed ensemble.\n\n"
                "Key caveat: the ensemble is a presentation sanity check, not a final production benchmark:\n"
                "single dirty start and 650 Adam steps are used to keep the scan lightweight."
            ),
            fontsize=10,
            va="top",
        )
        y = 0.48
        for source_key, rows in payload["feature_rows_by_source"].items():
            fig.text(0.06, y, f"{source_key} best single-seed metrics:", fontsize=12, weight="bold")
            y -= 0.045
            for row in rows:
                fig.text(
                    0.08,
                    y,
                    (
                        f"{row['strategy']}: global={float(row['global_corr']):.3f}, "
                        f"BLR={float(row['blr_corr']):.3f}, profile RMSE={float(row['profile_rmse']):.3f}, "
                        f"phase chi2={float(row['phase_chi2']):.3f}"
                    ),
                    fontsize=9,
                )
                y -= 0.035
            y -= 0.025
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for path in figure_paths:
            image_path = path.with_suffix(".png")
            if not image_path.exists():
                continue
            img = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(10.5, 7.2))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(image_path.name, fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def main() -> None:
    figure_pdfs: list[Path] = []
    feature_rows_all: list[dict[str, float | str]] = []
    payload: dict = {"feature_rows_by_source": {}, "figures": []}

    skip_full = os.environ.get("RML_DIAG_SKIP_FULL", "0") == "1"
    existing_feature_csv = OUT / "single_seed_feature_metrics.csv"
    if skip_full and existing_feature_csv.exists():
        with existing_feature_csv.open() as f:
            feature_rows_all = list(csv.DictReader(f))
        for row in feature_rows_all:
            payload["feature_rows_by_source"].setdefault(row["source"], []).append(row)
        for source_key in ("ngc4151", "3c273"):
            for suffix in ("residual_maps", "blr_azimuth_profile", "residual_cdf", "feature_errors"):
                pdf = OUT / f"{source_key}_{suffix}.pdf"
                png = OUT / f"{source_key}_{suffix}.png"
                if pdf.exists() and png.exists():
                    figure_pdfs.append(pdf)
                    payload["figures"].append({"pdf": str(pdf), "png": str(png)})
    else:
        for source_key in ("ngc4151", "3c273"):
            dataset = run_full_rml(source_key)
            rows = feature_summary(dataset)
            feature_rows_all.extend(rows)
            payload["feature_rows_by_source"][source_key] = rows
            for plotter in (plot_residual_maps, plot_azimuth_profiles, plot_residual_cdf):
                pdf, png = plotter(dataset)
                figure_pdfs.append(pdf)
                payload["figures"].append({"pdf": str(pdf), "png": str(png)})
            pdf, png = plot_feature_errors(dataset, rows)
            figure_pdfs.append(pdf)
            payload["figures"].append({"pdf": str(pdf), "png": str(png)})
        write_csv(OUT / "single_seed_feature_metrics.csv", feature_rows_all)

    ensemble_csv = OUT / "ngc4151_quick_noise_seed_ensemble.csv"
    ensemble_pdf = OUT / "ngc4151_quick_noise_seed_ensemble.pdf"
    ensemble_png = OUT / "ngc4151_quick_noise_seed_ensemble.png"
    if skip_full and ensemble_csv.exists() and ensemble_pdf.exists() and ensemble_png.exists():
        with ensemble_csv.open() as f:
            ensemble_rows = list(csv.DictReader(f))
        pdf, png = ensemble_pdf, ensemble_png
    else:
        ensemble_rows = run_quick_ensemble_ngc(n_seed=3)
        write_csv(ensemble_csv, ensemble_rows)
        pdf, png = plot_ensemble(ensemble_rows)
    figure_pdfs.append(pdf)
    payload["figures"].append({"pdf": str(pdf), "png": str(png)})
    payload["ensemble_note"] = (
        "NGC4151 only; 3 seeds; reduced data grid; single dirty start; 220 Adam steps. "
        "This is a visualization/stability sanity check, not a production benchmark."
    )

    summary_pdf = make_summary_pdf(figure_pdfs, payload)
    payload["summary_pdf"] = str(summary_pdf)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        "# RML presentation diagnostics\n\n"
        "This folder tests visualization strategies for showing direct-closure advantages when RML images look smooth.\n\n"
        "- `*_residual_maps.*`: images plus common-scale displayed residual maps.\n"
        "- `*_blr_azimuth_profile.*`: annular BLR brightness versus position angle.\n"
        "- `*_feature_errors.*`: feature-level metrics and errors.\n"
        "- `*_residual_cdf.*`: amplitude and phase/closure normalized residual CDFs.\n"
        "- `ngc4151_quick_noise_seed_ensemble.*`: lightweight seed-to-seed stability check.\n"
        "- `rml_presentation_diagnostics_summary.pdf`: one-file visual summary.\n\n"
        "No manuscript source was modified.\n",
    )
    print(OUT)
    print(summary_pdf)
    print(OUT / "single_seed_feature_metrics.csv")
    print(OUT / "ngc4151_quick_noise_seed_ensemble.csv")

    # Keep a lightweight copy of the script in the output folder for provenance.
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)


if __name__ == "__main__":
    main()
