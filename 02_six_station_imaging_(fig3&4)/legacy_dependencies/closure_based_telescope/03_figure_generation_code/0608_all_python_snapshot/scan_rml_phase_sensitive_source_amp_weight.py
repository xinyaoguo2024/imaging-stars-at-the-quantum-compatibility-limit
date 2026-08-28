from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import run_hawaii3_rml_strategy_comparison as strategy_run
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_phase_sensitive_source_amp_scan_20260525"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScanConfig:
    key: str
    morphology: str
    amp_weight: float
    phase_weight: float
    label: str


SCAN_CONFIGS = [
    ScanConfig(
        key="crescent_ampdom",
        morphology="lopsided_crescent",
        amp_weight=4.0,
        phase_weight=1.5,
        label="lopsided crescent, amp-dominated RML",
    ),
    ScanConfig(
        key="crescent_phaseled",
        morphology="lopsided_crescent",
        amp_weight=0.7,
        phase_weight=2.4,
        label="lopsided crescent, phase-led RML",
    ),
    ScanConfig(
        key="spotted_ampdom",
        morphology="lopsided_spotted",
        amp_weight=4.0,
        phase_weight=1.5,
        label="spotted BLR crescent, amp-dominated RML",
    ),
    ScanConfig(
        key="spotted_phaseled",
        morphology="lopsided_spotted",
        amp_weight=0.7,
        phase_weight=2.4,
        label="spotted BLR crescent, phase-led RML",
    ),
]

STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#8d99ae"),
    ("split", "Edge-first closure", "split_dirty", "#0077b6"),
    ("direct", "Direct closure", "direct_dirty", "#d00000"),
]


def configure_runtime(config: ScanConfig) -> None:
    amp_rml.SOURCE = ngc.NGC4151
    ngc.SOURCE_MORPHOLOGY = config.morphology
    amp_rml.N_RML = 64
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = 600.0
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.FIBER_LOSS_DB_PER_KM = 0.2
    amp_rml.MODE_FALSE_POSITIVE = 0.05
    amp_rml.PAIR_FALSE_POSITIVE = 0.0
    amp_rml.SNR_BOOST = 1.0
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.PRIOR_WEIGHT = 0.10
    amp_rml.TV_WEIGHT = 0.045
    amp_rml.ENTROPY_WEIGHT = 0.010
    amp_rml.AMP_GRAD_WEIGHT = config.amp_weight
    amp_rml.PHASE_GRAD_WEIGHT = config.phase_weight
    amp_rml.STEP = 0.018

    val.FIT_N_PIX = 32
    val.ADAM_ITER = 1600
    val.ADAM_LR = 0.012
    # Do not stop at chi^2 ~ 1; morphology keeps improving after that point.
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0


def simulate_by_morphology(morphology: str):
    dummy = ScanConfig(
        key=f"{morphology}_sim",
        morphology=morphology,
        amp_weight=4.0,
        phase_weight=1.5,
        label=morphology,
    )
    configure_runtime(dummy)
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    print(f"[simulate] morphology={morphology} case={case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return {
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "prior": prior,
        "starts": starts,
    }


def angular_profile(image: np.ndarray, axis_uas: np.ndarray, source: ngc.SourceModel, n_bin: int = 72):
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
    prof /= max(float(np.mean(prof[prof > 0])), 1e-30) if np.any(prof > 0) else 1.0
    return centers, prof


def profile_rmse(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    _, t_prof = angular_profile(truth, axis_uas, ngc.NGC4151)
    _, i_prof = angular_profile(image, axis_uas, ngc.NGC4151)
    return float(np.sqrt(np.mean((i_prof - t_prof) ** 2)))


def run_config(config: ScanConfig, data: dict) -> list[dict]:
    configure_runtime(config)
    results = []
    for strategy, label, start_name, _color in STRATEGIES:
        print(f"[rml] {config.key}: {strategy}", flush=True)
        result = strategy_run.run_strategy(
            strategy,
            label,
            start_name,
            data["case"],
            data["bands"],
            data["truth"],
            data["axis_uas"],
            data["prior"],
            data["starts"],
        )
        results.append(result)
    return results


def result_row(config: ScanConfig, result: dict, truth: np.ndarray, axis_uas: np.ndarray) -> dict:
    best = result["best"]
    metrics = best["metrics"]
    residuals = best["residuals"]
    return {
        "config": config.key,
        "morphology": config.morphology,
        "amp_weight": config.amp_weight,
        "phase_weight": config.phase_weight,
        "strategy": result["strategy"],
        "best_start": best["start"],
        "validation_score": float(best["validation_score"]),
        "global_corr": float(metrics["global_corr"]),
        "blr_corr": float(metrics["blr_corr"]),
        "radial_corr": float(metrics.get("radial_corr", np.nan)),
        "profile_rmse": profile_rmse(truth, best["image"], axis_uas),
        "amp_chi2": float(residuals["amp_reduced_chi2"]),
        "phase_chi2": float(residuals["phase_reduced_chi2"]),
    }


def plot_config(config: ScanConfig, data: dict, results: list[dict]) -> tuple[Path, Path]:
    axis = data["axis_uas"]
    truth = data["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    truth_disp = opt.normalize_blr_display(truth)
    result_by = {item["strategy"]: item for item in results}
    images = [result_by[strategy]["best"]["image"] for strategy, *_ in STRATEGIES]
    residuals = [opt.normalize_blr_display(image) - truth_disp for image in images]
    vmax = max(0.08, float(np.percentile(np.abs(np.concatenate([r.ravel() for r in residuals])), 99.0)))

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), constrained_layout=True)
    panels = [("truth", "Input source", truth)] + [
        (strategy, label, result_by[strategy]["best"]["image"]) for strategy, label, *_ in STRATEGIES
    ]
    for col, (strategy, label, image) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        if strategy == "truth":
            ax.set_title(label)
        else:
            m = result_by[strategy]["best"]["metrics"]
            r = result_by[strategy]["best"]["residuals"]
            ax.set_title(
                f"{label}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}\n"
                rf"$\chi^2_A$={r['amp_reduced_chi2']:.2f}, $\chi^2_\phi$={r['phase_reduced_chi2']:.2f}",
                fontsize=8.2,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.95,
        (
            f"{config.label}\n"
            f"amp weight={config.amp_weight:g}, phase weight={config.phase_weight:g}\n"
            "Bottom row: displayed-image residuals relative to the input."
        ),
        va="top",
        fontsize=9.0,
    )
    for col, ((strategy, label, *_), resid) in enumerate(zip(STRATEGIES, residuals), start=1):
        ax = axes[1, col]
        im = ax.imshow(resid, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} - input", fontsize=8.4)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"RML source/weight stress test: {config.key}", weight="bold")
    png = OUT / f"{config.key}_images_residuals.png"
    pdf = OUT / f"{config.key}_images_residuals.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_profile_overlay(config: ScanConfig, data: dict, results: list[dict]) -> tuple[Path, Path]:
    axis = data["axis_uas"]
    theta, truth_prof = angular_profile(data["truth"], axis, ngc.NGC4151)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)
    axes[0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    axes[1].axhline(0.0, color="0.2", lw=0.8)
    for strategy, label, _start, color in STRATEGIES:
        image = next(item for item in results if item["strategy"] == strategy)["best"]["image"]
        _, prof = angular_profile(image, axis, ngc.NGC4151)
        axes[0].plot(np.rad2deg(theta), prof, color=color, lw=1.6, label=label)
        axes[1].plot(np.rad2deg(theta), prof - truth_prof, color=color, lw=1.4, label=label)
    axes[0].set_title("BLR annular azimuthal profile")
    axes[1].set_title("Profile residual")
    for ax in axes:
        ax.set_xlabel("azimuth angle (deg)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean-normalized brightness")
    axes[1].set_ylabel("reconstructed - input")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(config.label, weight="bold")
    png = OUT / f"{config.key}_blr_profile.png"
    pdf = OUT / f"{config.key}_blr_profile.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_summary(rows: list[dict]) -> tuple[Path, Path]:
    configs = [config.key for config in SCAN_CONFIGS]
    metrics = ["blr_corr", "profile_rmse", "phase_chi2", "amp_chi2"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        x = np.arange(len(configs))
        width = 0.24
        for offset, (strategy, label, _start, color) in zip((-width, 0.0, width), STRATEGIES):
            vals = [
                next(row[metric] for row in rows if row["config"] == cfg and row["strategy"] == strategy)
                for cfg in configs
            ]
            ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=18, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"profile_rmse", "phase_chi2", "amp_chi2"}:
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("RML stress-test summary: source asymmetry and lower amplitude weight", weight="bold")
    png = OUT / "stress_test_metric_summary.png"
    pdf = OUT / "stress_test_metric_summary.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    simulated: dict[str, dict] = {}
    rows: list[dict] = []
    payload = {"configs": [], "figures": {}}
    for config in SCAN_CONFIGS:
        if config.morphology not in simulated:
            simulated[config.morphology] = simulate_by_morphology(config.morphology)
        data = simulated[config.morphology]
        results = run_config(config, data)
        plot_pdf, plot_png = plot_config(config, data, results)
        prof_pdf, prof_png = plot_profile_overlay(config, data, results)
        config_rows = [result_row(config, result, data["truth"], data["axis_uas"]) for result in results]
        rows.extend(config_rows)
        payload["configs"].append(
            {
                "key": config.key,
                "label": config.label,
                "morphology": config.morphology,
                "amp_weight": config.amp_weight,
                "phase_weight": config.phase_weight,
                "rows": config_rows,
                "figures": {
                    "images_pdf": str(plot_pdf),
                    "images_png": str(plot_png),
                    "profile_pdf": str(prof_pdf),
                    "profile_png": str(prof_png),
                },
            }
        )
    summary_pdf, summary_png = plot_summary(rows)
    payload["figures"]["summary_pdf"] = str(summary_pdf)
    payload["figures"]["summary_png"] = str(summary_png)
    payload["runtime"] = {
        "observing_days": amp_rml.OBSERVING_DAYS,
        "n_time_windows": amp_rml.N_TIME_WINDOWS,
        "exposure_s": amp_rml.EXPOSURE_S,
        "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
        "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
        "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
        "fit_n_pix": val.FIT_N_PIX,
        "shown_n_pix": amp_rml.N_RML,
        "adam_iter": val.ADAM_ITER,
        "adam_lr": val.ADAM_LR,
    }

    csv_path = OUT / "stress_test_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT / "stress_test_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        "# RML phase-sensitivity stress test\n\n"
        "This folder scans two source morphologies and two RML likelihood balances.\n"
        "The same simulated data are reused for both weight settings within each morphology.\n\n"
        "- `lopsided_crescent`: asymmetric but smooth BLR/core/jet benchmark.\n"
        "- `lopsided_spotted`: stronger closure-phase-sensitive BLR with bright irregular clumps.\n"
        "- `amp-dominated`: amplitude gradient weight 4.0, phase gradient weight 1.5.\n"
        "- `phase-led`: amplitude gradient weight 0.7, phase gradient weight 2.4.\n\n"
        "The key diagnostic is whether direct closure separates from edge-first in BLR correlation,\n"
        "azimuthal profile RMSE, and phase chi-square when phase information is made more decisive.\n"
    )
    print(summary_png)
    print(csv_path)
    print(json_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
