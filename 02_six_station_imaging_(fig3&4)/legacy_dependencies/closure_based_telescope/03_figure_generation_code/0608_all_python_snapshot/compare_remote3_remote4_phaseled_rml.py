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
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import run_hawaii3_rml_strategy_comparison as strategy_run
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_vs_remote4_phaseled_20260525"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    case: aug.NetworkCase


STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#8d99ae"),
    ("split", "Edge-first closure", "split_dirty", "#0077b6"),
    ("direct", "Direct closure", "direct_dirty", "#d00000"),
]


def make_hawaii4_compact_remote_case() -> aug.NetworkCase:
    """Compact Hawaii+4 variant matched to the current Hawaii+3 layout.

    The current Fig. 3-like Hawaii+3 uses three remote-station directions from
    the far-template layout, rescaled to 2, 4, and 9 km.  Here we keep those
    three directions and add the next available far-template direction at 6 km.
    This tests whether one extra angularly distinct remote station improves BLR
    azimuthal-profile recovery without changing the overall scale too much.
    """
    full = amp_rml.load_maunakea_case()
    core = [tel for tel in full.telescopes if not tel.is_added]
    added = [tel for tel in full.telescopes if tel.is_added]
    templates_and_radii = [
        (added[-3], 2.0),
        (added[-2], 4.0),
        (added[-4], 6.0),
        (added[-1], 9.0),
    ]
    remotes: list[aug.Telescope] = []
    for template, radius in templates_and_radii:
        direction = np.array([template.x_km, template.y_km], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-30)
        xy = radius * direction
        remotes.append(
            aug.Telescope(
                f"new 5 m r={radius:g}km compact",
                float(xy[0]),
                float(xy[1]),
                5.0,
                True,
            )
        )
    return aug.NetworkCase(
        key="hawaii_top4_remote4_compact_r2_4_6_9_ngc4151",
        title="Maunakea top-four core + compact 2/4/6/9 km 5 m outstations",
        latitude_deg=full.latitude_deg,
        center_latlon=full.center_latlon,
        telescopes=core + remotes,
        hub_km=full.hub_km,
        optimization_score=full.optimization_score,
    )


def configure_runtime() -> None:
    amp_rml.SOURCE = ngc.NGC4151
    ngc.SOURCE_MORPHOLOGY = "lopsided_crescent"
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
    amp_rml.AMP_GRAD_WEIGHT = 0.7
    amp_rml.PHASE_GRAD_WEIGHT = 2.4
    amp_rml.STEP = 0.018
    val.FIT_N_PIX = 32
    val.ADAM_ITER = 1600
    val.ADAM_LR = 0.012
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0


def angular_profile(image: np.ndarray, axis_uas: np.ndarray, n_bin: int = 72):
    source = ngc.NGC4151
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
    _, truth_prof = angular_profile(truth, axis_uas)
    _, image_prof = angular_profile(image, axis_uas)
    return float(np.sqrt(np.mean((image_prof - truth_prof) ** 2)))


def simulate_and_run(spec: CaseSpec) -> dict:
    configure_runtime()
    print(f"[simulate] {spec.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(spec.case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    results = []
    for strategy, label, start_name, _color in STRATEGIES:
        print(f"[rml] {spec.key}: {strategy}", flush=True)
        results.append(
            strategy_run.run_strategy(
                strategy,
                label,
                start_name,
                spec.case,
                bands,
                truth,
                axis_uas,
                prior,
                starts,
            )
        )
    return {
        "spec": spec,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "results": results,
    }


def metric_rows(dataset: dict) -> list[dict]:
    rows = []
    for result in dataset["results"]:
        best = result["best"]
        metrics = best["metrics"]
        residuals = best["residuals"]
        rows.append(
            {
                "case": dataset["spec"].key,
                "case_label": dataset["spec"].label,
                "n_station": len(dataset["spec"].case.telescopes),
                "n_baseline": len(dataset["spec"].case.telescopes) * (len(dataset["spec"].case.telescopes) - 1) // 2,
                "n_closure": (len(dataset["spec"].case.telescopes) - 1)
                * (len(dataset["spec"].case.telescopes) - 2)
                // 2,
                "strategy": result["strategy"],
                "global_corr": float(metrics["global_corr"]),
                "blr_corr": float(metrics["blr_corr"]),
                "profile_rmse": profile_rmse(dataset["truth"], best["image"], dataset["axis_uas"]),
                "amp_chi2": float(residuals["amp_reduced_chi2"]),
                "phase_chi2": float(residuals["phase_reduced_chi2"]),
            }
        )
    return rows


def plot_uv(ax, dataset: dict) -> None:
    stats = dataset["stats"]
    for wavelength, color, alpha in (("400", "#005f73", 0.55), ("800", "#ee9b00", 0.46)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.0, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.0, color=color, alpha=alpha * 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title(f"{dataset['spec'].label}: uv coverage")
    ax.legend(frameon=False, fontsize=7)


def plot_profiles(datasets: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.6, 4.2 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = axes[None, :]
    for row, dataset in enumerate(datasets):
        axis = dataset["axis_uas"]
        theta, truth_prof = angular_profile(dataset["truth"], axis)
        ax = axes[row, 0]
        ax.plot(np.rad2deg(theta), truth_prof, color="black", lw=2.4, label="input")
        for strategy, label, _start, color in STRATEGIES:
            image = next(item for item in dataset["results"] if item["strategy"] == strategy)["best"]["image"]
            _, prof = angular_profile(image, axis)
            ax.plot(np.rad2deg(theta), prof, color=color, lw=1.6, label=label)
        ax.set_title(f"{dataset['spec'].label}: BLR annular profile")
        ax.set_xlabel("azimuth angle (deg)")
        ax.set_ylabel("mean-normalized brightness")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        plot_uv(axes[row, 1], dataset)
    png = OUT / "remote3_vs_remote4_profiles_uv.png"
    pdf = OUT / "remote3_vs_remote4_profiles_uv.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_images(datasets: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(datasets), 4, figsize=(12.2, 3.2 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = axes[None, :]
    cols = [("truth", "Input source", None)] + [(s, label, color) for s, label, _start, color in STRATEGIES]
    for row, dataset in enumerate(datasets):
        axis = dataset["axis_uas"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]
        result_by = {item["strategy"]: item for item in dataset["results"]}
        for col, (key, label, _color) in enumerate(cols):
            ax = axes[row, col]
            if key == "truth":
                image = dataset["truth"]
                title = f"{dataset['spec'].label}\nInput"
            else:
                result = result_by[key]
                image = result["best"]["image"]
                metric = result["best"]["metrics"]
                resid = result["best"]["residuals"]
                title = (
                    f"{label}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}\n"
                    rf"$\chi^2_A$={resid['amp_reduced_chi2']:.2f}, $\chi^2_\phi$={resid['phase_reduced_chi2']:.2f}"
                )
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            ax.set_title(title, fontsize=8.2)
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    fig.suptitle("Remote-station test: phase-led RML, lopsided crescent source", weight="bold")
    png = OUT / "remote3_vs_remote4_images.png"
    pdf = OUT / "remote3_vs_remote4_images.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    specs = [
        CaseSpec("remote3", "top4 + remote3", hawaii3_compact_case.make_hawaii3_compact_remote_case()),
        CaseSpec("remote4", "top4 + remote4", make_hawaii4_compact_remote_case()),
    ]
    datasets = [simulate_and_run(spec) for spec in specs]
    rows = []
    for dataset in datasets:
        rows.extend(metric_rows(dataset))
    profile_pdf, profile_png = plot_profiles(datasets)
    image_pdf, image_png = plot_images(datasets)

    csv_path = OUT / "remote3_vs_remote4_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "runtime": {
            "source": amp_rml.SOURCE.name,
            "morphology": ngc.SOURCE_MORPHOLOGY,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "adam_iter": val.ADAM_ITER,
            "adam_lr": val.ADAM_LR,
        },
        "rows": rows,
        "figures": {
            "profiles_uv_pdf": str(profile_pdf),
            "profiles_uv_png": str(profile_png),
            "images_pdf": str(image_pdf),
            "images_png": str(image_png),
            "metrics_csv": str(csv_path),
        },
    }
    json_path = OUT / "remote3_vs_remote4_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        "# Remote3 vs remote4 phase-led RML test\n\n"
        "This test keeps the NGC 4151 lopsided-crescent source, true SNR, 30 days,\n"
        "0.2 dB/km fiber loss, and phase-led RML weights fixed.  It compares the\n"
        "current compact top4+remote3 layout against a compact top4+remote4 layout\n"
        "with an added 6 km remote station in an extra angular direction.\n"
    )
    print(profile_png)
    print(image_png)
    print(csv_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
