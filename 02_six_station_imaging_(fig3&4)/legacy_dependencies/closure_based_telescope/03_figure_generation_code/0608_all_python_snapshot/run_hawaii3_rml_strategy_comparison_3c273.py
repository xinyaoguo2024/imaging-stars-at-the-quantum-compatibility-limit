from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_hawaii3_rml_strategy_comparison as ngc_fig3
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


SOURCE_3C273 = ngc.SourceModel(
    key="3c273",
    name="3C 273",
    dec_deg=2.052388,
    tau_hbeta_days=260.0,
    distance_mpc=749.0,
    mbh_msun=8.0e8,
    blr_radius_uas=50.0,
    blr_width_uas=9.5,
    disc_sigma_major_uas=7.0,
    disc_sigma_minor_uas=4.5,
    position_angle_deg=-20.0,
    blr_fraction=base.SOURCE_COMPONENT_FRACTIONS["broad_line_region_lines"],
    sed_lambda_nm=tuple(float(x) for x in base.SOURCE_SED_LAMBDA_NM),
    sed_fnu_mjy=tuple(float(1.0e3 * x) for x in base.SOURCE_SED_FNU_JY),
    sed_reference=base.SOURCE_SPECTRUM_NOTE,
    note=(
        "3C 273 toy optical continuum + BLR + inner-jet model from "
        "plot_prl_broadband_clean.py; this preserves the original PA and "
        "source morphology rather than using the NGC lopsided-crescent factory."
    ),
)

SOURCE_3C273_FNU = base.source_fnu_jy
SOURCE_3C273_MAKE_SOURCE = base.make_source


STRATEGIES = [
    ("all", "All visibilities\nwith piston drift", "all_dirty"),
    ("split", "Edge-first\nclosure", "split_dirty"),
    ("direct", "Direct\nclosure", "direct_dirty"),
]


def configure_fig3_runtime() -> None:
    """Use the current manuscript-style RML settings without touching NGC code."""
    amp_rml.SOURCE = SOURCE_3C273
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
    # Match the stable manuscript-style runs: do not stop as soon as chi^2
    # reaches order unity, because the extra iterations noticeably improve the
    # image morphology even after the likelihood is statistically acceptable.
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0

    aug.FIBER_LOSS_DB_PER_KM = amp_rml.FIBER_LOSS_DB_PER_KM
    aug.MODE_FALSE_POSITIVE = amp_rml.MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = amp_rml.PAIR_FALSE_POSITIVE
    aug.N_TIME_WINDOWS = amp_rml.N_TIME_WINDOWS
    aug.EXPOSURE_S = amp_rml.EXPOSURE_S
    aug.EXPOSURE_GAP_S = amp_rml.EXPOSURE_GAP_S
    aug.POST_AVERAGE_DRIFT_STD = np.pi / 5.0

    wt.N_PIX = amp_rml.N_RML
    wt.SNR_BOOST = 1.0
    wt.OBSERVING_DAYS = amp_rml.OBSERVING_DAYS
    wt.AMPLITUDE_MODE_FALSE_POSITIVE = 0.05


@contextmanager
def patched_3c273_source():
    old_make_source = base.make_source
    old_fnu = base.source_fnu_jy
    old_dec = aug.SOURCE_DEC_DEG
    base.make_source = SOURCE_3C273_MAKE_SOURCE
    base.source_fnu_jy = SOURCE_3C273_FNU
    aug.SOURCE_DEC_DEG = SOURCE_3C273.dec_deg
    try:
        yield
    finally:
        base.make_source = old_make_source
        base.source_fnu_jy = old_fnu
        aug.SOURCE_DEC_DEG = old_dec


def make_3c273_case() -> aug.NetworkCase:
    base_case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    return replace(
        base_case,
        key=base_case.key.replace("ngc4151", "3c273"),
        title=base_case.title + " observing 3C 273",
    )


def simulate_3c273_case(case: aug.NetworkCase) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
    old = {
        "loss": aug.FIBER_LOSS_DB_PER_KM,
        "mode_fp": getattr(aug, "MODE_FALSE_POSITIVE", 0.05),
        "pair_fp": getattr(aug, "PAIR_FALSE_POSITIVE", 0.0),
        "n_time": aug.N_TIME_WINDOWS,
        "exp": aug.EXPOSURE_S,
        "gap": aug.EXPOSURE_GAP_S,
        "drift": getattr(aug, "POST_AVERAGE_DRIFT_STD", np.pi / 10.0),
        "wt_npix": wt.N_PIX,
        "wt_snr": wt.SNR_BOOST,
        "wt_days": wt.OBSERVING_DAYS,
        "wt_amp_fp": getattr(wt, "AMPLITUDE_MODE_FALSE_POSITIVE", 0.05),
    }
    configure_fig3_runtime()
    try:
        with patched_3c273_source():
            bands, stats, truth, axis_uas = wt.simulate_bands(case)
        stats.update(
            {
                "source": {
                    "key": SOURCE_3C273.key,
                    "name": SOURCE_3C273.name,
                    "declination_deg": SOURCE_3C273.dec_deg,
                    "effective_ab_mag_550nm": float(base.source_ab_magnitude(base.C_LIGHT / (550.0e-9))),
                    "sed_lambda_nm": list(SOURCE_3C273.sed_lambda_nm),
                    "sed_fnu_mjy": list(SOURCE_3C273.sed_fnu_mjy),
                    "sed_reference": SOURCE_3C273.sed_reference,
                    "blr_radius_uas": SOURCE_3C273.blr_radius_uas,
                    "blr_width_uas": SOURCE_3C273.blr_width_uas,
                    "position_angle_deg": SOURCE_3C273.position_angle_deg,
                    "note": SOURCE_3C273.note,
                },
                "source_morphology": "3c273_default_continuum_blr_jet_from_plot_prl_broadband_clean",
            }
        )
        return bands, stats, truth, axis_uas
    finally:
        aug.FIBER_LOSS_DB_PER_KM = old["loss"]
        aug.MODE_FALSE_POSITIVE = old["mode_fp"]
        aug.PAIR_FALSE_POSITIVE = old["pair_fp"]
        aug.N_TIME_WINDOWS = old["n_time"]
        aug.EXPOSURE_S = old["exp"]
        aug.EXPOSURE_GAP_S = old["gap"]
        aug.POST_AVERAGE_DRIFT_STD = old["drift"]
        wt.N_PIX = old["wt_npix"]
        wt.SNR_BOOST = old["wt_snr"]
        wt.OBSERVING_DAYS = old["wt_days"]
        wt.AMPLITUDE_MODE_FALSE_POSITIVE = old["wt_amp_fp"]


def plot_six_panel(case, stats, truth, axis_uas, results, tag: str) -> tuple[Path, Path]:
    stations, diameters, _names, is_added = aug.station_table_from_case(case)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
        }
    )
    fig = plt.figure(figsize=(7.55, 4.88), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.40, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new 5 m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax.scatter(
                stations[mask, 0],
                stations[mask, 1],
                s=30 if added else 25,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=55, marker="*", color="#ca6702", label="hub", zorder=4)
    for i, j in base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.83", lw=0.42, zorder=0)
    label_offsets = {
        0: (-1.45, 1.18),
        1: (0.08, 1.58),
        2: (-1.75, -0.72),
        3: (0.56, -0.80),
        4: (0.18, 0.44),
        5: (0.20, 0.46),
        6: (0.18, -0.92),
    }
    for i, (x, y) in enumerate(stations):
        dx, dy = label_offsets.get(i, (0.2, 0.2))
        ax.text(x + dx, y + dy, f"S{i + 1}\n{diameters[i]:g}m", fontsize=5.35)
    x_pad = max(0.9, 0.08 * (float(np.max(stations[:, 0])) - float(np.min(stations[:, 0]))))
    y_pad = max(0.9, 0.10 * (float(np.max(stations[:, 1])) - float(np.min(stations[:, 1]))))
    ax.set_xlim(float(np.min(stations[:, 0])) - x_pad, float(np.max(stations[:, 0])) + x_pad)
    ax.set_ylim(float(np.min(stations[:, 1])) - y_pad, float(np.max(stations[:, 1])) + 1.55 * y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("Maunakea top4 + remote3")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.15, borderpad=0.1)

    ax = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.50), ("800", "#ee9b00", 0.42)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.2, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.2, color=color, alpha=0.62 * alpha)
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 256)
    for theta_uas, ls in ((60.0, ":"), (30.0, "--"), (10.0, "-.")):
        radius_g_lambda = 1.0 / (theta_uas * base.UAS_TO_RAD) / 1.0e9
        ax.plot(
            radius_g_lambda * np.cos(theta_circle),
            radius_g_lambda * np.sin(theta_circle),
            ls=ls,
            lw=0.55,
            color="0.35",
            alpha=0.70,
            label=rf"{theta_uas:g} $\mu$as",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage for 3C 273")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Input 3C 273\ncontinuum + BLR + jet")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    result_by_strategy = {result["strategy"]: result for result in results}
    labels = {
        "all": "All visibilities + drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, strategy in enumerate(("all", "split", "direct")):
        result = result_by_strategy[strategy]
        ax = fig.add_subplot(gs[1, col])
        image = result["best"]["image"]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        metric = result["best"]["metrics"]
        residual = result["best"]["residuals"]
        ax.set_title(
            (
                f"{labels[strategy]}\n"
                f"BLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}\n"
                f"$\\chi^2_A$={residual['amp_reduced_chi2']:.2g}, "
                f"$\\chi^2_\\phi$={residual['phase_reduced_chi2']:.2g}"
            ),
            fontsize=7.4,
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.024,
        pad=0.018,
    )
    cbar.set_label("norm. brightness\n(BLR-emphasis arcsinh)", fontsize=6.8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    fig.suptitle(
        "3C 273 RML imaging benchmark: same Hawaii+remote3 topology",
        fontsize=9.6,
        weight="bold",
        y=0.995,
    )
    png = OUTFIG / f"{tag}_6panel.png"
    pdf = OUTFIG / f"{tag}_6panel.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def serializable_result(item: dict) -> dict:
    return {
        "strategy": item["strategy"],
        "start": item["start"],
        "validation_score": float(item["validation_score"]),
        "metrics": {key: float(value) for key, value in item["metrics"].items()},
        "residuals": {key: float(value) for key, value in item["residuals"].items()},
    }


def main() -> None:
    configure_fig3_runtime()
    case = make_3c273_case()
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = simulate_3c273_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)

    results = [
        ngc_fig3.run_strategy(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts)
        for strategy, label, start_name in STRATEGIES
    ]

    tag = (
        f"{case.key}_rml_strategy_comparison_3c273_default_"
        f"fit{val.FIT_N_PIX}_shown{amp_rml.N_RML}_"
        f"{amp_rml.sigma_tag()}_{ngc_fig3.drift_tag(float(stats.get('post_average_drift_std_rad', 0.0)))}_adam"
        f"i{val.ADAM_ITER}lr{val.ADAM_LR:g}_"
        f"aw{amp_rml.AMP_GRAD_WEIGHT:g}pw{amp_rml.PHASE_GRAD_WEIGHT:g}_"
        f"{amp_rml.OBSERVING_DAYS}d"
    ).replace(".", "p")

    pdf_path, png_path = plot_six_panel(case, stats, truth, axis_uas, results, tag)

    rows = []
    for result in results:
        best = serializable_result(result["best"])
        rows.append(
            {
                "strategy": result["strategy"],
                "label": result["label"].replace("\n", " "),
                "best_start": best["start"],
                "validation_score": best["validation_score"],
                **{f"metric_{key}": value for key, value in best["metrics"].items()},
                **{f"resid_{key}": value for key, value in best["residuals"].items()},
            }
        )
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "case": case.key,
        "source": SOURCE_3C273.name,
        "source_model": "3C273 default source from plot_prl_broadband_clean.py",
        "simulation_stats": stats,
        "environment": {
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "pair_false_positive": amp_rml.PAIR_FALSE_POSITIVE,
            "post_average_drift_std_rad": stats.get("post_average_drift_std_rad"),
            "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
            "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "optimizer": "adam",
            "adam_iter": val.ADAM_ITER,
            "adam_lr": val.ADAM_LR,
            "amp_grad_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_grad_weight": amp_rml.PHASE_GRAD_WEIGHT,
        },
        "results": [
            {
                "strategy": result["strategy"],
                "label": result["label"],
                "best": serializable_result(result["best"]),
                "candidates": [serializable_result(item) for item in result["candidates"]],
            }
            for result in results
        ],
        "figures": {
            "six_panel_png": str(png_path),
            "six_panel_pdf": str(pdf_path),
            "metrics_csv": str(csv_path),
        },
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
