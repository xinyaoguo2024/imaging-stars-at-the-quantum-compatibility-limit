from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as rml_cases
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

SOURCE = ngc.NGC4151
RECON_MODE = os.environ.get("RECON_MODE", "coarse_interp")
REMOTE_COUNT = int(os.environ.get("REMOTE_COUNT", "3"))
OBSERVING_DAYS = 30
N_TIME_WINDOWS = 36
EXPOSURE_S = 600.0
EXPOSURE_GAP_S = 150.0
FIBER_LOSS_DB_PER_KM = 0.20
FIBER_LENGTH_SCALE = 0.75
MODE_FALSE_POSITIVE = 0.05
PAIR_FALSE_POSITIVE = 0.0
SNR_BOOST = 1.0


def configure_simulation() -> None:
    """Use the same nightly exposure convention as the current manuscript Fig. 3."""
    aug.OBSERVING_DAYS = OBSERVING_DAYS
    aug.N_TIME_WINDOWS = N_TIME_WINDOWS
    aug.EXPOSURE_S = EXPOSURE_S
    aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.FIBER_LENGTH_SCALE = FIBER_LENGTH_SCALE
    aug.MODE_FALSE_POSITIVE = MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    aug.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE

    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    wt.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    wt.N_PIX = aug.N_PIX


def reconstruct_hawaii_plus3() -> dict:
    configure_simulation()
    if REMOTE_COUNT == 3:
        case = rml_cases.load_maunakea_plus3_case()
    elif REMOTE_COUNT == 5:
        case = rml_cases.load_maunakea_case()
    else:
        raise ValueError("REMOTE_COUNT must be 3 or 5 for the saved Maunakea layouts.")
    with ngc.patched_source(SOURCE):
        bands, base_stats, truth, axis_uas = wt.simulate_bands(case)

    images, stack_weights = wt.reconstruct_case(bands, truth)
    metrics: dict[str, dict[str, dict[str, float]]] = {mode: {} for mode in images}
    for strategy in ("all", "split", "direct"):
        for mode in images:
            metrics[mode][strategy] = ngc.image_metrics(truth, images[mode][strategy], axis_uas, SOURCE)

    stats = dict(base_stats)
    stations, diameters, names, is_added = aug.station_table_from_case(case)
    stats.update(
        {
            "case": case.key,
            "title": case.title,
            "source": {
                "key": SOURCE.key,
                "name": SOURCE.name,
                "declination_deg": SOURCE.dec_deg,
                "effective_ab_mag_550nm": ngc.sed_effective_ab_mag(SOURCE, 550.0),
                "sed_lambda_nm": list(SOURCE.sed_lambda_nm),
                "sed_fnu_mjy": list(SOURCE.sed_fnu_mjy),
                "sed_reference": SOURCE.sed_reference,
                "tau_hbeta_days": SOURCE.tau_hbeta_days,
                "blr_radius_uas": SOURCE.blr_radius_uas,
                "blr_width_uas": SOURCE.blr_width_uas,
                "note": SOURCE.note,
            },
            "station_names": names,
            "station_diameters_m": diameters.tolist(),
            "station_is_added": is_added.tolist(),
            "station_positions_km": stations.tolist(),
            "hub_km": list(case.hub_km),
            "observing_days": OBSERVING_DAYS,
            "n_time_windows": N_TIME_WINDOWS,
            "exposure_s": EXPOSURE_S,
            "exposure_gap_s": EXPOSURE_GAP_S,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "fiber_length_scale": FIBER_LENGTH_SCALE,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "snr_boost": SNR_BOOST,
            "reconstruction_mode": RECON_MODE,
            "reconstruction_mode_description": (
                "non-parametric coarse uv-cell interpolation/Wiener inversion from "
                "plot_augmented_far_snr100_weighting_test.py"
            ),
            "stack_weights": {
                mode: {strategy: float(value) for strategy, value in mode_weights.items()}
                for mode, mode_weights in stack_weights.items()
            },
            "noise_model": "pure fibre attenuation plus independent mode-local false positives",
            "metrics": metrics,
        }
    )
    return {"case": case, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images}


def make_figure(result: dict) -> tuple[Path, Path]:
    case = result["case"]
    stats = result["stats"]
    truth = result["truth"]
    axis_uas = result["axis_uas"]
    images = result["images"]
    stations, diameters, _names, is_added = aug.station_table_from_case(case)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]

    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.1,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )
    fig = plt.figure(figsize=(7.45, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.39, wspace=0.34)

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
                s=31 if added else 27,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=58, marker="*", color="#ca6702", label="hub", zorder=4)
    for i, j in base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.82", lw=0.42, zorder=0)
    for i, (x, y) in enumerate(stations):
        ax.text(x + 0.22, y + 0.18, f"S{i + 1}\n{diameters[i]:g}m", fontsize=5.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title(f"Hawaii4 + remote{REMOTE_COUNT}")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.15, borderpad=0.1)

    ax = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.50), ("800", "#ee9b00", 0.42)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.2, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.2, color=color, alpha=0.62 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title(f"Input {SOURCE.name}\nRM radius {SOURCE.blr_radius_uas:.0f} $\\mu$as")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    labels = {
        "all": "All visibilities + drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, strategy in enumerate(("all", "split", "direct")):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(opt.normalize_blr_display(images[RECON_MODE][strategy]), origin="lower", extent=extent, cmap="inferno")
        metric = stats["metrics"][RECON_MODE][strategy]
        ax.set_title(f"{labels[strategy]}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}")
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
        f"Maunakea optical core plus {REMOTE_COUNT} remote 5 m stations: NGC 4151, {RECON_MODE}, physical SNR",
        fontsize=9.7,
        weight="bold",
        y=0.995,
    )
    png = OUTFIG / f"prl_fig3_hawaii4_remote{REMOTE_COUNT}_ngc4151_{RECON_MODE}.png"
    pdf = OUTFIG / f"prl_fig3_hawaii4_remote{REMOTE_COUNT}_ngc4151_{RECON_MODE}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    result = reconstruct_hawaii_plus3()
    pdf, png = make_figure(result)
    stats_path = OUTFIG / f"prl_fig3_hawaii4_remote{REMOTE_COUNT}_ngc4151_{RECON_MODE}_stats.json"
    result["stats"]["figure_pdf"] = str(pdf)
    result["stats"]["figure_png"] = str(png)
    stats_path.write_text(json.dumps(result["stats"], indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(result["stats"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
