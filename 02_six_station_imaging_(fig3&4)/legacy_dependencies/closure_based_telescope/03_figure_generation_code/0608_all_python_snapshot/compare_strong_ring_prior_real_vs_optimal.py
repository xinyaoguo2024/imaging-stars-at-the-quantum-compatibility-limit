from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import latest_maunakea_closure_snr_clean_rml as latest
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

REAL_LAYOUT = OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json"
OPTIMAL_LAYOUT = OUTFIG / "optimized_array_topology_u50v50_lowuv15_near10_5_ydown1p5_hub_m3_m4.json"

SOURCE = ngc.NGC4151
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "1.0"))
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.2"))
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("STATION_FALSE_POSITIVE", "0.05")))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", os.environ.get("BASELINE_FALSE_POSITIVE", "0.0")))
PRIOR_ALPHAS = (0.65, 0.85)


def load_synthetic_case(path: Path) -> aug.NetworkCase:
    payload = json.loads(path.read_text())
    stations = np.asarray(payload["stations_km"], dtype=float)
    telescopes = [
        aug.Telescope(f"S{i + 1}", float(x), float(y), 5.0, True)
        for i, (x, y) in enumerate(stations)
    ]
    return aug.NetworkCase(
        key=path.stem,
        title="Synthetic 8-station optimized array",
        latitude_deg=35.0,
        center_latlon=(35.0, 0.0),
        telescopes=telescopes,
        hub_km=tuple(payload.get("hub_km", [0.0, 0.0])),
        optimization_score=0.0,
    )


def core_ring_prior(axis_uas: np.ndarray, source: ngc.SourceModel) -> np.ndarray:
    """A deliberately simple prior: circular compact core plus circular BLR ring."""
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    core_sigma = max(source.disc_sigma_major_uas, 7.5)
    ring_sigma = max(source.blr_width_uas, 6.0)
    core = np.exp(-0.5 * (rr / core_sigma) ** 2)
    ring = np.exp(-0.5 * ((rr - source.blr_radius_uas) / ring_sigma) ** 2)
    core /= np.sum(core)
    ring /= np.sum(ring)
    prior = 0.45 * core + 0.55 * ring
    prior /= np.sum(prior)
    return latest.normalize_stack(prior)


def positive_blend(data_image: np.ndarray, prior_image: np.ndarray, alpha: float) -> np.ndarray:
    """Stable strong-prior MAP proxy.

    The blend is intentionally transparent: alpha=0 is the data-only image and
    alpha=1 is the core+ring prior.  This avoids the gradient-scale instability
    that appears when imposing a very strong image prior on our dirty-map proxy.
    """
    data = latest.normalize_stack(data_image)
    prior = latest.normalize_stack(prior_image)
    mixed = (1.0 - alpha) * data + alpha * prior
    mixed = np.clip(mixed, 0.0, None)
    return latest.normalize_stack(mixed)


def radial_profile_corr(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.0, np.max(np.abs(axis_uas)), 34)
    truth_n = base.normalize_for_display(truth)
    image_n = base.normalize_for_display(image)
    t_prof = []
    x_prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        if np.count_nonzero(mask) < 4:
            continue
        t_prof.append(float(np.mean(truth_n[mask])))
        x_prof.append(float(np.mean(image_n[mask])))
    if len(t_prof) < 3 or np.std(t_prof) == 0.0 or np.std(x_prof) == 0.0:
        return 0.0
    return float(np.corrcoef(t_prof, x_prof)[0, 1])


def image_metrics(image: np.ndarray, truth: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    out = latest.image_metrics(SOURCE, truth, image, axis_uas)
    out["radial_corr"] = radial_profile_corr(truth, image, axis_uas)
    return out


def reconstruct_case(case: aug.NetworkCase) -> dict:
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    wt.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    old_loss = aug.FIBER_LOSS_DB_PER_KM
    old_false_positive = getattr(aug, "BASELINE_FALSE_POSITIVE", 0.0)
    old_mode_false_positive = getattr(aug, "MODE_FALSE_POSITIVE", 0.05)
    old_pair_false_positive = getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    aug.MODE_FALSE_POSITIVE = MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    try:
        with ngc.patched_source(SOURCE):
            bands, stats, truth, axis_uas = wt.simulate_bands(case)
    finally:
        aug.FIBER_LOSS_DB_PER_KM = old_loss
        aug.BASELINE_FALSE_POSITIVE = old_false_positive
        aug.MODE_FALSE_POSITIVE = old_mode_false_positive
        aug.PAIR_FALSE_POSITIVE = old_pair_false_positive

    prior = core_ring_prior(axis_uas, SOURCE)
    images: dict[str, np.ndarray] = {"truth": truth, "prior": prior}
    psfs: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for strategy in ("split", "direct"):
        sparse_dirty, sparse_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=False)
        filled_dirty, filled_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=True)
        clean, _ = base.multiscale_clean(
            sparse_dirty,
            sparse_psf,
            scales_pix=(0.0, 2.0, 4.5, 8.0, 14.0, 22.0),
            gain=0.10,
            max_iter=900,
            threshold_factor=1.3,
        )
        images[f"{strategy}_filled_dirty"] = latest.normalize_stack(filled_dirty)
        images[f"{strategy}_clean"] = latest.normalize_stack(clean)
        psfs[strategy] = sparse_psf
        metrics[f"{strategy}_filled_dirty"] = image_metrics(images[f"{strategy}_filled_dirty"], truth, axis_uas)
        metrics[f"{strategy}_clean"] = image_metrics(images[f"{strategy}_clean"], truth, axis_uas)
        for alpha in PRIOR_ALPHAS:
            key = f"{strategy}_prior_a{alpha:.2f}".replace(".", "p")
            images[key] = positive_blend(images[f"{strategy}_clean"], prior, alpha)
            metrics[key] = image_metrics(images[key], truth, axis_uas)

    metrics["prior_only"] = image_metrics(prior, truth, axis_uas)
    stats.update(
        {
            "source": SOURCE.name,
            "observing_days": OBSERVING_DAYS,
            "snr_boost": SNR_BOOST,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "noise_model": "pure fibre attenuation plus independent mode-local false positives",
            "prior_definition": "0.45 circular compact core + 0.55 circular Gaussian ring; no azimuthal structure",
            "prior_alphas": list(PRIOR_ALPHAS),
            "metrics": metrics,
        }
    )
    return {"case": case, "stats": stats, "images": images, "axis_uas": axis_uas, "metrics": metrics}


def draw_layout_uv(ax_layout: plt.Axes, ax_uv: plt.Axes, result: dict) -> None:
    case = result["case"]
    stats = result["stats"]
    stations, diameters, _, is_added = aug.station_table_from_case(case)
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new/synthetic"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax_layout.scatter(stations[mask, 0], stations[mask, 1], s=20, marker=marker, color=color, label=label, zorder=3)
    ax_layout.scatter([case.hub_km[0]], [case.hub_km[1]], s=48, marker="*", color="#ca6702", label="hub", zorder=4)
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_xlabel("east (km)")
    ax_layout.set_ylabel("north (km)")
    ax_layout.set_title("Stations + hub")
    ax_layout.legend(loc="best", frameon=False, fontsize=5.8)

    for wavelength, color in (("400", "#005f73"), ("800", "#ee9b00")):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax_uv.scatter(uu, vv, s=0.9, color=color, alpha=0.45, label=f"{wavelength} nm")
        ax_uv.scatter(-uu, -vv, s=0.9, color=color, alpha=0.28)
    ax_uv.set_aspect("equal", adjustable="box")
    ax_uv.set_xlabel(r"$u$ (G$\lambda$)")
    ax_uv.set_ylabel(r"$v$ (G$\lambda$)")
    ax_uv.set_title("UV coverage")
    ax_uv.legend(loc="upper right", frameon=False, fontsize=5.8)


def make_summary_figure(results: list[dict]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(10.4, 5.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 6, hspace=0.44, wspace=0.36)
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.4,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
        }
    )

    image_axes = []
    for row, result in enumerate(results):
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        draw_layout_uv(fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]), result)
        row_label = "Real Maunakea top4+5" if row == 0 else "Synthetic 8-station optimal"

        panels = [
            ("truth", "Input"),
            ("direct_clean", "Direct clean"),
            ("direct_prior_a0p65", r"Direct + prior $\alpha=0.65$"),
            ("direct_prior_a0p85", r"Direct + prior $\alpha=0.85$"),
        ]
        for col, (key, title) in enumerate(panels, start=2):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(opt.normalize_blr_display(result["images"][key]), origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(f"{row_label}\n{title}")
            else:
                metric = result["metrics"][key]
                ax.set_title(
                    f"{title}\n"
                    f"BLR r={metric['blr_corr']:.2f}, rad r={metric['radial_corr']:.2f}"
                )
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 2:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("norm. brightness\n(BLR stretch)", fontsize=6.6)
    fig.suptitle(
        f"Strong core+ring prior diagnostic for {SOURCE.name}: "
        f"{OBSERVING_DAYS} days, SNR x{SNR_BOOST:g}, fibre loss {FIBER_LOSS_DB_PER_KM:g} dB/km, "
        f"mode p_fp {MODE_FALSE_POSITIVE:g}, pair p_fp {PAIR_FALSE_POSITIVE:g}",
        fontsize=10.2,
        weight="bold",
        y=0.992,
    )
    tag = f"{SOURCE.key}_{OBSERVING_DAYS}d_loss{FIBER_LOSS_DB_PER_KM:g}_snr{SNR_BOOST:g}".replace(".", "p")
    png = OUTFIG / f"strong_ring_prior_real_vs_optimal_{tag}.png"
    pdf = OUTFIG / f"strong_ring_prior_real_vs_optimal_{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    real_case = latest.load_case(REAL_LAYOUT)
    opt_case = load_synthetic_case(OPTIMAL_LAYOUT)
    results = [reconstruct_case(real_case), reconstruct_case(opt_case)]
    pdf, png = make_summary_figure(results)

    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "source": SOURCE.name,
        "observing_days": OBSERVING_DAYS,
        "snr_boost": SNR_BOOST,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": MODE_FALSE_POSITIVE,
        "pair_false_positive": PAIR_FALSE_POSITIVE,
        "noise_model": "pure fibre attenuation plus independent mode-local false positives",
        "real_layout": str(REAL_LAYOUT),
        "optimal_layout": str(OPTIMAL_LAYOUT),
        "prior_alphas": list(PRIOR_ALPHAS),
        "cases": [
            {
                "case": result["case"].key,
                "title": result["case"].title,
                "stats": {
                    key: value
                    for key, value in result["stats"].items()
                    if key
                    in (
                        "n_station",
                        "n_baseline",
                        "n_closure",
                        "baseline_max_km",
                        "station_link_eff_min",
                        "station_link_eff_max",
                        "coverage_400nm_half_range_g_lambda",
                        "coverage_800nm_half_range_g_lambda",
                    )
                },
                "metrics": result["metrics"],
            }
            for result in results
        ],
    }
    out = OUTFIG / f"strong_ring_prior_real_vs_optimal_{SOURCE.key}_{OBSERVING_DAYS}d_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)
    print(png)
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
