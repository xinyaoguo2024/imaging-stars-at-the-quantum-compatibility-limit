from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)
from scan_augmented_far_hybrid_density_nearest import coarse_density_on_fine_grid


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

N_BIN = 40
FILL_ALPHA = 0.5
NOISE_AWARE_POWER = 1.0
WEIGHT_CLIP = (0.05, 4.0)


def maunakea_plus4_case_from_plus5(path: Path) -> aug.NetworkCase:
    """Build a quick four-outstation Hawaii variant from the saved +5 layout.

    We remove the lowest-efficiency added outstation from the existing optimized
    +5 design.  This is a controlled first test, not a full re-optimization.
    """
    case = wt.case_from_stats(path)
    added = [t for t in case.telescopes if t.is_added]
    keep_remove = max(
        added,
        key=lambda t: np.linalg.norm(np.array([t.x_km, t.y_km]) - np.array(case.hub_km, dtype=float)),
    )
    telescopes = [t for t in case.telescopes if t.name != keep_remove.name]
    return aug.NetworkCase(
        key="maunakea_plus4_far",
        title="Maunakea optical core + four far 5 m outstations",
        latitude_deg=case.latitude_deg,
        center_latlon=case.center_latlon,
        telescopes=telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def noise_aware_gate(
    cell_var: np.ndarray,
    label_y: np.ndarray,
    label_x: np.ndarray,
    fillable: np.ndarray,
    occupied: np.ndarray,
    *,
    q: float,
) -> tuple[np.ndarray, float]:
    """Return a smooth noise gate on the final Fourier grid.

    The gate is one for well-measured cells and decreases as the assigned
    inverse variance falls below the selected quantile of occupied-cell
    inverse variances.  Empty nearest-filled cells inherit the variance of their
    donor cell, but remain separately downweighted by FILL_ALPHA.
    """
    if q <= 0.0:
        return np.ones_like(cell_var, dtype=float), 0.0

    gate = np.zeros_like(cell_var, dtype=float)
    occupied_var = cell_var[occupied]
    finite_occ = np.isfinite(occupied_var) & (occupied_var > 0.0)
    if not np.any(finite_occ):
        return np.ones_like(cell_var, dtype=float), 1.0
    occupied_ivar = 1.0 / occupied_var[finite_occ]
    ivar0 = float(np.quantile(occupied_ivar, q))
    assigned_var = cell_var[label_y[fillable], label_x[fillable]]
    finite = np.isfinite(assigned_var) & (assigned_var > 0.0)
    assigned_ivar = np.zeros_like(assigned_var, dtype=float)
    assigned_ivar[finite] = 1.0 / assigned_var[finite]
    gate_values = assigned_ivar / (assigned_ivar + max(ivar0, 1e-30))
    gate[fillable] = gate_values
    gate[occupied] = (1.0 / np.maximum(cell_var[occupied], 1e-30)) / (
        (1.0 / np.maximum(cell_var[occupied], 1e-30)) + max(ivar0, 1e-30)
    )
    gate = np.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)
    return gate, ivar0


def reconstruct_band_noiseaware_p1(
    band: dict[str, np.ndarray],
    strategy: str,
    fov_rad: float,
    *,
    q: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    grid, occupied, cell_var = aggregate_cells(
        band["u"],
        band["v"],
        band[f"vis_{strategy}"],
        band[f"sigma_{strategy}"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        average_mode="noise",
    )
    support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=wt.aug.SUPPORT_MODE)
    label_y, label_x, fillable = nearest_label_map(occupied, support)
    filled_grid = np.zeros_like(grid)
    filled_grid[fillable] = grid[label_y[fillable], label_x[fillable]]
    filled_grid[occupied] = grid[occupied]

    area_weight = coarse_density_on_fine_grid(
        band["u"],
        band["v"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        n_bin=N_BIN,
        power=1.0,
    )
    gate, ivar0 = noise_aware_gate(cell_var, label_y, label_x, fillable, occupied, q=q)
    support_weight = np.zeros_like(area_weight)
    support_weight[fillable] = FILL_ALPHA * area_weight[fillable] * gate[fillable] ** NOISE_AWARE_POWER
    support_weight[occupied] = area_weight[occupied] * gate[occupied] ** NOISE_AWARE_POWER
    support_weight = np.clip(support_weight, 0.0, WEIGHT_CLIP[1])
    positive = support_weight[support_weight > 0.0]
    if len(positive):
        floor = WEIGHT_CLIP[0] * float(np.median(positive))
        support_weight[(support_weight > 0.0) & (support_weight < floor)] = floor
    support_weight[wt.N_PIX // 2, wt.N_PIX // 2] = max(support_weight[wt.N_PIX // 2, wt.N_PIX // 2], 1.0)

    image_grid = filled_grid * support_weight
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image_grid))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(support_weight))).real
    peak = psf[wt.N_PIX // 2, wt.N_PIX // 2]
    if peak > 0.0:
        image /= peak

    assigned_var = cell_var[label_y[fillable], label_x[fillable]]
    assigned_weight = support_weight[fillable]
    finite = np.isfinite(assigned_var) & (assigned_var > 0.0) & (assigned_weight > 0.0)
    if np.any(finite):
        mean_var = np.average(assigned_var[finite], weights=assigned_weight[finite])
        band_weight = 1.0 / float(mean_var)
    else:
        band_weight = 1.0
    diagnostics = {
        "occupied_fraction": float(np.mean(occupied)),
        "fillable_fraction": float(np.mean(fillable)),
        "median_support_weight": float(np.median(positive)) if len(positive) else 0.0,
        "ivar0": ivar0,
    }
    return image, band_weight, diagnostics


def reconstruct_stack(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    truth: np.ndarray,
    *,
    q: float,
) -> tuple[np.ndarray, dict[str, float]]:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    stack = np.zeros_like(truth)
    total_weight = 0.0
    diagnostics: dict[str, list[float]] = {}
    for band in bands:
        image, weight, diag = reconstruct_band_noiseaware_p1(band, strategy, fov_rad, q=q)
        stack += weight * image
        total_weight += weight
        for key, value in diag.items():
            diagnostics.setdefault(key, []).append(float(value))
    summary = {key: float(np.median(values)) for key, values in diagnostics.items()}
    return normalize_stack(stack / max(total_weight, 1e-30)), summary


def image_metrics(
    truth: np.ndarray,
    image: np.ndarray,
    axis_uas: np.ndarray,
    source: ngc.SourceModel,
) -> dict[str, float]:
    ring_mask, core_mask = ngc.blr_masks_for_source(axis_uas, source)
    return {
        "global_corr": float(base.corrcoef_positive(truth, image)),
        "blr_corr": float(opt.masked_corr(truth, image, ring_mask)),
        "ring_contrast": float(opt.ring_contrast(image, ring_mask, core_mask)),
    }


def plot_case(
    case: wt.aug.NetworkCase,
    source: ngc.SourceModel,
    stats: dict,
    images: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    *,
    q: float,
) -> tuple[Path, Path]:
    stations, diameters, _, is_added = wt.aug.station_table_from_case(case)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig = plt.figure(figsize=(7.55, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.38, wspace=0.34)
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )

    ax = fig.add_subplot(gs[0, 0])
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new 5 m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax.scatter(stations[mask, 0], stations[mask, 1], s=30 if added else 26, marker=marker, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=3)
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=58, marker="*", color="#ca6702", label="hub", zorder=4)
    for i in range(len(stations)):
        ax.text(stations[i, 0] + 0.18, stations[i, 1] + 0.18, f"S{i+1}\n{diameters[i]:g}m", fontsize=5.6)
    for i, j in base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.82", lw=0.42, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("stations and hub")
    ax.legend(loc="best", frameon=False, handletextpad=0.15)

    ax = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.50), ("800", "#ee9b00", 0.42)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.15, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.15, color=color, alpha=0.62 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title(f"Input {source.name}\nRM radius {source.blr_radius_uas:.0f} $\\mu$as")
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
        ax.imshow(opt.normalize_blr_display(images[strategy]), origin="lower", extent=extent, cmap="inferno")
        metric = stats["metrics"][strategy]
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
    method_label = "uniform p=1, no noise gate" if q <= 0.0 else f"noise-aware p=1 (q={q:.2f})"
    fig.suptitle(f"{case.title}: {source.name}, {method_label}", fontsize=10.1, weight="bold", y=0.995)
    safe = f"{case.key}_{source.key}_noiseaware_p1_q{int(round(100*q)):02d}"
    png = OUTFIG / f"augmented_existing_telescope_{safe}.png"
    pdf = OUTFIG / f"augmented_existing_telescope_{safe}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_case(case_or_path: aug.NetworkCase | Path, source: ngc.SourceModel, *, q_values: list[float]) -> dict:
    case = wt.case_from_stats(case_or_path) if isinstance(case_or_path, Path) else case_or_path
    with ngc.patched_source(source):
        bands, base_stats, truth, axis_uas = wt.simulate_bands(case)

    all_results = {}
    for q in q_values:
        images: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, float]] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for strategy in ("all", "split", "direct"):
            images[strategy], diagnostics[strategy] = reconstruct_stack(bands, strategy, truth, q=q)
            metrics[strategy] = image_metrics(truth, images[strategy], axis_uas, source)
        all_results[f"q{q:.2f}"] = {
            "q": q,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "images": images,
        }

    best_key = max(
        all_results,
        key=lambda key: (
            all_results[key]["metrics"]["direct"]["blr_corr"],
            all_results[key]["metrics"]["direct"]["global_corr"],
        ),
    )
    best = all_results[best_key]
    stats = dict(base_stats)
    stats.update(
        {
            "source": {
                "key": source.key,
                "name": source.name,
                "declination_deg": source.dec_deg,
                "effective_ab_mag_550nm": ngc.sed_effective_ab_mag(source, 550.0),
                "sed_lambda_nm": list(source.sed_lambda_nm),
                "sed_fnu_mjy": list(source.sed_fnu_mjy),
                "tau_hbeta_days": source.tau_hbeta_days,
                "distance_mpc": source.distance_mpc,
                "mbh_msun": source.mbh_msun,
                "blr_radius_uas": source.blr_radius_uas,
                "blr_width_uas": source.blr_width_uas,
                "blr_orbital_period_years": ngc.orbital_period_years(source),
            },
            "reconstruction": {
                "name": "noise-aware p=1 uniform-area weighting",
                "n_bin": N_BIN,
                "fill_alpha": FILL_ALPHA,
                "gate": "unity for q<=0; otherwise ivar / (ivar + quantile(occupied_ivar, q))",
                "q_values": q_values,
                "best_q_key": best_key,
            },
            "metrics_by_q": {
                key: {"q": payload["q"], "metrics": payload["metrics"], "diagnostics": payload["diagnostics"]}
                for key, payload in all_results.items()
            },
            "metrics": best["metrics"],
            "diagnostics": best["diagnostics"],
        }
    )
    pdf, png = plot_case(case, source, stats, best["images"], truth, axis_uas, q=best["q"])
    stats["figure_pdf"] = str(pdf)
    stats["figure_png"] = str(png)
    out_json = OUTFIG / f"augmented_existing_telescope_{case.key}_{source.key}_noiseaware_p1_stats.json"
    stats["stats_json"] = str(out_json)
    out_json.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    wt.SNR_BOOST = 1.0
    q_values = [0.25, 0.50, 0.75]
    maunakea_plus5_path = OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json"
    jobs = [
        (maunakea_plus5_path, ngc.NGC4151),
        (maunakea_plus4_case_from_plus5(maunakea_plus5_path), ngc.NGC4151),
        (OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json", ngc.NGC3783),
    ]
    summary = {}
    for case_or_path, source in jobs:
        label = case_or_path.stem if isinstance(case_or_path, Path) else case_or_path.key
        print(f"simulating {label} with {source.name}")
        stats = run_case(case_or_path, source, q_values=q_values)
        summary[f"{stats['case']}_{source.key}"] = stats
        print(json.dumps(stats["reconstruction"], indent=2))
        print(json.dumps(stats["metrics_by_q"], indent=2))
        print(stats["figure_pdf"])
        print(stats["figure_png"])
    out_summary = OUTFIG / "augmented_existing_telescope_ngc_sources_noiseaware_p1_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(out_summary)


if __name__ == "__main__":
    main()
