from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_prl_broadband_clean as base
import plot_prl_broadband_blr_optimized as opt
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    monochromatic_dirty_image,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles
from plot_uv_weighting_diagnostic import aggregate_to_coarse_uv_grid, aggregate_to_uv_cells


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "100.0"))
OBSERVING_DAYS = int(os.environ.get("AUGMENTED_OBSERVING_DAYS", os.environ.get("OBSERVING_DAYS", "30")))
BASELINE_FALSE_POSITIVE = float(
    os.environ.get("BASELINE_FALSE_POSITIVE", str(getattr(aug, "BASELINE_FALSE_POSITIVE", 0.05)))
)
AMPLITUDE_MODE_FALSE_POSITIVE = float(
    os.environ.get(
        "AMPLITUDE_MODE_FALSE_POSITIVE",
        os.environ.get("AMP_MODE_FALSE_POSITIVE", str(getattr(aug, "MODE_FALSE_POSITIVE", 0.05))),
    )
)
AMPLITUDE_SIGMA_ABS = os.environ.get("AMPLITUDE_SIGMA_ABS")
AMPLITUDE_SIGMA_ABS = None if AMPLITUDE_SIGMA_ABS is None else float(AMPLITUDE_SIGMA_ABS)
RNG_SEED = 20260515 + 771
N_PIX = aug.N_PIX
HALF_WIDTH_UAS = aug.HALF_WIDTH_UAS
SIGMA_FLOOR = 0.08
COARSE_SMOOTH_CELLS = 0.75
COARSE_BINS_U = int(os.environ.get("AUGMENTED_COARSE_BINS_U", os.environ.get("AUGMENTED_COARSE_BINS", "60")))
COARSE_BINS_V = int(os.environ.get("AUGMENTED_COARSE_BINS_V", os.environ.get("AUGMENTED_COARSE_BINS", "60")))
SPECTRAL_CUTOFF_G_LAMBDA = 95.0


def case_from_stats(path: Path) -> aug.NetworkCase:
    stats = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(
            station["name"],
            station["x_km"],
            station["y_km"],
            station["diameter_m"],
            station["is_added"],
        )
        for station in stats["stations"]
    ]
    return aug.NetworkCase(
        key=stats["case"],
        title=stats["title"],
        latitude_deg=stats["latitude_deg"],
        center_latlon=tuple(stats["center_latlon"]),
        telescopes=telescopes,
        hub_km=tuple(stats["hub_km"]),
        optimization_score=stats["optimization_score"],
    )


def simulate_bands(case: aug.NetworkCase) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED + (17 if "ctio" in case.key else 0))
    drift_rng = np.random.default_rng(RNG_SEED + (101 if "ctio" in case.key else 37))
    stations, diameters, _, _ = aug.station_table_from_case(case)
    hub = np.array(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)
    split_fraction = 1.0 / (n_station - 1.0)

    truth, axis_uas = base.make_source(N_PIX, HALF_WIDTH_UAS)
    fov_rad = 2.0 * HALF_WIDTH_UAS * base.UAS_TO_RAD
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    hub_dist = np.linalg.norm(stations - hub, axis=1)
    effective_hub_dist = aug.FIBER_LENGTH_SCALE * hub_dist
    station_eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full_like(station_eta, getattr(aug, "MODE_FALSE_POSITIVE", 0.05))
    amplitude_station_noise = np.full_like(station_eta, AMPLITUDE_MODE_FALSE_POSITIVE)
    direct_station_noise = station_noise
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    station_piston_std = aug.POST_AVERAGE_DRIFT_STD / np.sqrt(2.0)

    endpoint_coverage = {}
    for wavelength_nm in (aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=aug.SOURCE_DEC_DEG,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1e9).tolist(),
        }

    lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
    bands = []
    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        lam = math.sqrt(lo_nm * hi_nm) * 1e-9
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        df = freq_hi - freq_lo
        total_modes = aug.EXPOSURE_S * OBSERVING_DAYS * df
        u_station = aug.station_u_modes(freq, diameters)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=aug.SOURCE_DEC_DEG,
        )
        band = {"u": [], "v": []}
        vis = {key: [] for key in ("all", "split", "direct")}
        sig = {key: [] for key in ("all", "split", "direct")}
        amp_all = []
        amp_true_all = []
        amp_sigma_all = []
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            fisher_split = np.zeros(len(edges), dtype=float)
            fisher_amp = np.zeros(len(edges), dtype=float)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * math.sqrt(station_eta[i] * station_eta[j] * u_station[i] * u_station[j])
                load = split_fraction * (
                    station_eta[i] * u_station[i]
                    + station_eta[j] * u_station[j]
                    + station_noise[i]
                    + station_noise[j]
                ) + getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)
                amp_load = split_fraction * (
                    station_eta[i] * u_station[i]
                    + station_eta[j] * u_station[j]
                    + amplitude_station_noise[i]
                    + amplitude_station_noise[j]
                ) + getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)
                fisher_split[edge_index] = total_modes * 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-300)
                fisher_amp[edge_index] = total_modes * 4.0 * signal**2 * SNR_BOOST**2 / max(amp_load, 1e-300)
            sigma_amp = 1.0 / np.sqrt(np.maximum(fisher_amp, 1e-300))
            if AMPLITUDE_SIGMA_ABS is not None:
                sigma_amp = np.full_like(sigma_amp, AMPLITUDE_SIGMA_ABS)
            measured_amp = np.maximum(amp + rng.normal(scale=sigma_amp), 0.0)
            # Boost the actual CRB before clipping.  This avoids turning
            # essentially unmeasurable |V|~0 samples into artificially precise data.
            sigma_split = np.minimum(
                (1.0 / np.sqrt(np.maximum(fisher_split, 1e-300))) / SNR_BOOST,
                aug.SIGMA_CLIP_RAD,
            )
            raw_split_noise = rng.normal(scale=sigma_split)
            noise_split = q_basis @ (q_basis.T @ raw_split_noise)
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            sigma_split_projected = np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0))

            fisher_direct = (
                total_modes
                * aug.noisy_closure_fisher_station_u(vtrue, station_eta, direct_station_noise, u_station, q_basis, edges)
                * closure_rank_share
                * SNR_BOOST**2
            )
            noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)

            station_pistons = drift_rng.normal(scale=station_piston_std, size=n_station)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = raw_split_noise + residual_drift
            sigma_all = np.sqrt(sigma_split**2 + aug.POST_AVERAGE_DRIFT_STD**2)

            band["u"].append(uu)
            band["v"].append(vv)
            phase_amp = np.maximum(measured_amp, 1e-8)
            vis["all"].append(phase_amp * np.exp(1j * (phase + noise_all)))
            vis["split"].append(phase_amp * np.exp(1j * (phase_closure + noise_split)))
            vis["direct"].append(phase_amp * np.exp(1j * (phase_closure + noise_direct)))
            sig["all"].append(sigma_all)
            sig["split"].append(sigma_split_projected)
            sig["direct"].append(sigma_direct)
            amp_all.append(measured_amp)
            amp_true_all.append(amp)
            amp_sigma_all.append(sigma_amp)
        band["u"] = np.concatenate(band["u"])
        band["v"] = np.concatenate(band["v"])
        band["amp"] = np.concatenate(amp_all)
        band["amp_true"] = np.concatenate(amp_true_all)
        band["amp_sigma"] = np.concatenate(amp_sigma_all)
        for key in ("all", "split", "direct"):
            band[f"vis_{key}"] = np.concatenate(vis[key])
            band[f"sigma_{key}"] = np.concatenate(sig[key])
        bands.append(band)

    all_amp_sigma = np.concatenate([band["amp_sigma"] for band in bands])
    all_amp_true = np.concatenate([band["amp_true"] for band in bands])
    all_amp_data = np.concatenate([band["amp"] for band in bands])
    endpoint_keys = list(endpoint_coverage)
    coverage_endpoint_half_range = {
        key: {
            "u": float(np.max(np.abs(endpoint_coverage[key]["u"]))),
            "v": float(np.max(np.abs(endpoint_coverage[key]["v"]))),
        }
        for key in endpoint_keys
    }
    stats = {
        "case": case.key,
        "title": case.title,
        "snr_boost": SNR_BOOST,
        "observing_days": OBSERVING_DAYS,
        "fiber_loss_db_per_km": float(aug.FIBER_LOSS_DB_PER_KM),
        "baseline_false_positive": float(getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)),
        "mode_false_positive": float(getattr(aug, "MODE_FALSE_POSITIVE", 0.05)),
        "amplitude_mode_false_positive": float(AMPLITUDE_MODE_FALSE_POSITIVE),
        "post_average_drift_std_rad": float(getattr(aug, "POST_AVERAGE_DRIFT_STD", 0.0)),
        "direct_station_false_positive_equivalent": float(getattr(aug, "MODE_FALSE_POSITIVE", 0.05)),
        "n_station": n_station,
        "n_baseline": len(edges),
        "n_closure": int(n_closure),
        "closure_rank_share": float(closure_rank_share),
        "amplitude_sigma_model": (
            f"fixed Gaussian sigma={AMPLITUDE_SIGMA_ABS:g}"
            if AMPLITUDE_SIGMA_ABS is not None
            else "edge-readout Fisher without the visibility-amplitude factor"
        ),
        "amplitude_data_model": "Gaussian noisy |V| sample clipped at zero",
        "amplitude_sigma_abs": {
            "p10": float(np.percentile(all_amp_sigma, 10.0)),
            "median": float(np.median(all_amp_sigma)),
            "p90": float(np.percentile(all_amp_sigma, 90.0)),
        },
        "amplitude_snr": {
            "p10": float(np.percentile(all_amp_true / np.maximum(all_amp_sigma, 1e-300), 10.0)),
            "median": float(np.median(all_amp_true / np.maximum(all_amp_sigma, 1e-300))),
            "p90": float(np.percentile(all_amp_true / np.maximum(all_amp_sigma, 1e-300), 90.0)),
        },
        "amplitude_noise_realization": {
            "data_minus_truth_rms": float(np.sqrt(np.mean((all_amp_data - all_amp_true) ** 2))),
            "normalized_rms": float(np.sqrt(np.mean(((all_amp_data - all_amp_true) / np.maximum(all_amp_sigma, 1e-300)) ** 2))),
        },
        "baseline_max_km": float(np.max(np.linalg.norm(baselines, axis=1))),
        "station_link_eff_min": float(np.min(station_eta)),
        "station_link_eff_max": float(np.max(station_eta)),
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "coverage_endpoint_half_range_g_lambda": coverage_endpoint_half_range,
    }
    if "400" in endpoint_coverage:
        stats["coverage_400nm_half_range_g_lambda"] = coverage_endpoint_half_range["400"]
    if "800" in endpoint_coverage:
        stats["coverage_800nm_half_range_g_lambda"] = coverage_endpoint_half_range["800"]
    return bands, stats, truth, axis_uas


def reconstruct_band_nearest(band: dict[str, np.ndarray], key: str, fov_rad: float) -> tuple[np.ndarray, float]:
    _, occupied, _ = aggregate_cells(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n=N_PIX,
        fov_rad=fov_rad,
        average_mode="noise",
    )
    support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=aug.SUPPORT_MODE)
    label_y, label_x, fillable = nearest_label_map(occupied, support)
    return monochromatic_dirty_image(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n=N_PIX,
        fov_rad=fov_rad,
        average_mode="noise",
        label_y=label_y,
        label_x=label_x,
        fillable=fillable,
    )


def wiener_from_uv(u: np.ndarray, v: np.ndarray, vis: np.ndarray, weights: np.ndarray, fov_rad: float) -> np.ndarray:
    u = np.concatenate([u, np.array([0.0])])
    v = np.concatenate([v, np.array([0.0])])
    vis = np.concatenate([vis, np.array([1.0 + 0.0j])])
    weights = np.concatenate([weights, np.array([0.0035 * np.sum(weights)])])
    dirty, psf = base.grid_dirty(u, v, vis, weights, n=N_PIX, fov_rad=fov_rad)
    image = opt.raw_wiener_image(dirty, psf, alpha=8.0e-4, smooth_pix=0.12)
    return opt.spectral_taper_image(image, fov_rad, cutoff_g_lambda=SPECTRAL_CUTOFF_G_LAMBDA, power=5.0)


def reconstruct_band_cell_equal(band: dict[str, np.ndarray], key: str, fov_rad: float) -> tuple[np.ndarray, float]:
    u_cell, v_cell, vis_cell, weights = aggregate_to_uv_cells(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n=N_PIX,
        fov_rad=fov_rad,
        mode="cell_ivar_briggs",
    )
    image = wiener_from_uv(u_cell, v_cell, vis_cell, weights, fov_rad)
    return image, float(np.median(weights[weights > 0.0])) if np.any(weights > 0.0) else 1.0


def reconstruct_band_coarse_interp(band: dict[str, np.ndarray], key: str, fov_rad: float) -> tuple[np.ndarray, float]:
    u_cell, v_cell, vis_cell, weights = aggregate_to_coarse_uv_grid(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n_bin_u=COARSE_BINS_U,
        n_bin_v=COARSE_BINS_V,
        mode="coarse_ivar_briggs",
        smooth_cells=COARSE_SMOOTH_CELLS,
    )
    image = wiener_from_uv(u_cell, v_cell, vis_cell, weights, fov_rad)
    return image, float(np.median(weights[weights > 0.0])) if np.any(weights > 0.0) else 1.0


def reconstruct_case(bands: list[dict[str, np.ndarray]], truth: np.ndarray) -> tuple[dict, dict]:
    fov_rad = 2.0 * HALF_WIDTH_UAS * base.UAS_TO_RAD
    modes = {
        "nearest_fill": reconstruct_band_nearest,
        "fine_cell_briggs": reconstruct_band_cell_equal,
        "coarse_interp": reconstruct_band_coarse_interp,
    }
    images = {mode: {key: np.zeros_like(truth) for key in ("all", "split", "direct")} for mode in modes}
    weights = {mode: {key: 0.0 for key in ("all", "split", "direct")} for mode in modes}
    for band in bands:
        for mode, recon in modes.items():
            for key in ("all", "split", "direct"):
                image, weight = recon(band, key, fov_rad)
                images[mode][key] += weight * image
                weights[mode][key] += weight
    for mode in modes:
        for key in ("all", "split", "direct"):
            images[mode][key] = normalize_stack(images[mode][key] / max(weights[mode][key], 1e-30))
    return images, weights


def compute_metrics(images: dict, truth: np.ndarray, axis_uas: np.ndarray) -> dict:
    ring_mask, core_mask = opt.blr_masks(axis_uas)
    return {
        mode: {
            key: {
                "global_corr": float(base.corrcoef_positive(truth, images[mode][key])),
                "blr_corr": float(opt.masked_corr(truth, images[mode][key], ring_mask)),
                "ring_contrast": float(opt.ring_contrast(images[mode][key], ring_mask, core_mask)),
            }
            for key in ("all", "split", "direct")
        }
        for mode in images
    }


def plot_diagnostic(case: aug.NetworkCase, images: dict, metrics: dict, truth: np.ndarray, axis_uas: np.ndarray) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )
    modes = ["nearest_fill", "fine_cell_briggs", "coarse_interp"]
    mode_labels = {
        "nearest_fill": "nearest filled cells",
        "fine_cell_briggs": "fine-cell Briggs",
        "coarse_interp": f"{COARSE_BINS_U}x{COARSE_BINS_V} smoothed interp",
    }
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.9), constrained_layout=True)
    for row, mode in enumerate(modes):
        ax = axes[row, 0]
        ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(f"{mode_labels[mode]}\ninput")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        for col, key in enumerate(("all", "split", "direct"), start=1):
            ax = axes[row, col]
            ax.imshow(opt.normalize_blr_display(images[mode][key]), origin="lower", extent=extent, cmap="inferno")
            m = metrics[mode][key]
            label = {"all": "all-vis + drift", "split": "edge-first", "direct": "direct"}[key]
            ax.set_title(f"{label}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}")
        for col in range(4):
            axes[row, col].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col > 0:
                axes[row, col].set_yticklabels([])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("norm. brightness\n(BLR arcsinh)", fontsize=7.0)
    fig.suptitle(f"{case.title}: SNR x{SNR_BOOST:.0f} weighting/interpolation diagnostic", fontsize=10.5, weight="bold")
    snr_tag = f"snr{SNR_BOOST:g}".replace(".", "p")
    safe_key = case.key.replace("_far", f"_far_{snr_tag}_coarse{COARSE_BINS_U}x{COARSE_BINS_V}_weighting")
    png = OUTFIG / f"augmented_existing_telescope_{safe_key}.png"
    pdf = OUTFIG / f"augmented_existing_telescope_{safe_key}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_summary(summary: dict, truth: np.ndarray, axis_uas: np.ndarray) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    cases = list(summary)
    fig, axes = plt.subplots(len(cases), 4, figsize=(8.0, 2.05 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case_key in enumerate(cases):
        payload = summary[case_key]
        images = payload["images"]
        metrics = payload["metrics"]
        ax = axes[row, 0]
        ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(payload["short_label"])
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        for col, mode in enumerate(("nearest_fill", "fine_cell_briggs", "coarse_interp"), start=1):
            image = images[mode]["direct"]
            m = metrics[mode]["direct"]
            ax = axes[row, col]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            ax.set_title(f"{mode.replace('_', ' ')}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}")
            ax.set_yticklabels([])
        for col in range(4):
            axes[row, col].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("norm. brightness\n(BLR arcsinh)", fontsize=7.0)
    snr_tag = f"snr{SNR_BOOST:g}".replace(".", "p")
    fig.suptitle(
        f"Direct-closure reconstruction with SNR x{SNR_BOOST:g}: weighting/interpolation comparison",
        fontsize=10.5,
        weight="bold",
    )
    png = OUTFIG / f"augmented_existing_telescope_far_{snr_tag}_coarse{COARSE_BINS_U}x{COARSE_BINS_V}_weighting_summary.png"
    pdf = OUTFIG / f"augmented_existing_telescope_far_{snr_tag}_coarse{COARSE_BINS_U}x{COARSE_BINS_V}_weighting_summary.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    stats_paths = [
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus3_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json",
    ]
    all_stats = {}
    summary_for_plot = {}
    truth_ref = None
    axis_ref = None
    for path in stats_paths:
        case = case_from_stats(path)
        print(f"simulating {case.key} with SNR x{SNR_BOOST:g}")
        bands, stats, truth, axis_uas = simulate_bands(case)
        images, stack_weights = reconstruct_case(bands, truth)
        metrics = compute_metrics(images, truth, axis_uas)
        pdf, png = plot_diagnostic(case, images, metrics, truth, axis_uas)
        stats["metrics"] = metrics
        stats["stack_weights"] = {
            mode: {key: float(value) for key, value in mode_weights.items()}
            for mode, mode_weights in stack_weights.items()
        }
        stats["figure_pdf"] = str(pdf)
        stats["figure_png"] = str(png)
        snr_tag = f"snr{SNR_BOOST:g}".replace(".", "p")
        stats_path = OUTFIG / f"augmented_existing_telescope_{case.key}_{snr_tag}_coarse{COARSE_BINS_U}x{COARSE_BINS_V}_weighting_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        all_stats[case.key] = stats
        summary_for_plot[case.key] = {
            "short_label": case.key.replace("_", " "),
            "images": images,
            "metrics": metrics,
        }
        truth_ref = truth
        axis_ref = axis_uas
        print(pdf)
        print(png)
        print(stats_path)
        print(json.dumps(metrics, indent=2))
    assert truth_ref is not None and axis_ref is not None
    summary_pdf, summary_png = plot_summary(summary_for_plot, truth_ref, axis_ref)
    snr_tag = f"snr{SNR_BOOST:g}".replace(".", "p")
    out_path = OUTFIG / f"augmented_existing_telescope_far_{snr_tag}_coarse{COARSE_BINS_U}x{COARSE_BINS_V}_weighting_summary.json"
    for payload in all_stats.values():
        payload.pop("endpoint_coverage_g_lambda", None)
    out_path.write_text(json.dumps(all_stats, indent=2) + "\n")
    print(summary_pdf)
    print(summary_png)
    print(out_path)


if __name__ == "__main__":
    main()
