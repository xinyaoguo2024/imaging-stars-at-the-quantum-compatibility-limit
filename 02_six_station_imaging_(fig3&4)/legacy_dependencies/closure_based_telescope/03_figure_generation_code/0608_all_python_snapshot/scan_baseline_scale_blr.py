from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles
import plot_prl_broadband_blr_optimized as opt
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    monochromatic_dirty_image,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

PARENT_LAYOUT = Path(os.environ.get("SCALE_PARENT_LAYOUT", "output/figures/one08_uv80v35_realnight_layout.json"))
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_blr_scale_scan_v35")
SCALES_ENV = os.environ.get("SCALE_LIST", "")
SCALE_MIN = float(os.environ.get("SCALE_MIN", "0.60"))
SCALE_MAX = float(os.environ.get("SCALE_MAX", "1.50"))
SCALE_STEP = float(os.environ.get("SCALE_STEP", "0.10"))

N_PIX = int(os.environ.get("MONO_N_PIX", "256"))
HALF_WIDTH_UAS = float(os.environ.get("MONO_HALF_WIDTH_UAS", "80.0"))
LAMBDA_MIN_NM = float(os.environ.get("LAMBDA_MIN_NM", "400.0"))
LAMBDA_MAX_NM = float(os.environ.get("LAMBDA_MAX_NM", "800.0"))
LAMBDA_STEP_NM = float(os.environ.get("LAMBDA_STEP_NM", "10.0"))
SUPPORT_MODE = os.environ.get("MONO_SUPPORT_MODE", "ellipse").lower()
CELL_AVERAGE_MODES = tuple(os.environ.get("MONO_CELL_AVERAGE_MODES", "direct,noise").split(","))
ARRAY_LAT_DEG = float(os.environ.get("ARRAY_LAT_DEG", "35.0"))
SOURCE_DEC_DEG = float(os.environ.get("SOURCE_DEC_DEG", "2.052388"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "36"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "600.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "300.0"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "60"))
IMAGING_SNR_BOOST = float(os.environ.get("IMAGING_SNR_BOOST", "1.0"))
RNG_SEED = int(os.environ.get("SCALE_SCAN_RNG_SEED", "273"))


def parse_scales() -> np.ndarray:
    if SCALES_ENV.strip():
        return np.array([float(x) for x in SCALES_ENV.replace(",", " ").split()], dtype=float)
    n = int(round((SCALE_MAX - SCALE_MIN) / SCALE_STEP)) + 1
    return np.round(SCALE_MIN + SCALE_STEP * np.arange(n), 10)


def load_parent_layout() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((ROOT / PARENT_LAYOUT).read_text())
    stations = np.array(payload["stations_km"], dtype=float)
    hub = np.array(payload.get("hub_km", [2.0, 0.0]), dtype=float)
    return stations, hub


def scaled_layout(parent: np.ndarray, hub: np.ndarray, scale: float) -> np.ndarray:
    return hub + scale * (parent - hub)


def simulate_direct_band(
    *,
    lam_lo: float,
    lam_hi: float,
    rng: np.random.Generator,
    vgrid: np.ndarray,
    uv_axis: np.ndarray,
    baselines_km: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    station_link_eff: np.ndarray,
    station_channel_noise: np.ndarray,
    closure_rank_share: float,
    hour_angles: np.ndarray,
    noiseless: bool,
) -> dict[str, np.ndarray]:
    lam = np.sqrt(lam_lo * lam_hi)
    freq = base.C_LIGHT / lam
    freq_lo = base.C_LIGHT / lam_hi
    freq_hi = base.C_LIGHT / lam_lo
    df = freq_hi - freq_lo
    u_mode = base.mode_occupation_ab(base.SOURCE_AB_MAG, freq, diameter_m=base.TELESCOPE_DIAMETER_M)
    total_modes = EXPOSURE_S * OBSERVING_DAYS * df
    uu_rows, vv_rows = project_enu_baselines(
        baselines_km,
        hour_angles,
        lam,
        latitude_deg=ARRAY_LAT_DEG,
        declination_deg=SOURCE_DEC_DEG,
    )

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_vis: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    for uu, vv in zip(uu_rows, vv_rows):
        vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
        amp = np.abs(vtrue)
        phase = np.angle(vtrue)
        phase_closure = q_basis @ (q_basis.T @ phase)
        if noiseless:
            noise = np.zeros_like(amp)
            sigma = np.zeros_like(amp)
        else:
            fisher_direct = (
                total_modes
                * base.noisy_closure_fisher_from_station_modes(
                    vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                )
                * closure_rank_share
                * IMAGING_SNR_BOOST**2
            )
            noise, sigma = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)
        all_u.append(uu)
        all_v.append(vv)
        all_vis.append(amp * np.exp(1j * (phase_closure + noise)))
        all_sigma.append(sigma)
    return {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
        "vis": np.concatenate(all_vis),
        "sigma": np.concatenate(all_sigma),
    }


def reconstruct_scale(
    *,
    scale: float,
    parent_stations: np.ndarray,
    hub: np.ndarray,
    truth: np.ndarray,
    axis_uas: np.ndarray,
    vgrid: np.ndarray,
    uv_axis: np.ndarray,
    fov_rad: float,
    lam_edges_nm: np.ndarray,
    noiseless: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, float]]:
    rng = np.random.default_rng(RNG_SEED)
    stations = scaled_layout(parent_stations, hub, scale)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    baselines_km = np.array([stations[j] - stations[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    closure_rank_share = min(1.0, (n_station - 1.0) / w_basis.shape[1])
    hub_distances_km = np.linalg.norm(stations - hub, axis=1)
    effective_hub_distances_km = base.FIBER_LENGTH_SCALE * hub_distances_km
    station_link_eff = 10.0 ** (-base.FIBER_LOSS_DB_PER_KM * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)
    hour_angles = realnight_hour_angles(N_TIME_WINDOWS, EXPOSURE_S, EXPOSURE_GAP_S)

    stacks = {mode: np.zeros((N_PIX, N_PIX), dtype=float) for mode in CELL_AVERAGE_MODES}
    stack_weights = {mode: 0.0 for mode in CELL_AVERAGE_MODES}
    occupied_counts: list[int] = []
    support_counts: list[int] = []

    for lam_lo_nm, lam_hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        band = simulate_direct_band(
            lam_lo=lam_lo_nm * 1e-9,
            lam_hi=lam_hi_nm * 1e-9,
            rng=rng,
            vgrid=vgrid,
            uv_axis=uv_axis,
            baselines_km=baselines_km,
            q_basis=q_basis,
            edges=edges,
            station_link_eff=station_link_eff,
            station_channel_noise=station_channel_noise,
            closure_rank_share=closure_rank_share,
            hour_angles=hour_angles,
            noiseless=noiseless,
        )
        _, occupied, _ = aggregate_cells(
            band["u"],
            band["v"],
            band["vis"],
            band["sigma"],
            n=N_PIX,
            fov_rad=fov_rad,
            average_mode="direct",
        )
        support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=SUPPORT_MODE)
        label_y, label_x, fillable = nearest_label_map(occupied, support)
        occupied_counts.append(int(np.sum(occupied)))
        support_counts.append(int(np.sum(fillable)))
        for mode in CELL_AVERAGE_MODES:
            image, band_weight = monochromatic_dirty_image(
                band["u"],
                band["v"],
                band["vis"],
                band["sigma"],
                n=N_PIX,
                fov_rad=fov_rad,
                average_mode=mode,
                label_y=label_y,
                label_x=label_x,
                fillable=fillable,
            )
            stacks[mode] += band_weight * image
            stack_weights[mode] += band_weight

    images: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    ring_mask, core_mask = opt.blr_masks(axis_uas)
    for mode in CELL_AVERAGE_MODES:
        images[mode] = normalize_stack(stacks[mode] / max(stack_weights[mode], 1e-30))
        metrics[mode] = {
            "global_corr": float(base.corrcoef_positive(truth, images[mode])),
            "blr_corr": float(opt.masked_corr(truth, images[mode], ring_mask)),
            "ring_contrast": float(opt.ring_contrast(images[mode], ring_mask, core_mask)),
        }

    uu_400, vv_400 = project_enu_baselines(
        baselines_km,
        hour_angles,
        400e-9,
        latitude_deg=ARRAY_LAT_DEG,
        declination_deg=SOURCE_DEC_DEG,
    )
    extra = {
        "baseline_min_km": float(np.min(np.linalg.norm(baselines_km, axis=1))),
        "baseline_max_km": float(np.max(np.linalg.norm(baselines_km, axis=1))),
        "hub_distance_max_km": float(np.max(hub_distances_km)),
        "station_link_eff_min": float(np.min(station_link_eff)),
        "u400_max_g_lambda": float(np.max(np.abs(uu_400)) / 1e9),
        "v400_max_g_lambda": float(np.max(np.abs(vv_400)) / 1e9),
        "occupied_cells_median": float(np.median(occupied_counts)),
        "support_cells_median": float(np.median(support_counts)),
    }
    return images, metrics, extra


def plot_scan(results: list[dict]) -> tuple[Path, Path]:
    scales = np.array([r["scale"] for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    for mode, marker in zip(CELL_AVERAGE_MODES, ("o", "s", "^")):
        axes[0].plot(
            scales,
            [r["noisy"]["metrics"][mode]["blr_corr"] for r in results],
            marker=marker,
            label=f"{mode}, noisy",
        )
        axes[0].plot(
            scales,
            [r["noiseless"]["metrics"][mode]["blr_corr"] for r in results],
            marker=marker,
            linestyle="--",
            label=f"{mode}, noiseless",
        )
        axes[1].plot(
            scales,
            [r["noisy"]["metrics"][mode]["global_corr"] for r in results],
            marker=marker,
            label=f"{mode}, noisy",
        )
        axes[1].plot(
            scales,
            [r["noiseless"]["metrics"][mode]["global_corr"] for r in results],
            marker=marker,
            linestyle="--",
            label=f"{mode}, noiseless",
        )
    axes[0].set_xlabel("baseline scale about hub")
    axes[0].set_ylabel("BLR correlation")
    axes[0].set_title("BLR recovery")
    axes[1].set_xlabel("baseline scale about hub")
    axes[1].set_ylabel("global correlation")
    axes[1].set_title("Full image recovery")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.7, frameon=False)
    png = OUTDIR / f"baseline_scale_scan{OUTPUT_SUFFIX}.png"
    pdf = OUTDIR / f"baseline_scale_scan{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_best_images(
    truth: np.ndarray,
    axis_uas: np.ndarray,
    best_noisy: dict,
    best_noiseless: dict,
    images_by_scale: dict[tuple[float, str], dict[str, np.ndarray]],
) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    panels = [
        ("Input source", truth),
        (
            f"Best noisy s={best_noisy['scale']:.2f}",
            images_by_scale[(best_noisy["scale"], "noisy")][best_noisy["mode"]],
        ),
        (
            f"Best noiseless s={best_noiseless['scale']:.2f}",
            images_by_scale[(best_noiseless["scale"], "noiseless")][best_noiseless["mode"]],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    axes[0].set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    png = OUTDIR / f"baseline_scale_scan_best_images{OUTPUT_SUFFIX}.png"
    pdf = OUTDIR / f"baseline_scale_scan_best_images{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    parent_stations, hub = load_parent_layout()
    scales = parse_scales()
    fov_rad = 2.0 * HALF_WIDTH_UAS * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(N_PIX, HALF_WIDTH_UAS)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    lam_edges_nm = np.arange(LAMBDA_MIN_NM, LAMBDA_MAX_NM + 0.5 * LAMBDA_STEP_NM, LAMBDA_STEP_NM)
    if lam_edges_nm[-1] < LAMBDA_MAX_NM:
        lam_edges_nm = np.append(lam_edges_nm, LAMBDA_MAX_NM)
    lam_edges_nm[-1] = LAMBDA_MAX_NM

    results: list[dict] = []
    images_by_scale: dict[tuple[float, str], dict[str, np.ndarray]] = {}
    for index, scale in enumerate(scales, 1):
        print(f"scale {scale:.3f} ({index}/{len(scales)})")
        noisy_images, noisy_metrics, noisy_extra = reconstruct_scale(
            scale=float(scale),
            parent_stations=parent_stations,
            hub=hub,
            truth=truth,
            axis_uas=axis_uas,
            vgrid=vgrid,
            uv_axis=uv_axis,
            fov_rad=fov_rad,
            lam_edges_nm=lam_edges_nm,
            noiseless=False,
        )
        noiseless_images, noiseless_metrics, noiseless_extra = reconstruct_scale(
            scale=float(scale),
            parent_stations=parent_stations,
            hub=hub,
            truth=truth,
            axis_uas=axis_uas,
            vgrid=vgrid,
            uv_axis=uv_axis,
            fov_rad=fov_rad,
            lam_edges_nm=lam_edges_nm,
            noiseless=True,
        )
        images_by_scale[(float(scale), "noisy")] = noisy_images
        images_by_scale[(float(scale), "noiseless")] = noiseless_images
        row = {
            "scale": float(scale),
            "noisy": {"metrics": noisy_metrics, "extra": noisy_extra},
            "noiseless": {"metrics": noiseless_metrics, "extra": noiseless_extra},
        }
        results.append(row)
        best_mode = max(CELL_AVERAGE_MODES, key=lambda m: noisy_metrics[m]["blr_corr"])
        print(
            f"  noisy best {best_mode}: BLR={noisy_metrics[best_mode]['blr_corr']:.3f}, "
            f"global={noisy_metrics[best_mode]['global_corr']:.3f}"
        )

    best_noisy = max(
        (
            {"scale": r["scale"], "mode": mode, **r["noisy"]["metrics"][mode]}
            for r in results
            for mode in CELL_AVERAGE_MODES
        ),
        key=lambda x: x["blr_corr"],
    )
    best_noiseless = max(
        (
            {"scale": r["scale"], "mode": mode, **r["noiseless"]["metrics"][mode]}
            for r in results
            for mode in CELL_AVERAGE_MODES
        ),
        key=lambda x: x["blr_corr"],
    )
    scan_pdf, scan_png = plot_scan(results)
    best_pdf, best_png = plot_best_images(truth, axis_uas, best_noisy, best_noiseless, images_by_scale)

    stats = {
        "parent_layout": str(PARENT_LAYOUT),
        "scale_about_hub": True,
        "hub_km": hub.tolist(),
        "scales": scales.tolist(),
        "lambda_step_nm": LAMBDA_STEP_NM,
        "n_bands": len(lam_edges_nm) - 1,
        "n_time_windows": N_TIME_WINDOWS,
        "exposure_s": EXPOSURE_S,
        "exposure_gap_s": EXPOSURE_GAP_S,
        "observing_days": OBSERVING_DAYS,
        "array_latitude_deg": ARRAY_LAT_DEG,
        "source_declination_deg": SOURCE_DEC_DEG,
        "cell_average_modes": CELL_AVERAGE_MODES,
        "best_noisy_blr": best_noisy,
        "best_noiseless_blr": best_noiseless,
        "results": results,
    }
    json_path = OUTDIR / f"baseline_scale_scan{OUTPUT_SUFFIX}.json"
    json_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(scan_pdf)
    print(scan_png)
    print(best_pdf)
    print(best_png)
    print(json_path)
    print(json.dumps({"best_noisy_blr": best_noisy, "best_noiseless_blr": best_noiseless}, indent=2))


if __name__ == "__main__":
    main()
