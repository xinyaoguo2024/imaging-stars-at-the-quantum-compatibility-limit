from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_existing_telescope_ngc_sources_noiseaware_p1 as noiseaware
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

LAYOUT = OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json"
SOURCE = ngc.NGC4151
OBSERVING_DAYS = 30
SNR_BOOST = 1.0
Q_WEIGHT = 0.25
RNG_SEED = 20260517 + 4151
BASELINE_FALSE_POSITIVE = getattr(aug, "BASELINE_FALSE_POSITIVE", 0.0)


def load_case(path: Path) -> aug.NetworkCase:
    payload = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(
            station["name"],
            float(station["x_km"]),
            float(station["y_km"]),
            float(station["diameter_m"]),
            bool(station["is_added"]),
        )
        for station in payload["stations"]
    ]
    return aug.NetworkCase(
        key=payload.get("case_key", path.stem.replace("_layout", "")),
        title=payload.get("case_title", "Maunakea top-four core + five 5 m outstations"),
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=tuple(payload["hub_km"]),
        optimization_score=float(payload["metrics"]["score"]),
    )


def directed_edge_vector(
    edges: list[tuple[int, int]],
    directed_edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    w = np.zeros(len(edges), dtype=float)
    for a, b in directed_edges:
        if a < b:
            w[edge_index[(a, b)]] += 1.0
        else:
            w[edge_index[(b, a)]] -= 1.0
    return w


def wrap_phase(x: np.ndarray | float) -> np.ndarray | float:
    return np.angle(np.exp(1j * x))


def direct_cycle_covariance(fisher: np.ndarray, *, max_std: float = 2.5) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    safe = np.maximum(evals, 1.0 / max_std**2)
    return (evecs / safe) @ evecs.T


def closure_snr_tables(case: aug.NetworkCase, truth: np.ndarray, axis_uas: np.ndarray) -> dict:
    stations, diameters, names, _ = aug.station_table_from_case(case)
    hub = np.array(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)
    triangles = [(i, j, k) for i in range(n_station) for j in range(i + 1, n_station) for k in range(j + 1, n_station)]
    triangle_vectors = {
        tri: directed_edge_vector(edges, ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
        for tri in triangles
    }

    vgrid, uv_axis = base.visibility_grid(truth, 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD)
    hub_dist = np.linalg.norm(stations - hub, axis=1)
    effective_hub_dist = aug.FIBER_LENGTH_SCALE * hub_dist
    station_eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full_like(station_eta, getattr(aug, "MODE_FALSE_POSITIVE", 0.05))
    direct_station_noise = station_noise
    split_fraction = 1.0 / (n_station - 1.0)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    hour_angle_hours = hour_angles * 12.0 / np.pi

    lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
    records: list[dict] = []
    band_rows: list[dict] = []
    broadband_by_time_triangle: dict[tuple[int, tuple[int, int, int]], dict[str, float | int | str]] = {}
    broadband_by_triangle: dict[tuple[int, int, int], dict[str, float | int | str]] = {}

    for band_index, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
        lam_nm = math.sqrt(lo_nm * hi_nm)
        lam = lam_nm * 1e-9
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
            declination_deg=SOURCE.dec_deg,
        )

        band_detection_split: list[float] = []
        band_detection_direct: list[float] = []
        band_precision_split: list[float] = []
        band_precision_direct: list[float] = []

        for time_index, (uu, vv) in enumerate(zip(uu_rows, vv_rows)):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            phase = np.angle(vtrue)
            amp = np.abs(vtrue)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            fisher_split = np.zeros(len(edges), dtype=float)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * math.sqrt(station_eta[i] * station_eta[j] * u_station[i] * u_station[j])
                load = split_fraction * (
                    station_eta[i] * u_station[i]
                    + station_eta[j] * u_station[j]
                    + station_noise[i]
                    + station_noise[j]
                ) + getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)
                fisher_split[edge_index] = (
                    total_modes
                    * 4.0
                    * signal**2
                    * nu_eff[edge_index] ** 2
                    / max(load, 1e-300)
                    * SNR_BOOST**2
                )
            sigma_edge = np.minimum(1.0 / np.sqrt(np.maximum(fisher_split, 1e-300)), aug.SIGMA_CLIP_RAD)

            fisher_direct = (
                total_modes
                * aug.noisy_closure_fisher_station_u(vtrue, station_eta, direct_station_noise, u_station, q_basis, edges)
                * closure_rank_share
                * SNR_BOOST**2
            )
            cov_direct = direct_cycle_covariance(fisher_direct)

            for triangle_id, tri in enumerate(triangles, start=1):
                w = triangle_vectors[tri]
                phi = float(wrap_phase(np.dot(w, phase)))
                sigma_split = float(np.sqrt(np.sum((w * sigma_edge) ** 2)))
                q_vec = q_basis.T @ w
                sigma_direct = float(np.sqrt(max(float(q_vec.T @ cov_direct @ q_vec), 0.0)))
                precision_split = 1.0 / max(sigma_split, 1e-30)
                precision_direct = 1.0 / max(sigma_direct, 1e-30)
                detection_split = abs(phi) * precision_split
                detection_direct = abs(phi) * precision_direct
                band_precision_split.append(precision_split)
                band_precision_direct.append(precision_direct)
                band_detection_split.append(detection_split)
                band_detection_direct.append(detection_direct)
                time_key = (time_index, tri)
                if time_key not in broadband_by_time_triangle:
                    broadband_by_time_triangle[time_key] = {
                        "time_index": time_index,
                        "hour_angle_h": float(hour_angle_hours[time_index]),
                        "triangle_id": triangle_id,
                        "stations": "-".join(names[idx] for idx in tri),
                        "station_indices_1based": "-".join(str(idx + 1) for idx in tri),
                        "edge_first_precision_chi2": 0.0,
                        "direct_precision_chi2": 0.0,
                        "edge_first_detection_chi2": 0.0,
                        "direct_detection_chi2": 0.0,
                        "n_lambda_channels": 0,
                    }
                if tri not in broadband_by_triangle:
                    broadband_by_triangle[tri] = {
                        "triangle_id": triangle_id,
                        "stations": "-".join(names[idx] for idx in tri),
                        "station_indices_1based": "-".join(str(idx + 1) for idx in tri),
                        "edge_first_precision_chi2": 0.0,
                        "direct_precision_chi2": 0.0,
                        "edge_first_detection_chi2": 0.0,
                        "direct_detection_chi2": 0.0,
                        "n_lambda_time_samples": 0,
                    }
                for bucket in (broadband_by_time_triangle[time_key], broadband_by_triangle[tri]):
                    bucket["edge_first_precision_chi2"] = float(bucket["edge_first_precision_chi2"]) + precision_split**2
                    bucket["direct_precision_chi2"] = float(bucket["direct_precision_chi2"]) + precision_direct**2
                    bucket["edge_first_detection_chi2"] = float(bucket["edge_first_detection_chi2"]) + detection_split**2
                    bucket["direct_detection_chi2"] = float(bucket["direct_detection_chi2"]) + detection_direct**2
                broadband_by_time_triangle[time_key]["n_lambda_channels"] = int(
                    broadband_by_time_triangle[time_key]["n_lambda_channels"]
                ) + 1
                broadband_by_triangle[tri]["n_lambda_time_samples"] = int(broadband_by_triangle[tri]["n_lambda_time_samples"]) + 1
                records.append(
                    {
                        "band_index": band_index,
                        "lambda_center_nm": lam_nm,
                        "lambda_lo_nm": float(lo_nm),
                        "lambda_hi_nm": float(hi_nm),
                        "time_index": time_index,
                        "hour_angle_h": float(hour_angle_hours[time_index]),
                        "triangle_id": triangle_id,
                        "stations": "-".join(names[idx] for idx in tri),
                        "station_indices_1based": "-".join(str(idx + 1) for idx in tri),
                        "closure_phase_rad": phi,
                        "sigma_edge_first_rad": sigma_split,
                        "sigma_direct_rad": sigma_direct,
                        "precision_snr_edge_first": precision_split,
                        "precision_snr_direct": precision_direct,
                        "detection_snr_edge_first": detection_split,
                        "detection_snr_direct": detection_direct,
                    }
                )

        def pct(values: list[float], q: float) -> float:
            return float(np.percentile(np.asarray(values, dtype=float), q))

        band_rows.append(
            {
                "lambda_center_nm": lam_nm,
                "lambda_lo_nm": float(lo_nm),
                "lambda_hi_nm": float(hi_nm),
                "n_time_triangle_samples": len(band_detection_direct),
                "median_precision_snr_edge_first": pct(band_precision_split, 50),
                "median_precision_snr_direct": pct(band_precision_direct, 50),
                "p90_precision_snr_edge_first": pct(band_precision_split, 90),
                "p90_precision_snr_direct": pct(band_precision_direct, 90),
                "max_precision_snr_edge_first": pct(band_precision_split, 100),
                "max_precision_snr_direct": pct(band_precision_direct, 100),
                "median_detection_snr_edge_first": pct(band_detection_split, 50),
                "median_detection_snr_direct": pct(band_detection_direct, 50),
                "p90_detection_snr_edge_first": pct(band_detection_split, 90),
                "p90_detection_snr_direct": pct(band_detection_direct, 90),
                "max_detection_snr_edge_first": pct(band_detection_split, 100),
                "max_detection_snr_direct": pct(band_detection_direct, 100),
                "n_detection_gt3_edge_first": int(np.sum(np.asarray(band_detection_split) > 3.0)),
                "n_detection_gt3_direct": int(np.sum(np.asarray(band_detection_direct) > 3.0)),
                "n_precision_gt3_edge_first": int(np.sum(np.asarray(band_precision_split) > 3.0)),
                "n_precision_gt3_direct": int(np.sum(np.asarray(band_precision_direct) > 3.0)),
            }
        )

    band_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_closure_snr_by_10nm_band.csv"
    records_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_closure_snr_all_samples.csv"
    with band_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)
    with records_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    def finalize_bucket(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
        out = dict(row)
        out["broadband_precision_snr_edge_first"] = math.sqrt(float(row["edge_first_precision_chi2"]))
        out["broadband_precision_snr_direct"] = math.sqrt(float(row["direct_precision_chi2"]))
        out["broadband_detection_snr_edge_first"] = math.sqrt(float(row["edge_first_detection_chi2"]))
        out["broadband_detection_snr_direct"] = math.sqrt(float(row["direct_detection_chi2"]))
        return out

    time_triangle_rows = [finalize_bucket(row) for row in broadband_by_time_triangle.values()]
    triangle_rows = [finalize_bucket(row) for row in broadband_by_triangle.values()]
    time_triangle_rows.sort(key=lambda row: float(row["broadband_detection_snr_direct"]), reverse=True)
    triangle_rows.sort(key=lambda row: float(row["broadband_detection_snr_direct"]), reverse=True)

    time_triangle_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_broadband_time_triangle_snr.csv"
    triangle_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_broadband_triangle_snr.csv"
    gt3_time_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_broadband_time_triangle_direct_detection_snr_gt3.csv"
    gt3_triangle_csv = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_broadband_triangle_direct_detection_snr_gt3.csv"
    for path, rows in ((time_triangle_csv, time_triangle_rows), (triangle_csv, triangle_rows)):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    gt3_time = [row for row in time_triangle_rows if float(row["broadband_detection_snr_direct"]) > 3.0]
    gt3_triangle = [row for row in triangle_rows if float(row["broadband_detection_snr_direct"]) > 3.0]
    for path, rows in ((gt3_time_csv, gt3_time), (gt3_triangle_csv, gt3_triangle)):
        if rows:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("No direct-closure broadband detection_snr > 3 rows.\n")

    return {
        "band_csv": str(band_csv),
        "records_csv": str(records_csv),
        "time_triangle_csv": str(time_triangle_csv),
        "triangle_csv": str(triangle_csv),
        "gt3_time_triangle_csv": str(gt3_time_csv),
        "gt3_triangle_csv": str(gt3_triangle_csv),
        "band_rows": band_rows,
        "gt3_time_triangle_count": len(gt3_time),
        "gt3_triangle_count": len(gt3_triangle),
        "gt3_time_triangle_top": gt3_time[:20],
        "gt3_triangle_top": gt3_triangle[:20],
    }


def noiseaware_dirty_psf(
    band: dict[str, np.ndarray],
    strategy: str,
    fov_rad: float,
    *,
    fill: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    grid, occupied, cell_var = aggregate_cells(
        band["u"],
        band["v"],
        band[f"vis_{strategy}"],
        band[f"sigma_{strategy}"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        average_mode="noise",
    )
    support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=aug.SUPPORT_MODE)
    label_y, label_x, fillable = nearest_label_map(occupied, support)
    filled_grid = np.zeros_like(grid)
    if fill:
        filled_grid[fillable] = grid[label_y[fillable], label_x[fillable]]
    filled_grid[occupied] = grid[occupied]

    area_weight = noiseaware.coarse_density_on_fine_grid(
        band["u"],
        band["v"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        n_bin=noiseaware.N_BIN,
        power=1.0,
    )
    gate, _ = noiseaware.noise_aware_gate(cell_var, label_y, label_x, fillable, occupied, q=Q_WEIGHT)
    support_weight = np.zeros_like(area_weight)
    if fill:
        support_weight[fillable] = noiseaware.FILL_ALPHA * area_weight[fillable] * gate[fillable]
    support_weight[occupied] = area_weight[occupied] * gate[occupied]
    support_weight = np.clip(support_weight, 0.0, noiseaware.WEIGHT_CLIP[1])
    positive = support_weight[support_weight > 0.0]
    if len(positive):
        floor = noiseaware.WEIGHT_CLIP[0] * float(np.median(positive))
        support_weight[(support_weight > 0.0) & (support_weight < floor)] = floor
    support_weight[wt.N_PIX // 2, wt.N_PIX // 2] = max(support_weight[wt.N_PIX // 2, wt.N_PIX // 2], 1.0)

    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filled_grid * support_weight))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(support_weight))).real
    peak = psf[wt.N_PIX // 2, wt.N_PIX // 2]
    if peak > 0.0:
        image /= peak
        psf /= peak

    assigned_var = cell_var[label_y[fillable], label_x[fillable]]
    assigned_weight = support_weight[fillable]
    finite = np.isfinite(assigned_var) & (assigned_var > 0.0) & (assigned_weight > 0.0)
    if np.any(finite):
        weight = 1.0 / float(np.average(assigned_var[finite], weights=assigned_weight[finite]))
    else:
        weight = 1.0
    return image, psf, weight


def stack_dirty_psf(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    truth: np.ndarray,
    *,
    fill: bool,
) -> tuple[np.ndarray, np.ndarray]:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    image = np.zeros_like(truth)
    psf = np.zeros_like(truth)
    total_weight = 0.0
    for band in bands:
        band_image, band_psf, weight = noiseaware_dirty_psf(band, strategy, fov_rad, fill=fill)
        image += weight * band_image
        psf += weight * band_psf
        total_weight += weight
    image /= max(total_weight, 1e-30)
    psf /= max(total_weight, 1e-30)
    peak = psf[wt.N_PIX // 2, wt.N_PIX // 2]
    if peak > 0.0:
        image /= peak
        psf /= peak
    return image, psf


def centered_fft_convolve(image: np.ndarray, otf: np.ndarray, *, adjoint: bool = False) -> np.ndarray:
    spectrum = np.fft.fft2(np.fft.ifftshift(image))
    kernel = np.conj(otf) if adjoint else otf
    return np.fft.fftshift(np.fft.ifft2(spectrum * kernel)).real


def tv_gradient(image: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    dx = np.roll(image, -1, axis=1) - image
    dy = np.roll(image, -1, axis=0) - image
    norm = np.sqrt(dx * dx + dy * dy + eps * eps)
    px = dx / norm
    py = dy / norm
    div = px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)
    return -div


def rml_tv_reconstruct(
    dirty: np.ndarray,
    psf: np.ndarray,
    *,
    n_iter: int = 420,
    step: float = 0.11,
    tv_weight: float = 0.018,
    l2_weight: float = 0.004,
    smooth_every: int = 40,
) -> np.ndarray:
    otf = np.fft.fft2(np.fft.ifftshift(psf))
    x = dirty.copy()
    x -= np.percentile(x, 2.0)
    x = np.clip(base.gaussian_filter(x, 1.1), 0.0, None)
    scale = np.percentile(x, 99.5)
    if scale > 0:
        x /= scale
    d = dirty.copy()
    d -= np.percentile(d, 2.0)
    d_scale = np.percentile(np.abs(d), 99.5)
    if d_scale > 0:
        d /= d_scale

    for iteration in range(n_iter):
        model = centered_fft_convolve(x, otf)
        residual = model - d
        grad = centered_fft_convolve(residual, otf, adjoint=True)
        grad += l2_weight * x
        grad += tv_weight * tv_gradient(x)
        x -= step * grad
        x = np.clip(x, 0.0, None)
        if smooth_every > 0 and (iteration + 1) % smooth_every == 0:
            x = base.gaussian_filter(x, 0.22)
    return normalize_stack(x)


def image_metrics(source: ngc.SourceModel, truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    ring_mask, core_mask = ngc.blr_masks_for_source(axis_uas, source)
    return {
        "global_corr": float(base.corrcoef_positive(truth, image)),
        "blr_corr": float(opt.masked_corr(truth, image, ring_mask)),
        "ring_contrast": float(opt.ring_contrast(image, ring_mask, core_mask)),
    }


def reconstruction_comparison(case: aug.NetworkCase) -> dict:
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    with ngc.patched_source(SOURCE):
        bands, stats, truth, axis_uas = wt.simulate_bands(case)

    images: dict[str, np.ndarray] = {"truth": truth}
    psfs: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for strategy in ("split", "direct"):
        filled_dirty, filled_psf = stack_dirty_psf(bands, strategy, truth, fill=True)
        sparse_dirty, sparse_psf = stack_dirty_psf(bands, strategy, truth, fill=False)
        clean, _ = base.multiscale_clean(
            sparse_dirty,
            sparse_psf,
            scales_pix=(0.0, 2.0, 4.5, 8.0, 14.0, 22.0),
            gain=0.10,
            max_iter=900,
            threshold_factor=1.3,
        )
        rml = rml_tv_reconstruct(sparse_dirty, sparse_psf)
        images[f"{strategy}_filled_dirty"] = normalize_stack(filled_dirty)
        images[f"{strategy}_sparse_dirty"] = normalize_stack(sparse_dirty)
        images[f"{strategy}_multiscale_clean"] = normalize_stack(clean)
        images[f"{strategy}_rml_tv"] = normalize_stack(rml)
        psfs[f"{strategy}_sparse"] = sparse_psf
        for key in (
            f"{strategy}_filled_dirty",
            f"{strategy}_sparse_dirty",
            f"{strategy}_multiscale_clean",
            f"{strategy}_rml_tv",
        ):
            metrics[key] = image_metrics(SOURCE, truth, images[key], axis_uas)

    figure_pdf, figure_png = plot_reconstruction_comparison(images, metrics, axis_uas)
    stats.update(
        {
            "reconstruction_methods": {
                "filled_dirty": "current nearest-filled noise-aware p=1 dirty map",
                "sparse_dirty": "measured-cell-only dirty map using the same weights",
                "multiscale_clean": "multiscale CLEAN on sparse dirty map and sparse PSF",
                "rml_tv": "positive image-domain RML approximation with TV and L2 regularization",
            },
            "metrics_clean_rml": metrics,
            "figure_pdf": str(figure_pdf),
            "figure_png": str(figure_png),
        }
    )
    return {"stats": stats, "truth": truth, "axis_uas": axis_uas}


def plot_reconstruction_comparison(
    images: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    axis_uas: np.ndarray,
) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.75), constrained_layout=True)
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
        }
    )
    rows = [("direct", "Direct closure-space"), ("split", "Edge-first closure")]
    cols = [
        ("truth", "Input"),
        ("filled_dirty", "Filled dirty"),
        ("multiscale_clean", "Multiscale CLEAN"),
        ("rml_tv", "RML-TV"),
    ]
    image_axes = []
    for row_idx, (strategy, row_label) in enumerate(rows):
        for col_idx, (suffix, col_label) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            key = "truth" if suffix == "truth" else f"{strategy}_{suffix}"
            display = opt.normalize_blr_display(images[key])
            ax.imshow(display, origin="lower", extent=extent, cmap="inferno")
            if suffix == "truth":
                ax.set_title(f"{row_label}\n{col_label}")
            else:
                m = metrics[key]
                ax.set_title(f"{col_label}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col_idx == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.025,
        pad=0.012,
    )
    cbar.set_label("norm. brightness", fontsize=6.6)
    fig.suptitle("Maunakea top4+5, NGC 4151, 30 days, SNR boost = 1: deconvolution test", fontsize=10.4, weight="bold")
    png = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_clean_rml_comparison.png"
    pdf = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_clean_rml_comparison.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    case = load_case(LAYOUT)
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    with ngc.patched_source(SOURCE):
        truth, axis_uas = base.make_source(wt.N_PIX, wt.HALF_WIDTH_UAS)
        snr = closure_snr_tables(case, truth, axis_uas)
    recon = reconstruction_comparison(case)

    summary = {
        "case": case.key,
        "source": SOURCE.name,
        "observing_days": OBSERVING_DAYS,
        "snr_boost": SNR_BOOST,
        "lambda_step_nm": aug.LAMBDA_STEP_NM,
        "fiber_loss_db_per_km": float(aug.FIBER_LOSS_DB_PER_KM),
        "baseline_false_positive": float(BASELINE_FALSE_POSITIVE),
        "snr_outputs": {
            "band_csv": snr["band_csv"],
            "all_samples_csv": snr["records_csv"],
            "broadband_time_triangle_csv": snr["time_triangle_csv"],
            "broadband_triangle_csv": snr["triangle_csv"],
            "direct_broadband_time_triangle_detection_snr_gt3_csv": snr["gt3_time_triangle_csv"],
            "direct_broadband_triangle_detection_snr_gt3_csv": snr["gt3_triangle_csv"],
            "direct_broadband_time_triangle_detection_snr_gt3_count": snr["gt3_time_triangle_count"],
            "direct_broadband_triangle_detection_snr_gt3_count": snr["gt3_triangle_count"],
            "direct_broadband_time_triangle_detection_snr_gt3_top20": snr["gt3_time_triangle_top"],
            "direct_broadband_triangle_detection_snr_gt3_top20": snr["gt3_triangle_top"],
        },
        "clean_rml_outputs": {
            "figure_pdf": recon["stats"]["figure_pdf"],
            "figure_png": recon["stats"]["figure_png"],
            "metrics": recon["stats"]["metrics_clean_rml"],
        },
    }
    out = OUTFIG / "maunakea_top4_plus5_ngc4151_30day_snr_clean_rml_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)
    print(summary["snr_outputs"]["band_csv"])
    print(summary["snr_outputs"]["broadband_triangle_csv"])
    print(summary["snr_outputs"]["direct_broadband_triangle_detection_snr_gt3_csv"])
    print(summary["clean_rml_outputs"]["figure_png"])
    print(json.dumps(summary["clean_rml_outputs"]["metrics"], indent=2))
    print(
        "direct broadband gt3 counts:",
        "time-triangle=",
        summary["snr_outputs"]["direct_broadband_time_triangle_detection_snr_gt3_count"],
        "triangle=",
        summary["snr_outputs"]["direct_broadband_triangle_detection_snr_gt3_count"],
    )


if __name__ == "__main__":
    main()
