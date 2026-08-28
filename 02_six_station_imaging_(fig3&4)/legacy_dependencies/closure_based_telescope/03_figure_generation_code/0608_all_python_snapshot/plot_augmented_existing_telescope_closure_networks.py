from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_prl_broadband_clean as base
from make_chile_optical_zoom_panels import SITES as CHILE_SITES
from make_hawaii_optical_overview_figure import CLUSTER_CENTER, VISIBLE_400_800
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    monochromatic_dirty_image,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)
from plot_prl_broadband_blr_optimized import blr_masks, masked_corr, normalize_blr_display, ring_contrast
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

R_EARTH_KM = 6371.0
SOURCE_DEC_DEG = 2.052388
OBSERVING_DAYS = 30
N_TIME_WINDOWS = 72
EXPOSURE_S = 300.0
EXPOSURE_GAP_S = 150.0
LAMBDA_MIN_NM = 400.0
LAMBDA_MAX_NM = 800.0
LAMBDA_STEP_NM = 10.0
N_PIX = 256
HALF_WIDTH_UAS = 80.0
FIBER_LOSS_DB_PER_KM = 0.20
FIBER_LENGTH_SCALE = 0.75
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("STATION_FALSE_POSITIVE", "0.05")))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", "0.0"))
BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
POST_AVERAGE_DRIFT_STD = float(os.environ.get("POST_AVERAGE_DRIFT_STD_RAD", str(np.pi / 10.0)))
SIGMA_CLIP_RAD = 2.5
SUPPORT_MODE = "ellipse"
RNG_SEED = 20260515


@dataclass(frozen=True)
class Telescope:
    name: str
    x_km: float
    y_km: float
    diameter_m: float
    is_added: bool


@dataclass(frozen=True)
class NetworkCase:
    key: str
    title: str
    latitude_deg: float
    center_latlon: tuple[float, float]
    telescopes: list[Telescope]
    hub_km: tuple[float, float]
    optimization_score: float


def xy_km(lat: float, lon: float, center: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = center
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    y = R_EARTH_KM * math.radians(lat - lat0)
    return x, y


def station_u_modes(freq_hz: float, diameters_m: np.ndarray) -> np.ndarray:
    return np.array(
        [base.source_mode_occupation(freq_hz, diameter_m=float(d)) for d in diameters_m],
        dtype=float,
    )


def noisy_closure_fisher_station_u(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    station_u: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """Per-temporal-mode closure Fisher with station-gauge nuisance removed.

    The raw SLD Fisher matrix is first computed for all oriented edge phases.
    Station piston directions are then eliminated by a Schur complement before
    projecting onto the closure-cycle basis.  This avoids assigning finite
    closure information to a loop when one of its required coherences vanishes.
    """
    n_station = len(station_efficiencies)
    bmat = np.diag(station_efficiencies * station_u + station_noise).astype(complex)
    source_coherences = np.zeros(len(edges), dtype=complex)
    for edge_index, (i, j) in enumerate(edges):
        coherence = (
            math.sqrt(station_efficiencies[i] * station_efficiencies[j] * station_u[i] * station_u[j])
            * visibilities[edge_index]
        )
        source_coherences[edge_index] = coherence
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros_like(bmat, dtype=complex)
        deriv[i, j] = 1j * source_coherences[edge_index]
        deriv[j, i] = -1j * np.conj(source_coherences[edge_index])
        edge_derivatives.append(deriv)
    edge_fisher = base.qfi_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)
    return base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)


def station_table_from_case(case: NetworkCase) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    stations = np.array([[t.x_km, t.y_km] for t in case.telescopes], dtype=float)
    diameters = np.array([t.diameter_m for t in case.telescopes], dtype=float)
    names = [t.name for t in case.telescopes]
    is_added = np.array([t.is_added for t in case.telescopes], dtype=bool)
    return stations, diameters, names, is_added


def optimize_hub(stations: np.ndarray, diameters: np.ndarray) -> tuple[np.ndarray, float]:
    xmin, ymin = np.min(stations, axis=0) - 2.0
    xmax, ymax = np.max(stations, axis=0) + 2.0
    xs = np.linspace(xmin, xmax, 42)
    ys = np.linspace(ymin, ymax, 42)
    weights = diameters**2 / np.sum(diameters**2)
    best_score = -np.inf
    best = np.average(stations, axis=0, weights=weights)
    for x in xs:
        for y in ys:
            hub = np.array([x, y])
            distances = FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
            eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * distances / 10.0)
            score = float(np.sum(weights * np.log(np.maximum(eta, 1e-12))) + 0.22 * np.log(np.min(eta)))
            if score > best_score:
                best_score = score
                best = hub
    return best, best_score


def coverage_score(
    stations: np.ndarray,
    diameters: np.ndarray,
    *,
    latitude_deg: float,
    hub: np.ndarray,
    max_target_g_lambda: float,
) -> float:
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    hour_angles = realnight_hour_angles(N_TIME_WINDOWS, EXPOSURE_S, EXPOSURE_GAP_S)
    uu, vv = project_enu_baselines(
        baselines,
        hour_angles,
        500e-9,
        latitude_deg=latitude_deg,
        declination_deg=SOURCE_DEC_DEG,
    )
    uvg = np.sqrt(uu.reshape(-1) ** 2 + vv.reshape(-1) ** 2) / 1e9
    half_u = float(np.max(np.abs(uu)) / 1e9)
    half_v = float(np.max(np.abs(vv)) / 1e9)
    uv_balance = min(half_u, half_v) / max(max(half_u, half_v), 1e-9)
    v_reach = min(half_v / max_target_g_lambda, 1.0)
    theta = np.mod(np.arctan2(vv.reshape(-1), uu.reshape(-1)), np.pi)
    radial_bins = np.linspace(0.8, max_target_g_lambda, 15)
    angular_bins = np.linspace(0.0, np.pi, 17)
    r_hist = np.histogram(uvg, bins=radial_bins)[0].astype(float)
    a_hist = np.histogram(theta, bins=angular_bins)[0].astype(float)
    r_occ = np.mean(r_hist > 0)
    a_occ = np.mean(a_hist > 0)
    r_entropy = -np.sum((r_hist / max(np.sum(r_hist), 1.0)) * np.log(r_hist / max(np.sum(r_hist), 1.0) + 1e-12))
    a_entropy = -np.sum((a_hist / max(np.sum(a_hist), 1.0)) * np.log(a_hist / max(np.sum(a_hist), 1.0) + 1e-12))
    baseline_lengths = np.linalg.norm(baselines, axis=1)
    short_count = np.sum((baseline_lengths > 0.5) & (baseline_lengths < 4.0))
    mid_count = np.sum((baseline_lengths >= 4.0) & (baseline_lengths < 10.0))
    long_count = np.sum(baseline_lengths >= 10.0)
    distances = FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * distances / 10.0)
    aperture_link = np.sum((diameters**2) * eta) / np.sum(diameters**2)
    return float(
        1.8 * r_occ
        + 1.4 * a_occ
        + 0.25 * r_entropy
        + 0.18 * a_entropy
        + 0.05 * min(short_count, 5)
        + 0.04 * min(mid_count, 8)
        + 0.03 * min(long_count, 10)
        + 1.25 * uv_balance
        + 0.85 * v_reach
        + 0.9 * aperture_link
        - 0.004 * max(np.max(baseline_lengths) - 22.0, 0.0)
    )


def optimize_added_telescopes(
    existing: list[Telescope],
    *,
    n_added: int,
    center: tuple[float, float],
    latitude_deg: float,
    radius_range: tuple[float, float],
    max_target_g_lambda: float,
    rng: np.random.Generator,
    n_trials: int,
) -> NetworkCase:
    existing_positions = np.array([[t.x_km, t.y_km] for t in existing], dtype=float)
    existing_diameters = np.array([t.diameter_m for t in existing], dtype=float)
    origin = np.mean(existing_positions, axis=0)
    best_case: NetworkCase | None = None
    best_score = -np.inf
    for _ in range(n_trials):
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_added))
        if n_added > 1:
            gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
            if np.min(gaps) < np.deg2rad(55.0):
                continue
        radii = radius_range[0] + (radius_range[1] - radius_range[0]) * np.sqrt(
            rng.uniform(0.0, 1.0, size=n_added)
        )
        added_positions = origin + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        stations = np.vstack([existing_positions, added_positions])
        diameters = np.concatenate([existing_diameters, np.full(n_added, 5.0)])
        hub, hub_score = optimize_hub(stations, diameters)
        score = coverage_score(
            stations,
            diameters,
            latitude_deg=latitude_deg,
            hub=hub,
            max_target_g_lambda=max_target_g_lambda,
        )
        score += 0.15 * hub_score
        if score > best_score:
            telescopes = list(existing)
            for idx, pos in enumerate(added_positions, start=1):
                telescopes.append(Telescope(f"new 5 m {idx}", float(pos[0]), float(pos[1]), 5.0, True))
            best_case = NetworkCase(
                key="candidate",
                title="candidate",
                latitude_deg=latitude_deg,
                center_latlon=center,
                telescopes=telescopes,
                hub_km=(float(hub[0]), float(hub[1])),
                optimization_score=float(score),
            )
            best_score = score
    assert best_case is not None
    return best_case


def make_maunakea_case(rng: np.random.Generator) -> NetworkCase:
    center = CLUSTER_CENTER["Maunakea"]
    apertures = {
        "Keck I": 10.0,
        "Keck II": 10.0,
        "Subaru": 8.2,
        "Gemini North": 8.1,
        "CFHT": 3.6,
    }
    existing = []
    for name, lat, lon, _, cluster in VISIBLE_400_800:
        if cluster != "Maunakea" or name not in apertures:
            continue
        x, y = xy_km(lat, lon, center)
        existing.append(Telescope(name, x, y, apertures[name], False))
    case = optimize_added_telescopes(
        existing,
        n_added=3,
        center=center,
        latitude_deg=center[0],
        radius_range=(5.0, 10.0),
        max_target_g_lambda=32.0,
        rng=rng,
        n_trials=1500,
    )
    return NetworkCase(
        key="maunakea_augmented",
        title="Maunakea optical core + three 5 m outstations",
        latitude_deg=center[0],
        center_latlon=center,
        telescopes=case.telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def make_ctio_case(rng: np.random.Generator) -> NetworkCase:
    wanted = {
        "CTIO / Blanco": 4.0,
        "Gemini South": 8.1,
        "SOAR": 4.1,
        "Rubin / El Penon": 6.7,
    }
    rows = [row for row in CHILE_SITES if row[0] in wanted]
    center = (
        sum(row[1] for row in rows) / len(rows),
        sum(row[2] for row in rows) / len(rows),
    )
    existing = []
    for name, lat, lon, _ in rows:
        x, y = xy_km(lat, lon, center)
        existing.append(Telescope(name.replace(" / El Penon", ""), x, y, wanted[name], False))
    case = optimize_added_telescopes(
        existing,
        n_added=2,
        center=center,
        latitude_deg=center[0],
        radius_range=(5.0, 10.0),
        max_target_g_lambda=45.0,
        rng=rng,
        n_trials=1500,
    )
    return NetworkCase(
        key="ctio_augmented",
        title="CTIO/Pachon/Rubin core + two 5 m outstations",
        latitude_deg=center[0],
        center_latlon=center,
        telescopes=case.telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def simulate_case(case: NetworkCase) -> tuple[dict, dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED + (17 if "ctio" in case.key else 0))
    drift_rng = np.random.default_rng(RNG_SEED + (101 if "ctio" in case.key else 37))
    stations, diameters, names, is_added = station_table_from_case(case)
    hub = np.array(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)
    split_fraction = 1.0 / (n_station - 1.0)

    fov_rad = 2.0 * HALF_WIDTH_UAS * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(N_PIX, HALF_WIDTH_UAS)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    hub_dist = np.linalg.norm(stations - hub, axis=1)
    effective_hub_dist = FIBER_LENGTH_SCALE * hub_dist
    station_eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full_like(station_eta, MODE_FALSE_POSITIVE)
    hour_angles = realnight_hour_angles(N_TIME_WINDOWS, EXPOSURE_S, EXPOSURE_GAP_S)
    station_piston_std = POST_AVERAGE_DRIFT_STD / np.sqrt(2.0)

    lam_edges_nm = np.arange(LAMBDA_MIN_NM, LAMBDA_MAX_NM + 0.5 * LAMBDA_STEP_NM, LAMBDA_STEP_NM)
    lam_edges_nm[-1] = LAMBDA_MAX_NM
    stacks = {key: np.zeros((N_PIX, N_PIX), dtype=float) for key in ("all", "split", "direct")}
    stack_weights = {key: 0.0 for key in ("all", "split", "direct")}
    sigma_history = {key: [] for key in ("all", "split", "direct")}

    endpoint_coverage = {}
    for wavelength_nm in (LAMBDA_MIN_NM, LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=SOURCE_DEC_DEG,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1e9).tolist(),
        }

    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        lam = math.sqrt(lo_nm * hi_nm) * 1e-9
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        df = freq_hi - freq_lo
        total_modes = EXPOSURE_S * OBSERVING_DAYS * df
        u_station = station_u_modes(freq, diameters)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=SOURCE_DEC_DEG,
        )

        all_u: list[np.ndarray] = []
        all_v: list[np.ndarray] = []
        vis = {key: [] for key in ("all", "split", "direct")}
        sig = {key: [] for key in ("all", "split", "direct")}
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            fisher_split = np.zeros(len(edges), dtype=float)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * math.sqrt(station_eta[i] * station_eta[j] * u_station[i] * u_station[j])
                load = split_fraction * (
                    station_eta[i] * u_station[i]
                    + station_eta[j] * u_station[j]
                    + station_noise[i]
                    + station_noise[j]
                ) + PAIR_FALSE_POSITIVE
                fisher_split[edge_index] = total_modes * 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-30)
            sigma_split = np.minimum(1.0 / np.sqrt(np.maximum(fisher_split, 1e-18)), SIGMA_CLIP_RAD)
            raw_split_noise = rng.normal(scale=sigma_split)
            noise_split = q_basis @ (q_basis.T @ raw_split_noise)
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            sigma_split_projected = np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0))

            fisher_direct = (
                total_modes
                * noisy_closure_fisher_station_u(vtrue, station_eta, station_noise, u_station, q_basis, edges)
                * closure_rank_share
            )
            noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)

            station_pistons = drift_rng.normal(scale=station_piston_std, size=n_station)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = raw_split_noise + residual_drift
            sigma_all = np.sqrt(sigma_split**2 + POST_AVERAGE_DRIFT_STD**2)

            all_u.append(uu)
            all_v.append(vv)
            vis["all"].append(amp * np.exp(1j * (phase + noise_all)))
            vis["split"].append(amp * np.exp(1j * (phase_closure + noise_split)))
            vis["direct"].append(amp * np.exp(1j * (phase_closure + noise_direct)))
            sig["all"].append(sigma_all)
            sig["split"].append(sigma_split_projected)
            sig["direct"].append(sigma_direct)

        band = {"u": np.concatenate(all_u), "v": np.concatenate(all_v)}
        for key in ("all", "split", "direct"):
            band[f"vis_{key}"] = np.concatenate(vis[key])
            band[f"sigma_{key}"] = np.concatenate(sig[key])
            sigma_history[key].append(float(np.median(band[f"sigma_{key}"])))

        _, occupied, _ = aggregate_cells(
            band["u"],
            band["v"],
            band["vis_direct"],
            band["sigma_direct"],
            n=N_PIX,
            fov_rad=fov_rad,
            average_mode="noise",
        )
        support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=SUPPORT_MODE)
        label_y, label_x, fillable = nearest_label_map(occupied, support)
        for key in ("all", "split", "direct"):
            image, weight = monochromatic_dirty_image(
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
            stacks[key] += weight * image
            stack_weights[key] += weight

    for key in stacks:
        stacks[key] = normalize_stack(stacks[key] / max(stack_weights[key], 1e-30))

    ring_mask, core_mask = blr_masks(axis_uas)
    metrics = {
        key: {
            "global_corr": float(base.corrcoef_positive(truth, stacks[key])),
            "blr_corr": float(masked_corr(truth, stacks[key], ring_mask)),
            "ring_contrast": float(ring_contrast(stacks[key], ring_mask, core_mask)),
            "median_phase_sigma_rad": float(np.median(sigma_history[key])),
        }
        for key in ("all", "split", "direct")
    }
    baseline_lengths = np.linalg.norm(baselines, axis=1)
    stats = {
        "case": case.key,
        "title": case.title,
        "latitude_deg": case.latitude_deg,
        "center_latlon": list(case.center_latlon),
        "optimization_score": case.optimization_score,
        "observing_days": OBSERVING_DAYS,
        "n_time_windows_per_night": N_TIME_WINDOWS,
        "exposure_s": EXPOSURE_S,
        "exposure_gap_s": EXPOSURE_GAP_S,
        "wavelength_min_nm": LAMBDA_MIN_NM,
        "wavelength_max_nm": LAMBDA_MAX_NM,
        "wavelength_step_nm": LAMBDA_STEP_NM,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "fiber_length_scale": FIBER_LENGTH_SCALE,
        "post_average_drift_std_rad": float(POST_AVERAGE_DRIFT_STD),
        "hub_km": list(case.hub_km),
        "stations": [
            {
                "name": names[i],
                "x_km": float(stations[i, 0]),
                "y_km": float(stations[i, 1]),
                "diameter_m": float(diameters[i]),
                "is_added": bool(is_added[i]),
                "hub_distance_km": float(hub_dist[i]),
                "link_efficiency": float(station_eta[i]),
            }
            for i in range(n_station)
        ],
        "n_station": n_station,
        "n_baseline": len(edges),
        "n_closure": int(n_closure),
        "closure_rank_share": float(closure_rank_share),
        "baseline_min_km": float(np.min(baseline_lengths)),
        "baseline_median_km": float(np.median(baseline_lengths)),
        "baseline_max_km": float(np.max(baseline_lengths)),
        "effective_hub_distance_min_km": float(np.min(effective_hub_dist)),
        "effective_hub_distance_max_km": float(np.max(effective_hub_dist)),
        "station_link_eff_min": float(np.min(station_eta)),
        "station_link_eff_max": float(np.max(station_eta)),
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "coverage_400nm_half_range_g_lambda": {
            "u": float(np.max(np.abs(endpoint_coverage["400"]["u"]))),
            "v": float(np.max(np.abs(endpoint_coverage["400"]["v"]))),
        },
        "coverage_800nm_half_range_g_lambda": {
            "u": float(np.max(np.abs(endpoint_coverage["800"]["u"]))),
            "v": float(np.max(np.abs(endpoint_coverage["800"]["v"]))),
        },
        "metrics": metrics,
    }
    return stats, {"images": stacks}, truth, axis_uas


def plot_case(case: NetworkCase, stats: dict, image_pack: dict, truth: np.ndarray, axis_uas: np.ndarray) -> tuple[Path, Path]:
    stations, diameters, names, is_added = station_table_from_case(case)
    stacks = image_pack["images"]
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )
    fig = plt.figure(figsize=(7.55, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.38, wspace=0.34)

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
        ax.text(stations[i, 0] + 0.16, stations[i, 1] + 0.16, f"S{i+1}\n{diameters[i]:g}m", fontsize=5.6)
    for i, j in base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.82", lw=0.45, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("stations and optimized hub")
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
    ax.imshow(normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Input 3C273 model")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    labels = {
        "all": "All visibilities + drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, key in enumerate(("all", "split", "direct")):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(normalize_blr_display(stacks[key]), origin="lower", extent=extent, cmap="inferno")
        metric = stats["metrics"][key]
        ax.set_title(f"{labels[key]}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}")
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

    fig.suptitle(case.title, fontsize=10.4, weight="bold", y=0.995)
    png = OUTFIG / f"augmented_existing_telescope_{case.key}.png"
    pdf = OUTFIG / f"augmented_existing_telescope_{case.key}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    cases = [make_maunakea_case(rng), make_ctio_case(rng)]
    summary = {}
    for case in cases:
        print(f"simulating {case.key}")
        stats, image_pack, truth, axis_uas = simulate_case(case)
        pdf, png = plot_case(case, stats, image_pack, truth, axis_uas)
        stats["figure_pdf"] = str(pdf)
        stats["figure_png"] = str(png)
        stats_path = OUTFIG / f"augmented_existing_telescope_{case.key}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        summary[case.key] = stats
        print(pdf)
        print(png)
        print(stats_path)
        print(json.dumps(stats["metrics"], indent=2))
    summary_path = OUTFIG / "augmented_existing_telescope_closure_networks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
