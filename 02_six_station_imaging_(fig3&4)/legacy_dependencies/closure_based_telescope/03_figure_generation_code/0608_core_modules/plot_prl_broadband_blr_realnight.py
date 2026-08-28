from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])

ARRAY_LAT_DEG = float(os.environ.get("ARRAY_LAT_DEG", "-24.627"))
SOURCE_DEC_DEG = float(os.environ.get("SOURCE_DEC_DEG", "2.052388"))
TRANSIT_CENTER_HOUR = float(os.environ.get("TRANSIT_CENTER_HOUR", "0.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "150.0"))
IMAGING_SNR_BOOST = float(os.environ.get("IMAGING_SNR_BOOST", "1.0"))


def realnight_hour_angles(n_sample: int, exposure_s: float, gap_s: float) -> np.ndarray:
    """Exposure-midpoint hour angles for a night centered on meridian transit."""
    cadence_s = exposure_s + gap_s
    total_elapsed_s = n_sample * cadence_s
    first_start_s = -0.5 * total_elapsed_s
    mid_s = first_start_s + exposure_s / 2.0 + cadence_s * np.arange(n_sample)
    return (TRANSIT_CENTER_HOUR + mid_s / 3600.0) * (np.pi / 12.0)


def project_enu_baselines(
    baselines_km: np.ndarray,
    hour_angles_rad: np.ndarray,
    wavelength_m: float,
    *,
    latitude_deg: float,
    declination_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project local east-north baselines onto the source tangent plane.

    The station coordinates in the paper figure are local east/north coordinates.
    We form the source direction in the local ENU frame for each hour angle, then
    project each baseline onto the celestial east and north directions at the
    source.  This produces the usual non-circular Earth-rotation tracks.
    """
    phi = np.deg2rad(latitude_deg)
    dec = np.deg2rad(declination_deg)
    north_pole = np.array([0.0, np.cos(phi), np.sin(phi)])
    baseline_enu_m = np.column_stack(
        [
            baselines_km[:, 0] * 1000.0,
            baselines_km[:, 1] * 1000.0,
            np.zeros(len(baselines_km)),
        ]
    )

    u_rows: list[np.ndarray] = []
    v_rows: list[np.ndarray] = []
    for hour_angle in hour_angles_rad:
        source = np.array(
            [
                -np.cos(dec) * np.sin(hour_angle),
                np.cos(phi) * np.sin(dec) - np.sin(phi) * np.cos(dec) * np.cos(hour_angle),
                np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle),
            ]
        )
        source /= np.linalg.norm(source)
        east_on_sky = np.cross(north_pole, source)
        east_on_sky /= np.linalg.norm(east_on_sky)
        north_on_sky = np.cross(source, east_on_sky)
        north_on_sky /= np.linalg.norm(north_on_sky)

        u_rows.append(baseline_enu_m @ east_on_sky / wavelength_m)
        v_rows.append(baseline_enu_m @ north_on_sky / wavelength_m)

    return np.array(u_rows), np.array(v_rows)


def simulate_measurements_realnight() -> dict[str, np.ndarray | list[tuple[int, int]] | float]:
    rng = np.random.default_rng(273)
    drift_rng = np.random.default_rng(31415)
    n_station = len(opt.STATIONS_KM)
    edges = base.edge_list(n_station)
    baselines_km = np.array([opt.STATIONS_KM[j] - opt.STATIONS_KM[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)

    n_pix = 256
    half_width_uas = 80.0
    fov_rad = 2.0 * half_width_uas * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    hub_distances_km = np.linalg.norm(opt.STATIONS_KM - opt.HUB_KM, axis=1)
    effective_hub_distances_km = base.FIBER_LENGTH_SCALE * hub_distances_km
    station_link_eff = 10.0 ** (-base.FIBER_LOSS_DB_PER_KM * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)

    baseline_link_eff = np.array([np.sqrt(station_link_eff[i] * station_link_eff[j]) for i, j in edges])
    baseline_load_eff = np.array([(station_link_eff[i] + station_link_eff[j]) / 2.0 for i, j in edges])
    baseline_noise_eff = np.array([(station_channel_noise[i] + station_channel_noise[j]) / 2.0 for i, j in edges])
    split_fraction = 1.0 / (n_station - 1.0)
    edge_split_coherence_eff = split_fraction * baseline_link_eff
    edge_split_load_eff = 2.0 * split_fraction * baseline_load_eff
    edge_split_channel_noise = 2.0 * split_fraction * baseline_noise_eff + base.PAIR_FALSE_POSITIVE

    lam_edges = np.linspace(opt.LAMBDA_MIN_NM * 1e-9, opt.LAMBDA_MAX_NM * 1e-9, opt.N_LAMBDA_BINS + 1)
    lam_centers = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    hour_angles = realnight_hour_angles(opt.N_TIME_WINDOWS, opt.EXPOSURE_S, EXPOSURE_GAP_S)
    post_average_drift_std = float(os.environ.get("POST_AVERAGE_DRIFT_STD", str(np.pi / 10.0)))
    station_piston_std = post_average_drift_std / np.sqrt(2.0)

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_uv_radius: list[np.ndarray] = []
    all_vis_split: list[np.ndarray] = []
    all_vis_direct: list[np.ndarray] = []
    all_vis_all: list[np.ndarray] = []
    all_sigma_split_projected: list[np.ndarray] = []
    all_sigma_direct: list[np.ndarray] = []
    all_sigma_all: list[np.ndarray] = []

    for lam, lam_lo, lam_hi in zip(lam_centers, lam_edges[:-1], lam_edges[1:]):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        total_modes = opt.EXPOSURE_S * opt.OBSERVING_DAYS * df
        uu_rows, vv_rows = project_enu_baselines(
            baselines_km,
            hour_angles,
            lam,
            latitude_deg=ARRAY_LAT_DEG,
            declination_deg=SOURCE_DEC_DEG,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            uv_radius = np.sqrt(uu**2 + vv**2)
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            fisher_split = (
                total_modes
                * 4.0
                * (edge_split_coherence_eff * u_mode) ** 2
                * nu_eff**2
                / (edge_split_load_eff * u_mode + edge_split_channel_noise)
            )
            sigma_split = np.minimum(1.0 / np.sqrt(np.maximum(fisher_split, 1e-18)), 2.5)
            sigma_split /= IMAGING_SNR_BOOST
            raw_split_noise = rng.normal(scale=sigma_split)
            noise_split = q_basis @ (q_basis.T @ raw_split_noise)
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            sigma_split_projected = np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0))

            fisher_direct = (
                total_modes
                * base.noisy_closure_fisher_from_station_modes(
                    vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                )
                * closure_rank_share
                * IMAGING_SNR_BOOST**2
            )
            noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)

            station_pistons = drift_rng.normal(scale=station_piston_std, size=n_station)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = raw_split_noise + residual_drift
            sigma_all = np.sqrt(sigma_split**2 + post_average_drift_std**2)

            all_u.append(uu)
            all_v.append(vv)
            all_uv_radius.append(uv_radius)
            all_vis_split.append(amp * np.exp(1j * (phase_closure + noise_split)))
            all_vis_direct.append(amp * np.exp(1j * (phase_closure + noise_direct)))
            all_vis_all.append(amp * np.exp(1j * (phase + noise_all)))
            all_sigma_split_projected.append(sigma_split_projected)
            all_sigma_direct.append(sigma_direct)
            all_sigma_all.append(sigma_all)

    endpoint_coverage = {}
    for wavelength_nm in (opt.LAMBDA_MIN_NM, opt.LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines_km,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=ARRAY_LAT_DEG,
            declination_deg=SOURCE_DEC_DEG,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": uu_rows.reshape(-1),
            "v": vv_rows.reshape(-1),
        }

    return {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
        "uv_radius": np.concatenate(all_uv_radius),
        "vis_all": np.concatenate(all_vis_all),
        "vis_split": np.concatenate(all_vis_split),
        "vis_direct": np.concatenate(all_vis_direct),
        "sigma_all": np.concatenate(all_sigma_all),
        "sigma_split": np.concatenate(all_sigma_split_projected),
        "sigma_direct": np.concatenate(all_sigma_direct),
        "truth": truth,
        "axis_uas": axis_uas,
        "fov_rad": fov_rad,
        "edges": edges,
        "station_link_eff": station_link_eff,
        "effective_hub_distances_km": effective_hub_distances_km,
        "baseline_lengths_km": np.linalg.norm(baselines_km, axis=1),
        "closure_rank_share": closure_rank_share,
        "endpoint_coverage": endpoint_coverage,
        "hour_angles_rad": hour_angles,
    }


def main() -> None:
    opt.simulate_measurements = simulate_measurements_realnight
    opt.main()

    stats_path = opt.OUTDIR / f"prl_broadband_blr_optimized_stats{opt.OUTPUT_SUFFIX}.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        hour_angles = realnight_hour_angles(opt.N_TIME_WINDOWS, opt.EXPOSURE_S, EXPOSURE_GAP_S)
        stats["realnight_projection"] = {
            "array_latitude_deg": ARRAY_LAT_DEG,
            "source_declination_deg": SOURCE_DEC_DEG,
            "source_polar_angle_to_earth_axis_deg": 90.0 - SOURCE_DEC_DEG,
            "transit_center_hour": TRANSIT_CENTER_HOUR,
            "exposure_gap_s": EXPOSURE_GAP_S,
            "cadence_s": opt.EXPOSURE_S + EXPOSURE_GAP_S,
            "elapsed_time_h": opt.N_TIME_WINDOWS * (opt.EXPOSURE_S + EXPOSURE_GAP_S) / 3600.0,
            "integrated_time_h": opt.N_TIME_WINDOWS * opt.EXPOSURE_S / 3600.0,
            "hour_angle_min_h": float(np.min(hour_angles) * 12.0 / np.pi),
            "hour_angle_max_h": float(np.max(hour_angles) * 12.0 / np.pi),
            "projection": "local ENU baselines projected onto celestial east/north tangent basis",
            "imaging_snr_boost": IMAGING_SNR_BOOST,
            "snr_boost_note": "Quantum/readout phase-noise sigmas are divided by this factor; post-average piston drift is not boosted.",
        }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
